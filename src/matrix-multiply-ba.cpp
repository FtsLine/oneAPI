#include <sycl/sycl.hpp>
#include <vector>
#include <ctime>

using namespace sycl;
using namespace std;

class MyDeviceSelector {
public:
    MyDeviceSelector(string vendorName) : vendorName_(vendorName){};
    int operator()(const device &dev) {
        int rating = 0;
        if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != string::npos))
            rating = 3;
        else if (dev.is_gpu()) rating = 2;
        else if (dev.is_cpu()) rating = 1;
        return rating;
    };

private:
    string vendorName_;
};

vector<vector<vector<double>>> ParseMatrixFile(const string& filename){
    ifstream file(filename);
    vector<vector<vector<double>>> matrices;
    vector<vector<double>> matrix;
    string line;

    while (getline(file, line)){
        // 检查是否为维度行
        if (line.find("Dimension") != string::npos)
            continue;

        // 检查是否为矩阵行
        if (line.find("Matrix") != string::npos){
            if (!matrix.empty()){
                matrices.push_back(matrix);
                matrix.clear();
            }
            continue;
        }

        // 解析矩阵数据行
        istringstream iss(line);
        double value;
        vector<double> row;
        while (iss >> value){
            row.push_back(value);
        }
        if(!row.empty()){
            matrix.push_back(row);
        }
    }

    // 添加最后一个矩阵
    if (!matrix.empty()){
        matrices.push_back(matrix);
    }

    return matrices;
}

vector<vector<double>> MatrixMultiply(queue&q, vector<vector<double>>& matrix_a, vector<vector<double>>& matrix_b){
    unsigned long long dimension_1 = matrix_a.size(), dimension_2 = matrix_a[0].size(), dimension_3 = matrix_b[0].size();

    vector<vector<double>> matrix_c(dimension_1, vector<double>(dimension_3, 0));

    if (matrix_b.size() != dimension_2) {
        cout << "矩阵无法相乘！\n";
        return matrix_c;
    }

    vector<double> flat_data_a;
    for (const auto& row : matrix_a) {
        flat_data_a.insert(flat_data_a.end(), row.begin(), row.end());
    }
    vector<double> flat_data_b;
    for (const auto& row : matrix_b) {
        flat_data_b.insert(flat_data_b.end(), row.begin(), row.end());
    }
    vector<double> flat_data_c;
    for (const auto& row : matrix_c) {
        flat_data_c.insert(flat_data_c.end(), row.begin(), row.end());
    }

    try {
        // 创建设备访问器
        sycl::buffer<double, 2> buffer_a(flat_data_a.data(), range<2>(dimension_1, dimension_2));
        sycl::buffer<double, 2> buffer_b(flat_data_b.data(), range<2>(dimension_2, dimension_3));
        sycl::buffer<double, 2> buffer_c(flat_data_c.data(), range<2>(dimension_1, dimension_3));

        // 提交 SYCL 任务
        q.submit([&](handler &h) {
            accessor MA (buffer_a,h,read_only);
            accessor MB (buffer_b,h,read_only);
            accessor MC (buffer_c,h,write_only);

            h.parallel_for(nd_range<2>({dimension_1, dimension_3}, {64, 64}), [=](nd_item<2> item) {
                //# Multiplication
                size_t row = item.get_global_id(0);
                size_t col = item.get_global_id(1);
                for (size_t k = 0; k < dimension_2; ++k) {
                    MC[row][col] += MA[row][k] * MB[k][col];
                }
            });
        });

        // 等待 SYCL 任务完成
        q.wait_and_throw();

        //# Create a host accessor to copy data from device to host
        host_accessor h_a(buffer_c,read_only);
        for(unsigned long long i = 0; i < dimension_1; i++){
            for(unsigned long long j = 0; j < dimension_3; j++){
                matrix_c[i][j] = flat_data_c[i * dimension_3 + j];
            }
        }
    } catch (const sycl::exception& e) {
        cout << "SYCL 错误: " << e.what() << std::endl;
    }

    return matrix_c;
}

vector<vector<double>> MatricsMultiply(queue& q, vector<vector<vector<double>>>& matrices){
    vector<vector<double>> matrix_ans;

    if(matrices.empty()){
        return matrix_ans;
    }else if(matrices.size() == 1){
        return matrices[0];
    }

    matrix_ans = MatrixMultiply(q, matrices[0], matrices[1]);

    for(int i = 2; i < matrices.size(); i++){
        matrix_ans = MatrixMultiply(q, matrix_ans, matrices[i]);
    }

    return matrix_ans;
}

int main() {
    string vendor_name = "Intel";
    // string vendor_name = "AMD";
    // string vendor_name = "Nvidia";
    //# Submit task to multiply matrices
    MyDeviceSelector selector(vendor_name);
    queue q(selector);

    string filename = "../data/problem-1.txt";
    vector<vector<vector<double>>> matrices = ParseMatrixFile(filename);

    clock_t begin=clock();
    // 对每个矩阵进行并行运算
    vector<vector<double>> matrix_ans = MatricsMultiply(q, matrices);

    clock_t end=clock();
    double t = double(end-begin)/CLOCKS_PER_SEC;
    cout<<"RUN_TIME:"<<t<<"\n";

    ofstream outputFile("../data/problem-1-ba.txt");
    // 检查文件是否成功打开
    if (outputFile.is_open()) {
        outputFile << "Dimension 0: " << matrix_ans.size() << std::endl;
        outputFile << "Dimension 1: " << matrix_ans[0].size() << std::endl;
        outputFile << "Matrix 0:" << std::endl;
        // 遍历矩阵并将值写入文件
        for (const auto& row : matrix_ans) {
            for (const auto& value : row) {
                outputFile << value << " ";
            }
            outputFile << std::endl;
        }

        // 关闭文件
        outputFile.close();
    } else {
        cout << "无法打开输出文件" << std::endl;
    }

    return 0;
}