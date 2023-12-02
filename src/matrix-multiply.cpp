#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>

using namespace std;

// 函数：解析矩阵文本
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


vector<vector<double>> MatrixMultiply(vector<vector<double>>& matrix_a, vector<vector<double>>& matrix_b){
    unsigned long long dimension_1 = matrix_a.size(), dimension_2 = matrix_a[0].size(), dimension_3 = matrix_b[0].size();

    vector<vector<double>> matrix_c(dimension_1, vector<double>(dimension_3, 0));

    if(matrix_b.size() != dimension_2){
        cout << "矩阵无法相乘！" << endl;
        return matrix_c;
    }

    for(unsigned long long i = 0; i < dimension_1; i++){
        for(unsigned long long j = 0; j < dimension_3; j++){
            for(unsigned long long k = 0; k < dimension_2; k++){
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }

    return matrix_c;
}

vector<vector<double>> MatrixMultiply(vector<vector<vector<double>>>& matrices){
    vector<vector<double>> matrix_ans;

    if(matrices.empty()){
        return matrix_ans;
    }else if(matrices.size() == 1){
        return matrices[0];
    }

    matrix_ans = MatrixMultiply(matrices[0], matrices[1]);

    for(int i = 2; i < matrices.size(); i++){
        matrix_ans = MatrixMultiply(matrix_ans, matrices[i]);
    }

    return matrix_ans;
}

int main(){
    string filename = "../data/problem-1.txt";
    vector<vector<vector<double>>> matrices = ParseMatrixFile(filename);

    clock_t begin=clock();
    vector<vector<double>> matrix_ans = MatrixMultiply(matrices);

    clock_t end=clock();
    double t = double(end-begin)/CLOCKS_PER_SEC;
    cout<<"RUN_TIME:"<<t<<"\n";

    ofstream outputFile("../data/problem-1-ans.txt");
    // 检查文件是否成功打开
    if (outputFile.is_open()) {
        outputFile << "Dimension 0: " << matrix_ans.size() << endl;
        outputFile << "Dimension 1: " << matrix_ans[0].size() << endl;
        outputFile << "Matrix 0:" << endl;
        // 遍历矩阵并将值写入文件
        for (const auto& row : matrix_ans) {
            for (const auto& value : row) {
                outputFile << value << " ";
            }
            outputFile << endl;
        }

        // 关闭文件
        outputFile.close();
    } else {
        cout << "无法打开输出文件" << endl;
    }

    return 0;
}