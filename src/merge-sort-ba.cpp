#include <sycl/sycl.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
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

vector<double> GetData(const string& filename){
    ifstream file(filename);
    string line;

    vector<double> nums;

    while(getline(file, line)){
        istringstream iss(line);
        double value;
        while(iss >> value){
            nums.push_back(value);
        }
    }

    return nums;
}

void MergeSort(queue& q, vector<double>& nums){
    unsigned long long size = nums.size();
    unsigned long long step = 1;
    vector<double> temp;
    temp.insert(temp.begin(), nums.begin(), nums.end());
    while (step < size){
        unsigned long long pairs = (size / step) / 2;
        sycl::buffer<double, 1> buffer_a(nums.data(), range<1>(size));
        sycl::buffer<double, 1> buffer_b(temp.data(), range<1>(size));

        range<1> num_items{pairs + 1};

        // 提交 SYCL 任务
        q.submit([&](handler &h) {
            accessor acc_num (buffer_a,h,read_only);
            accessor acc_tmp (buffer_b,h,write_only);

            h.parallel_for(num_items, [=](auto i) {
                unsigned long long index_tmp;
                unsigned long long index1, end1, index2, end2;
                index1 = i * 2 * step, end1 = index1 + step;
                if(end1 < size){
                    index2 = end1, end2 = index2 + step < size ? index2 + step : size;
                    index_tmp = index1;
                    while(index1 < end1 && index2 < end2){
                        if(acc_num[index1] < acc_num[index2]){
                            acc_tmp[index_tmp] = acc_num[index1];
                            index1++;
                        }else{
                            acc_tmp[index_tmp] = acc_num[index2];
                            index2++;
                        }
                        index_tmp++;
                    }
                    while(index1 < end1){
                        acc_tmp[index_tmp] = acc_num[index1];
                        index1++;
                        index_tmp++;
                    }
                    while(index2 < end2){
                        acc_tmp[index_tmp] = acc_num[index2];
                        index2++;
                        index_tmp++;
                    }
                }
            });
        });

        // 等待 SYCL 任务完成
        q.wait_and_throw();

        step *= 2;

        host_accessor h_a(buffer_b,read_only);
        for(int i = 0; i < size; i++){
            nums[i] = temp[i];
        }
    }
}

int main() {
    string vendor_name = "Intel";
    // string vendor_name = "AMD";
    // string vendor_name = "Nvidia";
    MyDeviceSelector selector(vendor_name);
    queue q(selector);

    string filename = "../data/problem-2.txt";
    vector<double> nums = GetData(filename);

    clock_t begin=clock();
    MergeSort(q, nums);
    clock_t end=clock();
    double t = double(end-begin)/CLOCKS_PER_SEC;
    cout<<"RUN_TIME:"<<t<<"\n";

    ofstream outputFile("../data/problem-2-ba.txt");
    // 检查文件是否成功打开
    if (outputFile.is_open()) {
        for(auto num: nums){
            outputFile << num << " ";
        }

        // 关闭文件
        outputFile.close();
    } else {
        cout << "无法打开输出文件\n";
    }

    return 0;
}