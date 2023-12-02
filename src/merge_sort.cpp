#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>

using namespace std;

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

void MergeSort(vector<double>& nums){
    unsigned long long size = nums.size();
    unsigned long long step = 1;
    unsigned long long index1, end1, index2, end2;
    unsigned long long index_tmp;
    vector<double> temp;
    temp.insert(temp.begin(), nums.begin(), nums.end());
    while (step < size){
        unsigned long long pairs = (size / step) / 2;
        for(int i = 0; i <= pairs; i++){
            index1 = i * 2 * step, end1 = index1 + step;
            if(end1 >= size){
                break;
            }
            index2 = end1, end2 = index2 + step < size ? index2 + step : size;
            index_tmp = index1;
            while(index1 < end1 && index2 < end2){
                if(nums[index1] < nums[index2]){
                    temp[index_tmp] = nums[index1];
                    index1++;
                }else{
                    temp[index_tmp] = nums[index2];
                    index2++;
                }
                index_tmp++;
            }
            while(index1 < end1){
                temp[index_tmp] = nums[index1];
                index1++;
                index_tmp++;
            }
            while(index2 < end2){
                temp[index_tmp] = nums[index2];
                index2++;
                index_tmp++;
            }
        }
        step *= 2;
        for(int i = 0; i < size; i++){
            nums[i] = temp[i];
        }
    }
}

int main(){
    string filename = "../data/problem-2.txt";

    vector<double> nums = GetData(filename);

    clock_t begin=clock();
    MergeSort(nums);
    clock_t end=clock();
    double t = double(end-begin)/CLOCKS_PER_SEC;
    cout<<"RUN_TIME:"<<t<<"\n";

    ofstream outputFile("../data/problem-2-ans.txt");
    // 检查文件是否成功打开
    if (outputFile.is_open()) {
        for(auto num: nums){
            outputFile << num << " ";
        }

        // 关闭文件
        outputFile.close();
    } else {
        cout << "无法打开输出文件" << endl;
    }

    return 0;
}