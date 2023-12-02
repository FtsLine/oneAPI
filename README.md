
## Inter oneAPI 简介

Intel oneAPI是 Intel 推出的一个开放、统一的编程模型和软件开发工具套件。

### DPC++

DPC++ 是 Inter 基于 C++ 编写的跨架构编程语言，由一组 C++ 类、模板与库组成，同时兼容 SYCL 规范。

### SYCL

SYCL 是一种用于高性能计算的开放标准，其目标是提供一个用于异构计算的统一编程模型，使开发人员能够在不同类型的处理器上编写可移植的并行代码，包括多核CPU、GPU和FPGA等。

## 编程环境

使用 Inter oneAPI Developer Cloud 提供的 [JupyterLab](https://jupyter.oneapi.devcloud.intel.com) 编辑代码并运行。提供了代码预览与编辑，Linux 终端等功能。

## 矩阵乘法

### Buffer-Accessor Memory Model

核心代码是，使用 Buffer Accessor 来访问矩阵数据并行计算。

```cpp
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
```

### Unified Shared Memory

核心代码是，设置一块新的共享内存用来存放矩阵。

```cpp
try {
	//# USM allocation using malloc_shared
	double *MA = malloc_shared<double>(flat_data_a.size(), q);
	memcpy(MA, flat_data_a.data(), sizeof(double) * flat_data_a.size());
	double *MB = malloc_shared<double>(flat_data_b.size(), q);
	memcpy(MB, flat_data_b.data(), sizeof(double) * flat_data_b.size());
	double *MC = malloc_shared<double>(flat_data_c.size(), q);
	memcpy(MC, flat_data_c.data(), sizeof(double) * flat_data_c.size());

	q.parallel_for(nd_range<2>({dimension_1, dimension_3}, {64, 64}), [=](nd_item<2> item) {
	  //# Multiplication
	  size_t row = item.get_global_id(0);
	  size_t col = item.get_global_id(1);
	  for (size_t k = 0; k < dimension_2; ++k) {
		  MC[row * dimension_3 + col] += MA[row * dimension_2 + k] * MB[k * dimension_3 + col];
	  }
	}).wait();

	// 将数据从 MC 逐元素复制到 matrix_c
	for (size_t i = 0; i < dimension_1; ++i) {
		for (size_t j = 0; j < dimension_3; ++j) {
			matrix_c[i][j] = MC[i * dimension_3 + j];
		}
	}
} catch (const sycl::exception& e) {
	cout << "SYCL 错误: " << e.what() << std::endl;
}
```
### 数据

一个含有 10 个矩阵的文本文件，这 10 个矩阵可以连续相乘。

### 运行

```shell
$ icpx -fsycl matrix-multiply-ba.cpp -o run-ba
$ ./run-ba
RUN_TIME:0.155032
$ icpx -fsycl matrix-multiply-um.cpp -o run-um
$ ./run-um
RUN_TIME:0.141436
```
### 结果

得到的结果与串行计算的结果相比较，结果一致说明并行计算正确。

## 归并排序

### 串行实现

```cpp
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
```

每次合并两个有序数组的时候，剩余其他数组也可以进行合并，所以可以并行执行。

### oneAPI 实现

```cpp
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
```

其作用是将归并排序里，并行地实现多组两个有序数组的合并。