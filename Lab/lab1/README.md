项目目录：

- asset：各个规模矩阵下的运行时间记录文件
- bin：生成的二进制 test 文件，可直接运行
- build：生成的中间目标文件
- include：头文件
- src：源文件
- script：python 脚本，统计运行时间并作图

```.
├── asset
│   ├── elapsed_time_100
│   ├── elapsed_time_1000
│   ├── elapsed_time_1200
│   ├── elapsed_time_1500
│   ├── elapsed_time_1600
│   ├── elapsed_time_1800
│   ├── elapsed_time_200
│   ├── elapsed_time_2000
│   ├── elapsed_time_300
│   ├── elapsed_time_400
│   ├── elapsed_time_500
│   ├── elapsed_time_600
│   ├── elapsed_time_700
│   ├── elapsed_time_800
│   └── performance.png
├── bin
│   └── test
├── build
│   ├── main.o
│   ├── Matrix_Mul.o
│   └── Matrix.o
├── include
│   ├── Matrix_Mul.h
│   └── Matrix.h
├── script
│   └── plot.py
├── src
│   ├── main.cpp
│   ├── Matrix_Mul.cpp
│   └── Matrix.cpp
├── Makefile
├── README.pdf
└── report.pdf
```

注意，test 可执行文件可直接运行，如果要编译链接，请注意按照电脑的环境配置修改 Makefile 中 Intel MKL 路径的配置。程序在 Ubuntu 20.04 下测试，使用的是 apt 仓库的 intel mkl。**如果要在 Ubuntu 20.04 下测试，请先执行 `sudo apt install intel-mkl`，然后在项目目录下 `make` 即可。**

