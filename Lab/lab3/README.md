目录结构：

```
.
├── mpi-table
├── pthread-array-sum
├── pthread-integral
├── pthread-mat-mul
├── pthread-root
├── README.pdf
└── report.pdf
```

- mpi-table 存放用于统计 MPI 和软件优化版本的矩阵乘法耗时的代码，以及生成的结果 .svg 图片
- pthread-array-sum 是 pthread 数组求和
- pthread-integral 是多线程 Monte-Carlo 方法估算积分的值
- pthread-mat-mul 是 pthread 矩阵乘法
- pthread-root 是条件变量计算一元二次方程的根

编译运行方法：

- mpi-table

  - 编译：

    ```shell
    cd mpi-table
    mkdir build && mkdir bin
    make
    ```

  - 测试：

    ```shell
    make serial 	# 朴素矩阵乘法
    make p2p			#	MPI 点对点通信
    make collect	#	MPI 集合通信
    make optmm		#	软件优化矩阵乘法
    ```

    各个版本耗时（对应不同进程数、矩阵规模）会输出在终端。

- pthread-mat-mul

  - 编译：

    ```shell
    cd pthread-mat-mul
    mkdir build && mkdir bin
    make
    ```

  - 测试：

    ```shell
    make test
    ```

    线程数从 1 到 8，矩阵规模 512、1024、1536、2048，耗时会在子目录 `asset` 下保存在文件中。文件名为 `time-<thread>`，`<thread>` 表示线程数，文件每行是从各个规模的串行时间、并行时间。

    ```shell
    make plot
    ```

    执行作图脚本，统计数据并画出在线程数一定时，运行时间关于矩阵规模的折线图，保存在 `asset` 下，文件名为 `performance-<thread>.svg`，`<thread>` 表示对应的线程数。

- pthread-array-sum

  - 编译：

    ```shell
    cd pthread-array-sum
    gcc pthread-array-sum.c -o test -lpthread 
    ```

  - 测试：

    ```shell
    ./test
    ```

- pthread-root

  - 编译：

    ```shell
    cd pthread-root
    gcc pthread-root.c -o test -lpthread -lm
    ```

  - 测试：

    ```shell
    ./test <a> <b> <c>
    ```

    注意 a、b、c 要保证 delta 的值大于等于 0，且 a 不为 0。

- pthread-integral

  - 编译：

    ```shell
    cd pthread-integral
    gcc pthread-integral -o test -lpthread
    ```

  - 运行：

    ```shell
    ./test <number_toss> <thread_count>
    ```

    number_toss 是总的模拟次数，thread_count 是开启的线程数。