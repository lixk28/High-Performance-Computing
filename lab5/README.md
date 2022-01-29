任务一和任务三代码在 `fft` 下，任务二代码在 `heated-plate-mpi`。

编译测试方法：

- `fft`

  - 编译：

    ```shell
    mkdir bin && make
    ```

  - 测试：

    ```shell
    make test
    ```

    误差、运行时间和 MFLOPS 文件生成在 `asset` 下。

    ```shell
    make memory-test
    ```

    使用 valgrind 的 massif 工具对 `fft_pf` 和 `fft_openmp` 执行内存测试。

    ```shell
    make print
    ```

    打印 massif 生成的文件。

- `heated-plate-mpi`

  - 编译：

    ```shell
    mkdir bin && make
    ```

  - 测试：

    ```shell
    make test
    ```

    默认开启最大线程数和进程数，分别执行 openmp 版本和 mpi 版本。