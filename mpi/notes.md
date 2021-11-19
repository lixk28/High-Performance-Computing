



## Collective Communication

### MPI_Bcast



### MPI_Scatter

函数原型：

```C
int MPI_Scatter(
  const void *	sendbuf, 			// in
  int 					sendcount, 		// in
  MPI_Datatype 	sendtype,			// in
  void *				recvbuf, 			// out
  int 					recvcount, 		// in
  MPI_Datatype 	recvtype, 		// in
  int 					root,					// in
  MPI_Comm 			comm					// in
)
```

参数：

- 输入参数：
  - sendbuf: 发送缓冲区的起始地址
  - sendcount: 发送给每个进程的数据量
  - sendtype: 发送缓冲区的数据类型
  - recvcount: 每个进程接收到的数据量
  - recvtype: 接收缓冲区的数据类型
  - root: 执行发送操作的进程号
  - comm: 通信子
- 输出参数：
  - recvbuf: 接收缓冲区的起始地址

功能：

`MPI_Scatter` 函数的功能类似于 root 进程执行了 n 次 `MPI_Send` 操作，`MPI_Send(sendbuf + i * sendcount * sizeof(sendtype), sendcount, sendtype, i, ...)`，并且每个进程都执行 1 个 `MPI_Recv` 操作，`MPI_Recv(recvbuf, recvcount, recvtype, i, ...)`。

但是，`MPI_Scatter` 函数比这种通信方式高效得多，它采用树形通信结构，在进程数很大时，通信耗时会比上面的方式少得多。

注意到，对于 root 进程来说，所有的参数都是必要的，因为由它执行发送操作，而且 root 进程自己也要接受数据。但对于其他的进程，只有 `recvbuf`，`recvcount`，`recvtype`，`root`，`comm` 这些参数是必要的，因为它们只需要接受数据。

也就是说，对于非 root 进程，`sendbuf`，`sendcount`，`sendtype` 这 3 个参数的值可以随便取，`sendbuf` 的值可以设置为 `NULL`,其他两个最好与 root 进程保持一致。`sendbuf` 最好由 root 进程创建，其他进程无法获取到。一般来说，`sendcount` 和 `recvcount` 的值是一样的。`root` 和 `comm` 参数的值对于所有的进程应保持一致。

### MPI_Gather

函数原型：

```C
int MPI_Gather(
  const void *	sendbuf, 			// in
  int 					sendcount, 		// in
  MPI_Datatype 	sendtype,			// in
  void *				recvbuf, 			// out
  int 					recvcount, 		// in
  MPI_Datatype 	recvtype, 		// in
  int 					root,					// in
  MPI_Comm 			comm					// in
)
```

参数：

- 输入参数：
  - sendbuf: 发送缓冲区的起始地址
  - sendcount: 发送给 root 进程的数据量
  - sendtype: 发送缓冲区的数据类型
  - recvcount: root 进程接收每个进程的数据量
  - recvtype: 接收缓冲区的数据类型
  - root: 执行接收操作的进程号
- 输出参数：
  - recvbuf: 接收缓冲区的起始地址

功能：

`MPI_Gather` 函数是 `MPI_Scatter` 函数的逆操作。

