#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <string.h>

// #define DEBUG

#ifdef DEBUG
  #include <unistd.h>
#endif

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define M 500
#define N 500

void my_bound(int start, int end, int comm_sz, int my_rank, int *my_first_m, int *my_last_m, int *my_m)
{
  if (my_rank == comm_sz - 1) // if I'm the last process
  {
    *my_first_m = start + (end - start) / comm_sz * my_rank;
    *my_last_m = end;  // set my_last_m as M-1
  }
  else
  {
    *my_first_m = start + (end - start) / comm_sz * my_rank;
    *my_last_m = *my_first_m + (end - start) / comm_sz;
  }
  *my_m = *my_last_m - *my_first_m;
}

int main(int argc, char *argv[])
{
  double diff;
  double epsilon = 0.001;
  int iterations;
  int iterations_print;
  double mean;
  // double u[M][N];
  double w[M][N];
  double wtime;

  int comm_sz;
  int my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  /* Initialize w and broadcast w to slaves */
  if (my_rank == 0) // I'm the master
  {
    printf("\n");
    printf("HEATED_PLATE_MPI\n");
    printf("  C/MPI version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processes = %d\n", comm_sz);

    int i, j;
    #pragma omp parallel shared(w) private(i, j)
    {
      // 初始化 w 边界的值
      #pragma omp for
      for (i = 1; i < M - 1; i++)
      {
        w[i][0] = 100.0;      // left border
        w[i][N - 1] = 100.0;  // right border
      }
      #pragma omp for
      for (j = 0; j < N; j++)
      {
        w[M - 1][j] = 100.0;  // bottom border
        w[0][j] = 0.0;        // top border
      }
    
      // 计算边界 w 值之和，加到 mean 上
      #pragma omp for reduction(+ : mean)
      for (i = 1; i < M - 1; i++)
      {
        mean += w[i][0] + w[i][N - 1];
      }
      #pragma omp for reduction(+ : mean)
      for (j = 0; j < N; j++)
      {
        mean += w[M - 1][j] + w[0][j];
      }
    }

    // 算均值，除以边界节点的个数
    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);

    // 初始化 w 内部为 mean
    #pragma omp parallel shared(mean, w) private(i, j)
    {
      #pragma omp for
      for (i = 1; i < M - 1; i++)
        for (j = 1; j < N - 1; j++)
          w[i][j] = mean;
    }
  }
  /* End of initialization of w */

  /* Each process compute my_first_m and my_last_m */
  int my_first_m, my_last_m, my_m;
  my_bound(1, M - 1, comm_sz, my_rank, &my_first_m, &my_last_m, &my_m);

  #ifdef DEBUG
    // Just check
    MPI_Barrier(MPI_COMM_WORLD);
    printf("I'm process %d, my first m = %d, my last m = %d\n", my_rank, my_first_m, my_last_m);
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

  // set my_w as send buffer, the computation results will be saved in my_w
  double my_w[my_m][N];
  for (int i = 0; i < my_m; i++)
  {
    my_w[i][0] = 100;
    my_w[i][N-1] = 100;
  }

  // set w_buf as recv buffer
  double w_buf[my_m + 2][N];

  if (my_rank == 0)
  {
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
  }

  diff = epsilon;
  MPI_Barrier(MPI_COMM_WORLD);
  double local_begin = MPI_Wtime();
  while (diff >= epsilon)
  {
    // Optimization:
    // 1. use my_w as a send buffer, send my_w to master, then u is no longer needed (finished, performance x ~2)
    //    - Copy w to u every loop is expensive, O(MN)
    //    - Send the whole w to master is expensive, many data are unnecessary
    // 2. use w_buf as recv buffer, recv w_buf from master, not the whole w (finished, performance x ~2.5)
    // 3. Compute my_w and my_diff at the same time (finished, nearly no performance improvement)
    // 4. pack and unpack (?)

    // Broadcast w to slaves
    // MPI_Bcast(&w[0][0], M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Do not send the whole w
    if (my_rank == 0)
    {
      for (int slave = 1; slave < comm_sz; slave++)
      {
        int slave_first_m, slave_last_m, slave_m;
        my_bound(1, M - 1, comm_sz, slave, &slave_first_m, &slave_last_m, &slave_m);
        MPI_Send(&w[slave_first_m - 1][0], (slave_m + 2) * N, MPI_DOUBLE, slave, 0, MPI_COMM_WORLD);
      }
      memcpy(&w_buf[0][0], &w[my_first_m - 1][0], sizeof(double) * (my_m + 2) * N);
    }
    else
    {
      MPI_Recv(&w_buf[0][0], (my_m + 2) * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Compute my_w for each process and my_diff at the same time
    // Note that my_w[i][j] <=> w_buf[i+1][j]
    double my_diff = 0.0;
    for (int i = 0; i < my_m; i++)
    {
      for (int j = 1; j < N - 1; j++)
      {
        my_w[i][j] = (w_buf[i][j] + w_buf[i+2][j] + w_buf[i+1][j-1] + w_buf[i+1][j+1]) / 4.0;
        my_diff = MAX(my_diff, fabs(my_w[i][j] - w_buf[i+1][j]));
      }
    }

    // // Compute my_diff for each process
    // double my_diff = 0.0;
    // for (int i = 0; i < my_m; i++)
    //   for (int j = 1; j < N - 1; j++)
    //     my_diff = MAX(my_diff, fabs(my_w[i][j] - w_buf[i+1][j]));

    if (my_rank != 0) // if I'm slave, send my_w to master
    {
      // printf("I'm process %d, my begin row = %d\n", my_rank, my_first_m);
      MPI_Send(&my_w[0][0], my_m * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else  // if I'm the master, recv my_w from slaves
    {
      for (int slave = 1; slave < comm_sz; slave++)
      {
        int slave_first_m, slave_last_m, slave_m;
        my_bound(1, M - 1, comm_sz, slave, &slave_first_m, &slave_last_m, &slave_m);
        MPI_Recv(&w[slave_first_m][0], slave_m * N, MPI_DOUBLE, slave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      // master should also copy its own my_w to w
      memcpy(&w[1][0], &my_w[0][0], sizeof(double) * my_m * N);
    }

    #ifdef DEBUG
      if (my_rank == 0)
      {
        for (int i = 0; i < M; i++)
          for (int j = 0; j < N; j++)
            printf("%.2lf%c", w[i][j], j == N - 1 ? '\n' : '\t');
        printf("\n");
      }
      sleep(10);
    #endif

    // Reduce the maximum my_diff to diff, every process will get a copy
    MPI_Allreduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
      iterations++;
      if (iterations == iterations_print)
      {
        printf("  %8d  %f\n", iterations, diff);
        iterations_print = 2 * iterations_print;
      }
    }
  }
  double local_end = MPI_Wtime();
  double local_elapsed = local_end - local_begin;
  MPI_Reduce(&local_elapsed, &wtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    printf("\n");
    printf("HEATED_PLATE_OPENMP:\n");
    printf("  Normal end of execution.\n");
  }

  MPI_Finalize();
  return 0;
}
