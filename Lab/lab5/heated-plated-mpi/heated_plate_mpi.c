#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define M 500
#define N 500

int main(int argc, char *argv[])
{
  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double u[M][N];
  double w[M][N];
  double wtime;

  int comm_sz;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) // I'm the master
  {
    printf("\n");
    printf("HEATED_PLATE_OPENMP\n");
    printf("  C/MPI version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processes = %d\n", omp_get_num_procs());

    #pragma omp parallel shared(w) private(i, j)
    {
      // 初始化 w 边界的值
      #pragma omp for
      for (i = 1; i < M - 1; i++)
      {
        w[i][0] = 100.0;
      }
      #pragma omp for
      for (i = 1; i < M - 1; i++)
      {
        w[i][N - 1] = 100.0;
      }
      #pragma omp for
      for (j = 0; j < N; j++)
      {
        w[M - 1][j] = 100.0;
      }
      #pragma omp for
      for (j = 0; j < N; j++)
      {
        w[0][j] = 0.0;
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

    // Broadcast w to other slave processes
    MPI_Bcast(w, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else  // I'm slave
  {
    MPI_Bcast(w, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if (my_rank == 0)
  {
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
  }

  diff = epsilon;

  // 我们主要需要并行的部分
  // my_first_m, my_last_m
  while (epsilon <= diff)
  {
#pragma omp parallel shared(u, w) private(i, j)
    {
// Save the old solution in U.
#pragma omp for
      for (i = 0; i < M; i++)
      {
        for (j = 0; j < N; j++)
        {
          u[i][j] = w[i][j];
        }
      }

// w 更新为上下左右相邻四个的平均值
#pragma omp for
      for (i = 1; i < M - 1; i++)
      {
        for (j = 1; j < N - 1; j++)
        {
          w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
        }
      }
    }

    // diff 设置为 w[i][j] 相对误差最大的一项
    diff = 0.0;
#pragma omp parallel shared(diff, u, w) private(i, j, my_diff)
    {
      my_diff = 0.0;
#pragma omp for
      for (i = 1; i < M - 1; i++)
      {
        for (j = 1; j < N - 1; j++)
        {
          if (my_diff < fabs(w[i][j] - u[i][j]))
          {
            my_diff = fabs(w[i][j] - u[i][j]);
          }
        }
      }

#pragma omp critical
      {
        if (diff < my_diff)
        {
          diff = my_diff;
        }
      }
    }

    iterations++;
    if (iterations == iterations_print)
    {
      printf("  %8d  %f\n", iterations, diff);
      iterations_print = 2 * iterations_print;
    }
  }

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
