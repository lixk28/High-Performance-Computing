#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>

#define DEBUG

#define M 12
#define N 12

int main(int argc, char *argv[])
{
  double diff;
  double epsilon = 0.001;
  int iterations;
  int iterations_print;
  double mean;
  double u[M][N];
  double w[M][N];
  double wtime;

  int comm_sz;
  int my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
        w[i][0] = 100.0;  // left border
        w[i][N - 1] = 100.0;  // right border
      }
      #pragma omp for
      for (j = 0; j < N; j++)
      {
        w[M - 1][j] = 100.0;  // bottom border
        w[0][j] = 0.0;  // top border
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
    MPI_Bcast(&w[0][0], M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else  // I'm slave
  {
    MPI_Bcast(&w[0][0], M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // #ifdef DEBUG
  //   if (my_rank == 1)
  //     {
  //       for (int i = 0; i < M; i++)
  //         for (int j = 0; j < N; j++)
  //           printf("%.2lf%c", w[i][j], j == N - 1 ? '\n' : '\t');
  //     }
  // #endif

  // Compute my assignment boundary
  int my_first_m, my_last_m;
  if (my_rank == comm_sz - 1) // I'm the last process
  {
    my_first_m = 1 + (M - 2) / comm_sz * my_rank;
    my_last_m = M - 1;  // set my_last_m as M-1
  }
  else
  {
    my_first_m = 1 + (M - 2) / comm_sz * my_rank;
    my_last_m = my_first_m + (M - 2) / comm_sz;
  }

  #ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("I'm process %d, my first m = %d, my last m = %d\n", my_rank, my_first_m, my_last_m);
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

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
    // Save w as u
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        u[i][j] = w[i][j];

    // Compute my w for each process
    for (int i = my_first_m; i < my_last_m; i++)
      for (int j = 1; j < N - 1; j++)
        w[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4.0;
    
    // Compute my_diff for each process
    double my_diff = 0.0;
    for (int i = my_first_m; i < my_last_m; i++)
      for (int j = 1; j < N - 1; j++)
        if (my_diff < fabs(w[i][j] - u[i][j]))
          my_diff = fabs(w[i][j] - u[i][j]);

    // Send my_w to master
    if (my_rank != 0)
    {
      MPI_Send(&w[1 + (my_last_m - my_first_m) * my_rank][0], (M - 2) / comm_sz * N, \
              MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else  // I'm the master
    {
      for (int slave = 1; slave < comm_sz; slave++)
      {
        MPI_Recv(&w[1 + (M - 2) / comm_sz * slave][0], (M - 2) / comm_sz * N, \
                MPI_DOUBLE, slave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    // Broadcast w to slaves
    MPI_Bcast(&w[0][0], M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reduce the maximum my_diff to diff, each process will get a copy
    MPI_Allreduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
 
    #ifdef DEBUG
      if (my_rank == 0)
      {
        for (int i = 0; i < M; i++)
          for (int j = 0; j < N; j++)
            printf("%.2lf%c", w[i][j], j == N - 1 ? '\n' : '\t');
        printf("\n");
      }
    #endif

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
