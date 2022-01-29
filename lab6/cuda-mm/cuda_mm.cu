#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"

// #define DEBUG

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void matrix_mul_cuda_kernel(double *A, double *B, double *C, int m, int k, int n)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  for (int l = 0; l < k; l++) // switch the order of loop
  {
    for (int j = 0; j < n; j++)
      C[row * n + j] += A[row * k + l] * B[l * n + j];
  }
}

void matrix_mul_cuda(double *A, double *B, double *C, int m, int k, int n, int block_size)
{
  double *A_d;
  double *B_d;
  double *C_d;
  cudaMalloc((void **)&A_d, sizeof(double) * m * k);  // use double pointer
  cudaMalloc((void **)&B_d, sizeof(double) * k * n);
  cudaMalloc((void **)&C_d, sizeof(double) * m * n);

  cudaMemcpy(A_d, A, sizeof(double) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(double) * k * n, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, sizeof(double) * m * n, cudaMemcpyHostToDevice);

  int grid_x = m % block_size == 0 ? m / block_size : m / block_size + 1;
  dim3 dim_grid(grid_x);
  dim3 dim_block(block_size);
  
  #ifdef DEBUG
    printf("dim_grid(%d), dim_block(%d)\n", dim_grid.x, dim_block.x);
  #endif

  matrix_mul_cuda_kernel<<<dim_grid, dim_block>>>(A_d, B_d, C_d, m, k, n);

  cudaMemcpy(C, C_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void matrix_mul_cpu(double *A, double *B, double *C, int m, int k, int n)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int l = 0; l < k; l++)
        C[i * n + j] += A[i * k + l] * B[l * n + j];
}

void print_matrix(double *mat, int m, int n)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
      printf("%.2lf", mat[i * n + j]);
    printf("\n");
  }
  printf("\n");
}

double error(double *mat1, double *mat2, int m, int n)
{
  double e = 0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      e += fabs(mat1[i * n + j] - mat2[i * n + j]);
  return e;
}

int main(int argc, char *argv[])
{
  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);
  int block_size = strtol(argv[4], NULL, 10);

  #ifdef DEBUG
    printf("m = %d\n", m);
    printf("n = %d\n", k);
    printf("k = %d\n", n);
    printf("block size = %d\n", block_size);
  #endif

  double *A = (double *)malloc(sizeof(double) * m * k);
  double *B = (double *)malloc(sizeof(double) * k * n);
  double *C = (double *)malloc(sizeof(double) * m * n);

  srand(20211225);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[i * k + j] = 0 + rand() / (double) RAND_MAX * (10 - 0);
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[i * n + j] = 0 + rand() / (double) RAND_MAX * (10 - 0);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      C[i * n + j] = 0;

  double begin, end;

  begin = get_wall_time();
  matrix_mul_cuda(A, B, C, m, k, n, block_size);
  end = get_wall_time();
  printf("wall time of gemm_cuda, matrix size %d, block size %d: %.5lf\n", m, block_size, end - begin);

  #ifdef DEBUG
    double *D = (double *)malloc(sizeof(double) * m * n);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        D[i * n + j] = 0;

    begin = get_wall_time();
    matrix_mul_cpu(A, B, D, m, k, n);
    end = get_wall_time();
    printf("wall time of gemm_cpu: %.5lf\n", end - begin);
    
    printf("Error = %.5lf\n", error(C, D, m, n));
    
    free(D);
  #endif

  free(A);
  free(B);
  free(C);

  return 0;
}