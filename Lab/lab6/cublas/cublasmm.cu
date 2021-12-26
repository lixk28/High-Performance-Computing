#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

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
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  double value = 0;
  for (int l = 0; l < k; l++)
  {
    value += A[row * k + l] * B[l * n + col];
  }
  C[row * n + col] = value;
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

  dim3 dim_grid(m / block_size, n / block_size);
  dim3 dim_block(block_size, block_size);
  
  #ifdef DEBUG
    printf("dim_grid(%d), dim_block(%d)\n", dim_grid.x, dim_block.x);
  #endif

  matrix_mul_cuda_kernel<<<dim_grid, dim_block>>>(A_d, B_d, C_d, m, k, n);

  cudaMemcpy(C, C_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void matrix_mul_cublas(double *A, double *B, double *C, int m, int k, int n)
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

  cublasHandle_t handle;
  cublasCreate(&handle);
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_d, n, A_d, k, &beta, C_d, n);  // compute C^T, which is row major of C

  cudaMemcpy(C, C_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cublasDestroy(handle);
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
  double *D = (double *)malloc(sizeof(double) * m * n);

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
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      D[i * n + j] = 0;

  double begin, end;

  begin = get_wall_time();
  matrix_mul_cuda(A, B, C, m, k, n, block_size);
  end = get_wall_time();
  printf("wall time of gemm_cuda, matrix size %d, block size %d: %.5lf\n", m, block_size, end - begin);

  begin = get_wall_time();
  matrix_mul_cublas(A, B, D, m, k, n);
  end = get_wall_time();
  printf("wall time of cublas_gemm: %.5lf\n", end - begin);

  printf("error: %.5lf\n", error(C, D, m, n));

  #ifdef DEBUG
    print_matrix(C, m, n);
    print_matrix(D, m, n);
  #endif

  free(A);
  free(B);
  free(C);
  free(D);

  return 0;
}