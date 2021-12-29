#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"

#define DEBUG

#ifdef DEBUG
  void print_matrix(int *mat, int m, int n)
  {
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
        printf("%d%c", mat[i * n + j], j == n - 1 ? '\n' : '\t');
    }
    printf("\n");
  }
#endif

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (int)time.tv_sec + (int)time.tv_usec * .000001;
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

void im2col(int *im, int *im_col, int input_height, int input_width, int kernel_height, int kernel_width, int stride)
{
  // TODO
}

int main(int argc, char *argv[])
{
  int input_height = strtol(argv[1], NULL, 10);
  int input_width = strtol(argv[2], NULL, 10);
  int block_size = strtol(argv[3], NULL, 10);
  int stride = strtol(argv[4], NULL, 10);

  int kernel_height = 3;
  int kernel_width = 3;

  // just valid padding, for simplicity
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;

  int *input = (int *)malloc(sizeof(int) * input_height * input_width);
  int *output = (int *)malloc(sizeof(int) * output_height * output_width);
  int *kernel = (int *)malloc(sizeof(int) * kernel_height * kernel_width);

  srand(20211231);
  for (int i = 0; i < input_height * input_width; i++)
    input[i] = rand() % 5;

  srand(20220101);
  for (int i = 0; i < kernel_height * kernel_width; i++)
    kernel[i] = rand() % 5;

  #ifdef DEBUG
    printf("input:\n"); print_matrix(input, input_height, input_width);
    printf("kernel:\n"); print_matrix(kernel, kernel_height, kernel_width);
  #endif

  double begin, end;

  begin = get_wall_time();
  // TODO
  end = get_wall_time();

  printf("wall time of cuda_im2col_conv2d, input size = %d: %e\n", input_height, end - begin);

  #ifdef DEBUG
    printf("output:\n"); print_matrix(output, output_height, output_width);
  #endif

  // free allocated memory on host
  free(input);
  free(output);
  free(kernel);

  return 0;
}