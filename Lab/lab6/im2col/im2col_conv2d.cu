#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"

// #define DEBUG

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

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void matrix_mul_cuda_kernel(int *A, int *B, int *C, int m, int k, int n)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  int value = 0;
  for (int l = 0; l < k; l++)
  {
    value += A[row * k + l] * B[l * n + col];
  }
  C[row * n + col] = value;
}

void matrix_mul_cuda(int *A, int *B, int *C, int m, int k, int n, int block_size)
{
  int *A_d;
  int *B_d;
  int *C_d;

  cudaMalloc((void **)&A_d, sizeof(int) * m * k);  // use double pointer
  cudaMalloc((void **)&B_d, sizeof(int) * k * n);
  cudaMalloc((void **)&C_d, sizeof(int) * m * n);

  cudaMemcpy(A_d, A, sizeof(int) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(int) * k * n, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, sizeof(int) * m * n, cudaMemcpyHostToDevice);

  int grid_x = n % block_size == 0 ? n / block_size : n / block_size + 1;
  int grid_y = m % block_size == 0 ? m / block_size : m / block_size + 1;
  dim3 dim_grid(grid_x, grid_y);
  dim3 dim_block(block_size, block_size);
  
  #ifdef DEBUG
    printf("dim_grid(%d, %d), dim_block(%d, %d)\n", dim_grid.x, dim_grid.y, dim_block.x, dim_block.y);
  #endif

  matrix_mul_cuda_kernel<<<dim_grid, dim_block>>>(A_d, B_d, C_d, m, k, n);

  cudaMemcpy(C, C_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

__host__ void im2col(int *im, int *im_col, int input_height, int input_width, int output_height, int output_width, int kernel_height, int kernel_width, int stride)
{
	int im_col_height = kernel_height * kernel_width;
	int im_col_width = output_height * output_width;
  
  // output_count is the col index of im_col
  int output_count = 0;

	// valid padding, drop the rest cols and rows
	for (int i = 0; i + kernel_height - 1 < input_height; i += stride)
	{
		for (int j = 0; j + kernel_width - 1 < input_width; j += stride)
		{
			for (int i_k = 0; i_k < kernel_height; i_k++)
			{
				for (int j_k = 0; j_k < kernel_width; j_k++)
				{
					int row = i + i_k;
					int col = j + j_k;
					im_col[(i_k * kernel_width + j_k) * im_col_width + output_count] = im[row * input_width + col];
				}
			}
      output_count += 1;
		}
	}
}

void cuda_conv2d(int *input, int *output, int *kernel, int input_height, int input_width, int kernel_height, int kernel_width, int block_size, int stride)
{
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;
  
  int im_col_height = kernel_height * kernel_width;
	int im_col_width = output_height * output_width;

  int *im_col = (int *)malloc(sizeof(int) * im_col_height * im_col_width);
  im2col(input, im_col, input_height, input_width, output_height, output_width, kernel_height, kernel_width, stride);
  
  #ifdef DEBUG
    printf("im2col:\n"); print_matrix(im_col, kernel_height * kernel_width, output_height * output_width);
  #endif

  matrix_mul_cuda(kernel, im_col, output, 1, kernel_height * kernel_width, output_height * output_width, block_size);

  free(im_col);
}

int main(int argc, char *argv[])
{
  int input_height = strtol(argv[1], NULL, 10);
  int input_width = strtol(argv[2], NULL, 10);
  int stride = strtol(argv[3], NULL, 10);
  int block_size = strtol(argv[4], NULL, 10);

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
  cuda_conv2d(input, output, kernel, input_height, input_width, kernel_height, kernel_width, block_size, stride);
  end = get_wall_time();

  printf("wall time of im2col_conv2d, input size = %d, stride = %d: %e\n", input_height, stride, end - begin);

  #ifdef DEBUG
    printf("output:\n"); print_matrix(output, output_height, output_width);
  #endif

  // free allocated memory on host
  free(input);
  free(output);
  free(kernel);
  
  return 0;
}