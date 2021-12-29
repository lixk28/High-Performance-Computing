#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DEBUG

__global__ void cuda_conv2d_kernel(int *input, int *output, int *kernel, int input_height, int input_width, int kernel_height, int kernel_width, int stride)
{
  // output[row][col] is to be calculated in this thread
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  int sum = 0;
  for (int i = 0; i < kernel_height; i++)
  {
    for (int j = 0; j < kernel_width; j++)
    {
      sum += kernel[i * kernel_width + j] * input[(row * stride + i) * input_width + (col * stride + j)];
    }
  }
  output[row * ((input_width - kernel_width) / stride + 1) + col] = sum;
}

void cuda_conv2d(int *input, int *output, int *kernel, int input_height, int input_width, int kernel_height, int kernel_width, int block_size, int stride)
{
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;

  int *input_d = NULL;
  int *output_d = NULL;
  int *kernel_d = NULL;
  
  // allocate memory on device
  cudaMalloc((void **)&input_d, sizeof(int) * input_height * input_width);
  cudaMalloc((void **)&output_d, sizeof(int) * output_height * output_width);
  cudaMalloc((void **)&kernel_d, sizeof(int) * kernel_height * kernel_width);
  
  // copy memory from host to device
  cudaMemcpy(input_d, input, sizeof(int) * input_height * input_width, cudaMemcpyHostToDevice);
  cudaMemcpy(kernel_d, kernel, sizeof(int) * kernel_height * kernel_width, cudaMemcpyHostToDevice);

  // if block_size is too large, then we set it as the size of output
  // or, we will partition output into blocks
  block_size = output_width < block_size ? output_width : block_size;
  dim3 dim_block(block_size, block_size);

  // if grid dimensions is divisible by block_size, then we can perfectly partition it
  // or, we add one row and one col to contain the remain part of output
  int grid_x = output_width % block_size == 0 ? output_width / block_size : output_width / block_size + 1;
  int grid_y = output_height % block_size == 0 ? output_height / block_size : output_height / block_size + 1;
  dim3 dim_grid(grid_x, grid_y);

  #ifdef DEBUG
    printf("dim_grid(%d, %d)\n", dim_grid.x, dim_grid.y);
    printf("dim_block(%d, %d)\n", dim_block.x, dim_block.y);
  #endif

  // call cuda conv2d kernel
  cuda_conv2d_kernel<<<dim_grid, dim_block>>>(input_d, output_d, kernel_d, input_height, input_width, kernel_height, kernel_width, stride);

  // copy conv2d output from device to host
  cudaMemcpy(output, output_d, sizeof(int) * output_height * output_width, cudaMemcpyDeviceToHost);

  // free allocated memory on device
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(kernel_d);
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (int)time.tv_sec + (int)time.tv_usec * .000001;
}

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

int main(int argc, char const *argv[])
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
  cuda_conv2d(input, output, kernel, input_height, input_width, kernel_height, kernel_width, block_size, stride);
  end = get_wall_time();

  printf("wall time of cuda_conv2d, input size = %d: %e\n", input_height, end - begin);

  #ifdef DEBUG
    printf("output:\n"); print_matrix(output, output_height, output_width);
  #endif

  // free allocated memory on host
  free(input);
  free(output);
  free(kernel);

  return 0;
}
