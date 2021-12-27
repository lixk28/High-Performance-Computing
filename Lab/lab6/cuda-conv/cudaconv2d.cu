#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#include <cuda_runtime.h>

#define DEBUG

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void cuda_conv2d_kernel(int *input_d, int *ouput_d, int height, int width, int depth, int kernel_num, int kernel_size, int stride)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // int output_height = (height - kernel_size) / stride + 1;  // no padding
  // int output_width = (width - kernel_size) / stride + 1;
  // int output_depth = kernel_num;

}

void cuda_conv2d(int *input, int *output, int height, int width, int depth, int kernel_num, int kernel_size, int stride, int block_size)
{
  int *input_d;
  int *output_d;
  
  int output_height = (height - kernel_size) / stride + 1;  // no padding
  int output_width = (width - kernel_size) / stride + 1;
  int output_depth = kernel_num;

  cudaMalloc((void **)&input_d, sizeof(int) * height * width * depth);
  cudaMalloc((void **)&output_d, sizeof(int) * ouput_height * output_width * output_depth);

  cudaMemcpy(input_d, input, sizeof(int) * height * width * depth, cudaMemcpyHostToDevice);
  cudaMemcpy(output_d, output, sizeof(int) * ouput_height * output_width * output_depth, cudaMemcpyHostToDevice);

  dim3 dim_grid(height / block_size, width / block_size);
  dim3 dim_block(block_size, block_size);

  // todo
  cuda_conv2d_kernel<<<dim_grid, dim_block>>>();

  cudaMemcpy(output, output_d, sizeof(int) * ouput_height * output_width * output_depth, cudaMemcpyDeviceToHost);

  cudaFree(input_d);
  cudaFree(output_d);

}

int main(int argc, char *argv[])
{


  retrun 0;
}