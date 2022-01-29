#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cudnn.h"

#define DEBUG

#ifdef DEBUG
  void print_matrix(float *mat, int m, int n)
  {
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
        printf("%.2lf%c", mat[i * n + j], j == n - 1 ? '\n' : '\t');
    }
    printf("\n");
  }
#endif

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time, NULL))
  {
    return 0;
  }
  return (int)time.tv_sec + (int)time.tv_usec * .000001;
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
  int output_channel;
  int output_n;

  float *input = (float *)malloc(sizeof(float) * input_height * input_width);
  float *output = (float *)malloc(sizeof(float) * output_height * output_width);
  float *kernel = (float *)malloc(sizeof(float) * kernel_height * kernel_width);

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

  double begin_1, begin_2, end_1, end_2;

  begin_1 = get_wall_time();

  cudnnHandle_t cudnn_handle;
  cudnnCreate(&cudnn_handle);

  cudnnTensorDescriptor_t input_descriptor;
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, input_height, input_width);
  float *input_d = NULL;
  cudaMalloc((void **)&input_d, sizeof(float) * input_height * input_width);
  cudaMemcpy(input_d, input, sizeof(float) * input_height * input_width, cudaMemcpyHostToDevice);

  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, kernel_height, kernel_width);
  float *kernel_d = NULL;
  cudaMalloc((void **)&kernel_d, sizeof(float) * kernel_height * kernel_width);
  cudaMemcpy(kernel_d, kernel, sizeof(float) * kernel_height * kernel_width, cudaMemcpyHostToDevice);

  cudnnConvolutionDescriptor_t conv_descriptor;
  cudnnCreateConvolutionDescriptor(&conv_descriptor);
  cudnnSetConvolution2dDescriptor(conv_descriptor, 0, 0, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);  // valid padding, dilation = 1 ?

  cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor, &output_n, &output_channel, &output_height, &output_width);

  #ifdef DEBUG
    printf("output_n = %d\n", output_n);
    printf("output_channel = %d\n", output_channel);
    printf("output_height = %d\n", output_height);
    printf("output_width = %d\n\n", output_width);
  #endif

  cudnnTensorDescriptor_t output_descriptor;
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_channel, output_height, output_width);
  float *output_d = NULL;
  cudaMalloc((void **)&output_d, sizeof(float) * output_n * output_channel * output_height * output_width);

  cudnnConvolutionFwdAlgo_t alg;
  cudnnGetConvolutionForwardAlgorithm(cudnn_handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, \
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &alg);

  // workspace size && allocate memory
  size_t workspace_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, \
                                          alg, &workspace_size);
  void * workspace = nullptr;
  cudaMalloc(&workspace, workspace_size);

  // convolution
  float alpha = 1.0f;
  float beta = 0.0f;
  begin_2 = get_wall_time();
  cudnnConvolutionForward(cudnn_handle,
                          &alpha, input_descriptor, input_d,
                          kernel_descriptor, kernel_d,
                          conv_descriptor, alg,
                          workspace, workspace_size,
                          &beta, output_descriptor, output_d);
  end_2 = get_wall_time();

  cudaMemcpy(output, output_d, sizeof(float) * output_n * output_channel * output_height * output_width, cudaMemcpyDeviceToHost);

  // free and destroy
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(kernel_d);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(conv_descriptor);
  cudaFree(workspace);
  cudnnDestroy(cudnn_handle);

  end_1 = get_wall_time();

  printf("wall time of cudnn_conv2d, input size = %d, stride = %d: %e\n", input_height, stride, end_2 - begin_2);
  printf("wall time of cudnn_conv2d, input size = %d, stride = %d (including handle and descriptor): %e\n", input_height, stride, end_1 - begin_1);

  #ifdef DEBUG
    printf("output:\n"); print_matrix(output, output_height, output_width);
  #endif

  // free allocated memory on host
  free(input);
  free(output);
  free(kernel);
  
  return 0;
}