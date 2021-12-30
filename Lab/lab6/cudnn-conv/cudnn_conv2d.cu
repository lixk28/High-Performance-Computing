#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cudnn.h"

#define DEBUG

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
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