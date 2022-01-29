#include "../include/matrix.h"
#include "../include/matrix_mul.h"
#include <cstdio>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
// #define DEBUG

#define GET_WALL_TIME(now) { \
  struct timeval time; \
  gettimeofday(&time, NULL); \
  now = time.tv_sec + time.tv_usec / 1000000.0; \
}

int main(int argc, char *argv[])
{
  // std::cout << "Hello, World!" << std::endl;

  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);
  int thread_count = strtol(argv[4], NULL, 10);

  Matrix A(m, k, Matrix::RAND);
  Matrix B(k, n, Matrix::RAND);

#ifdef DEBUG
  std::cout << A;
  std::cout << B;
#endif

  double begin, end;
  double gemm_time;
  double ompmm_time;

  GET_WALL_TIME(begin);
  Matrix X = general_mat_mul(A, B);
  GET_WALL_TIME(end);
  gemm_time = end - begin;
  std::cout << "Elapsed time for gemm: " << gemm_time << std::endl;

  GET_WALL_TIME(begin);
  Matrix Y = omp_mat_mul(A, B, thread_count);
  GET_WALL_TIME(end);
  ompmm_time = end - begin;
  std::cout << "Elapsed time for ompmm: " << ompmm_time << std::endl;

  std::cout << "Error: " << X.error(Y) << std::endl;

  std::string file_name;
  file_name.append("./asset/time_");
  file_name.append(argv[4]);
#ifdef DEBUG
  std::cout << file_name << std::endl;
#endif

  if (access(file_name.c_str(), F_OK) == 0)
  {
    FILE *file = fopen(file_name.c_str(), "a");
    fprintf(file, "%lf\t%lf\n", gemm_time, ompmm_time);
    fclose(file);
  }
  else
  {
    FILE *file = fopen(file_name.c_str(), "w");
    fprintf(file, "%lf\t%lf\n", gemm_time, ompmm_time);
    fclose(file);
  }

#ifdef DEBUG
  std::cout << X;
  std::cout << Y;
#endif

  return 0;
}