#include "./include/matrix.h"
#include "./include/matrix_mul.h"
#include "./include/parallel_for.h"
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
// #define DEBUG

#define GET_WALL_TIME(now) { \
  struct timeval time; \
  gettimeofday(&time, NULL); \
  now = time.tv_sec + time.tv_usec / 1000000.0; \
}

int main(int argc, char *argv[])
{
  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);
  int thread_count = strtol(argv[4], NULL, 10);

  Matrix A(m, k, Matrix::RAND);
  Matrix B(k, n, Matrix::RAND);

  double begin, end;
  double gemm_time;
  double pfmm_time;

  GET_WALL_TIME(begin);
  Matrix X = general_mat_mul(A, B);
  GET_WALL_TIME(end);
  std::cout << "Elapsed time for gemm: " << end - begin << std::endl;

  GET_WALL_TIME(begin);
  Matrix Y = parallel_for_mat_mul(A, B, thread_count);
  GET_WALL_TIME(end);
  std::cout << "Elapsed time for pfmm: " << end - begin << std::endl;

  std::cout << "Error: " << X.error(Y) << std::endl;

#ifdef DEBUG
  std::cout << A;
  std::cout << B;
  std::cout << X;
  std::cout << Y;
#endif

  return 0;
}