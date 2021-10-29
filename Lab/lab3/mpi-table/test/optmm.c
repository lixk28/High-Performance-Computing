#include "matrix.h"
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

  Matrix *A = matrix_build(m, k, RAND);
  Matrix *B = matrix_build(k, n, RAND);

  double begin, end, elapsed;
  GET_WALL_TIME(begin);
  Matrix *C = my_optimized_mat_mul(A, B);
  GET_WALL_TIME(end);
  elapsed = end - begin;

  printf("order = %d: %lf\n", m, elapsed);

  matrix_destroy(A);
  matrix_destroy(B);
  matrix_destroy(C);

  return 0;
}