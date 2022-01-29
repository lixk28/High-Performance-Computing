#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "../include/matrix.h"
#include "../include/matrix_mul.h"
// #define DEBUG

#define GET_WALL_TIME(now) { \
  struct timeval time; \
  gettimeofday(&time, NULL); \
  now = time.tv_sec + time.tv_usec / 1000000.0; \
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage:" << std::endl;
    std::cout << "./bin/test <M> <K> <N> <T>" << std::endl;
    std::cout << "<M>: The number of rows of A." << std::endl;
    std::cout << "<K>: The number of cols of A. Note that it equals the number of rows of B." << std::endl;
    std::cout << "<N>: The number of cols of B." << std::endl;
    std::cout << "<T>: The number of threads." << std::endl;
  }

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
  double wall_time_serial, wall_time_pthread;

  GET_WALL_TIME(begin);
  Matrix C = general_mat_mul(A, B);
  GET_WALL_TIME(end);
  wall_time_serial = end - begin;
  std::cout << "Wall Time of General Matrix Multiplication: " << wall_time_serial << std::endl;

  GET_WALL_TIME(begin);
  Matrix D = pthread_mat_mul(A, B, thread_count);
  GET_WALL_TIME(end);
  wall_time_pthread = end - begin;
  std::cout << "Wall Time of Pthread Matrix Multiplication: " << wall_time_pthread << std::endl;

  std::cout << "Error: " << C.error(D) << std::endl;

#ifdef DEBUG
  std::cout << C;
  std::cout << D;
#endif

  char file_name[100] = "./asset/time-";
  strcat(file_name, argv[4]);
  // printf(file_name);
  FILE *file = fopen(file_name, "a");
  fprintf(file, "%lf\t%lf\n", wall_time_serial, wall_time_pthread);
  fclose(file);

  return 0; 
}