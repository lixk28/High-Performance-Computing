#ifndef MATRIX_MUL_PTHREAD_H
#define MATRIX_MUL_PTHREAD_H

#include "matrix.h"
#include <pthread.h>
#include <stdlib.h>

typedef struct
{
  int thread_count;
  long my_rank;
  double **A;
  double **B;
  double **C;
  int m;
  int k;
  int n;
} thread_arg;

// general matrix multiplication
Matrix general_mat_mul(Matrix &A, Matrix &B);

// pthread matrix multiplication
void *thread_mat_mul(void *arg); // thread function
void pthread_mat_mul_kernel(double **A, double **B, double **C, int m, int k, int n, int thread_count);

Matrix pthread_mat_mul(Matrix &A, Matrix &B, int thread_count);

#endif
