#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include "matrix.h"
#include <omp.h>

// general matrix multiplication
Matrix general_mat_mul(const Matrix &A, const Matrix &B);

// openmp matrix multiplication
void omp_mat_mul_kernel(double **A, double **B, double **C, int m, int k, int n);
Matrix omp_mat_mul(const Matrix &A, const Matrix &B, int thread_count);

#endif