#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include "matrix.h"
#include "parallel_for.h"

// general matrix multiplication
Matrix general_mat_mul(const Matrix &A, const Matrix &B);

// my parallel for matrix multiplication
void *parallel_for_mat_mul_kernel(void *arg);
Matrix parallel_for_mat_mul(const Matrix &A, const Matrix &B, int thread_count);

#endif