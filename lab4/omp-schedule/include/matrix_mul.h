#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include "matrix.h"
#include <omp.h>

// general matrix multiplication
Matrix general_mat_mul(const Matrix &A, const Matrix &B);

// openmp matrix multiplication
Matrix static_mat_mul(const Matrix &A, const Matrix &B, int thread_count);
Matrix dynamic_mat_mul(const Matrix &A, const Matrix &B, int thread_count);

#endif