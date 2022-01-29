#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

typedef union 
{
  __m128d v;
  double data[2];
} v2df_t;

#define MIN(X, Y) ((X) < (Y) ? (X): (Y))

// general matrix multiplication
Matrix general_mat_mul(const Matrix &A, const Matrix &B);

// strassen matrix multiplication
Matrix strassen_mat_mul(const Matrix &A, const Matrix &B);

Matrix opt_mat_mul(const Matrix &A, const Matrix &B);

Matrix opt_mat_mul_1(const Matrix &A, const Matrix &B);

#endif