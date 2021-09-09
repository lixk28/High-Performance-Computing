#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

// #include <mmintrin.h>
// #include <xmmintrin.h>
// #include <pmmintrin.h>
// #include <emmintrin.h>

// typedef union 
// {
//   __m128d v;
//   double data[2];
// } v2df_t;

#define MIN(X, Y) ((X) < (Y) ? (X): (Y))

// general matrix multiplication
Matrix general_mat_mul(const Matrix &A, const Matrix &B);

// strassen matrix multiplication
Matrix strassen_mat_mul(const Matrix &A, const Matrix &B);

// memory access optimized matrix multiplication
// compute 4 elements of C at a time
Matrix mat_mul_4x1(const Matrix &A, const Matrix &B);

// compute 4x4 block of C at a time
Matrix mat_mul_4x4(const Matrix &A, const Matrix &B);

// compute 4x4 block of C at a time, with register variables
Matrix mat_mul_4x4_reg(const Matrix &A, const Matrix &B);

Matrix mat_mul_4x4_pac_reg(const Matrix &A, const Matrix &B);

// Matrix mat_mul_4x4_vec(const Matrix &A, const Matrix &B);

#endif