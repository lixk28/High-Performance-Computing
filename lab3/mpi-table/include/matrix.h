#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

typedef union 
{
  __m128d v;
  double data[2];
} v2df_t;

typedef bool MAT_TYPE;
const static MAT_TYPE RAND = true;
const static MAT_TYPE ZERO = false;

typedef struct
{
  double * mat;
  int row;
  int col;
} Matrix;

// generate random row x col matrix
double * gen_rand_matrix(int row, int col);

// generate zero row x col matrix
double * gen_zero_matrix(int row, int col);

// build a row x col matrix with type RAND or ZERO
Matrix *matrix_build(int row, int col, MAT_TYPE type);

// matrix slice
Matrix *matrix_slice(Matrix *A, int row_begin, int col_begin, int row, int col);

// matrix multiplication
Matrix *matrix_multiplication(Matrix *A, Matrix *B);

// my optimized matrix multiplication
Matrix *my_optimized_mat_mul(Matrix *A, Matrix *B);

// destroy a matrix
void matrix_destroy(Matrix *A);

// print a matrix
void print_matrix(Matrix *A);

// error of two matrices
double error(Matrix *A, Matrix *B);

#endif