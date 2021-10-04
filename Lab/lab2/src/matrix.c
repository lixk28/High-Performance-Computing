#include "matrix.h"

// set random seed
void set_seed()
{
  srand((unsigned int)time(NULL));
}

// generate random row x col matrix
static double * gen_rand_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[row * i + j] = 0 + rand() / (double) RAND_MAX * (10 - 0);
  return mat;
}

// generate zero row x col matrix
static double * gen_zero_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[row * i + j] = 0.0;
  return mat;
}

// build a row x col matrix with type RAND or ZERO
Matrix *matrix_build(int row, int col, MAT_TYPE type)
{
  Matrix *A = malloc(sizeof(Matrix));
  A->row = row;
  A->col = col;
  if (type == ZERO)
    A->mat = gen_zero_matrix(row, col);
  else if (type == RAND)
    A->mat = gen_rand_matrix(row, col);
  return A;
}

// matrix slice
Matrix *matrix_slice(Matrix *A, int row_begin, int col_begin, int row, int col)
{
  Matrix *B = matrix_build(row, col, ZERO);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      B->mat[row * i + j] = A->mat[(row_begin + i) * A->row + (col_begin + j)];
  return B;
}

// destroy a matrix
void matrix_destroy(Matrix *A)
{
  free(A->mat);
  free(A);
}

// print a matrix
void print_matrix(Matrix *A)
{
  for (int i = 0; i < A->row; i++)
    for (int j = 0; j < A->col; j++)
      printf("%.2lf%c", A->mat[A->row * i + j], j == A->col - 1 ? '\n' : '\t');
  printf("\n");
}

Matrix *matrix_multiplication(Matrix *A, Matrix *B)
{
  int row = A->row;
  int col = B->col;
  int len = A->col;
  Matrix *C = matrix_build(row, col, ZERO);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      for (int k = 0; k < len; k++)
        C->mat[row * i + j] += A->mat[row * i + k] * B->mat[len * k + j];
  return C;
}

double error(Matrix *A, Matrix *B)
{
  int row = A->row;
  int col = A->col;
  double e = 0.0;
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      e += fabs(A->mat[row * i + j] - B->mat[row * i + j]);
  return e;
}