#include "matrix.h"

// generate random row x col matrix
double * gen_rand_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[col * i + j] = 0 + rand() / (double) RAND_MAX * (10 - 0);
  return mat;
}

// generate zero row x col matrix
double * gen_zero_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[col * i + j] = 0.0;
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
      B->mat[col * i + j] = A->mat[(row_begin + i) * A->row + (col_begin + j)];
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
      printf("%.2lf%c", A->mat[A->col * i + j], j == A->col - 1 ? '\n' : '\t');
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
        C->mat[col * i + j] += A->mat[len * i + k] * B->mat[col * k + j];
  return C;
}

double error(Matrix *A, Matrix *B)
{
  int row = A->row;
  int col = A->col;
  double e = 0.0;
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      e += fabs(A->mat[col * i + j] - B->mat[col * i + j]);
  return e;
}

Matrix *my_optimized_mat_mul(Matrix *A, Matrix *B)
{
  Matrix *C = matrix_build(A->row, B->col, ZERO);

  double *mat_a = A->mat;
  double *mat_b = B->mat;
  double *mat_c = C->mat;

  int row_a = A->row; int col_a = A->col;
  int row_b = B->row; int col_b = B->col;
  int row_c = C->row; int col_c = C->col;
  int len = col_a;

  int inner_block_row = MIN(row_c, 128);
  int inner_block_col = MIN(col_c, 128);
  
  for (int i_out = 0; i_out < row_c; i_out += inner_block_row)
  {
    for (int j_out = 0; j_out < col_c; j_out += inner_block_col)
    {
      // compute 4x4 block of C at a time
      for (int i = i_out; i < MIN(row_c, i_out + inner_block_row); i += 4)
      {
        for (int j = j_out; j < MIN(col_c, j_out + inner_block_col); j += 4)
        {
          // use vector registers to less memory access and faster computation
          v2df_t 
            C_i0_j01_vreg, C_i0_j23_vreg,
            C_i1_j01_vreg, C_i1_j23_vreg,
            C_i2_j01_vreg, C_i2_j23_vreg,
            C_i3_j01_vreg, C_i3_j23_vreg;

          v2df_t
            A_i0_k_vreg,
            A_i1_k_vreg,
            A_i2_k_vreg,
            A_i3_k_vreg;

          v2df_t
            B_k_j01_vreg,
            B_k_j23_vreg;

          C_i0_j01_vreg.v = _mm_setzero_pd();
          C_i0_j23_vreg.v = _mm_setzero_pd();
          C_i1_j01_vreg.v = _mm_setzero_pd();
          C_i1_j23_vreg.v = _mm_setzero_pd();
          C_i2_j01_vreg.v = _mm_setzero_pd();
          C_i2_j23_vreg.v = _mm_setzero_pd();
          C_i3_j01_vreg.v = _mm_setzero_pd();
          C_i3_j23_vreg.v = _mm_setzero_pd();

          for (int k = 0; k < len; k++)
          {
            // load elements of A and B to vector registers
            A_i0_k_vreg.v = _mm_loaddup_pd(&mat_a[i * col_a + k]);
            A_i1_k_vreg.v = _mm_loaddup_pd(&mat_a[(i+1) * col_a + k]);
            A_i2_k_vreg.v = _mm_loaddup_pd(&mat_a[(i+2) * col_a + k]);
            A_i3_k_vreg.v = _mm_loaddup_pd(&mat_a[(i+3) * col_a + k]);

            B_k_j01_vreg.v = _mm_load_pd(&mat_b[k * col_b + j]);
            B_k_j23_vreg.v = _mm_load_pd(&mat_b[k * col_b + j + 2]);

            // the first row
            C_i0_j01_vreg.v += A_i0_k_vreg.v * B_k_j01_vreg.v;
            C_i0_j23_vreg.v += A_i0_k_vreg.v * B_k_j23_vreg.v;

            // the second row
            C_i1_j01_vreg.v += A_i1_k_vreg.v * B_k_j01_vreg.v;
            C_i1_j23_vreg.v += A_i1_k_vreg.v * B_k_j23_vreg.v;

            // the third row
            C_i2_j01_vreg.v += A_i2_k_vreg.v * B_k_j01_vreg.v;
            C_i2_j23_vreg.v += A_i2_k_vreg.v * B_k_j23_vreg.v;

            // the fourth row
            C_i3_j01_vreg.v += A_i3_k_vreg.v * B_k_j01_vreg.v;
            C_i3_j23_vreg.v += A_i3_k_vreg.v * B_k_j23_vreg.v;
          }

          // store back to C
          // the first row
          mat_c[i * col_c + j] = C_i0_j01_vreg.data[0];
          mat_c[i * col_c + j + 1] = C_i0_j01_vreg.data[1];
          mat_c[i * col_c + j + 2] = C_i0_j23_vreg.data[0];
          mat_c[i * col_c + j + 3] = C_i0_j23_vreg.data[1];

          // the second row
          mat_c[(i+1) * col_c + j] = C_i1_j01_vreg.data[0];
          mat_c[(i+1) * col_c + j + 1] = C_i1_j01_vreg.data[1];
          mat_c[(i+1) * col_c + j + 2] = C_i1_j23_vreg.data[0];
          mat_c[(i+1) * col_c + j + 3] = C_i1_j23_vreg.data[1];

          // the third row
          mat_c[(i+2) * col_c + j] = C_i2_j01_vreg.data[0];
          mat_c[(i+2) * col_c + j + 1] = C_i2_j01_vreg.data[1];
          mat_c[(i+2) * col_c + j + 2] = C_i2_j23_vreg.data[0];
          mat_c[(i+2) * col_c + j + 3] = C_i2_j23_vreg.data[1];

          // the fourth row
          mat_c[(i+3) * col_c + j] = C_i3_j01_vreg.data[0];
          mat_c[(i+3) * col_c + j + 1] = C_i3_j01_vreg.data[1];
          mat_c[(i+3) * col_c + j + 2] = C_i3_j23_vreg.data[0];
          mat_c[(i+3) * col_c + j + 3] = C_i3_j23_vreg.data[1];
        }
      }
    }
  }

  return C;
}