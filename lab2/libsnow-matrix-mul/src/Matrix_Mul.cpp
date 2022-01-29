#include "../include/Matrix_Mul.h"

Matrix general_mat_mul(const Matrix &A, const Matrix &B)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);
  for (size_t i = 0; i < C.get_row(); i++)
    for (size_t j = 0; j < C.get_col(); j++)
      for (size_t k = 0; k < A.get_col(); k++)
        C(i, j) += A(i, k) * B(k, j);

  return C;
}

Matrix strassen_mat_mul(const Matrix &A, const Matrix &B)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  if (A.get_size() == 64 || B.get_size() == 64)
    return general_mat_mul(A, B);

  // partition
  size_t row_a = A.get_row();
  size_t col_a = A.get_col();
  size_t row_part_a = row_a / 2;
  size_t col_part_a = col_a / 2;
  Matrix A11(A, 0, 0, row_part_a, col_part_a);
  Matrix A12(A, 0, col_part_a, row_part_a, col_part_a);
  Matrix A21(A, row_part_a, 0, row_part_a, col_part_a);
  Matrix A22(A, row_part_a, col_part_a, row_part_a, col_part_a);

  size_t row_b = B.get_row();
  size_t col_b = B.get_col();
  size_t row_part_b = row_b / 2;
  size_t col_part_b = col_b / 2;
  Matrix B11(B, 0, 0, row_part_b, col_part_b);
  Matrix B12(B, 0, col_part_b, row_part_b, col_part_b);
  Matrix B21(B, row_part_b, 0, row_part_b, col_part_b);
  Matrix B22(B, row_part_b, col_part_b, row_part_b, col_part_b);

  // // compute strassen matrices recursively
  // // it involves allocating memory in heap constantly
  // // which will lead to lower performance
  // Matrix S1 = B12 - B22;
  // Matrix S2 = A11 + A12;
  // Matrix S3 = A21 + A22;
  // Matrix S4 = B21 - B11;
  // Matrix S5 = A11 + A22;
  // Matrix S6 = B11 + B22;
  // Matrix S7 = A12 - A22;
  // Matrix S8 = B21 + B22;
  // Matrix S9 = A11 - A21;
  // Matrix S10 = B11 + B12;

  // Matrix P1 = strassen_mat_mul(A11, S1);
  // Matrix P2 = strassen_mat_mul(S2, B22);
  // Matrix P3 = strassen_mat_mul(S3, B11);
  // Matrix P4 = strassen_mat_mul(A22, S4);
  // Matrix P5 = strassen_mat_mul(S5, S6);
  // Matrix P6 = strassen_mat_mul(S7, S8);
  // Matrix P7 = strassen_mat_mul(S9, S10);

  // Matrix C11 = P5 + P4 - P2 + P6;
  // Matrix C12 = P1 + P2;
  // Matrix C21 = P3 + P4;
  // Matrix C22 = P5 + P1 - P3 - P7;

  // compute strassen matrices directly
  Matrix S1 = (A11 + A22) * (B11 + B22);
  Matrix S2 = (A21 + A22) * B11;
  Matrix S3 = A11 * (B12 - B22);
  Matrix S4 = A22 * (B21 - B11);
  Matrix S5 = (A11 + A12) * B22;
  Matrix S6 = (A21 - A11) * (B11 + B12);
  Matrix S7 = (A12 - A22) * (B21 + B22);

  // compute the partitions of C
  Matrix C11 = S1 + S4 - S5 + S7;
  Matrix C12 = S3 + S5;
  Matrix C21 = S2 + S4;
  Matrix C22 = S1 - S2 + S3 + S6;

  // combine the partitions to get C
  Matrix C(C11, C12, C21, C22);
  return C;
}

Matrix opt_mat_mul(const Matrix &A, const Matrix &B)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

  size_t row_c = C.get_row();
  size_t col_c = C.get_col();
  size_t len = A.get_col();

  double ** mat_a = A.get_mat();
  double ** mat_b = B.get_mat();
  double ** mat_c = C.get_mat();

  size_t inner_block_row = MIN(row_c, 128);
  size_t inner_block_col = MIN(row_c, 128);
  
  for (size_t i_out = 0; i_out < row_c; i_out += inner_block_row)
  {
    for (size_t j_out = 0; j_out < col_c; j_out += inner_block_col)
    {
      // compute 4x4 block of C at a time
      for (size_t i = i_out; i < MIN(row_c, i_out + inner_block_row); i += 4)
      {
        for (size_t j = j_out; j < MIN(col_c, j_out + inner_block_col); j += 4)
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

          for (size_t k = 0; k < len; k++)
          {
            // load elements of A and B to vector registers
            A_i0_k_vreg.v = _mm_loaddup_pd(&mat_a[i][k]);
            A_i1_k_vreg.v = _mm_loaddup_pd(&mat_a[i+1][k]);
            A_i2_k_vreg.v = _mm_loaddup_pd(&mat_a[i+2][k]);
            A_i3_k_vreg.v = _mm_loaddup_pd(&mat_a[i+3][k]);

            B_k_j01_vreg.v = _mm_load_pd(&mat_b[k][j]);
            B_k_j23_vreg.v = _mm_load_pd(&mat_b[k][j+2]);

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
          mat_c[i][j] = C_i0_j01_vreg.data[0];
          mat_c[i][j+1] = C_i0_j01_vreg.data[1];
          mat_c[i][j+2] = C_i0_j23_vreg.data[0];
          mat_c[i][j+3] = C_i0_j23_vreg.data[1];

          // the second row
          mat_c[i+1][j] = C_i1_j01_vreg.data[0];
          mat_c[i+1][j+1] = C_i1_j01_vreg.data[1];
          mat_c[i+1][j+2] = C_i1_j23_vreg.data[0];
          mat_c[i+1][j+3] = C_i1_j23_vreg.data[1];

          // the third row
          mat_c[i+2][j] = C_i2_j01_vreg.data[0];
          mat_c[i+2][j+1] = C_i2_j01_vreg.data[1];
          mat_c[i+2][j+2] = C_i2_j23_vreg.data[0];
          mat_c[i+2][j+3] = C_i2_j23_vreg.data[1];

          // the fourth row
          mat_c[i+3][j] = C_i3_j01_vreg.data[0];
          mat_c[i+3][j+1] = C_i3_j01_vreg.data[1];
          mat_c[i+3][j+2] = C_i3_j23_vreg.data[0];
          mat_c[i+3][j+3] = C_i3_j23_vreg.data[1];
        }
      }
    }
  }

  return C;
}
