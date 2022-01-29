#include "../include/matrix.h"
#include "../include/matrix_mul.h"

Matrix general_mat_mul(const Matrix &A, const Matrix &B)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);
  for (size_t i = 0; i < C.get_row(); i++)
    for (size_t j = 0; j < C.get_col(); j++)
      for (size_t p = 0; p < A.get_col(); p++)
        C(i, j) += A(i, p) * B(p, j);

  return C;
}

void omp_mat_mul_kernel(double **A, double **B, double **C, int m, int k, int n)
{
  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();

  int my_m = m / thread_count;
  int my_first_m, my_last_m;

  if (my_rank == thread_count - 1)  // I'm the last thread
  {
    my_first_m = my_m * my_rank;
    my_last_m = m;
  }
  else
  {
    my_first_m = my_m * my_rank;
    my_last_m = my_first_m + my_m;
  }

  for (int i = my_first_m; i < my_last_m; i++)
    for (int j = 0; j < n; j++)
      for (int p = 0; p < k; p++)
        C[i][j] += A[i][p] * B[p][j];
}

Matrix omp_mat_mul(const Matrix &A, const Matrix &B, int thread_count)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

# pragma omp parallel num_threads(thread_count)
  // call the kernel thread function
  omp_mat_mul_kernel(A.get_mat(), B.get_mat(), C.get_mat(), A.get_row(), B.get_row(), B.get_col());

  return C;
}