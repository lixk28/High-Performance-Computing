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

Matrix static_mat_mul(const Matrix &A, const Matrix &B, int thread_count)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

# pragma omp parallel for num_threads(thread_count) schedule(static, 1)
  for (size_t i = 0; i < C.get_row(); i++)
    for (size_t j = 0; j < C.get_col(); j++)
      for (size_t p = 0; p < A.get_col(); p++)
        C(i, j) += A(i, p) * B(p, j);

  return C;
}

Matrix dynamic_mat_mul(const Matrix &A, const Matrix &B, int thread_count)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

# pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1)
  for (size_t i = 0; i < C.get_row(); i++)
    for (size_t j = 0; j < C.get_col(); j++)
      for (size_t p = 0; p < A.get_col(); p++)
        C(i, j) += A(i, p) * B(p, j);

  return C;
}
