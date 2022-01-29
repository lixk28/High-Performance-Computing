#include "../include/matrix.h"
#include "../include/matrix_mul.h"
#include "../include/parallel_for.h"

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

// parallel for matrix multiplication kernel
void *parallel_for_mat_mul_kernel(void *arg)
{
  parallel_for_arg *my_arg = (parallel_for_arg *) arg;

  // get args
  int my_start = my_arg->my_start;
  int my_end = my_arg->my_end;
  int my_increment = my_arg->my_increment;
  double ** A = my_arg->A;
  double ** B = my_arg->B;
  double ** C = my_arg->C;
  int m = my_arg->m;
  int k = my_arg->k;
  int n = my_arg->n;

  for (int i = my_start; i < my_end; i += my_increment)
    for (int j = 0; j < n; j++)
      for (int p = 0; p < k; p++)
        C[i][j] += A[i][p] * B[p][j];
  
  return NULL;
}

// my parallel for matrix multiplication
Matrix parallel_for_mat_mul(const Matrix &A, const Matrix &B, int thread_count)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

  parallel_for_arg *arg = (parallel_for_arg *) malloc(sizeof(parallel_for_arg) * thread_count);

  for (long thread = 0; thread < thread_count; thread++)
  {
    arg[thread].A = A.get_mat();
    arg[thread].B = B.get_mat();
    arg[thread].C = C.get_mat();
    arg[thread].m = A.get_row();
    arg[thread].k = A.get_col();
    arg[thread].n = B.get_col();
  }

  parallel_for(0, C.get_row(), 1, parallel_for_mat_mul_kernel, (void *)arg, thread_count);

  free(arg);
  return C;
}
