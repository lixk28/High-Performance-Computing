#include "../include/matrix_mul.h"

Matrix general_mat_mul(Matrix &A, Matrix &B)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);

  double ** mat_A = A.get_mat();
  double ** mat_B = B.get_mat();
  double ** mat_C = C.get_mat();

  int m = A.get_row();
  int k = A.get_col();
  int n = B.get_col();

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int l = 0; l < k; l++)
        mat_C[i][j] += mat_A[i][l] * mat_B[l][j];

  return C;
}

void *thread_mat_mul(void *arg)
{
  thread_arg *my_arg = (thread_arg *)arg;

  int thread_count = my_arg->thread_count;
  long my_rank = my_arg->my_rank;
  int m = my_arg->m;
  int k = my_arg->k;
  int n = my_arg->n;
  double ** A = my_arg->A;
  double ** B = my_arg->B;
  double ** C = my_arg->C;

  int my_m = m / thread_count;
  int my_first_i, my_last_i;

  if (my_rank == thread_count - 1)  // I'm the last thread
  {
    my_first_i = my_m * my_rank;
    my_last_i = m;
  }
  else 
  {
    my_first_i = my_m * my_rank;
    my_last_i = my_first_i + my_m;
  }

  for (int i = my_first_i; i < my_last_i; i++)
    for (int j = 0; j < n; j++)
      for (int l = 0; l < k; l++)
        C[i][j] += A[i][l] * B[l][j];

  return NULL;  
}

void pthread_mat_mul_kernel(double **A, double **B, double **C, int m, int k, int n, int thread_count)
{
  pthread_t *thread_handles = (pthread_t *) malloc(sizeof(pthread_t) * thread_count);
  thread_arg *thread_args = (thread_arg *) malloc(sizeof(thread_arg) * thread_count);

  // set thread arguments
  for (long thread = 0; thread < thread_count; thread++)
  {
    thread_args[thread].thread_count = thread_count;
    thread_args[thread].my_rank = thread;
    thread_args[thread].A = A;
    thread_args[thread].B = B;
    thread_args[thread].C = C;
    thread_args[thread].m = m;
    thread_args[thread].k = k;
    thread_args[thread].n = n;
  }

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles[thread], NULL, thread_mat_mul, (void *)&thread_args[thread]);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles[thread], NULL);

  free(thread_handles);
  free(thread_args);
}

Matrix pthread_mat_mul(Matrix &A, Matrix &B, int thread_count)
{
  if (A.get_col() != B.get_row())
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(A.get_row(), B.get_col(), Matrix::ZERO);
  pthread_mat_mul_kernel(A.get_mat(), B.get_mat(), C.get_mat(), A.get_row(), A.get_col(), B.get_col(), thread_count);
  return C;
}

