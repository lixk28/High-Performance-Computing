#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define DEBUG

double * gen_zero_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[col * i + j] = 0.0;
  return mat;
}

double * gen_rand_matrix(int row, int col)
{
  double * mat = malloc(sizeof(double) * row * col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      mat[col * i + j] = rand() / (double) RAND_MAX * 10; // generate random floating number in 0~10
  return mat;
}

void print_matrix(double *A, int row, int col)
{
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      printf("%.2lf%c", A[col * i + j], j == col - 1 ? '\n' : '\t');
  printf("\n");
}

int main(int argc, char * argv[])
{
  int my_rank;
  int comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int m = strtol(argv[1], NULL, 10);
  int n = strtol(argv[2], NULL, 10);

#ifdef DEBUG
  if (my_rank == 0)
  {
    printf("m = %d\n", m);
    printf("n = %d\n", n);
  }
#endif

  int my_m = m / comm_sz;

  double * A = NULL;
  double * my_A = NULL;

  if (my_rank == 0)
  {
    // generate random matrix A
    srand((unsigned int)time(NULL));
    A = gen_rand_matrix(m, n);

    #ifdef DEBUG
      printf("A:\n"); print_matrix(A, m, n);
    #endif

    for (int rank = 1; rank < comm_sz; rank++)
      // the offset is my_m * n * rank
      MPI_Send(A + my_m * n * rank, my_m * n, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);

    // copy the first my_m rows of A to my_A
    my_A = malloc(sizeof(double) * my_m * n);
    memcpy(my_A, A, sizeof(double) * my_m * n);
    print_matrix(my_A, my_m, n);

    // free A
    free(A);
  }
  else
  {
    my_A = gen_zero_matrix(my_m, n);

    // receive my_A from process 0
    MPI_Recv(my_A, my_m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);

    #ifdef DEBUG
      if (my_rank == 1)
      {
        print_matrix(my_A, my_m, n);
      }
    #endif
  }

  MPI_Finalize();
  return 0;
}