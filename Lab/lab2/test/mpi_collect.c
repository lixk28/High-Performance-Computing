#include "matrix.h"
#include <mpi.h>
#include <string.h>
#define DEBUG

int main(int argc, char *argv[])
{
  int comm_sz;
  int my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);

  int my_m = m / comm_sz;
  Matrix *my_A = matrix_build(my_m, k, ZERO);
  Matrix *my_B = NULL;

  if (my_rank == 0)
  {
    // generate random matrices by process 0
    set_seed();
    Matrix *A = matrix_build(m, k, RAND);
    my_B = matrix_build(k, n, RAND);

    #ifdef DEBUG
      printf("A:\n"); print_matrix(A);
      printf("B:\n"); print_matrix(my_B);
      printf("C:\n"); print_matrix(matrix_multiplication(A, my_B));
    #endif

    // send B to each process
    // send partition of A to each process
    MPI_Scatter(A->mat, my_m * k, MPI_DOUBLE, my_A->mat, my_m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(my_B->mat, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_destroy(A);
  }
  else
  {
    my_B = matrix_build(k, n, ZERO);
    // receive my_A and my_B from process 0
    MPI_Scatter(NULL, my_m * k, MPI_DOUBLE, my_A->mat, my_m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(my_B->mat, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // compute my_C using my_A and my_B
  Matrix *my_C = matrix_multiplication(my_A, my_B);

  if (my_rank == 0)
  {
    Matrix *C = matrix_build(m, n, ZERO);
    // receive my_C from other processes
    MPI_Gather(my_C->mat, my_m * n, MPI_DOUBLE, C->mat, my_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // print C
    print_matrix(C);
    matrix_destroy(C);
  }
  else
  {
    // send my_C to process 0
    MPI_Gather(my_C->mat, my_m * n, MPI_DOUBLE, NULL, my_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  matrix_destroy(my_A);
  matrix_destroy(my_B);
  matrix_destroy(my_C);

  MPI_Finalize();

  return 0;
}