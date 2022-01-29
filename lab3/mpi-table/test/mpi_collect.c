#include "matrix.h"
#include <mpi.h>
#include <string.h>
// #define DEBUG

int main(int argc, char *argv[])
{
  int comm_sz;
  int my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // parse command line arguments to get matrix scale
  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);

  Matrix *A = NULL;
  Matrix *B = NULL;
  Matrix *Y = NULL;   // Y is mpi result

  Matrix *my_A = NULL;
  Matrix *my_Y = NULL;

  int my_m = m / comm_sz;

  double local_begin, local_end, local_elapsed, elapsed;

  MPI_Barrier(MPI_COMM_WORLD);
  local_begin = MPI_Wtime();
  if (my_rank == 0)
  {
    // generate random matrices by process 0
    srand((unsigned int)time(NULL));
    A = matrix_build(m, k, RAND);
    B = matrix_build(k, n, RAND);

    #ifdef DEBUG
      printf("A:\n"); print_matrix(A);
      printf("B:\n"); print_matrix(B);
    #endif

    // send B to each process
    // send partition of A to each process
    my_A = matrix_build(my_m, k, ZERO);
    MPI_Scatter(A->mat, my_m * k, MPI_DOUBLE, my_A->mat, my_m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B->mat, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else
  {
    my_A = matrix_build(my_m, k, ZERO);
    B = matrix_build(k, n, ZERO);
    // receive my_A and B from process 0
    MPI_Scatter(NULL, my_m * k, MPI_DOUBLE, my_A->mat, my_m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B->mat, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // compute my_Y using my_A and B
  my_Y = matrix_multiplication(my_A, B);

  if (my_rank == 0)
  {
    Y = matrix_build(m, n, ZERO);
    // receive my_Y from oth er processes
    MPI_Gather(my_Y->mat, my_m * n, MPI_DOUBLE, Y->mat, my_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #ifdef DEBUG
      printf("Y:\n");
      print_matrix(Y);
    #endif
  }
  else
  {
    // send my_Y to process 0
    MPI_Gather(my_Y->mat, my_m * n, MPI_DOUBLE, NULL, my_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  local_end = MPI_Wtime();
  local_elapsed = local_end - local_begin;
  MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    printf("proc = %d, order = %d: %lf\n", comm_sz, m, elapsed);
  }

  if(my_rank == 0)
  {
    matrix_destroy(A);
    matrix_destroy(Y);
  }

  matrix_destroy(my_A);
  matrix_destroy(B);
  matrix_destroy(my_Y);

  MPI_Finalize();

  return 0;
}