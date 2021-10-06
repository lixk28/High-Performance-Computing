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
  Matrix *X = NULL;   // X is serial result
  Matrix *Y = NULL;   // Y is mpi result

  Matrix *my_A = NULL;
  Matrix *my_B = NULL;
  Matrix *my_Y = NULL;
  int my_m = m / comm_sz;


  double local_distb_begin, local_distb_end, local_distb_time, distb_time;  // matrix distribution time
  double local_merge_begin, local_merge_end, local_merge_time, merge_time;  // matrix merge time
  double local_begin, local_end, local_elapsed_time, elapsed_time_mpi;
  double elapsed_time_serial;

  MPI_Barrier(MPI_COMM_WORLD);
  local_begin = MPI_Wtime();
  local_distb_begin = MPI_Wtime();
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
    for (int rank = 1; rank < comm_sz; rank++)
    {
      MPI_Send(A->mat + my_m * k * rank, my_m * k, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD);
      MPI_Send(B->mat, k * n, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
    }

    my_A = matrix_slice(A, 0, 0, my_m, k);
    my_B = matrix_slice(B, 0, 0, k, n);
  }
  else
  {
    my_A = matrix_build(my_m, k, ZERO);
    my_B = matrix_build(k, n, ZERO);

    // receive my_A and my_B from process 0
    MPI_Recv(my_A->mat, my_m * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, NULL);
    MPI_Recv(my_B->mat, k * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
  }
  local_distb_end = MPI_Wtime();
  local_distb_time = local_distb_end - local_distb_begin;
  MPI_Reduce(&local_distb_time, &distb_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // compute my_C using my_A and my_B
  my_Y = matrix_multiplication(my_A, my_B);

  MPI_Barrier(MPI_COMM_WORLD);
  local_merge_begin = MPI_Wtime();
  if (my_rank == 0)
  {
    Y = matrix_build(m, n, ZERO);
    // copy my_Y to the first my_m rows of Y
    memcpy(Y->mat, my_Y->mat, sizeof(double) * my_m * n);
    // receive my_Y from other processes
    for (int rank = 1; rank < comm_sz; rank++)
      MPI_Recv(Y->mat + my_m * n * rank, my_m * n, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, NULL);
    #ifdef DEBUG
      printf("Y:\n");
      print_matrix(Y);
    #endif
  }
  else
  {
    // send my_Y to process 0
    MPI_Send(my_Y->mat, my_m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  local_merge_end = MPI_Wtime();
  local_end = MPI_Wtime();
  local_merge_time = local_merge_end - local_merge_begin;
  MPI_Reduce(&local_merge_time, &merge_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  local_elapsed_time = local_end - local_begin;
  MPI_Reduce(&local_elapsed_time, &elapsed_time_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    printf("Distribution time:          %es\n", distb_time);
    printf("Gather time:                %es\n", merge_time);
    printf("Total communication time:   %es\n", distb_time + merge_time);
    printf("MPI elapsed time:           %es\n", elapsed_time_mpi);
  }

  if (my_rank == 0)
  {
    double begin = MPI_Wtime();
    X = matrix_multiplication(A, B);
    double end = MPI_Wtime();
    elapsed_time_serial = end - begin;
    #ifdef DEBUG
      printf("X:\n");
      print_matrix(X);
    #endif
    printf("Serial elapsed time:        %es\n", elapsed_time_serial);
    printf("Error: %.2lf\n", error(X, Y));

    char file_name[100] = "./asset/p2p_";
    char proc[5];
    sprintf(proc, "_%d", comm_sz);
    strcat(file_name, argv[1]);
    strcat(file_name, proc);
    FILE* file = fopen(file_name, "a");
    fprintf(file, "%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\n",
                  distb_time,
                  merge_time,
                  distb_time + merge_time,
                  elapsed_time_mpi,
                  elapsed_time_serial);
  }

  if(my_rank == 0)
  {
    matrix_destroy(A);
    matrix_destroy(B);
    matrix_destroy(X);
    matrix_destroy(Y);
  }

  matrix_destroy(my_A);
  matrix_destroy(my_B);
  matrix_destroy(my_Y);

  MPI_Finalize();

  return 0;
}