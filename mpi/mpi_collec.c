#include <stdio.h>
#include <mpi.h>

int main()
{
  int my_rank;
  int comm_sz;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int local_a[4];

  if (my_rank == 0)
  {
    int a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    MPI_Scatter(a, 4, MPI_INT, local_a, 4, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Scatter(NULL, 100, MPI_CHAR, local_a, 4, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // printf("process %d: ", my_rank);
  // for (int i = 0; i < 4; i++)
  //   printf("%d%c", local_a[i], i == 3 ? '\n' : ' ');

  if (my_rank == 0)
  {
    int a[16] = {0};
    MPI_Gather(local_a, 4, MPI_INT, a, 4, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < 16; i++)
      printf("%d%c", a[i], i == 15 ? '\n' : ' ');
  }
  else
  {
    MPI_Gather(local_a, 4, MPI_INT, NULL, 4, MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}