#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int MAX_STRING_LEN = 100;

int main()
{
  char greeting[MAX_STRING_LEN];
  int comm_sz;  // number of processes
  int my_rank;  // my process rank

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank != 0)
  {
    sprintf(greeting, "Greetings from process %d of %d!", my_rank, comm_sz);
    MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else  // 0 process
  {
    printf("Greetings from process %d of %d!\n", my_rank, comm_sz);
    for (int i = 1; i < comm_sz; i++) // recieve greetings from 1 ~ comm_sz-1 process
    {
      MPI_Recv(greeting, MAX_STRING_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("%s\n", greeting);
    }
  }

  MPI_Finalize();

  return 0;
}