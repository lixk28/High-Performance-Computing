#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_len(int *n, int my_rank, MPI_Comm comm)
{
  if (my_rank == 0)
  {
    printf("Enter vector length:\n");
    scanf("%d", n);
  }
  MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

void read_vec(double local_x[], double local_y[], int n, int my_rank, int comm_sz, MPI_Comm comm)
{
  double *x = NULL;
  double *y = NULL;

  if (my_rank == 0)
  {
    // only root process will apply for memory to operating system
    // less memory consumption and other process have no access to x and y 
    x = malloc(sizeof(double) * n);
    y = malloc(sizeof(double) * n);

    printf("Enter vector x:\n");
    for (int i = 0; i < n; i++)
      scanf("%lf", &x[i]);

    printf("Enter vector y:\n");
    for (int i = 0; i < n; i++)
      scanf("%lf", &y[i]);

    MPI_Scatter(x, n / comm_sz, MPI_DOUBLE, local_x, n / comm_sz, MPI_DOUBLE, 0, comm);
    MPI_Scatter(y, n / comm_sz, MPI_DOUBLE, local_y, n / comm_sz, MPI_DOUBLE, 0, comm);

    free(x);
    free(y);
  }
  else
  {
    MPI_Scatter(x, n / comm_sz, MPI_DOUBLE, local_x, n / comm_sz, MPI_DOUBLE, 0, comm);
    MPI_Scatter(y, n / comm_sz, MPI_DOUBLE, local_y, n / comm_sz, MPI_DOUBLE, 0, comm);
  }
}

void vec_sum(double local_x[], double local_y[], double local_z[], int local_n)
{
  for (int local_i = 0; local_i < local_n; local_i++)
    local_z[local_i] = local_x[local_i] + local_y[local_i];
}

void print_vec(double local_z[], int local_n, int n, int my_rank, MPI_Comm comm)
{
  double *z = NULL;
  
  if (my_rank == 0)
  {
    z = malloc(sizeof(double) * n);
    MPI_Gather(local_z, local_n, MPI_DOUBLE, z, local_n, MPI_DOUBLE, 0, comm);
    for (int i = 0; i < n; i++)
      printf("%.2lf%c", z[i], i == n - 1 ? '\n' : ' ');
    free(z);
  }
  else
  {
    MPI_Gather(local_z, local_n, MPI_DOUBLE, z, local_n, MPI_DOUBLE, 0, comm);
  }
}

int main()
{
  int my_rank;
  int comm_sz;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  int n;
  read_len(&n, my_rank, MPI_COMM_WORLD);

  double *local_x = malloc(sizeof(double) * (n / comm_sz));
  double *local_y = malloc(sizeof(double) * (n / comm_sz));
  read_vec(local_x, local_y, n, my_rank, comm_sz, MPI_COMM_WORLD);
  
  double *local_z = malloc(sizeof(double) * (n / comm_sz));
  vec_sum(local_x, local_y, local_z, n / comm_sz);

  print_vec(local_z, n / comm_sz, n, my_rank, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}