#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_size(int *m, int *n, int my_rank, MPI_Comm comm)
{
  if (my_rank == 0)
  {
    printf("Enter m and n:\n");
    scanf("%d%d", m, n);
  }
  MPI_Bcast(m, 1, MPI_INT, 0, comm);
  MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

void read_mat(int local_A[], int m, int n, int local_m, int my_rank, MPI_Comm comm)
{
  int *A = NULL;

  if (my_rank == 0)
  {
    A = malloc(sizeof(int) * m * n);
    printf("Enter matrix A:\n");
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        scanf("%d", &A[i * n + j]);
    
    MPI_Scatter(A, local_m * n, MPI_INT, local_A, local_m * n, MPI_INT, 0, comm);
    free(A);
  }
  else
  {
    MPI_Scatter(A, local_m * n, MPI_INT, local_A, local_m * n, MPI_INT, 0, comm);
  }
}


void read_vec(int *x, int n, int my_rank, MPI_Comm comm)
{
  if (my_rank == 0)
  {
    printf("Enter vector x:\n");
    for (int i = 0; i < n; i++)
      scanf("%d", &x[i]);
  }
  MPI_Bcast(x, n, MPI_INT, 0, comm);
}

void mat_vec_mul(int local_A[], int x[], int local_y[], int local_m, int n)
{
  for (int i = 0; i < local_m; i++)
  {
    local_y[i] = 0;
    for (int j = 0; j < n; j++)
      local_y[i] += local_A[i * n + j] * x[j];
  }
}

void print_res(int local_y[], int local_m, int m, int my_rank, MPI_Comm comm)
{
  int *y = NULL;
  
  if (my_rank == 0)
  {
    y = malloc(sizeof(int) * m);
    MPI_Gather(local_y, local_m, MPI_INT, y, local_m, MPI_INT, 0, comm);
    printf("y: ");
    for (int i = 0; i < m; i++)
      printf("%d%c", y[i], i == m - 1 ? '\n' : ' ');
    free(y);
  }
  else
  {
    MPI_Gather(local_y, local_m, MPI_INT, y, local_m, MPI_INT, 0, comm);
  }
}

int main()
{
  int my_rank;
  int comm_sz;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  int m;
  int n;
  read_size(&m, &n, my_rank, MPI_COMM_WORLD);

  int *local_A = malloc(sizeof(int) * (m / comm_sz) * n);
  read_mat(local_A, m, n, m / comm_sz, my_rank, MPI_COMM_WORLD);

  int *x = malloc(sizeof(int) * n);
  read_vec(x, n, my_rank, MPI_COMM_WORLD);

  int *local_y = malloc(sizeof(int) * (m / comm_sz));
  mat_vec_mul(local_A, x, local_y, m / comm_sz, n);

  print_res(local_y, m / comm_sz, m, my_rank, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}