#include "../include/Matrix.h"
#include "../include/Matrix_Mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
// #define DEBUG

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: "
              << "./bin/test <M> <N> <K>" << std::endl;
    std::cout << "Arguments:" << std::endl
              << "* <M>: The number of rows of matrix A." << std::endl
              << "* <N>: The number of columns of matrix A (rows of matrix B)." << std::endl
              << "* <K>: The number of columns of matrix B." << std::endl;
    std::cout << "Attention:" << std::endl
              << "M, N and K must be the power of 2!" << std::endl;
    return 0;
  }

  long M = strtol(argv[1], NULL, 10);
  long N = strtol(argv[2], NULL, 10);
  long K = strtol(argv[3], NULL, 10);

  Matrix A(M, K, Matrix::RAND);
  Matrix B(K, N, Matrix::RAND);

#ifdef DEBUG
  std::cout << "Matrix A:" << std::endl
            << A;
  std::cout << "Matrix B:" << std::endl
            << B;
#endif

  char file_name[100] = "./asset/elapsed_time_";
  strcat(file_name, argv[1]);
  FILE *fp = fopen(file_name, "a");

  // general mat mul
  auto start_general = std::chrono::system_clock::now();
  Matrix X = general_mat_mul(A, B);
  auto end_general = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_general = end_general - start_general;
  std::cout << "Elapsed Time of GEMM: " 
            << elapsed_seconds_general.count() << 's' << std::endl;

  // strassen mat mul
  auto start_strassen = std::chrono::system_clock::now();
  Matrix Y = strassen_mat_mul(A, B);
  auto end_strassen = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_strassen = end_strassen - start_strassen;
  std::cout << "Elapsed Time of Strassen: " 
            << elapsed_seconds_strassen.count() << 's' 
            << " with error " << X.error(Y) << std::endl;

  // optimized mat mul
  auto start_opt = std::chrono::system_clock::now();
  Matrix Z = opt_mat_mul(A, B);
  auto end_opt = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_opt = end_opt - start_opt;
  std::cout << "Elapsed Time of OPT-GEMM: " 
            << elapsed_seconds_opt.count() << 's'
            << " with error " << X.error(Z) << std::endl;

  // intel mkl
  double *a, *b, *I;
  double alpha = 1.0;
  double beta = 0.0;
  a = mat_convert(A);
  b = mat_convert(B);
  I = mat_convert(I, M, N);
  auto start_I = std::chrono::system_clock::now();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, K, b, N, beta, I, N);
  auto end_I = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_I = end_I - start_I;
  std::cout << "Elapsed Time of Intel-MKL: "
            << elapsed_seconds_I.count() << 's'
            << " with error " << X.error(Matrix(I, M, N)) << std::endl;

  fprintf(fp, "%lf\t%lf\t%lf\t%lf\n",  
                elapsed_seconds_general.count(),  
                elapsed_seconds_strassen.count(), 
                elapsed_seconds_opt.count(), 
                elapsed_seconds_I.count());

#ifdef DEBUG
  std::cout << "General Matrix Multiplication:" << std::endl
            << X;
  std::cout << "Strassen Matrix Multiplication:" << std::endl
            << Y;
  std::cout << "Optimized Matrix Multiplication:" << std::endl
            << Z;
  std::cout << "Intel MKL:" << std::endl
            << Matrix(I, M, N);
#endif

  fclose(fp);

  return 0;
}
