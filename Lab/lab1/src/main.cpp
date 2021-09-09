#include "../include/Matrix.h"
#include "../include/Matrix_Mul.h"
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
  long K = strtol(argv[3], NULL, 10);
  long N = strtol(argv[2], NULL, 10);

  Matrix A(M, K, Matrix::RAND);
  Matrix B(K, N, Matrix::RAND);

#ifdef DEBUG
  std::cout << "Matrix A:" << std::endl
            << A;
  std::cout << "Matrix B:" << std::endl
            << B;
#endif

  auto start_X = std::chrono::system_clock::now();
  Matrix X = general_mat_mul(A, B);
  auto end_X = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_X = end_X - start_X;
  std::cout << "Elapsed Time of GEMM: " 
            << elapsed_seconds_X.count() << 's' << std::endl;

  auto start_Y = std::chrono::system_clock::now();
  Matrix Y = strassen_mat_mul(A, B);
  auto end_Y = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_Y = end_Y - start_Y;
  std::cout << "Elapsed Time of Strassen: " 
            << elapsed_seconds_Y.count() << 's' << std::endl;

  auto start_Z = std::chrono::system_clock::now();
  Matrix Z = mat_mul_4x1(A, B);
  auto end_Z = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_Z = end_Z - start_Z;
  std::cout << "Elapsed Time of OPT-GEMM 4x1: " 
            << elapsed_seconds_Z.count() << 's' << std::endl;

  auto start_R = std::chrono::system_clock::now();
  Matrix R = mat_mul_4x4(A, B);
  auto end_R = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_R = end_R - start_R;
  std::cout << "Elapsed Time of OPT-GEMM 4x4: " 
            << elapsed_seconds_R.count() << 's' << std::endl;

  auto start_S = std::chrono::system_clock::now();
  Matrix S = mat_mul_4x4_reg(A, B);
  auto end_S = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_S = end_S - start_S;
  std::cout << "Elapsed Time of OPT-GEMM 4x4 wit reg: " 
            << elapsed_seconds_S.count() << 's' << std::endl;

  auto start_T = std::chrono::system_clock::now();
  Matrix T = mat_mul_4x4_pac_reg(A, B);
  auto end_T = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_T = end_T - start_T;
  std::cout << "Elapsed Time of OPT-GEMM 4x4 wit pac reg: " 
            << elapsed_seconds_T.count() << 's' << std::endl;

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
            << elapsed_seconds_I.count() << 's' << std::endl;

#ifdef DEBUG
  std::cout << "General Matrix Multiplication:" << std::endl
            << X;
  std::cout << "Strassen Matrix Multiplication:" << std::endl
            << Y;
  std::cout << "Optimized 4x1 Matrix Multiplication:" << std::endl
            << Z;
  std::cout << "Optimized 4x4 Matrix Multiplication:" << std::endl
            << R;
  std::cout << "Optimized 4x4 Matrix Multiplication with reg:" << std::endl
            << S;
  std::cout << "Optimized 4x4 Matrix Multiplication with pac and reg:" << std::endl
            << T;
  std::cout << "Intel MKL:" << std::endl
            << Matrix(I, M, N);
#endif

  return 0;
}
