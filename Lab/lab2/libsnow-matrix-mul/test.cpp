#include <iostream>
#include "include/Matrix_Mul.h"

int main()
{
  Matrix A(4, 4, Matrix::RAND);
  Matrix B(4, 4, Matrix::RAND);

  std::cout << "A:" << std::endl << A;
  std::cout << "B:" << std::endl << B;

  std::cout << "General Matrix Multiplication:" << std::endl << general_mat_mul(A, B);
  std::cout << "Strassen Algorithm:" << std::endl << strassen_mat_mul(A, B);
  std::cout << "My Optimized Matrix Multiplication:" << std::endl << opt_mat_mul(A, B); 

  return 0;
}