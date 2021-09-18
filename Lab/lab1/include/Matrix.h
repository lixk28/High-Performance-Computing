#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>
#include "mkl.h"

class Matrix
{
public:
  typedef bool MAT_TYPE;
  const static MAT_TYPE RAND = true;
  const static MAT_TYPE ZERO = false;

private:
  size_t row;
  size_t col;
  size_t size; // row x col

  double **mat; // 2-dim dynamic array

  void gen_rand_mat();
  void gen_zero_mat();

public:
  Matrix(size_t _row, size_t _col, const MAT_TYPE);

  Matrix(double *A, size_t M, size_t N);

  // copy constructor
  Matrix(const Matrix &A);

  // matrix slicing
  Matrix(const Matrix &A,
         size_t row_begin, size_t col_begin,
         size_t _row, size_t _col);

  // matrix combination using 4 same size matrices
  Matrix(const Matrix &A, const Matrix &B,
         const Matrix &C, const Matrix &D);

  // destructor
  ~Matrix();

  // getters
  size_t get_row() const;
  size_t get_col() const;
  size_t get_size() const;
  double ** get_mat() const;

  // operators
  double &operator()(const int i, const int j) const;
  Matrix operator=(const Matrix &A);
  Matrix operator+(const Matrix &A) const;
  Matrix operator-(const Matrix &A) const;
  Matrix operator*(const Matrix &A) const;

  // print matrix
  friend std::ostream &operator<<(std::ostream &os, const Matrix &A);

  // error
  double error(const Matrix &A) const;
};

double *mat_convert(const Matrix &A);
double *mat_convert(double *A, long M, long N);

#endif