#include "../include/matrix.h"

void Matrix::gen_rand_mat()
{ 
  mat = new double *[row];
  for (size_t i = 0; i < row; i++)
    mat[i] = new double[col];

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 10);
  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      mat[i][j] = dist(e2);
}

void Matrix::gen_zero_mat()
{
  mat = new double *[row];
  for (size_t i = 0; i < row; i++)
    mat[i] = new double[col];

  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      mat[i][j] = 0.0;
}

Matrix::Matrix(size_t _row, size_t _col, const Matrix::MAT_TYPE _mat_type)
{
  row = _row;
  col = _col;
  size = _row * _col;
  if (_mat_type == RAND)
    gen_rand_mat();
  else if (_mat_type == ZERO)
    gen_zero_mat();
}

Matrix::Matrix(double *A, size_t M, size_t N)
{
  row = M;
  col = N;
  size = M * N;
  gen_zero_mat();
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++)
      mat[i][j] = A[i * M + j];
}


Matrix::Matrix(const Matrix &A)
{
  row = A.row;
  col = A.col;
  size = A.size;
  gen_zero_mat();

  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      (*this)(i, j) = A(i, j);
}

Matrix::Matrix(const Matrix &A, size_t row_begin, size_t col_begin, size_t _row, size_t _col)
{
  row = _row;
  col = _col;
  size = _row * _col;
  gen_zero_mat();

  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      mat[i][j] = A(i + row_begin, j + col_begin);
}

Matrix::~Matrix()
{
  for (size_t i = 0; i < row; i++)
    delete[] mat[i];
  delete[] mat;
}

size_t Matrix::get_row() const
{
  return row;
}

size_t Matrix::get_col() const
{
  return col;
}

size_t Matrix::get_size() const
{
  return size;
}

double **Matrix::get_mat() const
{
  return mat;
}

double &Matrix::operator()(const int x, const int y) const
{
  if (x < 0 || x >= row || y < 0 || y >= col)
    throw std::logic_error("Matrix indices out of border!");

  return mat[x][y];
}

Matrix Matrix::operator=(const Matrix &A)
{
  if (this->row != A.row || this->col != A.col)
    throw std::logic_error("Matrices have different size!");

  for (size_t i = 0; i < this->row; i++)
    for (size_t j = 0; j < this->col; j++)
      (*this)(i, j) = A(i, j);

  return *this;
}

Matrix Matrix::operator+(const Matrix &A) const
{
  if (this->row != A.row || this->col != A.col)
    throw std::logic_error("Matrices have different size!");

  Matrix C(this->row, this->col, ZERO);
  for (size_t i = 0; i < this->row; i++)
    for (size_t j = 0; j < this->col; j++)
      C(i, j) = (*this)(i, j) + A(i, j);

  return C;
}

Matrix Matrix::operator-(const Matrix &A) const
{
  if (this->row != A.row || this->col != A.col)
    throw std::logic_error("Matrices have different size!");

  Matrix C(this->row, this->col, ZERO);
  for (size_t i = 0; i < this->row; i++)
    for (size_t j = 0; j < this->col; j++)
      C(i, j) = (*this)(i, j) - A(i, j);

  return C;
}

Matrix Matrix::operator*(const Matrix &A) const
{
  if (this->col != A.row)
    throw std::logic_error("Inconsistent matrices for multiplication!");

  Matrix C(this->row, A.col, ZERO);
  for (size_t i = 0; i < C.row; i++)
  {
    for (size_t j = 0; j < C.col; j++)
    {
      C(i, j) = 0;
      for (size_t k = 0; k < this->col; k++)
        C(i, j) += (*this)(i, k) * A(k, j);
    }
  }

  return C;
}

std::ostream &operator<<(std::ostream &os, const Matrix &A)
{
  os << "row: " << A.row << std::endl;
  os << "col: " << A.col << std::endl;
  os << "size: " << A.size << std::endl;

  for (size_t i = 0; i < A.row; i++)
  {
    for (size_t j = 0; j < A.col; j++)
      os << std::setw(8) << A(i, j) << '\t';
    os << std::endl;
  }
  os << std::endl;

  return os;
}

double Matrix::error(const Matrix &A) const
{
  if (this->row != A.row || this->col != A.col)
    throw std::logic_error("Matrices have different size!");

  double e = 0.0;
  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      e += fabsf32((*this)(i, j) - A(i, j));
    
  return e;
}