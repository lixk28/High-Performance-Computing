#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include "parallel_for.h"

#define DEBUG

#ifdef DEBUG
  #include <unistd.h>
#endif

void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[],
          double w[], double sgn);
void timestamp();

void *step_pf(void *arg);

typedef struct
{
  double aw;
  double *w;
}cffti_arg_t;

// typedef struct
// {
//   double *x;
//   double *y;
// }ccopy_arg_t;

typedef struct
{
  // int n;
  int mj;
  int mj2;
  double *a;
  double *b;
  double *c;
  double *d;
  double *w;
  double sgn;
}step_arg_t;

int thread_count;

int main(int argc, char *argv[])
/* 
  Purpose:
    MAIN is the main program for FFT_SERIAL.
  Discussion:
    The "complex" vector A is actually stored as a double vector B.
    The "complex" vector entry A[I] is stored as:
      B[I*2+0], the real part,
      B[I*2+1], the imaginary part.
*/
{
  // double ctime;
  // double ctime1;
  // double ctime2;
  double wtime;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;

  thread_count = strtol(argv[1], NULL, 10);

  timestamp();
  printf("\n");
  printf("FFT_PF\n");
  printf("  C/PARALLEL-FOR version\n");
  printf("\n");
  printf("  Demonstrate an implementation of the Fast Fourier Transform\n");
  printf("  of a complex data vector.\n");

  printf("\n");
  printf("  Number of threads =              %d\n", thread_count);
  /*
  Prepare for tests.
*/
  printf("\n");
  printf("  Accuracy check:\n");
  printf("\n");
  printf("    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n");
  printf("\n");
  printf("             N      NITS    Error         Time          Time/Call     MFLOPS\n");
  printf("\n");

  seed = 331.0;
  n = 1;
  /*
  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
*/
  for (ln2 = 1; ln2 <= 20; ln2++) // 无法并行，有数据依赖
  {
    n = 2 * n; // n = 2^(ln2)
               /*
  Allocate storage for the complex arrays W, X, Y, Z.  

  We handle the complex arithmetic,
  and store a complex number as a pair of doubles, a complex vector as a doubly
  dimensioned array whose second dimension is 2. 
*/
    w = (double *)malloc(n * sizeof(double));
    x = (double *)malloc(2 * n * sizeof(double));
    y = (double *)malloc(2 * n * sizeof(double));
    z = (double *)malloc(2 * n * sizeof(double));

    first = 1;

    for (icase = 0; icase < 2; icase++)
    {
      if (first)
      {
        for (i = 0; i < 2 * n; i = i + 2) // 可以并行
        {
          z0 = ggl(&seed); // ggl 生成一个随机的浮点数
          z1 = ggl(&seed); // z0 和 z1 是临时变量
          x[i] = z0;       // 偶数是实部
          z[i] = z0;
          x[i + 1] = z1; // 奇数是虚部
          z[i + 1] = z1;
          // 向量 x 和向量 z 相同
        }
      }
      else
      {
        for (i = 0; i < 2 * n; i = i + 2) // 可以并行
        {
          z0 = 0.0; /* real part of array */
          z1 = 0.0; /* imaginary part of array */
          x[i] = z0;
          z[i] = z0; /* copy of initial real data */
          x[i + 1] = z1;
          z[i + 1] = z1; /* copy of initial imag data */
        }
      }

      cffti(n, w); // 初始化 sin cos 表
                   /* 
  Transform forward, back 
*/
      if (first)
      {
        sgn = +1.0;
        cfft2(n, x, y, w, sgn);
        sgn = -1.0;
        cfft2(n, y, x, w, sgn);
        /* 
  Results should be same as the initial data multiplied by N.
*/
        fnm1 = 1.0 / (double)n;
        error = 0.0;
        for (i = 0; i < 2 * n; i = i + 2) // 可以并行
        {
          error = error + pow(z[i] - fnm1 * x[i], 2) + pow(z[i + 1] - fnm1 * x[i + 1], 2);
        }
        error = sqrt(fnm1 * error);
        printf("  %12d  %8d  %12e", n, nits, error);
        first = 0;
      }
      else
      {
        wtime = omp_get_wtime();
        // ctime1 = cpu_time();
        for (it = 0; it < nits; it++) // 有数据依赖
        {
          sgn = +1.0;
          cfft2(n, x, y, w, sgn);
          sgn = -1.0;
          cfft2(n, y, x, w, sgn);
        }
        // ctime2 = cpu_time();
        // ctime = ctime2 - ctime1;
        wtime = omp_get_wtime() - wtime;

        flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

        mflops = flops / 1.0E+06 / wtime;

        printf("  %12e  %12e  %12f\n", wtime, wtime / (double)(2 * nits), mflops);
      }
    }
    if ((ln2 % 4) == 0)
    {
      nits = nits / 10;
    }
    if (nits < 1)
    {
      nits = 1;
    }
    free(w);
    free(x);
    free(y);
    free(z);
  }
  printf("\n");
  printf("FFT_SERIAL:\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();

  return 0;
}

// void *ccopy_pf(void *arg)
// {
//   pf_arg_t *my_arg = (pf_arg_t *)arg;
//   int my_start = my_arg->my_start;
//   int my_end = my_arg->my_end;
//   int my_increment = my_arg->my_increment;

//   ccopy_arg_t *ccopy_arg = (ccopy_arg_t *)my_arg->func_arg;
//   double *x = ccopy_arg->x;
//   double *y = ccopy_arg->y;

//   for (int i = my_start; i < my_end; i += my_increment)
//   {
//     y[i * 2 + 0] = x[i * 2 + 0];
//     y[i * 2 + 1] = x[i * 2 + 1];
//   }

//   return NULL;
// }

void ccopy(int n, double x[], double y[])
/*
  Purpose:
    CCOPY copies a complex vector.
  Discussion:
    The "complex" vector A[N] is actually stored as a double vector B[2*N].
    The "complex" vector entry A[I] is stored as:
      B[I*2+0], the real part,
      B[I*2+1], the imaginary part.
  Parameters:
    Input, int N, the length of the vector.
    Input, double X[2*N], the vector to be copied.
    Output, double Y[2*N], a copy of X.
*/
{
  int i;

  for (i = 0; i < n; i++) // 并行提升很小，甚至降低
  {
    y[i * 2 + 0] = x[i * 2 + 0];
    y[i * 2 + 1] = x[i * 2 + 1];
  }
  return;
}

void cfft2(int n, double x[], double y[], double w[], double sgn)
/*
  Purpose:
    CFFT2 performs a complex Fast Fourier Transform.
  Parameters:
    Input, int N, the size of the array to be transformed.
    Input/output, double X[2*N], the data to be transformed.
    On output, the contents of X have been overwritten by work information.
    Input, double W[N], a table of sines and cosines.
    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
    Output, double Y[2*N], the forward or backward FFT of X.
*/
{
  int j;
  int m;
  int mj;
  int tgle;

  m = (int)(log((double)n) / log(1.99)); // m = log_2^n
  mj = 1;

  tgle = 1; // Toggling switch for work array.
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  if (n == 2)
  {
    return;
  }

  int mj2;
  int lj;
  for (j = 0; j < m - 2; j++) // m = log_2^n，这里的 step 要被重复执行很多次，线程开销过大
  {
    mj = mj * 2;
    if (tgle)
    {
      mj2 = 2 * mj;
      lj = n / mj2;

      step_arg_t step_arg;
      step_arg.mj = mj;
      step_arg.mj2 = mj2;
      step_arg.a = &y[0 * 2 + 0];
      step_arg.b = &y[(n / 2) * 2 + 0];
      step_arg.c = &x[0 * 2 + 0];
      step_arg.d = &x[mj * 2 + 0];
      step_arg.w = w;
      step_arg.sgn = sgn;

      parallel_for(0, lj, 1, step_pf, (void *)&step_arg, thread_count);
      // step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
      tgle = 0;
    }
    else
    {
      mj2 = 2 * mj;
      lj = n / mj2;

      step_arg_t step_arg;
      step_arg.mj = mj;
      step_arg.mj2 = mj2;
      step_arg.a = &y[0 * 2 + 0];
      step_arg.b = &x[(n / 2) * 2 + 0];
      step_arg.c = &y[0 * 2 + 0];
      step_arg.d = &y[mj * 2 + 0];
      step_arg.w = w;
      step_arg.sgn = sgn;

      parallel_for(0, lj, 1, step_pf, (void *)&step_arg, thread_count);
      // step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
      tgle = 1;
    }
  }

  if (tgle) // Last pass through data: move Y to X if needed.
  {
    ccopy(n, y, x);
  }

  mj = n / 2;
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  return;
}

void *cffti_pf(void *arg)
{
  pf_arg_t * my_arg = (pf_arg_t *)arg;
  int my_start = my_arg->my_start;
  int my_end = my_arg->my_end;
  int my_increment = my_arg->my_increment;

  cffti_arg_t * cffti_arg = (cffti_arg_t *)my_arg->func_arg;
  double aw = cffti_arg->aw;
  double *w = cffti_arg->w;
  double x;

  // #ifdef DEBUG
  //   int my_rank = my_arg->my_rank;
  //   printf("I'm thread %d, my_start = %d, my_end = %d\n", my_rank, my_start, my_end);
  // #endif

  for (int i = my_start; i < my_end; i += my_increment)
  {
    x = aw * ((double)i);
    w[i * 2 + 0] = cos(x);
    w[i * 2 + 1] = sin(x);
  }

  return NULL;
}

void cffti(int n, double w[])
/*
  Purpose:
    CFFTI sets up sine and cosine tables needed for the FFT calculation.
  Parameters:
    Input, int N, the size of the array to be transformed.
    Output, double W[N], a table of sines and cosines.
*/
{
  double aw;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ((double)n); // aw = 2pi / n

  cffti_arg_t cffti_arg;
  cffti_arg.aw = aw;
  cffti_arg.w = w;

  parallel_for(0, n2, 1, cffti_pf, (void *)&cffti_arg, thread_count);
}

// double cpu_time(void)
// {
//   double value;

//   value = (double)clock() / (double)CLOCKS_PER_SEC; // 返回 CPU 时间

//   return value;
// }

double ggl(double *seed)
/* 
  Purpose:
    GGL generates uniformly distributed pseudorandom real numbers in [0,1]. 
  Parameters:
    Input/output, double *SEED, used as a seed for the sequence.
    Output, double GGL, the next pseudorandom value.
*/
{
  double d2 = 0.2147483647e10; // 生成一个随机浮点数并返回
  double t;
  double value;

  t = *seed;
  t = fmod(16807.0 * t, d2);
  *seed = t;
  value = (t - 1.0) / (d2 - 1.0);

  return value;
}

void *step_pf(void *arg)
{
  pf_arg_t *my_arg = (pf_arg_t *)arg;
  int my_start = my_arg->my_start;
  int my_end = my_arg->my_end;
  int my_increment = my_arg->my_increment;

  step_arg_t *step_arg = (step_arg_t *)my_arg->func_arg;
  int mj = step_arg->mj;
  int mj2 = step_arg->mj2;
  double *a = step_arg->a;
  double *b = step_arg->b;
  double *c = step_arg->c;
  double *d = step_arg->d;
  double *w = step_arg->w;
  int sgn = step_arg->sgn;

  double ambr;  
  double ambu;  
  int j;  
  int ja; 
  int jb; 
  int jc; 
  int jd; 
  int jw; 
  int k;  
  double wjw[2];

  for (j = my_start; j < my_end; j += my_increment)
  {
    jw = j * mj;
    ja = jw;
    jb = ja;
    jc = j * mj2;
    jd = jc;

    wjw[0] = w[jw * 2 + 0];
    wjw[1] = w[jw * 2 + 1];

    if (sgn < 0.0)
    {
      wjw[1] = -wjw[1];
    }

    for (k = 0; k < mj; k++)
    {
      c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
      c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

      ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
      ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

      d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return NULL;                 
}

void step(int n, int mj, double a[], double b[], double c[],
          double d[], double w[], double sgn)
/*
  Purpose:
    STEP carries out one step of the workspace version of CFFT2.
  Parameters:
    ???
*/
{
  int mj2;
  int lj;

  mj2 = 2 * mj;
  lj = n / mj2;

  step_arg_t step_arg;
  step_arg.mj = mj;
  step_arg.mj2 = mj2;
  step_arg.a = a;
  step_arg.b = b;
  step_arg.c = c;
  step_arg.d = d;
  step_arg.w = w;
  step_arg.sgn = sgn;

  parallel_for(0, lj, 1, step_pf, (void *)&step_arg, thread_count);

  return;
}
// {
//   double ambr;
//   double ambu;
//   int j;
//   int ja;
//   int jb;
//   int jc;
//   int jd;
//   int jw;
//   int k;
//   int lj;
//   int mj2;
//   double wjw[2];

//   mj2 = 2 * mj;
//   lj = n / mj2;

// #pragma omp parallel \
//     shared(a, b, c, d, lj, mj, mj2, sgn, w) private(ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw)

// #pragma omp for nowait

//   for (j = 0; j < lj; j++)
//   {
//     jw = j * mj;
//     ja = jw;
//     jb = ja;
//     jc = j * mj2;
//     jd = jc;

//     wjw[0] = w[jw * 2 + 0];
//     wjw[1] = w[jw * 2 + 1];

//     if (sgn < 0.0)
//     {
//       wjw[1] = -wjw[1];
//     }

//     for (k = 0; k < mj; k++)
//     {
//       c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
//       c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

//       ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
//       ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

//       d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
//       d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
//     }
//   }
//   return;
// }

void timestamp()
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}
