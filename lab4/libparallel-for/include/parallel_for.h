#ifndef PARALLEL_H
#define PARALLEL_H

#include <pthread.h>
#include <cstdlib>

typedef struct
{
  int my_start;
  int my_end;
  int my_increment;

  int m;
  int k;
  int n;

  double ** A;
  double ** B;
  double ** C;
}parallel_for_arg;

void parallel_for(int start, int end, int increment, \
                   void *(*functor)(void *), void *arg, int thread_count);

#endif