#ifndef PARALLEL_H
#define PARALLEL_H

#include <pthread.h>
#include <cstdlib>

typedef struct
{
  int my_start;
  int my_end;
  int my_increment;

  void * func_arg;  // thread function arguments, specified by user
} pf_arg_t;

void parallel_for(int start, int end, int increment, \
                  void *(*functor)(void *), void *arg, int thread_count);

#endif