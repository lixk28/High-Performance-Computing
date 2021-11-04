#include "../include/parallel_for.h"

void parallel_for(int start, int end, int increment, \
                   void *(*functor)(void *), void *arg, int thread_count)
{
  pthread_t *thread_handles = (pthread_t *) malloc(sizeof(pthread_t) * thread_count);
  parallel_for_arg *args = (parallel_for_arg *) arg;

  int my_interval = (end - start) / thread_count;
  for (long thread = 0; thread < thread_count; thread++)
  {
    if (thread == thread_count - 1) // I'm the last thread
    {
      args[thread].my_start = my_interval * thread;
      args[thread].my_end = end;
      args[thread].my_increment = increment;
    }
    else
    {
      args[thread].my_start = my_interval * thread;
      args[thread].my_end = my_interval * (thread + 1);
      args[thread].my_increment = increment;
    }
  }

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles[thread], NULL, functor, (void *)&args[thread]);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles[thread], NULL); 

  free(thread_handles);
}