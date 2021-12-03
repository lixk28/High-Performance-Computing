#include "parallel_for.h"

void parallel_for(int start, int end, int increment, \
                  void *(*functor)(void *), void *func_arg, int thread_count)
// functor is a function pointer, defined by user, executed by thread
// func_arg is the arguments of functor, defined and allocated by user
{
  pthread_t *thread_handles = (pthread_t *) malloc(sizeof(pthread_t) * thread_count);
  pf_arg_t *pf_args = (pf_arg_t *) malloc(sizeof(pf_arg_t) * thread_count);

  int loop_count; // total loop times
  if ((end - start) / increment == 0)
    loop_count = (end - start) / increment;
  else
    loop_count = (end - start) / increment + 1;

  if (loop_count <= thread_count) // if loop_count smaller than thread_count
  {
    thread_count = 1;
  }
    
  int my_count = loop_count / thread_count;   // expected average thread loop times

  for (long thread = 0; thread < thread_count; thread++)
  {
    if (thread == thread_count - 1) // I'm the last thread
    {
      pf_args[thread].my_rank = thread;
      pf_args[thread].my_start = start + increment * my_count * thread;
      pf_args[thread].my_end = end;
      pf_args[thread].my_increment = increment;
      pf_args[thread].func_arg = func_arg;
    }
    else  // I'm not the last thread
    {
      pf_args[thread].my_rank = thread;
      pf_args[thread].my_start = start + increment * my_count * thread;
      pf_args[thread].my_end = pf_args[thread].my_start + increment * my_count;
      pf_args[thread].my_increment = increment;
      pf_args[thread].func_arg = func_arg;
    }
  }

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles[thread], NULL, functor, (void *)&pf_args[thread]);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles[thread], NULL); 

  free(thread_handles);
  free(pf_args);
}