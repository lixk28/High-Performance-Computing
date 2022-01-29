#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
// #define DEBUG

int thread_count;
int number_toss;
int number_hit;

pthread_mutex_t mutex;

void *thread_throw(void *rank)
{
  long my_rank = (long) rank;
  int my_number_toss;
  int my_number_hit = 0;

  // handle the situation when my_number_toss is not divisble by thread_count
  if (my_rank == thread_count - 1)
  {
    my_number_toss = number_toss - number_toss / thread_count * my_rank;
  }
  else
  {
    my_number_toss = number_toss / thread_count;
  }

  #ifdef DEBUG
    printf("I'm thread %ld, my number in toss is %d\n", my_rank, my_number_toss);
  #endif

  for (int i = 0; i < my_number_toss; i++)
  {
    double x = rand() / (double) RAND_MAX;
    double y = rand() / (double) RAND_MAX;
    if (x * x > y)
      my_number_hit++;
  }

  pthread_mutex_lock(&mutex);
  number_hit += my_number_hit;
  pthread_mutex_unlock(&mutex);

  return NULL;
}


int main(int argc, char *argv[])
{
  number_toss = strtol(argv[1], NULL, 10);
  thread_count = strtol(argv[2], NULL, 10);

  pthread_t *thread_handles = malloc(sizeof(pthread_t) * thread_count);
  pthread_mutex_init(&mutex, NULL);

  srand(time(NULL));

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles[thread], NULL, thread_throw, (void *)thread);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles[thread], NULL);

  printf("Monte Carlo Integral Value = %lf\n", (double) number_hit / number_toss);

  free(thread_handles);
  pthread_mutex_destroy(&mutex);

  return 0;
}