#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
// #define DEBUG
#define LEN 1000

#define GET_WALL_TIME(now) { \
  struct timeval time; \
  gettimeofday(&time, NULL); \
  now = time.tv_sec + time.tv_usec / 1000000.0; \
}

int thread_count;

int a[LEN];
int global_sum;
int global_index = 0;

pthread_mutex_t mutex;
pthread_mutex_t mutex_index;
pthread_mutex_t mutex_sum;

void *thread_sum_byone(void *rank)
{
  // critical section
  while(global_index < LEN)
  {
    pthread_mutex_lock(&mutex);
    global_sum += a[global_index++];
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

void *thread_sum_bygroup(void *rank)
{
  while (global_index < LEN)
  {
    int local_begin, local_end;
    int local_sum = 0;

    // access of global_index
    pthread_mutex_lock(&mutex_index);
    local_begin = global_index;
    local_end = global_index + 10;
    global_index += 10;
    pthread_mutex_unlock(&mutex_index);

    for (int i = local_begin; i < local_end; i++)
      local_sum += a[i];

    // update global_sum
    pthread_mutex_lock(&mutex_sum);
    global_sum += local_sum;
    pthread_mutex_unlock(&mutex_sum);
  }

  return NULL;
}

void sum_byone()
{
  pthread_t *thread_handles_byone = malloc(sizeof(pthread_t) * thread_count);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles_byone[thread], NULL, thread_sum_byone, (void *)thread);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles_byone[thread], NULL);

  free(thread_handles_byone);

  printf("Sum of sum_byone = %d\n", global_sum);
}

void sum_bygroup()
{
  pthread_t *thread_handles_bygroup = malloc(sizeof(pthread_t) * thread_count);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles_bygroup[thread], NULL, thread_sum_bygroup, (void *)thread);

  for (long thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles_bygroup[thread], NULL);
  
  free(thread_handles_bygroup);

  printf("Sum of sum_bygroup = %d\n", global_sum);
}

void serial_sum()
{
  int sum = 0;
  for (int i = 0; i < LEN; i++)
    sum += a[i];
  printf("Sum of serial_sum: %d\n", sum);
}

void reset()
{
  global_sum = 0;
  global_index = 0;
}

int main(int argc, char *argv[])
{
  srand(time(NULL));
  int sum = 0;
  for (int i = 0; i < LEN; i++)
  {
    a[i] = rand() % 10;
    #ifdef DEBUG
      printf("%d%c", a[i], i == LEN - 1 ? '\n' : ' ');
    #endif
    sum += a[i];
  }

  thread_count = strtol(argv[1], NULL, 10);

  double begin, end;

  GET_WALL_TIME(begin);
  serial_sum();
  GET_WALL_TIME(end);
  printf("Wall time of serial_sum: %lf\n", end - begin);

  GET_WALL_TIME(begin);
  sum_byone();
  GET_WALL_TIME(end);
  printf("Wall time of sum_byone: %lf\n", end - begin);

  reset();  // reset global_sum and global_index

  GET_WALL_TIME(begin);
  sum_bygroup();
  GET_WALL_TIME(end);
  printf("Wall time of sum_bygroup: %lf\n", end - begin);

  return 0;
}