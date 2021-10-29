#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
// #define DEBUG

double a;
double b;
double c;

double delta;
bool flag = false;
double root1;
double root2;

pthread_mutex_t mutex_delta;
pthread_cond_t cond_delta;

void *compute_delta(void *rank)
{
  pthread_mutex_lock(&mutex_delta);
  delta = b * b - 4 * a * c;
  flag = true;  // delta finished
  printf("Finish delta computation...\n");
  printf("Wake up thread2...\n");
  pthread_cond_signal(&cond_delta);  
  pthread_mutex_unlock(&mutex_delta);
  return NULL;
}

void *compute_root(void *rank)
{
  pthread_mutex_lock(&mutex_delta);
  while (!flag)
    pthread_cond_wait(&cond_delta, &mutex_delta);
  root1 = (-b + sqrt(delta)) / (2 * a);
  root2 = (-b - sqrt(delta)) / (2 * a);
  pthread_mutex_unlock(&mutex_delta);
  return NULL;
}

int main(int argc, char *argv[])
{
  a = strtold(argv[1], NULL);
  b = strtold(argv[2], NULL);
  c = strtold(argv[3], NULL);

#ifdef DEBUG
  printf("a = %lf\n", a);
  printf("b = %lf\n", b);
  printf("c = %lf\n", c);
#endif

  pthread_mutex_init(&mutex_delta, NULL);
  pthread_cond_init(&cond_delta, NULL);

  pthread_t thread1;
  pthread_t thread2;
  
  pthread_create(&thread1, NULL, compute_delta, NULL);
  pthread_create(&thread2, NULL, compute_root, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  printf("root1 = %lf\n", root1);
  printf("root2 = %lf\n", root2);

  pthread_mutex_destroy(&mutex_delta);
  pthread_cond_destroy(&cond_delta);

  return 0;
}