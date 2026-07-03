#ifndef KTHREAD_H
#define KTHREAD_H

#ifdef __cplusplus
extern "C" {
#endif

void kt_for(int n_threads, void (*func)(void*,long,int), void *data, long n);
void kt_for_async(int n_threads, void (*producer_func)(void*,long,int), void *producer_data, long n,
                  void (*consumer_func)(void*), void *consumer_data);
void kt_pipeline(int n_threads, void *(*func)(void*, int, void*), void *shared_data, int n_steps);

#ifdef __cplusplus
}
#endif

#endif
