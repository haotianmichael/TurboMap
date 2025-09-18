#ifndef _PL_MANAGER_CUH_
#define _PL_MANAGER_CUH_
#include "plutils.h"
#include <pthread.h>

// Global seeded queue structure
typedef struct {
    chain_read_t *reads;
    int capacity;
    int count;
    int head, tail;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int finished_workers;  // count of workers that finished seeding
    int total_workers;     // total number of seeding workers
} seeded_queue_t;


seeded_queue_t* init_seeded_queue(int capacity, int n_workers) {
    seeded_queue_t *q = (seeded_queue_t*)malloc(sizeof(seeded_queue_t));
    q->reads = (chain_read_t*)malloc(capacity * sizeof(chain_read_t));
    q->capacity = capacity;
    q->count = 0;
    q->head = 0;
    q->tail = 0;
    q->finished_workers = 0;
    q->total_workers = n_workers;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    return q;
}

// Destroy global seeded queue
void destroy_seeded_queue(seeded_queue_t *q) {
    if (q) {
        free(q->reads);
        pthread_mutex_destroy(&q->mutex);
        pthread_cond_destroy(&q->not_empty);
        pthread_cond_destroy(&q->not_full);
        free(q);
    }
}


// Push seeded read to global queue (blocking if full)
void push_seeded_read(seeded_queue_t *q, chain_read_t *read) {
    pthread_mutex_lock(&q->mutex);
    
    // Wait if queue is full
    while (q->count == q->capacity) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    
    // Copy read to queue
    q->reads[q->tail] = *read;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    
    // Signal that queue is not empty
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

// Pop seeded read from global queue (blocking if empty, unless all workers finished)
int pop_seeded_read(seeded_queue_t *q, chain_read_t *read) {
    pthread_mutex_lock(&q->mutex);
    
    // Wait if queue is empty and not all workers finished
    while (q->count == 0 && q->finished_workers < q->total_workers) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    
    // Return 0 if queue is empty and all workers finished
    if (q->count == 0 && q->finished_workers >= q->total_workers) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }
    
    // Pop read from queue
    *read = q->reads[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    
    // Signal that queue is not full
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

// Mark worker as finished
void mark_worker_finished(seeded_queue_t *q) {
    pthread_mutex_lock(&q->mutex);
    q->finished_workers++;
    pthread_cond_broadcast(&q->not_empty); // Wake up all waiting consumers
    pthread_mutex_unlock(&q->mutex);
}


#endif // __PL_MANAGER_CUH_