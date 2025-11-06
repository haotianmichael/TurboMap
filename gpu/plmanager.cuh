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
    
    // Deep copy read to queue - 使用malloc分配独立内存
    chain_read_t *queue_read = &q->reads[q->tail];
    
    // Copy basic fields
    *queue_read = *read;
    
    // Deep copy qlens
    if (read->n_seg > 0) {
        queue_read->qlens = (int*)malloc(sizeof(int) * read->n_seg);
        memcpy(queue_read->qlens, read->qlens, sizeof(int) * read->n_seg);
        
        queue_read->qseqs = (const char**)malloc(sizeof(const char*) * read->n_seg);
        memcpy(queue_read->qseqs, read->qseqs, sizeof(const char*) * read->n_seg);
    }
    
    // Deep copy mini_pos
    if (read->n_mini_pos > 0) {
        queue_read->mini_pos = (uint64_t*)malloc(read->n_mini_pos * sizeof(uint64_t));
        memcpy(queue_read->mini_pos, read->mini_pos, read->n_mini_pos * sizeof(uint64_t));
    }
    
    // Deep copy anchors array
    if (read->n > 0) {
        queue_read->a = (mm128_t*)malloc(read->n * sizeof(mm128_t));
        memcpy(queue_read->a, read->a, read->n * sizeof(mm128_t));
    }
    
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

// Free the independently allocated memory for a queue read
void free_queue_read(chain_read_t *read) {
    if (read->qlens) {
        free(read->qlens);
        read->qlens = NULL;
    }
    if (read->qseqs) {
        free(read->qseqs);
        read->qseqs = NULL;
    }
    if (read->mini_pos) {
        free(read->mini_pos);
        read->mini_pos = NULL;
    }
    if (read->a) {
        free(read->a);
        read->a = NULL;
    }
} 

#endif // __PL_MANAGER_CUH_