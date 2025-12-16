#ifndef _PLALIGN_H_
#define _PLALIGN_H_

#include "plutils.h"
#include "../ksw2.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

/*ChainQueue*/


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
    q->reads = (chain_read_t*)calloc(capacity, sizeof(chain_read_t));  // Use calloc to zero-initialize
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
        // Free any remaining allocated memory in queue slots
        for (int i = 0; i < q->capacity; i++) {
            if (q->reads[i].qlens) free(q->reads[i].qlens);
            if (q->reads[i].qseqs) free(q->reads[i].qseqs);
            if (q->reads[i].mini_pos) free(q->reads[i].mini_pos);
            if (q->reads[i].a) free(q->reads[i].a);
        }
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
    
   // Free any old data at this position (should be NULL after pop, but check anyway)
    if (queue_read->qlens) {
        free(queue_read->qlens);
        queue_read->qlens = NULL;
    }
    if (queue_read->qseqs) {
        free(queue_read->qseqs);
        queue_read->qseqs = NULL;
    }
    if (queue_read->mini_pos) {
        free(queue_read->mini_pos);
        queue_read->mini_pos = NULL;
    }
    if (queue_read->a) {
        free(queue_read->a);
        queue_read->a = NULL;
    }

    // Copy non-pointer fields manually to avoid copying pointers from kmalloc
    queue_read->seq = read->seq;
    queue_read->n_seg = read->n_seg;
    queue_read->rep_len = read->rep_len;
    queue_read->frag_gap = read->frag_gap;
    queue_read->n_mini_pos = read->n_mini_pos;
    queue_read->n = read->n;
    queue_read->n_u = read->n_u;
    queue_read->thread_id = read->thread_id;

    // Now initialize all pointers to NULL first
    queue_read->qlens = NULL;
    queue_read->qseqs = NULL;
    queue_read->mini_pos = NULL;
    queue_read->a = NULL;
    queue_read->u = NULL;

    // Deep copy qlens and qseqs
    if (read->n_seg > 0 && read->qlens && read->qseqs) { 
        queue_read->qlens = (int*)malloc(sizeof(int) * read->n_seg);
        memcpy(queue_read->qlens, read->qlens, sizeof(int) * read->n_seg);
        
        queue_read->qseqs = (const char**)malloc(sizeof(const char*) * read->n_seg);
        memcpy(queue_read->qseqs, read->qseqs, sizeof(const char*) * read->n_seg);
    }
    
    // Deep copy mini_pos
    if (read->n_mini_pos > 0 && read->mini_pos) {
        queue_read->mini_pos = (uint64_t*)malloc(read->n_mini_pos * sizeof(uint64_t));
        memcpy(queue_read->mini_pos, read->mini_pos, read->n_mini_pos * sizeof(uint64_t));
    }
    
    // Deep copy anchors array
    if (read->n > 0  && read->a) {
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
    // Clear the pointers in queue to prevent double free
    q->reads[q->head].qlens = NULL;
    q->reads[q->head].qseqs = NULL;
    q->reads[q->head].mini_pos = NULL;
    q->reads[q->head].a = NULL;
    q->reads[q->head].u = NULL;
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



/*KSW */
typedef struct __attribute__((aligned(8))){
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
    int32_t *mqe;           // max score when reaching end of query
	int32_t *mqe_t;         // target position when reaching end of query
	int32_t *mte;           // max score when reaching end of target
	int32_t *mte_q;         // query position when reaching end of target
	uint8_t *cigar;
	uint32_t *n_cigar_ops;
}gasal_res_t;

// Configuration structure for alignment parameters
typedef struct {
    int32_t blocks;
    int32_t threads;
    int32_t slice_width;
    int32_t z_threshold;
    int32_t band_width;
    int8_t match_score;
    int8_t mismatch_score;
    int8_t gap_open;
    int8_t gap_extend;
    int8_t gap_open_long;
    int8_t gap_extend_long;
} align_config_t;

//match/mismatch and gap penalties
typedef struct{
	int8_t match;
	int8_t mismatch;
	int8_t gap_open;
	int8_t gap_extend;
	int8_t gap_open_long;
	int8_t gap_extend_long;
	int32_t slice_width;
	int32_t z_threshold;
	int32_t band_width;
} gasal_subst_scores;


void gasal_copy_subst_scores(gasal_subst_scores *subst);
void gpu_align_cleanup();

// Set the device memory pointer for alignment operations
// This should be called before gpu_align_batch_execute
void gpu_align_set_device_mem(void *dev_mem_ptr);

#endif