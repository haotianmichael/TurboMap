#ifndef _PLALIGN_H_
#define _PLALIGN_H_

#include "plutils.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>


struct gasal_res{
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
	uint8_t *cigar;
	uint32_t *n_cigar_ops;
};
typedef struct gasal_res gasal_res_t;

typedef struct {
    // === Sequence Data ===
    // Unpacked sequences (raw)
    uint8_t *d_unpacked_query;
    uint8_t *d_unpacked_target;
    
    // Packed sequences (AGATHA format)
    uint32_t *d_packed_query;
    uint32_t *d_packed_target;
    
    // === Metadata Arrays ===
    uint32_t *d_query_offsets;
    uint32_t *d_target_offsets;
    uint32_t *d_query_lens;
    uint32_t *d_target_lens;
    
    // === Results ===
    gasal_res_t *device_res;       // Device-side result struct
    gasal_res_t *device_res_ptrs;  // Host-side copy of device pointers
    gasal_res_t *host_res;         // Host-side results
    
    // === AGATHA Specific ===
    short2 *d_global_buffer;        // AGATHA kernel working buffer
    short2 *h_sort_buffer;         // Host buffer for sorting
    uint4 *d_packed_tb_matrices;   // Traceback matrices (if needed)
    
    // === Task Mapping ===
    int32_t *d_task_to_align_id;   // Maps framework task ID to AGATHA alignment ID
    int32_t *h_task_to_align_id;   // Host copy for mapping
    
    // === Memory Management ===
    size_t max_tasks;
    size_t max_query_bytes;
    size_t max_target_bytes;
    uint32_t max_query_len;        // Maximum single query length
    
    // === CUDA Resources ===
    cudaStream_t stream;
    
    // === Configuration ===
    int kernel_blocks;
    int kernel_threads;
    int slice_width;
    int z_threshold;
    int band_width;
} gpu_align_storage_t;

// Configuration structure for alignment parameters
typedef struct {
    int blocks;
    int threads;
    int slice_width;
    int z_threshold;
    int band_width;
    int8_t match_score;
    int8_t mismatch_score;
    int8_t gap_open;
    int8_t gap_extend;
} align_config_t;

//match/mismatch and gap penalties
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
	int32_t slice_width;
	int32_t z_threshold;
	int32_t band_width;
} gasal_subst_scores;


void gasal_copy_subst_scores(gasal_subst_scores *subst);
void gpu_align_cleanup();

#endif