#ifndef _PLALIGN_H_
#define _PLALIGN_H_

#include "plutils.h"
#include "../ksw2.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>


typedef struct {
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
	uint8_t *cigar;
	uint32_t *n_cigar_ops;
}gasal_res_t;

typedef struct {

    // === kernel configuration ===
    int32_t kernel_blocks;
    int32_t kernel_threads;
    size_t max_tasks;
    size_t max_query_bytes;
    size_t max_target_bytes;
    uint32_t max_query_len;        
    
    // === Sequence Data ===
    uint8_t *d_unpacked_query;
    uint8_t *d_unpacked_target;
    uint32_t *d_packed_query;
    uint32_t *d_packed_target;
    uint32_t *d_query_offsets;
    uint32_t *d_target_offsets;
    uint32_t *d_query_lens;
    uint32_t *d_target_lens;

    // === CUDA Resources ===
    cudaStream_t stream;
    
    // === Results ===
    gasal_res_t *device_res;       // Device-side result struct
    gasal_res_t *device_res_ptrs;  // Host-side copy of device pointers
    gasal_res_t *host_res;         // Host-side results
    ksw_extz_t *d_ez_array;
    
    // === KSW Backtracking Arrays ===
    uint8_t *d_backtrack_p;        // Backtrack matrix buffer (per task)
    int *d_backtrack_off;          // Offset array buffer (per task)
    int *d_backtrack_off_end;          // Offset array buffer end(per task)
    int *d_backtrack_n_col;        // Number of columns for each task
    uint32_t *d_cigar_buffer;      // CIGAR operations buffer
    int *d_cigar_lengths;          // Length of CIGAR for each task
    uint32_t *h_cigar_buffer;      // Host CIGAR buffer
    int *h_cigar_lengths;          // Host CIGAR lengths
    size_t max_backtrack_size;     // Max backtrack matrix size per task
    size_t max_cigar_len;          // Max CIGAR length per task
    int8_t* mat;

    // === AGATHA Specific ===
    short2 *d_global_buffer;        // AGATHA kernel working buffer
    short2 *h_sort_buffer;         // Host buffer for sorting
    uint4 *d_packed_tb_matrices;   // Traceback matrices (if needed)
    
    // === Task Mapping ===
    int32_t *d_task_to_align_id;   // Maps framework task ID to AGATHA alignment ID
    int32_t *h_task_to_align_id;   // Host copy for mapping
    
    
    void *d_ksw_temp_buffer;
    size_t ksw_temp_per_task; 
} gpu_align_storage_t;

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

#endif