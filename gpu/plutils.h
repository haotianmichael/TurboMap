#ifndef _PLUTILS_H_
#define _PLUTILS_H_

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kalloc.h"
#include "minimap.h"

/* Chaining Options */

/* structure for metadata and hits */
// Sequence meta data
typedef struct {
    long i;          // read id
    int seg_id;      // seg id
    char name[200];  // name of the sequence
    uint32_t len;    // name of the sequence

    // mi data
    int n_alt;
    int is_alt;  // reference sequences only

    // sequence info
    int qlen_sum;
} mm_seq_meta_t;

typedef struct {
    int max_iter, max_dist_x, max_dist_y, max_skip, bw, min_cnt, min_score,
        is_cdna, n_seg;
    float chn_pen_gap, chn_pen_skip;
} Misc;

typedef struct {
    mm_seq_meta_t *refs;
    int n_refs;
    Misc misc;
} input_meta_t;

typedef struct {
    mm_seq_meta_t seq;

    // minimap2 input data for reads
    const char **qseqs;  // sequences for each segment          <- allocated in worker_for, freed in free_read after seeding
    int *qlens;          // query length for each segment       <- allocated in worker_for, freed in free_read after seeding
    int n_seg;           // number of segs

    int rep_len;
    int frag_gap;

    // seeding outputs
    uint64_t *mini_pos;  // minimizer positions                 <- allocated in 
    int n_mini_pos;

    // seeding output, updated in chaining
    mm128_t *a;  // array of anchors
    int64_t n;   // number of anchors = n_a

    // chaining outputs
    uint64_t *u;      // scores for chains
    int n_u;          // number of chains formed from anchors == n_reg0

    int thread_id;

} chain_read_t;

typedef struct seg_t {
    size_t start_idx;
    size_t end_idx;
} seg_t;

/* Align Options */
// Single GPU alignment task within a batch
typedef struct{

    int32_t ref_qs, ref_qe;
    int32_t ref_rs, ref_re;

    int32_t qs0, qe0;
    int32_t rs0, re0;
    int32_t rev;
    int32_t rid;

    // Anchor indices for z-drop split (mm_split_reg)
    int32_t as1;            // first valid anchor index (after bad-end filtering)
    int32_t cnt1;           // valid anchor count
}task_ctx_t;

typedef struct {
    // Input sequences (offsets into batch buffers)
    size_t qseq_offset;     // query sequence offset in batch buffer
    size_t tseq_offset;     // target sequence offset in batch buffer
    size_t junc_offset;     // junction info offset in batch buffer (0 if NULL)
    
    // Sequence lengths
    int32_t qlen;           // query length
    int32_t tlen;           // target length
    
    // Alignment parameters
    int8_t mat[25];         // scoring matrix
    int32_t w;              // bandwidth
    int32_t end_bonus;      // end bonus
    int32_t zdrop;          // z-drop
    int32_t flag;           // alignment flags
    
    // Task identification
    int32_t read_idx;       // which read this task belongs to
    int32_t reg_idx;        // which region within the read
    int32_t task_type;      // 0=left_ext, 1=gap_fill, 2=right_ext, 3=inv_align
    int32_t task_sub_idx;   // sub-index for gap filling tasks
    
    // Result storage (to be filled by GPU)
    int32_t score;          // alignment score
    int32_t max_q, max_t;   // max positions
    int32_t mqe, mqe_t;     // max score when reaching end of query
    int32_t mte, mte_q;     // max score when reaching end of target
    int32_t n_cigar;        // number of CIGAR operations
    size_t cigar_offset;    // offset in batch CIGAR buffer
    int32_t max_cigar;      // max CIGAR capacity
    uint8_t zdropped;       // whether zdropped
    uint8_t reach_end;      // whether reached end

    // P2/P3: GPU-computed alignment statistics (gpu_fix_cigar_and_stats kernel)
    // Precision note: dp_max uses integer log2 (31-__clz) vs CPU float polynomial → ±1 difference
    // gpu_stats_valid=0 when CIGAR starts with leading I/D (rare) → CPU mm_update_extra fallback
    int32_t blen;            // aligned block length (excl. ambiguous bases)
    int32_t mlen;            // match length (excl. mismatches + ambiguous)
    int32_t n_ambi;          // ambiguous base count
    int32_t dp_max;          // max local DP score (rounded from double)
    int32_t gpu_stats_valid; // 1 = GPU stats valid; 0 = use CPU mm_update_extra

    task_ctx_t task_ctx;    // coor
} gpu_align_task_t;

// Per-read alignment context
typedef struct {
    // Original mm_align_skeleton state
    mm_reg1_t *regs0;       // regions from chaining
    int32_t n_regs;         // number of regions
    uint8_t *qseq0[2];      // encoded query sequences
    int32_t n_a;            // number of anchors after squeeze
    mm128_t *a;             // anchor array
    int32_t qlen;           // query length (total length of the read)

 
    // Results will be written back to regs0
} read_align_ctx_t;

// GPU alignment batch for multiple reads
typedef struct {
    // Tasks
    int32_t n_tasks;        // number of alignment tasks
    int32_t max_tasks;      // maximum tasks capacity
    gpu_align_task_t *tasks; // array of alignment tasks
    
    // Unified sequence buffer for all tasks
    uint8_t *seq_buffer;    // buffer for all sequences
    size_t seq_buffer_size;
    size_t seq_buffer_used;
    
    // Unified CIGAR buffer for all results
    uint32_t *cigar_buffer; // buffer for all CIGAR results
    size_t cigar_buffer_size;
    size_t cigar_buffer_used;
    
    // Read contexts
    int32_t n_reads;        // number of reads being processed
    read_align_ctx_t *read_ctxs; // context for each read
    
} gpu_align_batch_t;


// Task type definitions
#define GPU_TASK_LEFT_EXT   0
#define GPU_TASK_GAP_FILL   1
#define GPU_TASK_RIGHT_EXT  2
#define GPU_TASK_INV_ALIGN  3

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
/* GPU chaining methods */
// // <lchain.h> backward, original chaining methods
// void chain_backword_cpu(const input_meta_t *meta, chain_read_t *read_arr,
//                         int n_read);
// // <fchain.c> forward chaining methods
// void chain_forward_cpu(const input_meta_t *meta, chain_read_t *read_arr,
//                        int n_read);

// <plchain.cu> gpu chaining methods
// initialization and cleanup
void init_stream_gpu(size_t *max_total_n, int *max_reads,
                     int *min_n, char gpu_config_file[],  Misc misc);  // for stream_gpu
void finish_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **batches,
                        int *num_reads, int num_batch, void *km);  // for stream_gpu
void free_stream_gpu(int n_threads); // for stream_gpu free pinned memory
// chaining method
void chain_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **in_arr_ptr, int *n_read_ptr, int thread_id, void* km);
// fine-grained pipeline API
int  launch_chain_gpu(chain_read_t *reads, int n_read, int stream_id);
void sync_chain_gpu(int stream_id);
void start_backtrack_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                         chain_read_t *reads, int stream_id,
                         void *km, int *out_n_reads);
void finish_backtrack_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                          chain_read_t *reads, int n_read,
                          int stream_id, void *km);
// multi-stream support
int  gpu_get_num_streams(void);
void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                             uint8_t *seq_buffer, uint32_t *cigar_buffer, int stream_id);
/* GPU voting-based re-chaining (replaces gpu_rechain_batch / mg_lchain_rmq).
 * Declared here so map.c and plchain.cu can share the same header.
 * Implemented in gpu/plvoting.cu. */
// plvoting_rechain_batch declared in plvoting.cuh (needs cudaStream_t)

/* <lchain.c> Chaining backtracking methods */
uint64_t *mg_chain_backtrack(void *km, int64_t n, const int32_t *f,
                             const int64_t *p, int32_t *v, int32_t *t,
                             int32_t min_cnt, int32_t min_sc, int32_t max_drop,
                             int32_t *n_u_, int32_t *n_v_);
mm128_t *compact_a(void *km, int32_t n_u, uint64_t *u, int32_t n_v, int32_t *v, mm128_t *a);


/* <map.c> Post Chaining helpers */
Misc build_misc(const mm_idx_t *mi, const mm_mapopt_t *opt, const int64_t qlen_sum, const int n_seg);
void post_chaining_helper(const mm_idx_t *mi, const mm_mapopt_t *opt,
                          chain_read_t *read, Misc misc, void *km);
int needs_rmq_rechain(const mm_mapopt_t *opt, chain_read_t* read);
void prepare_rechain_anchors(chain_read_t* read, void *km);

#ifdef __cplusplus
}
#endif  // __cplusplus

/////////////////////////////////////////////////////
///////////         Free Input Struct   /////////////
/////////////////////////////////////////////////////
// free input_iter pointers except a, because it is freed seperately.
static inline void free_read(chain_read_t *in, void* km) {
    if (in->qseqs) kfree(km, in->qseqs);
    if (in->qlens) kfree(km, in->qlens);
    in->qseqs = 0, in->qlens = 0;
    in->a = 0, in->u = 0;
}

static inline void free_meta_struct(input_meta_t *meta, void *km) {
    if (meta->refs) kfree(km, meta->refs);
}
#endif  // _PLUTILS_H_
