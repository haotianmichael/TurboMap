#ifndef _PLMEM_CUH_
#define _PLMEM_CUH_
#include "hipify.cuh"
#include "plchain.h"
#include "plutils.h"

#ifndef MAX_MICRO_BATCH
#define MAX_MICRO_BATCH 8
#endif // MAX_MICRO_BATCH

#define OneK 1024
#define OneM (OneK*1024)
#define OneG (OneM*1024)

/* ======== GPU Memory Arena ========
 * Single large allocation shared between chain and alignment phases.
 * Phase transitions just reassign pointers (zero cudaMalloc/cudaFree overhead).
 */
typedef struct {
    void  *base;        // single cudaMalloc'd chunk
    size_t total_size;  // total bytes allocated
    size_t offset;      // current bump-pointer offset
} gpu_arena_t;

typedef enum {
    GPU_PHASE_CHAIN = 0,   // chain + backtrack + voting
    GPU_PHASE_ALIGN = 1    // KSW alignment
} gpu_mem_phase_t;

static inline void *arena_alloc(gpu_arena_t *a, size_t bytes) {
    size_t aligned_off = (a->offset + 255) & ~(size_t)255;  // 256-byte alignment for coalescing
    void *ptr = (char*)a->base + aligned_off;
    a->offset = aligned_off + bytes;
    if (a->offset > a->total_size) {
        fprintf(stderr, "[FATAL] GPU arena OOM: need %zu, have %zu (alloc %zu at offset %zu)\n",
                a->offset, a->total_size, bytes, aligned_off);
        abort();
    }
    return ptr;
}

static inline void arena_reset(gpu_arena_t *a) {
    a->offset = 0;
}

static inline size_t arena_remaining(gpu_arena_t *a) {
    return a->total_size - a->offset;
}

typedef struct {
    int index;       // read index / batch index
    int griddim;     // grid for range selection kernel. 
    int size;        // number of reads in the batch
    size_t total_n;  // number of anchors in the batch
    size_t cut_num;  // number of cuts in the batch

    // array size: number of anchors in the batch
    int32_t *ax;  // (int32_t) a[].x
    int32_t *ay;  // (int32_t) a[].y
    int8_t* sid;  // a[].y >> 40 & 0xff
    int32_t *xrev; // a[].x >> 32
    int32_t *yrev; // a[].y >> 32 (contains q_span and seg_id)
    // outputs
    int32_t *f;   // score
    uint16_t *p;  // predecessor

    // array size: number of cuts in the batch / long_seg_cut
    // total long segs number till this batch
    unsigned int *long_segs_num;

    // start index for each block in range selection
    /***** range selection block assiagnment
     * One block only gets assgined one read or part of one read.
     *  start_idx:      idx of the first anchor assigned to each block
     *  read_end_idx:   idx of the last anchor OF THE READ assigned to each
     * block if a read is devided into several blocks, all the blocks take the
     * last anchor index of the read cut_start_idx:  idx of the first cut this
     * block needs to make
     */
    // array size: grid dimension
    size_t *start_idx;
    size_t *read_end_idx;
    size_t *cut_start_idx;
} hostMemPtr;

typedef struct {
    // array size: number of cuts in the batch / long_seg_cut
    seg_t *long_segs_og_idx;                   // start & end idx of long segs in the original micro batch
    unsigned int *total_long_segs_num; // sum of mini batch long_segs_num
    size_t *total_long_segs_n; // number of anchors in all the long segs
    int32_t *f_long;   // score for long segs
    uint16_t *p_long;  // predecessor for long segs
} longMemPtr;

typedef struct {
    // ========== Arena for dynamic phase-based memory sharing ==========
    gpu_arena_t arena;             // single GPU allocation shared between phases
    gpu_mem_phase_t current_phase; // which phase's buffers are currently set up
    // Saved chain-phase allocation parameters (for re-setup after alignment)
    size_t saved_anchor_per_batch;
    int    saved_range_grid_size;
    int    saved_num_cut;

    int size;
    int griddim;
    size_t total_n;
    size_t num_cut;
    // device memory ptrs
    // data array
    int32_t *d_ax;
    int32_t *d_ay;
    int8_t *d_sid;  // a[].y >> 40 & 0xff
    int32_t *d_xrev; // a[].x >> 32
    int32_t *d_yrev; // a[].y >> 32 (contains q_span and seg_id)
    int32_t *d_range;
    int32_t *d_f;   // score
    uint16_t *d_p;  // predecessor

    // range selection index
    size_t *d_start_idx;
    size_t *d_read_end_idx;
    size_t *d_cut_start_idx;

    // cut
    size_t *d_cut;  // cut
    unsigned int *d_long_seg_count; // total number of long seg (aggregated accross micro batches)
    seg_t *d_long_seg;              // start & end idx of long segs in the long seg buffer (aggregated across micro batches)
    seg_t *d_long_seg_og;           // start & end idx of long seg in the micro batch. (aggregated accross micro batches)
    unsigned int *d_mid_seg_count;  // private to micro batch
    seg_t *d_mid_seg;               // private to micro batch

    // long segement buffer
    unsigned *d_map;            // pre-allocated in plmem.cu (max_long_segs elements)
    size_t d_map_capacity;      // capacity in elements
    int32_t *d_ax_long, *d_ay_long;
    int8_t *d_sid_long;
    int32_t *d_range_long;
    size_t *d_total_n_long;
    size_t buffer_size_long;
    int32_t *d_f_long;  // score, size: buffer_size_long * sizeof(int32_t)
    uint16_t *d_p_long;  // predecessor, size: buffer_size_long * sizeof(uint16_t)

    // ========== Chain Backtrack Pre-allocated Buffers ==========
    // Pre-allocated to eliminate hot-path cudaMalloc/cudaFree in plbacktrack_gpu.
    // d_bt_*_in: SEPARATE input buffers for backtrack (sized: anchor_per_batch * micro_batch)
    //   Separate from d_ax/d_ay/d_f/d_p so host_mems can be reused across batches.
    // d_bt_* working buffers: also sized anchor_per_batch * micro_batch.

    // Backtrack input (read from host, separate from d_ax so chain can overlap)
    int32_t  *d_bt_ax_in;    // anchor x
    int32_t  *d_bt_ay_in;    // anchor y
    int32_t  *d_bt_xrev_in;  // anchor xrev
    int32_t  *d_bt_yrev_in;  // anchor yrev
    int32_t  *d_bt_f_in;     // chain score
    uint16_t *d_bt_p_in;     // predecessor

    // Anchor-sized working (int64_t × anchor_per_batch * micro_batch):
    int64_t  *d_bt_zx;       // z-scores x (filter output + sort in-place)
    int64_t  *d_bt_zy;       // z-scores y (sort key)
    int64_t  *d_bt_v;        // w_y scratch (sort value), then w array y
    int64_t  *d_bt_p_abs;    // expanded int64 predecessors; reused as w_x
    // Anchor-sized (other types):
    int32_t  *d_bt_t;        // track array (anchor filter)
    uint64_t *d_bt_u;        // chain u-array output (per-anchor)
    int32_t  *d_bt_ax_out;   // compacted anchor x
    int32_t  *d_bt_ay_out;   // compacted anchor y
    int32_t  *d_bt_xrev_out; // compacted xrev
    int32_t  *d_bt_yrev_out; // compacted yrev
    // Read-sized (int × max_reads * micro_batch):
    int      *d_bt_n_a;          // anchors per read
    int      *d_bt_offset;       // read anchor start offset
    int      *d_bt_ofs_end;      // read filter endpoint
    int      *d_bt_num_elements; // per-read count scratch
    int      *d_bt_n_v;          // # valid anchors per read
    int      *d_bt_n_u;          // # chains per read
    // CUB segmented-sort temp:
    void     *d_bt_cub_tmp;
    size_t    d_bt_cub_tmp_size;
    // Capacities stored for assertions:
    size_t    d_bt_max_total_n;  // anchor capacity of d_bt_* anchor arrays
    size_t    d_bt_max_n_reads;  // read capacity of d_bt_* read arrays

    // ========== Alignment Buffers (unified allocation) ==========
    size_t max_align_tasks;       // max number of alignment tasks
    size_t max_align_seq_bytes;   // max sequence bytes
    size_t max_align_query_len;   // max query length

    // Two-tier batch processing configuration
    size_t short_task_batch_size; // batch size for short tasks (max(qlen,tlen) <= 1000bp)
    size_t long_task_batch_size;  // batch size for long tasks (max(qlen,tlen) > 1000bp)
    size_t short_task_max_len;    // max sequence length for short tasks (1000bp)
    
    // Sequence data
    uint8_t *d_align_unpacked_query;
    uint8_t *d_align_unpacked_target;
    uint32_t *d_align_packed_query;
    uint32_t *d_align_packed_target;
    uint32_t *d_align_query_offsets;
    uint32_t *d_align_target_offsets;
    uint32_t *d_align_query_lens;
    uint32_t *d_align_target_lens;
    int32_t *d_align_flag;
    int32_t *d_align_bw;

    // Working buffers
    void *d_align_global_buffer;     // AGATHA working buffer
    void *d_align_ksw_temp_buffer;   // KSW temp buffer
    size_t align_ksw_temp_per_task;  // KSW temp size per task

    // Backtrack buffers
    uint8_t *d_align_backtrack_p;
    int *d_align_backtrack_off;
    int *d_align_backtrack_off_end;
    int *d_align_backtrack_n_col;
    uint32_t *d_align_cigar_buffer;
    int *d_align_cigar_lengths;
    size_t max_align_backtrack_size; // per task
    size_t max_align_cigar_len;      // per task

    // P1: Compact CIGAR buffers (GPU compaction eliminates stride padding)
    uint32_t *d_align_compact_cigar;   // compact CIGAR (no padding), same max size as stride buffer
    uint32_t *d_align_compact_offsets; // per-task start offset in compact buffer (exclusive prefix sum)
    void     *d_align_cub_tmp;         // CUB DeviceScan temporary storage
    size_t    align_cub_tmp_size;      // size of CUB temporary storage

    // P2: GPU alignment statistics (computed by gpu_fix_cigar_and_stats kernel)
    int32_t *d_align_blen;            // blen per task
    int32_t *d_align_mlen;            // mlen per task
    int32_t *d_align_n_ambi;          // n_ambi per task
    int32_t *d_align_dp_max;          // dp_max per task
    int32_t *d_align_gpu_stats_valid; // 1 if GPU stats valid (no leading I/D), 0 = CPU fallback

    // Result buffers
    void *d_align_device_res;        // gasal_res_t structure
    void *d_align_ez_array;          // ksw_extz_t array
    int32_t *d_align_scores;
    int32_t *d_align_query_ends;
    int32_t *d_align_target_ends;
    int32_t *d_align_mqe;            // max score when reaching end of query
    int32_t *d_align_mqe_t;          // target position when reaching end of query
    int32_t *d_align_mte;            // max score when reaching end of target
    int32_t *d_align_mte_q;          // query position when reaching end of target
    int32_t *d_align_zdropped;       // z-drop flag per task
    int32_t *d_align_task_to_align_id;
    int8_t *d_align_mat;             // scoring matrix

    // Persistent kernel configuration
    int   n_align_concurrent_blocks; // number of slots for persistent kernel
    int  *d_align_task_counter;      // atomic task counter (reset before each kernel launch)

    // ========== Voting Pre-allocated Buffers ==========
    // Pre-allocated to eliminate hot-path cudaMalloc/cudaFree in run_voting_minibatch.
    // Sized by d_vt_max_anchors (= anchor_per_batch * micro_batch, same as backtrack).
    // n_bins upper bound: same as max_anchors (each anchor can map to at most 1 bin).
    size_t d_vt_max_anchors;         // capacity of anchor-sized arrays
    // Anchor-sized (uint64_t):
    uint64_t *d_vt_ax;               // upload anchor x
    uint64_t *d_vt_ay;               // upload anchor y
    uint64_t *d_vt_bx;               // output compacted x
    uint64_t *d_vt_by;               // output compacted y
    // Anchor-sized (int32_t):
    int32_t  *d_vt_mark;             // anchor keep/discard flag
    int32_t  *d_vt_anchor_seg;       // anchor → segment mapping
    int32_t  *d_vt_out_pos;          // scatter output positions
    // Bin-sized (reuse max_anchors as upper bound for bins):
    int32_t  *d_vt_votes;            // voting histogram
    int8_t   *d_vt_keep_bin;         // bin keep flag
    int32_t  *d_vt_seg_start;        // segment start flags
    int32_t  *d_vt_seg_id;           // segment IDs
    int32_t  *d_vt_seg_cnt_flat;     // per-segment anchor counts
    // Read-sized (int32_t × max_reads, reuse d_bt_max_n_reads):
    int32_t  *d_vt_bin_off;          // bin offset per group (+1)
    int32_t  *d_vt_anchor_off;       // anchor offset per group (+1)
    int32_t  *d_vt_ref_min;          // min ref pos per group
    int32_t  *d_vt_bin_size;         // effective bin size per group
    int32_t  *d_vt_nsegs;            // segments per group
    int32_t  *d_vt_ncompact;         // compacted anchors per group

    // ========== Deferred D2H metadata (backtrack → voting fusion) ==========
    // Set by plbacktrack_gpu, freed by plbacktrack_d2h_finish.
    // Allows anchor data to stay on GPU between backtrack and voting stages.
    int       bt_n_reads;            // reads in current batch
    size_t    bt_total_n;            // total anchors (pre-backtrack count, array capacity)
    int      *bt_h_offset;           // host: per-read offset in d_bt_*_out
    int      *bt_h_n_u;              // host: chains per read

    // ========== Pre-allocated Pinned Host Buffers (alignment D2H/H2D) ==========
    // Allocated once during init, reused across all gpu_align_batch_execute calls.
    // Eliminates 22× cudaMallocHost/cudaFreeHost per batch (each is a mlock syscall).
    size_t    h_align_max_batch;          // max_batch_size used to size these buffers
    uint32_t *h_align_compact_cigar;     // compact CIGAR data
    uint32_t *h_align_compact_offsets;   // per-task CIGAR offsets
    int      *h_align_cigar_lengths;     // per-task CIGAR lengths
    int32_t  *h_align_blen;
    int32_t  *h_align_mlen;
    int32_t  *h_align_n_ambi;
    int32_t  *h_align_dp_max;
    int32_t  *h_align_gpu_stats_valid;
    int32_t  *h_align_scores;
    int32_t  *h_align_query_ends;
    int32_t  *h_align_target_ends;
    int32_t  *h_align_mqe;
    int32_t  *h_align_mqe_t;
    int32_t  *h_align_mte;
    int32_t  *h_align_mte_q;
    int32_t  *h_align_zdropped;
    uint32_t *h_align_query_offsets;
    uint32_t *h_align_target_offsets;
    uint32_t *h_align_query_lens;
    uint32_t *h_align_target_lens;
    int32_t  *h_align_flag;
    int32_t  *h_align_bw;
    int32_t  *h_align_task_to_align_id;

} deviceMemPtr;

typedef struct stream_ptr_t{
    chain_read_t *reads;
    size_t n_read;
    hostMemPtr host_mems[MAX_MICRO_BATCH];
    int cur_hm = 0;  // unused, kept for ABI compat
    longMemPtr long_mem;
    deviceMemPtr dev_mem;
    cudaStream_t cudastream;
    cudaEvent_t stopevent, startevent;
    bool busy = false;
} stream_ptr_t;

typedef struct gputSetup_t {
    int num_stream;
    stream_ptr_t *streams;
    size_t max_anchors_stream, max_num_cut, long_seg_buffer_size_stream;
    int max_range_grid;
} streamSetup_t;

extern streamSetup_t stream_setup;

/* per-stream accessors (defined in plchain.cu) */
#ifdef __cplusplus
extern "C" {
#endif
deviceMemPtr* gpu_get_dev_mem(int stream_id);
cudaStream_t  gpu_get_cudastream(int stream_id);
#ifdef __cplusplus
}
#endif

/* memory management methods */
// initialization and cleanup
void plmem_initialize(size_t *max_total_n, int *max_read, int *min_n);
void plmem_stream_initialize(size_t *max_total_n, int *max_read, int *min_n, char* gpu_config_file);
void plmem_stream_cleanup();

// alloc and free
void plmem_malloc_host_mem(hostMemPtr *host_mem, size_t anchor_per_batch,
                           int range_grid_size, size_t buffer_size_long);
void plmem_malloc_long_mem(longMemPtr *long_mem, size_t buffer_size_long);
void plmem_free_host_mem(hostMemPtr *host_mem);
void plmem_free_long_mem(longMemPtr *long_mem);
void plmem_malloc_device_mem(deviceMemPtr *dev_mem, size_t anchor_per_batch,
                             int range_grid_size, int num_cut);
void plmem_free_device_mem(deviceMemPtr *dev_mem);

// Phase transitions: reclaim chain memory for alignment and vice versa.
// Call plmem_phase_to_align() after finish_backtrack_gpu() returns.
// Call plmem_phase_to_chain() after alignment completes.
void plmem_phase_to_align(deviceMemPtr *dev_mem);
void plmem_phase_to_chain(deviceMemPtr *dev_mem);

// data movement
void plmem_reorg_input_arr(chain_read_t *reads, int n_read,
                           hostMemPtr *host_mem, range_kernel_config_t config);
void plmem_async_h2d_memcpy(stream_ptr_t *stream_ptrs);
void plmem_async_h2d_short_memcpy(stream_ptr_t *stream_ptrs, size_t uid);
void plmem_sync_h2d_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem);
void plmem_async_d2h_memcpy(stream_ptr_t *stream_ptrs);
void plmem_async_d2h_short_memcpy(stream_ptr_t *stream_ptrs, size_t uid);
void plmem_async_d2h_long_memcpy(stream_ptr_t *stream_ptrs);
void plmem_sync_d2h_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem);
#endif  // _PLMEM_CUH_