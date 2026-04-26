/* GPU memory management  */


#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <cub/cub.cuh>
#include "plmem.cuh"
#include "plrange.cuh"
#include "plscore.cuh"
#include <time.h>
#define CUDA_DEVICE 0
typedef struct {
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
	int32_t *mqe;
	int32_t *mqe_t;
	int32_t *mte;
	int32_t *mte_q;
	int32_t *zdropped;
	uint8_t *cigar;
	uint32_t *n_cigar_ops;
}gasal_res_t;
typedef struct {
    int32_t max;
    uint32_t zdropped:1;
    int32_t max_q, max_t;
    int32_t mqe, mqe_t;
    int32_t mte, mte_q;
    int32_t score;
    int32_t m_cigar, n_cigar;
    int32_t reach_end;
    uint32_t *cigar;
} ksw_extz_t;
void plmem_malloc_host_mem(hostMemPtr *host_mem, size_t anchor_per_batch,
                           int range_grid_size, size_t buffer_size_long) {
    // data array
    cudaMallocHost((void**)&host_mem->ax, anchor_per_batch * sizeof(int32_t));
    cudaMallocHost((void**)&host_mem->ay, anchor_per_batch * sizeof(int32_t));
    cudaMallocHost((void**)&host_mem->sid, anchor_per_batch * sizeof(int8_t));
    cudaMallocHost((void**)&host_mem->xrev, anchor_per_batch * sizeof(int32_t));
    cudaMallocHost((void**)&host_mem->yrev, anchor_per_batch * sizeof(int32_t));
    cudaMallocHost((void**)&host_mem->f, anchor_per_batch * sizeof(int32_t));
    cudaMallocHost((void**)&host_mem->p, anchor_per_batch * sizeof(uint16_t));

    //index
    cudaMallocHost((void**)&host_mem->start_idx, range_grid_size * sizeof(size_t));
    cudaMallocHost((void**)&host_mem->read_end_idx, range_grid_size * sizeof(size_t));
    cudaMallocHost((void**)&host_mem->cut_start_idx, range_grid_size * sizeof(size_t));

    cudaMallocHost((void**)&host_mem->long_segs_num, sizeof(unsigned int));
    cudaCheck();
}

void plmem_malloc_long_mem(longMemPtr *long_mem, size_t buffer_size_long) {
    // data array
    cudaMallocHost((void**)&long_mem->long_segs_og_idx, buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t));
    cudaMallocHost((void**)&long_mem->f_long, buffer_size_long * sizeof(int32_t));
    cudaMallocHost((void**)&long_mem->p_long, buffer_size_long * sizeof(uint16_t));
    cudaMallocHost((void**)&long_mem->total_long_segs_num, sizeof(unsigned int));
    cudaMallocHost((void**)&long_mem->total_long_segs_n, sizeof(size_t));
    cudaCheck();
}

void plmem_free_host_mem(hostMemPtr *host_mem) {
    cudaFreeHost(host_mem->ax);
    cudaFreeHost(host_mem->ay);
    cudaFreeHost(host_mem->sid);
    cudaFreeHost(host_mem->xrev);
    cudaFreeHost(host_mem->yrev);
    cudaFreeHost(host_mem->f);
    cudaFreeHost(host_mem->p);
    cudaFreeHost(host_mem->start_idx);
    cudaFreeHost(host_mem->read_end_idx);
    cudaFreeHost(host_mem->cut_start_idx);
    cudaFreeHost(host_mem->long_segs_num);
    cudaCheck();
}

void plmem_free_long_mem(longMemPtr *long_mem) { 
    cudaFreeHost(long_mem->long_segs_og_idx);
    cudaFreeHost(long_mem->f_long);
    cudaFreeHost(long_mem->p_long);
    cudaFreeHost(long_mem->total_long_segs_num);
    cudaFreeHost(long_mem->total_long_segs_n);
    cudaCheck();
}


/* ======== Arena-based GPU memory helpers ======== */

static size_t cub_sort_tmp_size(size_t n, size_t n_segs) {
    size_t tmp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, tmp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int)n, (int)n_segs,
        (int*)nullptr, (int*)nullptr,
        0, (int)(sizeof(int64_t) * 8));
    return tmp_bytes;
}

static size_t cub_scan_tmp_size(size_t n) {
    size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                  (int*)nullptr, (int*)nullptr, (int)n);
    return tmp_bytes;
}

/* Set up chain + backtrack + voting buffers from arena.
 * Called during init (chain phase first) and after alignment completes. */
static void setup_chain_phase(deviceMemPtr *dev_mem, size_t anchor_per_batch,
                               int range_grid_size, int num_cut) {
    gpu_arena_t *a = &dev_mem->arena;
    arena_reset(a);

    int mb = score_kernel_config.micro_batch;
    size_t bt_anchor_total = anchor_per_batch * mb;

    // ---- Chain anchor buffers ----
    dev_mem->d_ax    = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_ay    = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_sid   = (int8_t*)arena_alloc(a, anchor_per_batch * sizeof(int8_t));
    dev_mem->d_xrev  = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_yrev  = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_range = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_f     = (int32_t*)arena_alloc(a, anchor_per_batch * sizeof(int32_t));
    dev_mem->d_p     = (uint16_t*)arena_alloc(a, anchor_per_batch * sizeof(uint16_t));

    // ---- Index buffers ----
    dev_mem->d_start_idx     = (size_t*)arena_alloc(a, range_grid_size * sizeof(size_t));
    dev_mem->d_read_end_idx  = (size_t*)arena_alloc(a, range_grid_size * sizeof(size_t));
    dev_mem->d_cut_start_idx = (size_t*)arena_alloc(a, range_grid_size * sizeof(size_t));

    // ---- Cut buffers ----
    size_t max_long_segs = dev_mem->buffer_size_long /
        (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit);
    size_t long_seg_size = max_long_segs * sizeof(seg_t);
    size_t mid_seg_size = num_cut / (score_kernel_config.mid_seg_cutoff + 1) * sizeof(seg_t);

    dev_mem->d_cut            = (size_t*)arena_alloc(a, num_cut * sizeof(size_t));
    dev_mem->d_long_seg_count = (unsigned int*)arena_alloc(a, sizeof(unsigned int));
    dev_mem->d_long_seg       = (seg_t*)arena_alloc(a, long_seg_size);
    dev_mem->d_long_seg_og    = (seg_t*)arena_alloc(a, long_seg_size);
    dev_mem->d_map            = (unsigned*)arena_alloc(a, max_long_segs * sizeof(unsigned));
    dev_mem->d_map_capacity   = max_long_segs;
    dev_mem->d_mid_seg_count  = (unsigned int*)arena_alloc(a, sizeof(unsigned int));
    dev_mem->d_mid_seg        = (seg_t*)arena_alloc(a, mid_seg_size);

    // ---- Long segment buffers ----
    dev_mem->d_ax_long      = (int32_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int32_t));
    dev_mem->d_ay_long      = (int32_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int32_t));
    dev_mem->d_sid_long     = (int8_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int8_t));
    dev_mem->d_range_long   = (int32_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int32_t));
    dev_mem->d_total_n_long = (size_t*)arena_alloc(a, sizeof(size_t));
    dev_mem->d_f_long       = (int32_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int32_t));
    dev_mem->d_p_long       = (uint16_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(uint16_t));

    // ---- Backtrack buffers ----
    size_t bt_n = bt_anchor_total;
    size_t bt_r = (size_t)range_grid_size * mb;
    dev_mem->d_bt_max_total_n = bt_n;
    dev_mem->d_bt_max_n_reads = bt_r;

    dev_mem->d_bt_ax_in       = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_ay_in       = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_xrev_in     = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_yrev_in     = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_f_in        = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_p_in        = (uint16_t*)arena_alloc(a, bt_n * sizeof(uint16_t));

    dev_mem->d_bt_zx          = (int64_t*)arena_alloc(a, bt_n * sizeof(int64_t));
    dev_mem->d_bt_zy          = (int64_t*)arena_alloc(a, bt_n * sizeof(int64_t));
    dev_mem->d_bt_v           = (int64_t*)arena_alloc(a, bt_n * sizeof(int64_t));
    dev_mem->d_bt_p_abs       = (int64_t*)arena_alloc(a, bt_n * sizeof(int64_t));
    dev_mem->d_bt_t           = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_u           = (uint64_t*)arena_alloc(a, bt_n * sizeof(uint64_t));
    dev_mem->d_bt_ax_out      = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_ay_out      = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_xrev_out    = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_yrev_out    = (int32_t*)arena_alloc(a, bt_n * sizeof(int32_t));
    dev_mem->d_bt_n_a          = (int*)arena_alloc(a, bt_r * sizeof(int));
    dev_mem->d_bt_offset       = (int*)arena_alloc(a, bt_r * sizeof(int));
    dev_mem->d_bt_ofs_end      = (int*)arena_alloc(a, bt_r * sizeof(int));
    dev_mem->d_bt_num_elements = (int*)arena_alloc(a, bt_r * sizeof(int));
    dev_mem->d_bt_n_v          = (int*)arena_alloc(a, bt_r * sizeof(int));
    dev_mem->d_bt_n_u          = (int*)arena_alloc(a, bt_r * sizeof(int));

    dev_mem->d_bt_cub_tmp_size = cub_sort_tmp_size(bt_n, bt_r);
    dev_mem->d_bt_cub_tmp      = arena_alloc(a, dev_mem->d_bt_cub_tmp_size);

    // ---- Voting buffers ----
    size_t vt_a = bt_anchor_total;
    size_t vt_r = dev_mem->d_bt_max_n_reads + 1;
    dev_mem->d_vt_max_anchors = vt_a;

    dev_mem->d_vt_ax          = (uint64_t*)arena_alloc(a, vt_a * sizeof(uint64_t));
    dev_mem->d_vt_ay          = (uint64_t*)arena_alloc(a, vt_a * sizeof(uint64_t));
    dev_mem->d_vt_bx          = (uint64_t*)arena_alloc(a, vt_a * sizeof(uint64_t));
    dev_mem->d_vt_by          = (uint64_t*)arena_alloc(a, vt_a * sizeof(uint64_t));
    dev_mem->d_vt_mark        = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_anchor_seg  = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_out_pos     = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_votes        = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_keep_bin     = (int8_t*)arena_alloc(a, vt_a * sizeof(int8_t));
    dev_mem->d_vt_seg_start    = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_seg_id       = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_seg_cnt_flat = (int32_t*)arena_alloc(a, vt_a * sizeof(int32_t));
    dev_mem->d_vt_bin_off     = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));
    dev_mem->d_vt_anchor_off  = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));
    dev_mem->d_vt_ref_min     = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));
    dev_mem->d_vt_bin_size    = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));
    dev_mem->d_vt_nsegs       = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));
    dev_mem->d_vt_ncompact    = (int32_t*)arena_alloc(a, vt_r * sizeof(int32_t));

    dev_mem->current_phase = GPU_PHASE_CHAIN;
}

/* Set up alignment buffers from arena.
 * Called when transitioning from chain→align phase. */
static void setup_align_phase(deviceMemPtr *dev_mem) {
    gpu_arena_t *a = &dev_mem->arena;
    arena_reset(a);

    // ---- Sequence data ----
    size_t seq_unpacked_size = dev_mem->max_align_seq_bytes;
    size_t seq_packed_size = (dev_mem->max_align_seq_bytes / 8) * sizeof(uint32_t);
    size_t metadata_size = dev_mem->max_align_tasks * sizeof(uint32_t);

    dev_mem->d_align_unpacked_query  = (uint8_t*)arena_alloc(a, seq_unpacked_size);
    dev_mem->d_align_unpacked_target = (uint8_t*)arena_alloc(a, seq_unpacked_size);
    dev_mem->d_align_packed_query    = (uint32_t*)arena_alloc(a, seq_packed_size);
    dev_mem->d_align_packed_target   = (uint32_t*)arena_alloc(a, seq_packed_size);
    dev_mem->d_align_query_offsets   = (uint32_t*)arena_alloc(a, metadata_size);
    dev_mem->d_align_target_offsets  = (uint32_t*)arena_alloc(a, metadata_size);
    dev_mem->d_align_query_lens      = (uint32_t*)arena_alloc(a, metadata_size);
    dev_mem->d_align_target_lens     = (uint32_t*)arena_alloc(a, metadata_size);
    dev_mem->d_align_flag            = (int32_t*)arena_alloc(a, metadata_size);
    dev_mem->d_align_bw              = (int32_t*)arena_alloc(a, metadata_size);

    // ---- Global DP buffer ----
    size_t global_buffer_size = 28 * (256 / 8) * dev_mem->max_align_query_len * 4;
    size_t global_buffer_bytes = global_buffer_size * sizeof(short2);
    dev_mem->d_align_global_buffer = arena_alloc(a, global_buffer_bytes);

    // ---- KSW temp buffer (slot-indexed by blockIdx.x) ----
    // Layout: H[tlen]*int32 (max tracking) + u/v/x/y/x2/y2[(tlen+1)]*int8 + qr + target (uint8)
    size_t max_len = dev_mem->max_align_query_len;
    size_t h_array_size = max_len * sizeof(int32_t);                 // H for max tracking
    size_t sk_arrays_size = (max_len + 1) * 6 * sizeof(int8_t);     // u,v,x,y,x2,y2
    size_t seq_size = max_len * 2 * sizeof(uint8_t);                 // qr + target
    size_t raw_size = h_array_size + sk_arrays_size + seq_size;
    dev_mem->align_ksw_temp_per_task = (raw_size + 7) & ~7ULL;
    size_t ksw_temp_bytes = (size_t)dev_mem->n_align_concurrent_blocks *
                            dev_mem->align_ksw_temp_per_task;
    dev_mem->d_align_ksw_temp_buffer = arena_alloc(a, ksw_temp_bytes);

    // ---- Backtrack buffers ----
    size_t alloc_slots = (size_t)dev_mem->n_align_concurrent_blocks;
    size_t max_antidiag_short = 2 * dev_mem->short_task_max_len;
    size_t max_n_col = dev_mem->short_task_max_len + 1;  // bandwidth + 1 (capped by short task length)
    dev_mem->max_align_backtrack_size = max_antidiag_short * max_n_col;
    dev_mem->max_align_cigar_len = 2 * dev_mem->short_task_max_len;

    size_t bt_p_bytes   = alloc_slots * dev_mem->max_align_backtrack_size;
    size_t bt_off_bytes = alloc_slots * max_antidiag_short * sizeof(int);

    dev_mem->d_align_backtrack_p       = (uint8_t*)arena_alloc(a, bt_p_bytes);
    dev_mem->d_align_backtrack_off     = (int*)arena_alloc(a, bt_off_bytes);
    dev_mem->d_align_backtrack_off_end = (int*)arena_alloc(a, bt_off_bytes);
    dev_mem->d_align_backtrack_n_col   = (int*)arena_alloc(a, alloc_slots * sizeof(int));

    // ---- Long-task backtrack_off buffers ----
    // Short-task off buffers use stride = max_antidiag_short (2000), which is far too small for
    // long tasks (antidiag up to 2*max_align_query_len = 100000).  We allocate a separate pair
    // with the full long-task antidiag stride.  The number of concurrent long-task slots is
    // bounded by both the bt_p pool capacity and this allocation.
    // n_long_concurrent_slots = bt_p_total / (max_antidiag_long * max_n_col_long_upper)
    // where max_n_col_long_upper is a safe upper bound for the largest expected bandwidth.
    // We use 128 as a practical cap: for bw_long=7501 tasks each need ~183 MB of bt_p, so
    // the actual concurrency will be limited by bt_p (~26 slots) not by this cap.
    size_t max_antidiag_long = 2 * dev_mem->max_align_query_len;  // 100,000
    dev_mem->n_long_concurrent_slots = 128;
    size_t bt_off_long_bytes = (size_t)dev_mem->n_long_concurrent_slots *
                               max_antidiag_long * sizeof(int);
    dev_mem->d_align_backtrack_off_long     = (int*)arena_alloc(a, bt_off_long_bytes);
    dev_mem->d_align_backtrack_off_end_long = (int*)arena_alloc(a, bt_off_long_bytes);

    // ---- CIGAR buffers ----
    size_t cigar_buf_bytes = dev_mem->max_align_tasks *
                             dev_mem->max_align_cigar_len * sizeof(uint32_t);
    size_t cigar_len_bytes = dev_mem->max_align_tasks * sizeof(int);

    dev_mem->d_align_cigar_buffer  = (uint32_t*)arena_alloc(a, cigar_buf_bytes);
    dev_mem->d_align_cigar_lengths = (int*)arena_alloc(a, cigar_len_bytes);

    // ---- Compact CIGAR ----
    dev_mem->d_align_compact_cigar   = (uint32_t*)arena_alloc(a, cigar_buf_bytes);
    dev_mem->d_align_compact_offsets = (uint32_t*)arena_alloc(a,
        (dev_mem->max_align_tasks + 1) * sizeof(uint32_t));

    // ---- CUB temp ----
    dev_mem->align_cub_tmp_size = cub_scan_tmp_size(dev_mem->max_align_tasks);
    dev_mem->d_align_cub_tmp = arena_alloc(a, dev_mem->align_cub_tmp_size);

    // ---- Stats buffers ----
    size_t stats_bytes = dev_mem->max_align_tasks * sizeof(int32_t);
    dev_mem->d_align_blen            = (int32_t*)arena_alloc(a, stats_bytes);
    dev_mem->d_align_mlen            = (int32_t*)arena_alloc(a, stats_bytes);
    dev_mem->d_align_n_ambi          = (int32_t*)arena_alloc(a, stats_bytes);
    dev_mem->d_align_dp_max          = (int32_t*)arena_alloc(a, stats_bytes);
    dev_mem->d_align_gpu_stats_valid = (int32_t*)arena_alloc(a, stats_bytes);

    // ---- Result buffers ----
    dev_mem->d_align_device_res       = arena_alloc(a, sizeof(gasal_res_t));
    dev_mem->d_align_ez_array         = arena_alloc(a,
        sizeof(ksw_extz_t) * dev_mem->short_task_batch_size);
    dev_mem->d_align_scores           = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_query_ends       = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_target_ends      = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mqe              = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mqe_t            = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mte              = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mte_q            = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_zdropped         = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_task_to_align_id = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mat              = (int8_t*)arena_alloc(a, 25 * sizeof(int8_t));

    // ---- Persistent kernel counter ----
    dev_mem->d_align_task_counter = (int*)arena_alloc(a, sizeof(int));

    dev_mem->current_phase = GPU_PHASE_ALIGN;
}

/* ======== Public API ======== */

void plmem_malloc_device_mem(deviceMemPtr *dev_mem, size_t anchor_per_batch,
                              int range_grid_size, int num_cut,
                              int num_streams) {
    // Only print detailed allocation info for the first stream; all streams are identical.
    static int arena_info_printed = 0;
    int print_info = !arena_info_printed;
    arena_info_printed = 1;
    cudaSetDevice(CUDA_DEVICE);

    // Save parameters for phase transitions
    dev_mem->saved_anchor_per_batch = anchor_per_batch;
    dev_mem->saved_range_grid_size  = range_grid_size;
    dev_mem->saved_num_cut          = num_cut;

    // Pre-compute alignment config values (needed for arena sizing)
    dev_mem->max_align_tasks     = 120000;
    dev_mem->max_align_seq_bytes = (size_t)1*1024*1024*1024;  // 1 GB
    dev_mem->max_align_query_len = 50000;
    dev_mem->short_task_batch_size = 4000;
    dev_mem->long_task_batch_size  = 128;
    dev_mem->short_task_max_len    = 1000;

    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks_per_sm = 32;
    dev_mem->n_align_concurrent_blocks = numSMs * max_blocks_per_sm;

    // ---- Dry run: measure chain phase size ----
    gpu_arena_t dry = {(void*)256, SIZE_MAX, 0};
    dev_mem->arena = dry;
    setup_chain_phase(dev_mem, anchor_per_batch, range_grid_size, num_cut);
    size_t chain_size = dev_mem->arena.offset;

    // ---- Dry run: measure align phase size ----
    dev_mem->arena = dry;
    setup_align_phase(dev_mem);
    size_t align_size = dev_mem->arena.offset;

    // ---- Allocate arena using needed size ----
    // plmem_config_batch already sized max_total_n to fit in per-stream VRAM,
    // so just allocate the max of chain and align phase needs + small margin.
    size_t min_arena = (chain_size > align_size ? chain_size : align_size);
    size_t arena_size = min_arena + 4 * 1024 * 1024;  // 4 MB margin for alignment padding

    void *arena_base = nullptr;
    cudaMalloc(&arena_base, arena_size);
    if (!arena_base) {
        // Fall back to minimum needed
        arena_size = min_arena;
        cudaMalloc(&arena_base, arena_size);
    }
    if (!arena_base) {
        fprintf(stderr, "[FATAL] Failed to allocate GPU arena: %.2f GB\n",
                arena_size / (1024.0*1024.0*1024.0));
        abort();
    }

    dev_mem->arena.base       = arena_base;
    dev_mem->arena.total_size = arena_size;
    dev_mem->arena.offset     = 0;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    // ---- Dynamic scaling: use extra arena memory for larger align batches ----
    // Per-task align cost: 2×CIGAR(8KB) + metadata/stats/results(84B) + ez(48B) ≈ 16.2KB
    // Use 17KB to account for 256-byte alignment padding per arena_alloc
    size_t per_task_bytes = (size_t)2 * dev_mem->short_task_max_len * 2 * sizeof(uint32_t)
                          + 21 * sizeof(int32_t) + sizeof(ksw_extz_t) + 256;  // ~16.5KB
    if (arena_size > align_size) {
        size_t extra = arena_size - align_size;
        size_t extra_tasks = extra / per_task_bytes;
        size_t old_tasks = dev_mem->max_align_tasks;
        dev_mem->max_align_tasks += extra_tasks;
        // Cap: HOST memory for compact CIGAR = max_align_tasks × max_cigar_len × 4
        // Keep HOST alloc reasonable (< 2GB → max_align_tasks ≈ 250K)
        size_t max_tasks_cap = 250000;
        if (dev_mem->max_align_tasks > max_tasks_cap)
            dev_mem->max_align_tasks = max_tasks_cap;
        // Scale short batch size to match (fewer kernel launches = faster)
        dev_mem->short_task_batch_size = (int)dev_mem->max_align_tasks;
        // Also scale long batch size proportionally
        dev_mem->long_task_batch_size = (int)(128.0 * dev_mem->max_align_tasks / (double)old_tasks);
        if (dev_mem->long_task_batch_size > (int)dev_mem->max_align_tasks)
            dev_mem->long_task_batch_size = (int)dev_mem->max_align_tasks;
        if (dev_mem->long_task_batch_size < 128)
            dev_mem->long_task_batch_size = 128;

        // Verify: re-run align dry run with scaled params
        gpu_arena_t verify = {(void*)256, SIZE_MAX, 0};
        dev_mem->arena = verify;
        setup_align_phase(dev_mem);
        size_t scaled_align_size = dev_mem->arena.offset;
        if (scaled_align_size > arena_size) {
            // Scale back if doesn't fit
            dev_mem->max_align_tasks = old_tasks;
            dev_mem->short_task_batch_size = 4000;
            dev_mem->long_task_batch_size = 128;
            if (print_info)
                fprintf(stderr, " [Arena] Align scaling failed, using defaults\n");
        } else {
            align_size = scaled_align_size;
        }
        // Restore arena for real use
        dev_mem->arena.base       = arena_base;
        dev_mem->arena.total_size = arena_size;
        dev_mem->arena.offset     = 0;
    }

    if (print_info) {
        double GB = 1024.0*1024.0*1024.0;
        double bt_p_gb = (double)dev_mem->n_align_concurrent_blocks *
                         dev_mem->max_align_backtrack_size / GB;
        double bt_off_long_gb = (double)dev_mem->n_long_concurrent_slots *
                                2 * dev_mem->max_align_query_len * sizeof(int) / GB;
        fprintf(stderr, "[Info] GPU arena: %.2f GB total, %.2f GB free  (%d stream%s, %.2f GB each)\n",
                total_mem / GB, free_mem / GB,
                num_streams, num_streams > 1 ? "s" : "",
                arena_size / GB);
        fprintf(stderr, "[Info]   Chain phase: %.2f GB  |  Align phase: %.2f GB\n",
                chain_size / GB, align_size / GB);
        fprintf(stderr, "[Info]   Align config: max_tasks=%zu  short_batch=%d  long_batch=%d  n_concurrent=%d\n",
                dev_mem->max_align_tasks, dev_mem->short_task_batch_size,
                dev_mem->long_task_batch_size, dev_mem->n_align_concurrent_blocks);
        fprintf(stderr, "[Info]   Backtrack pool: %.2f GB  |  bt_off long: %.2f GB (%d slots × 100k stride)\n",
                bt_p_gb, bt_off_long_gb, dev_mem->n_long_concurrent_slots);
    }

    // Set up chain phase initially
    setup_chain_phase(dev_mem, anchor_per_batch, range_grid_size, num_cut);

    // ---- Pre-allocate pinned host buffers for alignment D2H/H2D ----
    // These were previously allocated/freed inside every gpu_align_batch_execute call
    // (22× cudaMallocHost + 22× cudaFreeHost per batch = expensive mlock syscalls).
    {
        size_t short_bp = (size_t)dev_mem->max_align_tasks;
        size_t long_cigar_l = 2 * dev_mem->max_align_query_len;
        size_t long_bp = dev_mem->max_align_tasks * dev_mem->max_align_cigar_len / long_cigar_l;
        size_t mbs = (short_bp > long_bp) ? short_bp : long_bp;
        size_t cigar_buf_sz = mbs * dev_mem->max_align_cigar_len;
        dev_mem->h_align_max_batch = mbs;

        cudaMallocHost(&dev_mem->h_align_compact_cigar,   cigar_buf_sz * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_compact_offsets,  mbs * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_cigar_lengths,    mbs * sizeof(int));
        cudaMallocHost(&dev_mem->h_align_blen,             mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_mlen,             mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_n_ambi,           mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_dp_max,           mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_gpu_stats_valid,  mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_scores,           mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_query_ends,       mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_target_ends,      mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_mqe,              mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_mqe_t,            mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_mte,              mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_mte_q,            mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_zdropped,         mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_query_offsets,    mbs * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_target_offsets,   mbs * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_query_lens,       mbs * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_target_lens,      mbs * sizeof(uint32_t));
        cudaMallocHost(&dev_mem->h_align_flag,             mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_bw,                mbs * sizeof(int32_t));
        cudaMallocHost(&dev_mem->h_align_task_to_align_id, mbs * sizeof(int32_t));

        if (print_info)
            fprintf(stderr, "[Info]   Pinned host buffers: %.2f GB (max_batch=%zu)\n",
                    (cigar_buf_sz * 4 + mbs * 21 * 4) / (1024.0*1024.0*1024.0), mbs);
    }

    cudaCheck();
}

void plmem_free_device_mem(deviceMemPtr *dev_mem) {
    if (dev_mem->arena.base) {
        cudaFree(dev_mem->arena.base);
        dev_mem->arena.base       = nullptr;
        dev_mem->arena.total_size = 0;
        dev_mem->arena.offset     = 0;
    }
    // Free pre-allocated pinned host buffers
    cudaFreeHost(dev_mem->h_align_compact_cigar);
    cudaFreeHost(dev_mem->h_align_compact_offsets);
    cudaFreeHost(dev_mem->h_align_cigar_lengths);
    cudaFreeHost(dev_mem->h_align_blen);
    cudaFreeHost(dev_mem->h_align_mlen);
    cudaFreeHost(dev_mem->h_align_n_ambi);
    cudaFreeHost(dev_mem->h_align_dp_max);
    cudaFreeHost(dev_mem->h_align_gpu_stats_valid);
    cudaFreeHost(dev_mem->h_align_scores);
    cudaFreeHost(dev_mem->h_align_query_ends);
    cudaFreeHost(dev_mem->h_align_target_ends);
    cudaFreeHost(dev_mem->h_align_mqe);
    cudaFreeHost(dev_mem->h_align_mqe_t);
    cudaFreeHost(dev_mem->h_align_mte);
    cudaFreeHost(dev_mem->h_align_mte_q);
    cudaFreeHost(dev_mem->h_align_zdropped);
    cudaFreeHost(dev_mem->h_align_query_offsets);
    cudaFreeHost(dev_mem->h_align_target_offsets);
    cudaFreeHost(dev_mem->h_align_query_lens);
    cudaFreeHost(dev_mem->h_align_target_lens);
    cudaFreeHost(dev_mem->h_align_flag);
    cudaFreeHost(dev_mem->h_align_bw);
    cudaFreeHost(dev_mem->h_align_task_to_align_id);
    cudaCheck();
}

void plmem_phase_to_align(deviceMemPtr *dev_mem) {
    if (dev_mem->current_phase == GPU_PHASE_ALIGN) return;
    setup_align_phase(dev_mem);
}

void plmem_phase_to_chain(deviceMemPtr *dev_mem) {
    if (dev_mem->current_phase == GPU_PHASE_CHAIN) return;
    setup_chain_phase(dev_mem, dev_mem->saved_anchor_per_batch,
                      dev_mem->saved_range_grid_size, dev_mem->saved_num_cut);
}


/**
 * Input
 *  reads[]:    array
 *  n_reads
 *  config:     range kernel configuartions
 * Output
 *  *host_mem   populate host_mem
*/
void plmem_reorg_input_arr(chain_read_t *reads, int n_read,
                           hostMemPtr *host_mem, range_kernel_config_t config) {
    size_t total_n = 0, cut_num = 0;
    size_t griddim = 0;

    host_mem->size = n_read;
    for (int i = 0; i < n_read; i++) {
        total_n += reads[i].n;
    }
    host_mem->total_n = total_n;

    size_t idx = 0;
    for (int i = 0; i < n_read; i++) {
        int n = reads[i].n;
        int block_num = (n - 1) / config.anchor_per_block + 1;

        host_mem->start_idx[griddim] = idx;
        size_t end_idx = idx + config.anchor_per_block;
        host_mem->read_end_idx[griddim] = idx + n;
        host_mem->cut_start_idx[griddim] = cut_num;
        for (int j = 1; j < block_num; j++) {
            cut_num += (config.anchor_per_block / config.blockdim);
            host_mem->start_idx[griddim + j] = end_idx;
            end_idx =
                host_mem->start_idx[griddim + j] + config.anchor_per_block;
            host_mem->read_end_idx[griddim + j] = idx + n;
            host_mem->cut_start_idx[griddim + j] = cut_num;
        }
        cut_num += (n - (block_num - 1) * config.anchor_per_block - 1) /
                       config.blockdim;
        end_idx = idx + n;

        griddim += block_num;
        
        for (int j = 0; j < n; j++) {
            host_mem->ax[idx] = (int32_t)reads[i].a[j].x;
            host_mem->ay[idx] = (int32_t)reads[i].a[j].y;
            host_mem->sid[idx] = (reads[i].a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
            host_mem->xrev[idx] = reads[i].a[j].x >> 32;
            host_mem->yrev[idx] = reads[i].a[j].y >> 32;
            ++idx;
        }
    }
    host_mem->cut_num = cut_num;
    host_mem->griddim = griddim;
}

void plmem_async_h2d_short_memcpy(stream_ptr_t* stream_ptrs, size_t uid) {
    hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
    deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
    cudaStream_t *stream = &stream_ptrs->cudastream;
    cudaMemcpyAsync(dev_mem->d_ax, host_mem->ax,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_ay, host_mem->ay,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_sid, host_mem->sid,
                    sizeof(int8_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_xrev, host_mem->xrev,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_yrev, host_mem->yrev,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_start_idx, host_mem->start_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_read_end_idx, host_mem->read_end_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemsetAsync(dev_mem->d_cut, 0xff,
                    sizeof(size_t) * host_mem->cut_num, *stream);
    cudaMemsetAsync(dev_mem->d_f, 0, sizeof(int32_t) * host_mem->total_n,
                    *stream);
    cudaMemsetAsync(dev_mem->d_p, 0, sizeof(uint16_t) * host_mem->total_n,
                    *stream);
    cudaCheck();
    dev_mem->total_n = host_mem->total_n;
    dev_mem->num_cut = host_mem->cut_num;
    dev_mem->size = host_mem->size;
    dev_mem->griddim = host_mem->griddim;
}

void plmem_async_h2d_memcpy(stream_ptr_t* stream_ptrs) {
    size_t uid = 0;
    hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
    deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
    cudaStream_t *stream = &stream_ptrs->cudastream;
    cudaMemcpyAsync(dev_mem->d_ax, host_mem->ax,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_ay, host_mem->ay,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_sid, host_mem->sid,
                    sizeof(int8_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_xrev, host_mem->xrev,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_yrev, host_mem->yrev,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_start_idx, host_mem->start_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_read_end_idx, host_mem->read_end_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemcpyAsync(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
                    sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice,
                    *stream);
    cudaMemsetAsync(dev_mem->d_cut, 0xff,
                    sizeof(size_t) * host_mem->cut_num, *stream);
    cudaMemsetAsync(dev_mem->d_f, 0, sizeof(int32_t) * host_mem->total_n,
                    *stream);
    cudaMemsetAsync(dev_mem->d_p, 0, sizeof(uint16_t) * host_mem->total_n,
                    *stream);
    cudaCheck();
    dev_mem->total_n = host_mem->total_n;
    dev_mem->num_cut = host_mem->cut_num;
    dev_mem->size = host_mem->size;
    dev_mem->griddim = host_mem->griddim;
}

void plmem_sync_h2d_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem) {
    cudaMemcpy(dev_mem->d_ax, host_mem->ax, sizeof(int32_t) * host_mem->total_n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_ay, host_mem->ay, sizeof(int32_t) * host_mem->total_n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_sid, host_mem->sid, sizeof(int8_t) * host_mem->total_n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_xrev, host_mem->xrev,
               sizeof(int32_t) * host_mem->total_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_start_idx, host_mem->start_idx,
               sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_read_end_idx, host_mem->read_end_idx,
               sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
               sizeof(size_t) * host_mem->griddim, cudaMemcpyHostToDevice);
    cudaMemset(dev_mem->d_cut, 0xff, sizeof(size_t) * host_mem->cut_num);
    dev_mem->total_n = host_mem->total_n;
    dev_mem->num_cut = host_mem->cut_num;
    dev_mem->size = host_mem->size;
    dev_mem->griddim = host_mem->griddim;
    cudaCheck();
}

void plmem_async_d2h_memcpy(stream_ptr_t *stream_ptrs) {
    size_t uid = 0;
    hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
    longMemPtr *long_mem = &stream_ptrs->long_mem;
    deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
    cudaStream_t *stream = &stream_ptrs->cudastream;
    cudaMemcpyAsync(host_mem->f, dev_mem->d_f,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyDeviceToHost,
                    *stream);
    cudaMemcpyAsync(host_mem->p, dev_mem->d_p,
                    sizeof(uint16_t) * host_mem->total_n,
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->long_segs_og_idx, dev_mem->d_long_seg_og,
                    dev_mem->buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t),
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(host_mem->long_segs_num, dev_mem->d_long_seg_count,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->f_long, dev_mem->d_f_long, sizeof(int32_t)*dev_mem->buffer_size_long,
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->p_long, dev_mem->d_p_long, sizeof(uint16_t)*dev_mem->buffer_size_long,
                    cudaMemcpyDeviceToHost, *stream);
    cudaCheck();
}

void plmem_async_d2h_short_memcpy(stream_ptr_t *stream_ptrs, size_t uid) {
    hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
    deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
    cudaStream_t *stream = &stream_ptrs->cudastream;
    cudaMemcpyAsync(host_mem->f, dev_mem->d_f,
                    sizeof(int32_t) * host_mem->total_n, cudaMemcpyDeviceToHost,
                    *stream);
    cudaMemcpyAsync(host_mem->p, dev_mem->d_p,
                    sizeof(uint16_t) * host_mem->total_n,
                    cudaMemcpyDeviceToHost, *stream);
    // copy back d_long_seg_count to long_segs_num, this is an accumulative value
    cudaMemcpyAsync(host_mem->long_segs_num, dev_mem->d_long_seg_count,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
    cudaCheck();
}

void plmem_async_d2h_long_memcpy(stream_ptr_t *stream_ptrs) {
    size_t uid = 0;
    longMemPtr *long_mem = &stream_ptrs->long_mem;
    deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
    cudaStream_t *stream = &stream_ptrs->cudastream;
    cudaMemcpyAsync(long_mem->long_segs_og_idx, dev_mem->d_long_seg_og,
                    dev_mem->buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t),
                    cudaMemcpyDeviceToHost, *stream);
    // cudaMemcpyAsync(&long_mem->total_long_segs_num, dev_mem->d_long_seg_count,
    //                 sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->f_long, dev_mem->d_f_long, sizeof(int32_t)*dev_mem->buffer_size_long,
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->p_long, dev_mem->d_p_long, sizeof(uint16_t)*dev_mem->buffer_size_long,
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->total_long_segs_n, dev_mem->d_total_n_long, sizeof(size_t),
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->total_long_segs_num, dev_mem->d_long_seg_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, *stream);
    cudaCheck();
}

void plmem_sync_d2h_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem){
    cudaMemcpy(host_mem->f, dev_mem->d_f, sizeof(int32_t) * host_mem->total_n,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mem->p, dev_mem->d_p, sizeof(uint16_t) * host_mem->total_n,
               cudaMemcpyDeviceToHost);
    cudaCheck();
}

//////////////////// Initialization and Cleanup <mmpriv.h>////////////////////////
streamSetup_t stream_setup;

#include "cJSON.h"
cJSON *plmem_parse_gpu_config(const char filename[]){
    // read json file to cstring
    char *buffer = 0;
    long length;
    FILE *f = fopen(filename, "rb");

    if (f) {
        fseek(f, 0, SEEK_END);
        length = ftell(f);
        fseek(f, 0, SEEK_SET);
        buffer = (char*)malloc(length);
        if (buffer) {
            fread(buffer, 1, length, f);
        }
        fclose(f);
    }

    if (!buffer) {
        fprintf(stderr, "[Error] fail to open gpu config file %s\n", filename);
        exit(1);
    }

    cJSON *json = cJSON_Parse(buffer);
    if (!json) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "[Error] cJSON error before %s\n", error_ptr);
        }
        exit(1);
    }

    return json;
}

int get_json_int(cJSON *json, const char name[]) {
    cJSON *elt = cJSON_GetObjectItem(json, name);
    if (!cJSON_IsNumber(elt)) {
        fprintf(stderr, "[Error] cJSON error failed to get field %s\n", name);
        exit(1);
    }
    return elt->valueint;
}

void plmem_config_kernels(cJSON *json) {
    cJSON *range_config_json = cJSON_GetObjectItem(json, "range_kernel");
    range_kernel_config.blockdim = get_json_int(range_config_json, "blockdim");
    range_kernel_config.cut_check_anchors =
        get_json_int(range_config_json, "cut_check_anchors");
    range_kernel_config.anchor_per_block =
        get_json_int(range_config_json, "anchor_per_block");

    cJSON *score_config_json = cJSON_GetObjectItem(json, "score_kernel");
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, CUDA_DEVICE);
    score_kernel_config.short_blockdim = device_prop.warpSize;
    score_kernel_config.long_blockdim = device_prop.maxThreadsPerBlock;
    score_kernel_config.mid_blockdim =
        get_json_int(score_config_json, "mid_blockdim");
    score_kernel_config.short_griddim =
        get_json_int(score_config_json, "short_griddim");
    score_kernel_config.long_griddim =
        get_json_int(score_config_json, "long_griddim");
    score_kernel_config.mid_griddim =
        get_json_int(score_config_json, "mid_griddim");
    score_kernel_config.long_seg_cutoff =
        get_json_int(score_config_json, "long_seg_cutoff");
    score_kernel_config.mid_seg_cutoff =
        get_json_int(score_config_json, "mid_seg_cutoff");
    score_kernel_config.cut_unit = range_kernel_config.blockdim;
    score_kernel_config.micro_batch = 
        get_json_int(score_config_json, "micro_batch");
    if (score_kernel_config.micro_batch > MAX_MICRO_BATCH) {
        fprintf(stderr, "[Error: gpu config] score_kernel:micro_batch should be less than %d\n"
                "\t\t or recompile with MAX_MICRO_BATCH=%d"
                , MAX_MICRO_BATCH, score_kernel_config.micro_batch);
        exit(1);
    }
    
}

void plmem_config_stream(size_t *max_range_grid_, size_t *max_num_cut_, size_t max_total_n, size_t max_read, size_t min_n){
    size_t max_range_grid, max_num_cut;
    max_range_grid =
        (max_total_n - 1) / range_kernel_config.anchor_per_block + 1 + max_read;
    max_num_cut = (max_total_n - 1) / range_kernel_config.blockdim + 1 + max_read;
    *max_range_grid_ = max_range_grid;
    *max_num_cut_ = max_num_cut;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, CUDA_DEVICE);
    cudaCheck();

    if (*max_range_grid_ > prop.maxGridSize[0]) {
        fprintf(stderr, "Invalid memory config!\n");
        exit(1);
    }

}


template <bool is_blooking = false>
void plmem_config_batch(cJSON *json, int *num_stream_,
                         int *min_n_, size_t *max_total_n_,
                         int *max_read_, size_t *long_seg_buffer_size_) {
    if (is_blooking) 
        *num_stream_ = 16;
    else
        *num_stream_ = get_json_int(json, "num_streams");

    size_t min_anchors = get_json_int(json, "min_n");
    *min_n_ = min_anchors;

    /* Auto-compute max_total_n, max_read, and long_seg_buffer_size from VRAM */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, CUDA_DEVICE);
    size_t gpu_free_mem, gpu_total_mem;
    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

    // Per-stream VRAM budget: leave 256MB global headroom
    size_t global_reserve = (size_t)256 * 1024 * 1024;
    size_t usable = (gpu_free_mem > global_reserve) ? (gpu_free_mem - global_reserve) : gpu_free_mem;
    size_t avail_mem_per_stream = usable / (*num_stream_);

    // Exact per-anchor memory costs matching setup_chain_phase() allocations:
    //   Chain anchors (N):  ax(4)+ay(4)+sid(1)+xrev(4)+yrev(4)+range(4)+f(4)+p(2) = 27
    //   Backtrack (N*mb):   ax,ay,xrev,yrev,f,p(22) + zx,zy,v,p_abs(32) + t,u(12) + ax,ay,xrev,yrev_out(16) = 82
    //   Voting (N*mb):      ax,ay,bx,by(32) + mark,anchor_seg,out_pos,votes(16) + keep_bin(1) + seg_start,seg_id,seg_cnt_flat(12) = 61
    //   CUB sort temp (N*mb): ~16 bytes (double-buffer for int64 key+value pairs)
    //   Long seg data (L):  ax(4)+ay(4)+sid(1)+range(4)+f(4)+p(2) = 19
    //   Long seg index (L): seg_t*2 + map(4) per (long_seg_cutoff*cut_unit) entries = 36/10240 per L
    //   Index/cut:          per-grid(24) + per-cut(12) + per bt_r/vt_r(24+24=48 per mb*G)
    int mb = score_kernel_config.micro_batch;

    // N-proportional cost
    size_t chain_per_n = 27;
    size_t bt_per_n    = (size_t)mb * 82;   // backtrack: 82 bytes × micro_batch
    size_t vt_per_n    = (size_t)mb * 61;   // voting: 61 bytes × micro_batch
    size_t cub_per_n   = (size_t)mb * 16;   // CUB sort temp: ~16 bytes × micro_batch (key+value double-buffer)
    size_t per_anchor_total = chain_per_n + bt_per_n + vt_per_n + cub_per_n;

    // L-proportional cost (long segment buffers)
    size_t per_long_entry = 19;  // ax,ay,sid,range,f,p long arrays

    // Avg anchors per read (for index/cut overhead estimate)
    cJSON *avg_n_json = cJSON_GetObjectItem(json, "avg_read_n");
    size_t avg_read_n = avg_n_json ? (size_t)avg_n_json->valueint : 1000;

    // Long seg buffer ratio: L as fraction of N.
    // Default L = 2*N (empirically reasonable for ONT data).
    cJSON *long_seg_json = cJSON_GetObjectItem(json, "long_seg_buffer_size");
    double long_ratio = 2.0;  // L = long_ratio * N

    // Per-read overhead (index + cut + bt_r + vt_r arrays):
    //   G ≈ N/anchor_per_block + N/avg_read_n
    //   C ≈ N/blockdim + N/avg_read_n
    //   per G: 24 (index) + mb*24 (bt_r) + mb*24 (vt_r) = 24 + 48*mb
    //   per C: 8 (d_cut) + 4*sizeof(seg_t)/(mid_seg_cutoff+1) ≈ 12
    double grids_per_n = 1.0 / range_kernel_config.anchor_per_block + 1.0 / avg_read_n;
    double cuts_per_n  = 1.0 / range_kernel_config.blockdim + 1.0 / avg_read_n;
    size_t per_grid = 24 + (size_t)mb * 48;
    size_t per_cut  = 12;
    double overhead_per_n = grids_per_n * per_grid + cuts_per_n * per_cut;

    // Total per anchor: anchor_cost + long_ratio * long_cost + overhead
    // Solve: N * total_per_n ≤ avail_mem_per_stream
    // Apply 0.98 margin for arena alignment padding
    double total_per_n = (double)per_anchor_total
                       + long_ratio * (double)per_long_entry
                       + overhead_per_n;
    size_t budget = (size_t)(avail_mem_per_stream * 0.98);

    *max_total_n_ = (size_t)(budget / total_per_n);
    // Cap to INT32_MAX safety (anchors are often int-indexed)
    if (*max_total_n_ > (size_t)2000000000) *max_total_n_ = 2000000000;

    *max_read_ = (int)(*max_total_n_ / avg_read_n);
    if (*max_read_ < 1000) *max_read_ = 1000;

    // Long seg buffer size
    if (long_seg_json) {
        *long_seg_buffer_size_ = (size_t)long_seg_json->valueint / (*num_stream_);
    } else {
        *long_seg_buffer_size_ = (size_t)(*max_total_n_ * long_ratio);
    }
    if (*long_seg_buffer_size_ < 1000000) *long_seg_buffer_size_ = 1000000;

    fprintf(stderr, "[Info::plmem] Auto-config for %d streams (%.1f GB free, %.1f GB/stream, "
            "%.0f B/anchor): max_total_n=%zu, max_read=%d, long_seg_buf=%zu\n",
            *num_stream_, gpu_free_mem / (1024.0*1024.0*1024.0),
            avail_mem_per_stream / (1024.0*1024.0*1024.0), total_per_n,
            *max_total_n_, *max_read_, *long_seg_buffer_size_);
}

// intialize and config kernels for gpu blocking setup
void plmem_initialize(size_t *max_total_n_, int *max_read_,
                      int *min_anchors_) {
#ifndef GPU_CONFIG
    cJSON *json = plmem_parse_gpu_config("gpu/gpu_config.json");
#else 
    cJSON *json = plmem_parse_gpu_config(GPU_CONFIG);
#endif
    plmem_config_kernels(json);
    int num_streams;
    size_t buffer_size_long;
    plmem_config_batch<true>(json, &num_streams, min_anchors_, max_total_n_,
                             max_read_, &buffer_size_long);
}

// initialize global variable stream_setup
void plmem_stream_initialize(size_t *max_total_n_,
                             int *max_read_, int *min_anchors_, char* gpu_config_file) {

    cudaSetDevice(CUDA_DEVICE);
    int num_stream;
    size_t max_anchors_stream, max_range_grid, max_num_cut, long_seg_buffer_size;

    cJSON *json = plmem_parse_gpu_config(gpu_config_file);

    plmem_config_kernels(json);
    size_t gpu_free_mem, gpu_total_mem;
    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
    plmem_config_batch<false>(json, &num_stream, min_anchors_, &max_anchors_stream,
                              max_read_, &long_seg_buffer_size);
    plmem_config_stream(&max_range_grid, &max_num_cut, max_anchors_stream,
                        *max_read_, *min_anchors_);

    stream_setup.num_stream = num_stream;
    // assert(num_stream > 1);

    stream_setup.streams = new stream_ptr_t[num_stream];

    for (int i = 0; i < num_stream; i++) {
        stream_setup.streams[i].busy = false;
        cudaStreamCreate(&stream_setup.streams[i].cudastream);
        cudaEventCreate(&stream_setup.streams[i].stopevent);
        cudaEventCreate(&stream_setup.streams[i].startevent);
        cudaCheck();
        stream_setup.streams[i].dev_mem.buffer_size_long = long_seg_buffer_size;
        // one stream has multiple host mems
        for (int j = 0; j < score_kernel_config.micro_batch; j++) {
            plmem_malloc_host_mem(&stream_setup.streams[i].host_mems[j], max_anchors_stream,
                              max_range_grid, long_seg_buffer_size);
        }
        // one stream has one long mem and one device mem
        plmem_malloc_long_mem(&stream_setup.streams[i].long_mem, long_seg_buffer_size);
        plmem_malloc_device_mem(&stream_setup.streams[i].dev_mem, max_anchors_stream,
                                max_range_grid, max_num_cut, num_stream);
        cudaMemset(stream_setup.streams[i].dev_mem.d_long_seg_count, 0, sizeof(unsigned int));
        cudaMemset(stream_setup.streams[i].dev_mem.d_mid_seg_count, 0, sizeof(unsigned int));
        cudaMemset(stream_setup.streams[i].dev_mem.d_total_n_long, 0, sizeof(size_t));
        cudaCheck();
    }

cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

    *max_total_n_ = max_anchors_stream * score_kernel_config.micro_batch;
    *max_read_ = *max_read_ * score_kernel_config.micro_batch;

    stream_setup.max_anchors_stream = max_anchors_stream;
    stream_setup.max_range_grid = max_range_grid;
    stream_setup.max_num_cut = max_num_cut;
    stream_setup.long_seg_buffer_size_stream = long_seg_buffer_size;
    cudaCheck();
}

void plmem_stream_cleanup() {
    // Synchronize all streams before cleanup to ensure all GPU operations are complete
    for (int i = 0; i < stream_setup.num_stream; i++) {
        cudaStreamSynchronize(stream_setup.streams[i].cudastream);
    }
    cudaDeviceSynchronize();
    cudaCheck();
    for (int i = 0; i < stream_setup.num_stream; i++) {
        cudaStreamDestroy(stream_setup.streams[i].cudastream);
        cudaEventDestroy(stream_setup.streams[i].stopevent);
        cudaEventDestroy(stream_setup.streams[i].startevent);
        cudaCheck();
        // free multiple host mems
        for (int j = 0; j < score_kernel_config.micro_batch; j++) {
            plmem_free_host_mem(&stream_setup.streams[i].host_mems[j]);
        }
        plmem_free_long_mem(&stream_setup.streams[i].long_mem);
        plmem_free_device_mem(&stream_setup.streams[i].dev_mem);
    }
    delete[] stream_setup.streams;
}
