/* GPU memory management  */


#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <cub/cub.cuh>
#include "plmem.cuh"
#include "plrange.cuh"
#include "plscore.cuh"
#include "pllog.h"
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
    size_t max_long_segs = buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit);
    cudaMallocHost((void**)&long_mem->long_segs_og_idx,  max_long_segs * sizeof(seg_t));
    cudaMallocHost((void**)&long_mem->long_segs_buf_idx, max_long_segs * sizeof(seg_t));
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
    cudaFreeHost(long_mem->long_segs_buf_idx);
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

// 256-byte alignment helper used when carving the concurrent Long-C context.
static size_t a256(size_t x) { return (x + 255) & ~(size_t)255; }

#define LONG_C_BATCH_MAX 512   // max C tasks per concurrent batch
#define LONG_C_SLOTS_MAX 512   // max C concurrent slots (bt_off/ksw_temp pre-alloc)

// ======== Module-level configuration globals ========
// Set by plmem_config_batch() (JSON config) before any arena allocation.
// Declared here so all static setup_*_phase() helpers can see them.

// VRAM headroom to leave free (for OS, cuda-gdb, etc.).
// Used when sizing the long bt_p pool and when computing per-stream VRAM budgets.
static size_t g_vram_global_reserve = (size_t)512 * 1024 * 1024;  // default 512 MB

// Manual override for the long-task CIGAR batch cap.
// 0 = auto (use compute_long_batch_size formula).
// Non-zero = use this value directly; smaller values leave more arena for bt_p,
// increasing concurrent slots for large tasks at the cost of more kernel launches
// for small tasks.  Set via JSON key "long_cigar_batch" in gpu_config.json.
static size_t g_long_cigar_batch_override = 0;

// Max sequence length per alignment task (query or target side).
// Controls ksw_temp, bt_off_long, and seq/CIGAR buffer sizes — all linear in this value.
// Reduce to e.g. 10000 for data where gap-fills/extensions are short → more concurrent slots.
// Set via JSON key "max_align_task_len" in gpu_config.json.
static size_t g_max_align_task_len = 50000;  // default 50,000 bp
static double g_bytes_per_anchor   = 0.0;    // set by plmem_stream_initialize, used by plmem_malloc_device_mem

// Conservative estimate of bt_p cost per concurrent slot, used only to size n_long_cap
// Used only in compute_long_batch_size to estimate CIGAR-buffer-bound batch size.
static const size_t TYPICAL_BT_STRIDE_INTERNAL = (size_t)6 << 20;  // 6 MB

// Returns the max long-task H2D batch size given available arena and max task length.
// Derived from CIGAR buffer budget: batch × K1_per_task ≤ arena.
static size_t compute_long_batch_size(size_t arena_bytes, size_t max_len) {
    const size_t L = 3;                                    // latency-hide factor
    const size_t TYPICAL_BT_STRIDE = TYPICAL_BT_STRIDE_INTERNAL;
    const size_t MAX_CAP = 8192;
    const size_t MIN_CAP = 64;

    size_t long_cigar     = 2 * max_len;
    size_t max_antidiag   = 2 * max_len;

    // per_slot_var: ksw_temp + bt_off_long × 2 (per concurrent slot)
    size_t h_arr  = max_len * sizeof(int32_t);
    size_t sk_arr = (max_len + 1) * 6 * sizeof(int8_t);
    size_t sq_arr = max_len * 2 * sizeof(uint8_t);
    size_t ksw_temp = ((h_arr + sk_arr + sq_arr) + 7) & ~(size_t)7;
    size_t per_slot_var = ksw_temp + 2 * max_antidiag * sizeof(int);

    // K1: per-task arena cost (seq buffers + CIGAR raw+compact + meta/stats/results)
    size_t K1 = 2 * max_len * sizeof(uint8_t)              // seq_unp  (q + t)
              + 2 * (max_len / 8) * sizeof(uint32_t)       // seq_pack (q + t)
              + 2 * long_cigar * sizeof(uint32_t)           // cigar raw + compact
              + (size_t)20 * sizeof(uint32_t);              // meta, stats, results (rough)

    size_t denom = L * K1 + per_slot_var + TYPICAL_BT_STRIDE;
    if (denom == 0) return MIN_CAP;
    size_t batch = L * arena_bytes / denom;
    if (batch > MAX_CAP) batch = MAX_CAP;
    if (batch < MIN_CAP) batch = MIN_CAP;
    return batch;
}

// Wrapper: returns g_long_cigar_batch_override when set, otherwise auto formula.
static size_t long_batch_size(size_t arena_bytes, size_t max_len) {
    if (g_long_cigar_batch_override > 0) return g_long_cigar_batch_override;
    return compute_long_batch_size(arena_bytes, max_len);
}

/* Set up chain + backtrack buffers from arena.
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
    dev_mem->d_xrev_long    = (int32_t*)arena_alloc(a, dev_mem->buffer_size_long * sizeof(int32_t));
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

    // Voting buffers removed (dead GPU voting path): freed mb*61 B/anchor here and
    // vt_per_n in plmem_config_batch — keep both in sync.

    dev_mem->current_phase = GPU_PHASE_CHAIN;
}

/* Set up alignment buffers optimised for LONG tasks.
 * Called once after the short-task phase completes (via plmem_phase_to_long_align).
 *
 * KEY IDEA: the short arena (~15 GB) was partitioned for 250 K short tasks.
 * arena_reset() reclaims all memory and re-partitions for long tasks:
 *
 *   Fixed overhead (batch-proportional, MAX_LONG_BATCH = auto from compute_long_batch_size):
 *     seq data (2 × batch × 50 KB):     ~360 MB  (at batch≈3600)
 *     CIGAR   (2 × batch × 100 K ops):  ~2.9 GB  (at batch≈3600)
 *     stats/results/misc:               ~100 MB
 *
 *   Per-slot variable (n_long_cap = MAX_LONG_BATCH / LATENCY_HIDE_FACTOR ≈ 1200 slots):
 *     ksw_temp  (600 KB/slot):          ~720 MB  (at 1200 slots)
 *     bt_off_long (800 KB/slot × 2):    ~1.9 GB  (at 1200 slots)
 *
 *   bt_p pool (everything remaining):   ~8+ GB  ← primary purpose of this transition
 *
 * MAX_LONG_BATCH is derived analytically from arena size so that pool_cap ≈ n_long_cap
 * (neither bt_off_long nor bt_p is the bottleneck).  No user config required.
 *
 * No cudaFree/cudaMalloc: same physical memory, new layout. */
static void setup_long_align_phase(deviceMemPtr *dev_mem) {
    gpu_arena_t *a = &dev_mem->arena;
    arena_reset(a);

    // ---- Batch cap: derived from arena size to balance CIGAR, slots, and bt_p ----
    // compute_long_batch_size() targets MAX_LONG_BATCH = n_long_cap × LATENCY_HIDE_FACTOR
    // so pool_cap ≈ n_long_cap ≈ MAX_LONG_BATCH / 3.  The per-batch kernel launch in
    // plalign.cu further constrains actual batch_size dynamically (see comment there).
    size_t MAX_LONG_BATCH = long_batch_size(a->total_size,
                                            dev_mem->max_align_task_len);
    size_t max_len        = dev_mem->max_align_task_len; // 50,000
    size_t long_cigar     = 2 * max_len;                  // 100,000
    size_t max_antidiag_long = 2 * max_len;               // 100,000 (bt_off stride)

    // ---- Per-task KSW temp size (arithmetic only; arena alloc deferred below) ----
    size_t h_arr  = max_len * sizeof(int32_t);
    size_t sk_arr = (max_len + 1) * 6 * sizeof(int8_t);
    size_t sq_arr = max_len * 2 * sizeof(uint8_t);
    dev_mem->align_ksw_temp_per_task = ((h_arr + sk_arr + sq_arr) + 7) & ~7ULL;

    // ---- Step 1: allocate all batch-sized fixed overhead first ----
    // (So that remaining arena can be measured before sizing per-slot arrays.)

    // Sequence data (MAX_LONG_BATCH tasks × max_len bases)
    size_t seq_unp  = MAX_LONG_BATCH * max_len * sizeof(uint8_t);
    size_t seq_pack = MAX_LONG_BATCH * (max_len / 8) * sizeof(uint32_t);
    size_t meta     = MAX_LONG_BATCH * sizeof(uint32_t);

    dev_mem->d_align_unpacked_query  = (uint8_t*)arena_alloc(a, seq_unp);
    dev_mem->d_align_unpacked_target = (uint8_t*)arena_alloc(a, seq_unp);
    dev_mem->d_align_packed_query    = (uint32_t*)arena_alloc(a, seq_pack);
    dev_mem->d_align_packed_target   = (uint32_t*)arena_alloc(a, seq_pack);
    dev_mem->d_align_query_offsets   = (uint32_t*)arena_alloc(a, meta);
    dev_mem->d_align_target_offsets  = (uint32_t*)arena_alloc(a, meta);
    dev_mem->d_align_query_lens      = (uint32_t*)arena_alloc(a, meta);
    dev_mem->d_align_target_lens     = (uint32_t*)arena_alloc(a, meta);
    dev_mem->d_align_flag            = (int32_t*)arena_alloc(a, meta);
    dev_mem->d_align_bw              = (int32_t*)arena_alloc(a, meta);

    // Short-stride bt_off stubs (not used in long phase — long tasks use bt_off_long)
    dev_mem->d_align_backtrack_off     = (int*)arena_alloc(a, sizeof(int));
    dev_mem->d_align_backtrack_off_end = (int*)arena_alloc(a, sizeof(int));

    // CIGAR buffers (MAX_LONG_BATCH × long_cigar ops × 4 B × 2 for raw+compact)
    size_t cigar_bytes = MAX_LONG_BATCH * long_cigar * sizeof(uint32_t);
    dev_mem->max_align_cigar_len     = long_cigar;
    dev_mem->d_align_cigar_buffer    = (uint32_t*)arena_alloc(a, cigar_bytes);
    dev_mem->d_align_cigar_lengths   = (int*)arena_alloc(a, MAX_LONG_BATCH * sizeof(int));
    dev_mem->d_align_compact_cigar   = (uint32_t*)arena_alloc(a, cigar_bytes);
    dev_mem->d_align_compact_offsets = (uint32_t*)arena_alloc(a,
                                         (MAX_LONG_BATCH + 1) * sizeof(uint32_t));

    // CUB temp
    dev_mem->align_cub_tmp_size = cub_scan_tmp_size(MAX_LONG_BATCH);
    dev_mem->d_align_cub_tmp = arena_alloc(a, dev_mem->align_cub_tmp_size);

    // Stats + result buffers
    size_t res = MAX_LONG_BATCH * sizeof(int32_t);
    dev_mem->d_align_blen            = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_mlen            = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_n_ambi          = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_dp_max          = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_gpu_stats_valid = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_device_res      = arena_alloc(a, sizeof(gasal_res_t));
    dev_mem->d_align_scores          = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_query_ends      = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_target_ends     = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_mqe             = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_mqe_t           = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_mte             = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_mte_q           = (int32_t*)arena_alloc(a, res);
    dev_mem->d_align_zdropped        = (int32_t*)arena_alloc(a, res);
    // d_align_task_to_align_id removed — it was always the identity mapping (i→i).
    dev_mem->d_align_mat             = (int8_t*)arena_alloc(a, 25 * sizeof(int8_t));
    dev_mem->d_align_task_counter    = (int*)arena_alloc(a, sizeof(int));

    // ---- Step 2: slot count = GPU hardware concurrent blocks (set by caller) ----
    size_t n_long_cap = (size_t)dev_mem->n_align_concurrent_blocks;
    if (n_long_cap < 64) n_long_cap = 64;  // sanity floor

    dev_mem->n_long_concurrent_slots = (int)n_long_cap;

    // ---- Step 3: allocate per-slot arrays ----
    size_t ksw_temp_bytes    = n_long_cap * dev_mem->align_ksw_temp_per_task;
    size_t bt_off_long_bytes = n_long_cap * max_antidiag_long * sizeof(int);

    dev_mem->d_align_ksw_temp_buffer        = arena_alloc(a, ksw_temp_bytes);
    dev_mem->d_align_backtrack_off_long     = (int*)arena_alloc(a, bt_off_long_bytes);
    dev_mem->d_align_backtrack_off_end_long = (int*)arena_alloc(a, bt_off_long_bytes);

    // ---- Step 4: bt_p pool gets ALL remaining arena space ----
    // Leave 16 MB headroom for arena alignment padding of the last alloc.
    const size_t BT_P_HEADROOM = (size_t)16 * 1024 * 1024;
    size_t bt_p_avail = arena_remaining(a);
    if (bt_p_avail > BT_P_HEADROOM) bt_p_avail -= BT_P_HEADROOM;
    else                             bt_p_avail = 0;

    dev_mem->d_align_backtrack_p     = (uint8_t*)arena_alloc(a, bt_p_avail);
    dev_mem->max_align_backtrack_size = 0;   // stride computed per-batch in plalign.cu

    // ALWAYS record the arena bt_p size separately so the dispatch code can
    // compare it against the dedicated pool and pick the larger.  Without this,
    // a tiny dedicated pool (e.g. 250 MB on a 40GB A100 where arena ate most
    // free VRAM) would be unconditionally used by plalign.cu, even when arena
    // has 14+ GB of bt_p sitting idle — leading to OOB writes when the batch's
    // bt_stride exceeds the dedicated pool size.
    dev_mem->long_arena_bt_p_bytes = bt_p_avail;

    // long_bt_p_pool_bytes still defaults to the arena size when no dedicated
    // pool exists.  Backward-compat path for existing dispatch.
    if (dev_mem->d_align_backtrack_p_long == nullptr)
        dev_mem->long_bt_p_pool_bytes = bt_p_avail;

    // Store long batch cap in long_task_batch_size.
    // DO NOT touch max_align_tasks — setup_align_phase() reads it and would break
    // if it were changed here (called again next round of short alignment).
    dev_mem->long_task_batch_size = (int)MAX_LONG_BATCH;


    dev_mem->current_phase = GPU_PHASE_ALIGN;
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

    // ---- KSW temp buffer (slot-indexed by blockIdx.x) ----
    // Layout: H[tlen]*int32 (max tracking) + u/v/x/y/x2/y2[(tlen+1)]*int8 + qr + target (uint8)
    size_t max_len = dev_mem->max_align_task_len;
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

    // ---- Long-task backtrack_off buffers ----
    // bt_off_long uses full long-task antidiag stride; placeholder allocation of 512 slots
    // during short phase — upgraded to gpu_max_slots when setup_long_align_phase is called.
    size_t max_antidiag_long = 2 * dev_mem->max_align_task_len;
    dev_mem->n_long_concurrent_slots = 512;
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
    dev_mem->d_align_scores           = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_query_ends       = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_target_ends      = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mqe              = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mqe_t            = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mte              = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_mte_q            = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    dev_mem->d_align_zdropped         = (int32_t*)arena_alloc(a, dev_mem->max_align_tasks * sizeof(int32_t));
    // d_align_task_to_align_id removed — it was always the identity mapping (i→i).
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
    // Track stream allocation order so we can partition remaining VRAM evenly.
    static int g_stream_alloc_idx = 0;
    int this_stream_idx = g_stream_alloc_idx++;
    cudaSetDevice(CUDA_DEVICE);

    // Save parameters for phase transitions
    dev_mem->saved_anchor_per_batch = anchor_per_batch;
    dev_mem->saved_range_grid_size  = range_grid_size;
    dev_mem->saved_num_cut          = num_cut;

    // Pre-compute alignment config values (needed for arena sizing)
    dev_mem->max_align_tasks     = 120000;
    dev_mem->max_align_seq_bytes = (size_t)1*1024*1024*1024;  // 1 GB
    dev_mem->max_align_task_len = g_max_align_task_len;
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
    cudaError_t alloc_err = cudaMalloc(&arena_base, arena_size);
    if (alloc_err != cudaSuccess || !arena_base) {
        // Fall back to minimum needed
        arena_size = min_arena;
        alloc_err = cudaMalloc(&arena_base, arena_size);
    }
    if (alloc_err != cudaSuccess || !arena_base) {
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
                PLOG_INFO(stderr, " [Arena] Align scaling failed, using defaults\n");
        } else {
            align_size = scaled_align_size;
        }
        // Restore arena for real use
        dev_mem->arena.base       = arena_base;
        dev_mem->arena.total_size = arena_size;
        dev_mem->arena.offset     = 0;
    }

    // ---- Allocate dedicated long-task bt_p pool from remaining VRAM ----
    // The shared arena bt_p is sized for short tasks (2 MB/slot × n_concurrent ≈ 5 GB).
    // For long tasks each slot needs antidiag × n_col bytes; using the shared pool limits
    // concurrency.  We grab remaining VRAM (after reserving space for ALL future arenas)
    // to provide a dedicated pool for more concurrent long-task slots.
    //
    // Reservation rule: after this stream's arena, subtract the arena size needed by
    // every stream that hasn't been initialized yet, plus a 512 MB safety margin.
    // Only allocate if the resulting budget is at least 256 MB (otherwise pointless).
    {
        size_t free_after_arena = 0, total_tmp = 0;
        cudaMemGetInfo(&free_after_arena, &total_tmp);

        // How many arenas still need to be allocated after this stream?
        int streams_after_this = num_streams - 1 - this_stream_idx;  // ≥0
        // Use g_vram_global_reserve so the pool respects the same headroom as auto-config.
        // This is critical for cuda-gdb: if global_vram_reserve_mb is set to e.g. 2048,
        // the pool will leave 2 GB free for the debugger instead of just 512 MB.
        size_t pool_safety = (g_vram_global_reserve > (size_t)512 * 1024 * 1024)
                             ? g_vram_global_reserve
                             : (size_t)512 * 1024 * 1024;
        size_t reserve_for_future = (size_t)streams_after_this * arena_size + pool_safety;

        size_t usable_for_long_pools = 0;
        if (free_after_arena > reserve_for_future)
            usable_for_long_pools = free_after_arena - reserve_for_future;

        // Split equally among ALL streams so each gets a consistent pool size.
        size_t this_pool = (num_streams > 0) ? (usable_for_long_pools / num_streams) : 0;

        dev_mem->d_align_backtrack_p_long = nullptr;
        dev_mem->long_bt_p_pool_bytes = 0;
        // Only bother if the pool is large enough to provide at least a few extra slots.
        const size_t MIN_USEFUL_LONG_POOL = (size_t)256 * 1024 * 1024;  // 256 MB
        if (this_pool >= MIN_USEFUL_LONG_POOL) {
            void *long_ptr = nullptr;
            cudaError_t cerr = cudaMalloc(&long_ptr, this_pool);
            if (cerr == cudaSuccess && long_ptr) {
                dev_mem->d_align_backtrack_p_long = (uint8_t*)long_ptr;
                dev_mem->long_bt_p_pool_bytes = this_pool;
            } else {
                // Non-fatal: fall back to shared arena bt_p for long tasks (fewer concurrent slots)
                fprintf(stderr, "[Warn] Failed to allocate long bt_p pool (%.2f GB); "
                        "long-task concurrency limited to arena pool.\n",
                        this_pool / (1024.0*1024.0*1024.0));
                cudaGetLastError();  // clear the error
            }
        }
        // If this_pool < MIN_USEFUL_LONG_POOL (e.g. arena fills nearly all VRAM with 2 streams),
        // skip the allocation entirely — long tasks fall back to the shared arena bt_p pool.
    }

    // ---- Set up concurrent Long-C context in dedicated pool ----
    // Carve fixed buffers from the start of d_align_backtrack_p_long.
    // The remaining pool bytes become d_long_c_bt_p for C-class backtrack.
    // align_ksw_temp_per_task was set by the earlier setup_align_phase dry run.
    {
        const size_t CMBT = LONG_C_BATCH_MAX;
        const size_t CSLT = LONG_C_SLOTS_MAX;
        dev_mem->d_long_c_unpacked_query = nullptr;
        dev_mem->d_long_c_max_tasks      = 0;
        dev_mem->d_long_c_n_slots        = 0;
        dev_mem->d_long_c_bt_p           = nullptr;
        dev_mem->d_long_c_bt_p_avail     = 0;
        dev_mem->h_long_c_unpacked_query  = nullptr;
        dev_mem->h_long_c_unpacked_target = nullptr;

        if (dev_mem->d_align_backtrack_p_long != nullptr) {
            size_t max_len      = dev_mem->max_align_task_len;
            size_t max_antidiag = 2 * max_len;
            size_t cigar_len    = 2 * max_len;
            size_t ksw_sz       = dev_mem->align_ksw_temp_per_task;
            size_t cub_sz       = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, cub_sz,
                                          (int*)nullptr, (int*)nullptr, (int)CMBT);

            size_t fixed = 0;
            fixed += a256(CMBT * max_len);                               // unp_q
            fixed += a256(CMBT * max_len);                               // unp_t
            fixed += a256(CMBT * (max_len / 8) * sizeof(uint32_t));      // pack_q
            fixed += a256(CMBT * (max_len / 8) * sizeof(uint32_t));      // pack_t
            fixed += a256(CMBT * sizeof(uint32_t));                      // q_offsets
            fixed += a256(CMBT * sizeof(uint32_t));                      // t_offsets
            fixed += a256(CMBT * sizeof(uint32_t));                      // q_lens
            fixed += a256(CMBT * sizeof(uint32_t));                      // t_lens
            fixed += a256(CMBT * sizeof(int32_t));                       // flag
            fixed += a256(CMBT * sizeof(int32_t));                       // bw
            fixed += a256(CMBT * cigar_len * sizeof(uint32_t));          // cigar_raw
            fixed += a256(CMBT * sizeof(int));                           // cigar_lengths
            fixed += a256(CMBT * cigar_len * sizeof(uint32_t));          // cigar_compact
            fixed += a256((CMBT + 1) * sizeof(uint32_t));                // compact_offsets
            fixed += a256(cub_sz);                                       // cub_tmp
            fixed += a256(CMBT * sizeof(int32_t));                       // blen
            fixed += a256(CMBT * sizeof(int32_t));                       // mlen
            fixed += a256(CMBT * sizeof(int32_t));                       // n_ambi
            fixed += a256(CMBT * sizeof(int32_t));                       // dp_max
            fixed += a256(CMBT * sizeof(int32_t));                       // gpu_stats_valid
            fixed += a256(CMBT * sizeof(int32_t));                       // scores
            fixed += a256(CMBT * sizeof(int32_t));                       // query_ends
            fixed += a256(CMBT * sizeof(int32_t));                       // target_ends
            fixed += a256(CMBT * sizeof(int32_t));                       // mqe
            fixed += a256(CMBT * sizeof(int32_t));                       // mqe_t
            fixed += a256(CMBT * sizeof(int32_t));                       // mte
            fixed += a256(CMBT * sizeof(int32_t));                       // mte_q
            fixed += a256(CMBT * sizeof(int32_t));                       // zdropped
            fixed += a256(sizeof(uint64_t) * 12);                        // gasal_res_t
            fixed += a256(25);                                           // mat
            fixed += a256(sizeof(int));                                  // task_counter
            fixed += a256(CSLT * max_antidiag * sizeof(int));            // bt_off
            fixed += a256(CSLT * max_antidiag * sizeof(int));            // bt_off_end
            fixed += a256(CSLT * ksw_sz);                               // ksw_temp

            const size_t MIN_C_BT_P = (size_t)256 * 1024 * 1024;
            if (fixed + MIN_C_BT_P <= dev_mem->long_bt_p_pool_bytes) {
                uint8_t *base = dev_mem->d_align_backtrack_p_long;
                size_t  pool  = dev_mem->long_bt_p_pool_bytes;
                size_t  off   = 0;
#define CNEXT(sz) ({ void *_p = base + off; off += a256(sz); _p; })
                dev_mem->d_long_c_unpacked_query  = (uint8_t*)CNEXT(CMBT * max_len);
                dev_mem->d_long_c_unpacked_target = (uint8_t*)CNEXT(CMBT * max_len);
                dev_mem->d_long_c_packed_query    = (uint32_t*)CNEXT(CMBT * (max_len / 8) * sizeof(uint32_t));
                dev_mem->d_long_c_packed_target   = (uint32_t*)CNEXT(CMBT * (max_len / 8) * sizeof(uint32_t));
                dev_mem->d_long_c_query_offsets   = (uint32_t*)CNEXT(CMBT * sizeof(uint32_t));
                dev_mem->d_long_c_target_offsets  = (uint32_t*)CNEXT(CMBT * sizeof(uint32_t));
                dev_mem->d_long_c_query_lens      = (uint32_t*)CNEXT(CMBT * sizeof(uint32_t));
                dev_mem->d_long_c_target_lens     = (uint32_t*)CNEXT(CMBT * sizeof(uint32_t));
                dev_mem->d_long_c_flag            = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_bw              = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_cigar_buffer    = (uint32_t*)CNEXT(CMBT * cigar_len * sizeof(uint32_t));
                dev_mem->d_long_c_cigar_lengths   = (int*)CNEXT(CMBT * sizeof(int));
                dev_mem->d_long_c_compact_cigar   = (uint32_t*)CNEXT(CMBT * cigar_len * sizeof(uint32_t));
                dev_mem->d_long_c_compact_offsets = (uint32_t*)CNEXT((CMBT + 1) * sizeof(uint32_t));
                dev_mem->d_long_c_cub_tmp         = CNEXT(cub_sz);
                dev_mem->d_long_c_cub_tmp_size    = cub_sz;
                dev_mem->d_long_c_blen            = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_mlen            = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_n_ambi          = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_dp_max          = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_gpu_stats_valid = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_scores          = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_query_ends      = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_target_ends     = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_mqe             = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_mqe_t           = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_mte             = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_mte_q           = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_zdropped        = (int32_t*)CNEXT(CMBT * sizeof(int32_t));
                dev_mem->d_long_c_device_res      = CNEXT(sizeof(uint64_t) * 12);
                dev_mem->d_long_c_mat             = (int8_t*)CNEXT(25);
                dev_mem->d_long_c_task_counter    = (int*)CNEXT(sizeof(int));
                dev_mem->d_long_c_bt_off          = (int*)CNEXT(CSLT * max_antidiag * sizeof(int));
                dev_mem->d_long_c_bt_off_end      = (int*)CNEXT(CSLT * max_antidiag * sizeof(int));
                dev_mem->d_long_c_ksw_temp        = CNEXT(CSLT * ksw_sz);
#undef CNEXT
                dev_mem->d_long_c_bt_p       = base + off;
                dev_mem->d_long_c_bt_p_avail = pool - off;
                dev_mem->d_long_c_max_tasks  = CMBT;
                dev_mem->d_long_c_n_slots    = CSLT;
                PLOG_INFO(stderr, "[Info::LongC] Concurrent C context: fixed=%.2fGB "
                          "bt_p=%.2fGB max_tasks=%zu slots=%zu\n",
                          off / (1024.0*1024.0*1024.0),
                          dev_mem->d_long_c_bt_p_avail / (1024.0*1024.0*1024.0),
                          CMBT, CSLT);
            } else {
                PLOG_INFO(stderr, "[Info::LongC] Dedicated pool too small for concurrent C context "
                          "(need %.2fGB + 256MB, have %.2fGB); C runs after Short.\n",
                          fixed / (1024.0*1024.0*1024.0),
                          dev_mem->long_bt_p_pool_bytes / (1024.0*1024.0*1024.0));
            }
        }
    }

    // Compute long_batch_max here (outside the pinned-buffer block) so it is visible
    // to the print_info log below as well as the cudaMallocHost calls that follow.
    size_t long_batch_max = long_batch_size(arena_size, dev_mem->max_align_task_len);

    if (print_info) {
        PLOG_INFO(stderr, "[Info::Align::Config]: h2d_max_tasks_short=%zu  h2d_max_tasks_long=%zu (%s)"
                "  gpu_max_slots=%d\n",
                dev_mem->max_align_tasks,
                long_batch_max,
                g_long_cigar_batch_override > 0 ? "manual" : "auto",
                dev_mem->n_align_concurrent_blocks);
        PLOG_INFO(stderr, "[Info::Align::Per Batch]: slots × (bt_p + bt_off + cigar_buf + ksw_temp)\n");
    }

    // Set up chain phase initially
    setup_chain_phase(dev_mem, anchor_per_batch, range_grid_size, num_cut);

    // ---- Pre-allocate pinned host buffers for alignment D2H/H2D ----
    // These were previously allocated/freed inside every gpu_align_batch_execute call
    // (22× cudaMallocHost + 22× cudaFreeHost per batch = expensive mlock syscalls).
    {
        // mbs: max batch size across both phases (drives per-task array sizes).
        // long_batch_max already computed above via compute_long_batch_size().
        size_t mbs = dev_mem->max_align_tasks;
        if (long_batch_max > mbs) mbs = long_batch_max;

        // cigar_buf_sz: max total CIGAR entries across both phases.
        // Short phase: max_align_tasks × short_cigar_len (2000)
        // Long phase:  long_batch_max × long_cigar_len (100000)
        size_t short_cigar_total = dev_mem->max_align_tasks * dev_mem->max_align_cigar_len;
        size_t long_cigar_total  = long_batch_max * 2 * dev_mem->max_align_task_len;
        size_t cigar_buf_sz = (long_cigar_total > short_cigar_total) ? long_cigar_total : short_cigar_total;
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

        // Pinned sequence staging buffers for H2D — replaces per-batch calloc/free.
        // Sized to cover the largest possible batch: short or long phase, whichever needs more.
        // long_batch_max computed above via compute_long_batch_size() (same formula as
        // setup_long_align_phase, avoids referencing a global that no longer exists).
        {
            size_t short_seq = (size_t)dev_mem->short_task_batch_size *
                               (((size_t)dev_mem->short_task_max_len + 7) & ~(size_t)7);
            size_t long_seq  = long_batch_max *
                               (size_t)dev_mem->max_align_task_len;
            size_t seq_staging = (short_seq > long_seq) ? short_seq : long_seq;
            dev_mem->h_align_seq_staging_bytes = seq_staging;
            cudaMallocHost(&dev_mem->h_align_unpacked_query,  seq_staging);
            cudaMallocHost(&dev_mem->h_align_unpacked_target, seq_staging);
        }

        // Separate pinned staging for concurrent C H2D
        // (h_align_unpacked_* can't be shared when C and Short H2D overlap)
        {
            size_t c_staging = (size_t)LONG_C_BATCH_MAX * dev_mem->max_align_task_len;
            cudaMallocHost(&dev_mem->h_long_c_unpacked_query,  c_staging);
            cudaMallocHost(&dev_mem->h_long_c_unpacked_target, c_staging);
        }

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
    // Free dedicated long-task bt_p pool (allocated outside arena)
    if (dev_mem->d_align_backtrack_p_long) {
        cudaFree(dev_mem->d_align_backtrack_p_long);
        dev_mem->d_align_backtrack_p_long = nullptr;
        dev_mem->long_bt_p_pool_bytes = 0;
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
    cudaFreeHost(dev_mem->h_align_unpacked_query);
    cudaFreeHost(dev_mem->h_align_unpacked_target);
    cudaFreeHost(dev_mem->h_long_c_unpacked_query);
    cudaFreeHost(dev_mem->h_long_c_unpacked_target);
    // BT anchor D2H staging buffers are per-batch cudaMallocHost (not pre-allocated).
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

void plmem_phase_to_long_align(deviceMemPtr *dev_mem) {
    // Transition from short-align arena layout to long-align layout.
    // Reuses the same physical cudaMalloc'd block: no cudaFree, no cudaMalloc.
    // After this call dev_mem->d_align_backtrack_p points to ~13 GB of bt_p
    // and dev_mem->long_bt_p_pool_bytes reflects its size.
    // Callers MUST refresh all local GPU pointers from dev_mem after this call.
    setup_long_align_phase(dev_mem);
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
            /* +1: range kernel uses one extra cut slot (RB slot) per block */
            cut_num += (config.anchor_per_block / config.blockdim) + 1;
            host_mem->start_idx[griddim + j] = end_idx;
            end_idx =
                host_mem->start_idx[griddim + j] + config.anchor_per_block;
            host_mem->read_end_idx[griddim + j] = idx + n;
            host_mem->cut_start_idx[griddim + j] = cut_num;
        }
        /* ceiling division: ensure small reads (n <= blockdim) get ≥1 slot,
         * plus 1 for the read-boundary cut slot; prevents cut_start_idx sharing
         * between consecutive reads that would race and lose RB cuts. */
        int n_last = n - (block_num - 1) * config.anchor_per_block;
        cut_num += (n_last + config.blockdim - 1) / config.blockdim + 1;
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

    // Update device-side metadata first so plrange/plscore see correct values.
    dev_mem->total_n = host_mem->total_n;
    dev_mem->num_cut = host_mem->cut_num;
    dev_mem->size    = host_mem->size;
    dev_mem->griddim = host_mem->griddim;

    // Skip CUDA ops when there are no anchors. cudaMemsetAsync/cudaMemcpyAsync
    // with count=0 returns cudaErrorInvalidValue on older CUDA drivers (pre-11.1).
    // This happens with partial-reference inputs (e.g. chr3.mmi) where reads
    // mapping to other chromosomes have n=0 anchors.
    if (host_mem->total_n == 0) return;

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
    size_t max_long_segs_bytes = dev_mem->buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t);
    cudaMemcpyAsync(long_mem->long_segs_og_idx,  dev_mem->d_long_seg_og, max_long_segs_bytes, cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->long_segs_buf_idx, dev_mem->d_long_seg,    max_long_segs_bytes, cudaMemcpyDeviceToHost, *stream);
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
    if (host_mem->total_n == 0) return;
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
    size_t max_long_segs_bytes2 = dev_mem->buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t);
    cudaMemcpyAsync(long_mem->long_segs_og_idx,  dev_mem->d_long_seg_og, max_long_segs_bytes2, cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(long_mem->long_segs_buf_idx, dev_mem->d_long_seg,    max_long_segs_bytes2, cudaMemcpyDeviceToHost, *stream);
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
    /* ceiling division + 2 extra slots per read (RB slot + one overflow guard)
     * matches the corrected plmem_reorg_input_arr allocation */
    max_num_cut = (max_total_n + range_kernel_config.blockdim - 1) /
                      range_kernel_config.blockdim +
                  3 * max_read +
                  (max_total_n / range_kernel_config.anchor_per_block + 1);
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

    // Per-stream VRAM budget: configurable headroom (JSON key "global_vram_reserve_mb").
    // Also propagated to plmem_malloc_device_mem for the long bt_p pool.
    // Increase to 1024+ MB when using cuda-gdb so the pool leaves room for the debugger.
    cJSON *vram_reserve_json = cJSON_GetObjectItem(json, "global_vram_reserve_mb");
    size_t global_reserve = vram_reserve_json
        ? (size_t)vram_reserve_json->valueint * 1024 * 1024
        : (size_t)256 * 1024 * 1024;  // default 256 MB
    g_vram_global_reserve = global_reserve;  // propagate to bt_p pool sizing

    // Optional manual override for the long-task CIGAR batch cap.
    // 0 (default) = auto via compute_long_batch_size(); set to e.g. 1024 to shrink CIGAR
    // buffers and leave more arena for bt_p, increasing concurrent slots for large tasks.
    cJSON *cigar_batch_json = cJSON_GetObjectItem(json, "long_cigar_batch");
    g_long_cigar_batch_override = cigar_batch_json
        ? (size_t)cigar_batch_json->valueint
        : 0;

    cJSON *task_len_json = cJSON_GetObjectItem(json, "max_align_task_len");
    if (task_len_json && task_len_json->valueint > 0)
        g_max_align_task_len = (size_t)task_len_json->valueint;

    size_t usable = (gpu_free_mem > global_reserve) ? (gpu_free_mem - global_reserve) : gpu_free_mem;
    size_t avail_mem_per_stream = usable / (*num_stream_);

    // Exact per-anchor memory costs matching setup_chain_phase() allocations:
    //   Chain anchors (N):  ax(4)+ay(4)+sid(1)+xrev(4)+yrev(4)+range(4)+f(4)+p(2) = 27
    //   Backtrack (N*mb):   ax,ay,xrev,yrev,f,p(22) + zx,zy,v,p_abs(32) + t,u(12) + ax,ay,xrev,yrev_out(16) = 82
    //   CUB sort temp (N*mb): ~16 bytes (double-buffer for int64 key+value pairs)
    //   Long seg data (L):  ax(4)+ay(4)+sid(1)+range(4)+f(4)+p(2) = 19
    //   Long seg index (L): seg_t*2 + map(4) per (long_seg_cutoff*cut_unit) entries = 36/10240 per L
    //   Index/cut:          per-grid(24) + per-cut(12) + per bt_r(mb*24) + vt reserve(mb*24)
    int mb = score_kernel_config.micro_batch;

    // N-proportional cost
    size_t chain_per_n = 27;
    size_t bt_per_n    = (size_t)mb * 82;   // backtrack: 82 bytes × micro_batch
    size_t cub_per_n   = (size_t)mb * 16;   // CUB sort temp: ~16 bytes × micro_batch (key+value double-buffer)
    // The voting DEVICE arrays were removed (setup_chain_phase no longer allocates
    // them), but KEEP this mb*61 term as a budget reserve.  It caps max_total_n at the
    // pre-removal value, which bounds the per-batch HOST allocation in
    // gpu_align_batch_init (~8 MB/read × reads-fitted-per-batch).  Dropping it raised
    // max_total_n ~1.5x and OOM-killed the host on large inputs.  The reclaimed device
    // VRAM still benefits extension: it falls through to the long bt_p pool (which takes
    // all remaining arena space).
    size_t vt_reserve_per_n = (size_t)mb * 61;
    size_t per_anchor_total = chain_per_n + bt_per_n + cub_per_n + vt_reserve_per_n;

    // L-proportional cost (long segment buffers)
    size_t per_long_entry = 23;  // ax(4)+ay(4)+sid(1)+range(4)+xrev(4)+f(4)+p(2) long arrays

    // Avg anchors per read (for index/cut overhead estimate)
    cJSON *avg_n_json = cJSON_GetObjectItem(json, "avg_read_n");
    size_t avg_read_n = avg_n_json ? (size_t)avg_n_json->valueint : 1000;

    // Long seg buffer ratio: L as fraction of N.
    // Default L = 2*N (empirically reasonable for ONT data).
    cJSON *long_seg_json = cJSON_GetObjectItem(json, "long_seg_buffer_size");
    double long_ratio = 2.0;  // L = long_ratio * N

    // Per-read overhead (index + cut + bt_r + vt-reserve arrays):
    //   G ≈ N/anchor_per_block + N/avg_read_n
    //   C ≈ N/blockdim + N/avg_read_n
    //   per G: 24 (index) + mb*24 (bt_r) + mb*24 (vt reserve) = 24 + 48*mb
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

    g_bytes_per_anchor = total_per_n;
    PLOG_INFO(stderr, "[Info::Arena] Auto-config for %d stream%s (%.2f GB free, %.2f GB/stream)\n",
            *num_stream_, *num_stream_ > 1 ? "s" : "",
            gpu_free_mem / (1024.0*1024.0*1024.0),
            avail_mem_per_stream / (1024.0*1024.0*1024.0));
    PLOG_INFO(stderr, "[Info::Chain::Config] max_total_n=%zu, max_read=%d, long_seg_buf=%zu, %.0fB/anchor\n",
            *max_total_n_, *max_read_, *long_seg_buffer_size_, total_per_n);
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

    /* Force CPU spin on GPU sync (vs the default Auto, which on A100 picks   */
    /* BlockingSync — every cudaStreamSynchronize then becomes a syscall and  */
    /* thread enters kernel-mode wait, accumulating huge sys time and adding  */
    /* ~50 μs latency per sync.  In a multi-stream pipeline with 100+ batches */
    /* per second this adds tens of seconds of wall time.  Spin wastes CPU    */
    /* cycles but reduces sync latency to ~μs, which dominates here.          */
    /* MUST be called before any other CUDA call (i.e. before cudaSetDevice). */
    cudaError_t flag_err = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if (flag_err != cudaSuccess && flag_err != cudaErrorSetOnActiveProcess) {
        fprintf(stderr, "[Warn] cudaSetDeviceFlags(Spin) failed: %s — keeping default sync mode.\n",
                cudaGetErrorString(flag_err));
        cudaGetLastError();  /* clear */
    }

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
