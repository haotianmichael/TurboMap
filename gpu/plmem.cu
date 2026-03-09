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

void plmem_malloc_device_mem(deviceMemPtr *dev_mem, size_t anchor_per_batch, int range_grid_size, int num_cut){
    fprintf(stderr, "[Info] GPU MEMORY ALLOCATION BREAKDOWN ========== ");
    fprintf(stderr, "Configuration: anchor_per_batch=%zu, range_grid_size=%d, num_cut=%d, ",
            anchor_per_batch, range_grid_size, num_cut);
    fprintf(stderr, "buffer_size_long=%zu\n", dev_mem->buffer_size_long);

    // data array
    cudaSetDevice(CUDA_DEVICE);

    size_t chain_ax_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_ay_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_sid_size = anchor_per_batch * sizeof(int8_t);
    size_t chain_xrev_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_yrev_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_range_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_f_size = anchor_per_batch * sizeof(int32_t);
    size_t chain_p_size = anchor_per_batch * sizeof(uint16_t);
    size_t chain_total = chain_ax_size + chain_ay_size + chain_sid_size + chain_xrev_size + chain_yrev_size +
                         chain_range_size + chain_f_size + chain_p_size;

    cudaMalloc(&dev_mem->d_ax, chain_ax_size);
    cudaMalloc(&dev_mem->d_ay, chain_ay_size);
    cudaMalloc(&dev_mem->d_sid, chain_sid_size);
    cudaMalloc(&dev_mem->d_xrev, chain_xrev_size);
    cudaMalloc(&dev_mem->d_yrev, chain_yrev_size);
    cudaMalloc(&dev_mem->d_range, chain_range_size);
    cudaMalloc(&dev_mem->d_f, chain_f_size);
    cudaMalloc(&dev_mem->d_p, chain_p_size);
    fprintf(stderr, " [Chain] Total anchor buffers: %.2f MB\n", chain_total / (1024.0*1024.0));

    //index
    size_t idx_total = range_grid_size * sizeof(size_t) * 3;
    cudaMalloc(&dev_mem->d_start_idx, range_grid_size * sizeof(size_t));
    cudaMalloc(&dev_mem->d_read_end_idx, range_grid_size * sizeof(size_t));
    cudaMalloc(&dev_mem->d_cut_start_idx, range_grid_size * sizeof(size_t));

    // cut
    size_t cut_size = num_cut * sizeof(size_t);
    size_t long_seg_size = dev_mem->buffer_size_long / (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) * sizeof(seg_t);
    size_t mid_seg_size = num_cut/(score_kernel_config.mid_seg_cutoff + 1) * sizeof(seg_t);
    cudaMalloc(&dev_mem->d_cut, cut_size);
    cudaMalloc(&dev_mem->d_long_seg_count, sizeof(unsigned int));
    cudaMalloc(&dev_mem->d_long_seg, long_seg_size);
    cudaMalloc(&dev_mem->d_long_seg_og, long_seg_size);
    cudaMalloc(&dev_mem->d_mid_seg_count, sizeof(unsigned int));
    cudaMalloc(&dev_mem->d_mid_seg, mid_seg_size);
    fprintf(stderr, " [Chain] Total Cut buffers: %.2f MB\n", (idx_total + cut_size + 2*long_seg_size + mid_seg_size + 2*sizeof(unsigned int)) / (1024.0*1024.0));

    // long seg buffer
    size_t long_ax_size = dev_mem->buffer_size_long * sizeof(int32_t);
    size_t long_ay_size = dev_mem->buffer_size_long * sizeof(int32_t);
    size_t long_sid_size = dev_mem->buffer_size_long * sizeof(int8_t);
    size_t long_range_size = dev_mem->buffer_size_long * sizeof(int32_t);
    size_t long_f_size = dev_mem->buffer_size_long * sizeof(int32_t);
    size_t long_p_size = dev_mem->buffer_size_long * sizeof(uint16_t);
    size_t long_total = long_ax_size + long_ay_size + long_sid_size + long_range_size +
                        long_f_size + long_p_size + sizeof(size_t);

    cudaMalloc(&dev_mem->d_ax_long, long_ax_size);
    cudaMalloc(&dev_mem->d_ay_long, long_ay_size);
    cudaMalloc(&dev_mem->d_sid_long, long_sid_size);
    cudaMalloc(&dev_mem->d_range_long, long_range_size);
    cudaMalloc(&dev_mem->d_total_n_long, sizeof(size_t));
    cudaMalloc(&dev_mem->d_f_long, long_f_size);
    cudaMalloc(&dev_mem->d_p_long, long_p_size);
    fprintf(stderr, " [Chain] Total long seg buffers: %.2f MB (%.2f GB)\n",
            long_total / (1024.0*1024.0), long_total / (1024.0*1024.0*1024.0)); 

    // ========== Chain Backtrack Pre-allocated Buffers ==========
    // Eliminates 17 cudaMalloc + 16 cudaFree per plbacktrack_gpu call.
    // Each cudaMalloc stalls the entire GPU; pre-allocating removes this bottleneck.
    // Anchor-sized arrays: anchor_per_batch = 50M → ~3 GB total.
    // Read-sized arrays:   range_grid_size ≥ max_reads → negligible.
    {
        size_t bt_n = anchor_per_batch;
        size_t bt_r = (size_t)range_grid_size;  // >= max_read; read-sized arrays are tiny

        dev_mem->d_bt_max_total_n = bt_n;
        dev_mem->d_bt_max_n_reads = bt_r;

        cudaMalloc(&dev_mem->d_bt_zx,          bt_n * sizeof(int64_t));
        cudaMalloc(&dev_mem->d_bt_zy,          bt_n * sizeof(int64_t));
        cudaMalloc(&dev_mem->d_bt_v,           bt_n * sizeof(int64_t));
        cudaMalloc(&dev_mem->d_bt_p_abs,       bt_n * sizeof(int64_t));
        cudaMalloc(&dev_mem->d_bt_t,           bt_n * sizeof(int32_t));
        cudaMalloc(&dev_mem->d_bt_u,           bt_n * sizeof(uint64_t));
        cudaMalloc(&dev_mem->d_bt_ax_out,      bt_n * sizeof(int32_t));
        cudaMalloc(&dev_mem->d_bt_ay_out,      bt_n * sizeof(int32_t));
        cudaMalloc(&dev_mem->d_bt_xrev_out,    bt_n * sizeof(int32_t));
        cudaMalloc(&dev_mem->d_bt_yrev_out,    bt_n * sizeof(int32_t));
        cudaMalloc(&dev_mem->d_bt_n_a,          bt_r * sizeof(int));
        cudaMalloc(&dev_mem->d_bt_offset,       bt_r * sizeof(int));
        cudaMalloc(&dev_mem->d_bt_ofs_end,      bt_r * sizeof(int));
        cudaMalloc(&dev_mem->d_bt_num_elements, bt_r * sizeof(int));
        cudaMalloc(&dev_mem->d_bt_n_v,          bt_r * sizeof(int));
        cudaMalloc(&dev_mem->d_bt_n_u,          bt_r * sizeof(int));

        // CUB DeviceSegmentedRadixSort temp storage (size query with max dimensions)
        dev_mem->d_bt_cub_tmp      = nullptr;
        dev_mem->d_bt_cub_tmp_size = 0;
        {
            size_t tmp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, tmp_bytes,
                dev_mem->d_bt_zx, dev_mem->d_bt_zx,
                dev_mem->d_bt_zy, dev_mem->d_bt_zy,
                (int)bt_n, (int)bt_r,
                dev_mem->d_bt_offset, dev_mem->d_bt_ofs_end,
                0, (int)(sizeof(int64_t) * 8));
            dev_mem->d_bt_cub_tmp_size = tmp_bytes;
            cudaMalloc(&dev_mem->d_bt_cub_tmp, tmp_bytes);
        }

        size_t bt_total = bt_n * (4*sizeof(int64_t) + sizeof(int32_t) + sizeof(uint64_t) + 4*sizeof(int32_t))
                        + bt_r * 6 * sizeof(int)
                        + dev_mem->d_bt_cub_tmp_size;
        fprintf(stderr, " [Chain] Backtrack pre-alloc buffers: %.2f MB (%.2f GB)\n",
                bt_total / (1024.0*1024.0), bt_total / (1024.0*1024.0*1024.0));
    }

    // ========== Alignment Buffers ==========
    // Configuration for alignment
    // OPTIMIZATION: Adjusted for 24GB GPU memory constraints
    // Key bottleneck: backtrack_p buffer = alloc_tasks × (2×max_query_len) × 752
    dev_mem->max_align_tasks = 120000;        // For DP phase batching
    dev_mem->max_align_seq_bytes = 1*1024*1024*1024;  // 1GB for sequences (reduced from 2GB)
    dev_mem->max_align_query_len = 50000;    // max query length (reduced from 100000 to save ~9GB)


    // Sequence data
    size_t seq_unpacked_size = dev_mem->max_align_seq_bytes;
    size_t seq_packed_size = (dev_mem->max_align_seq_bytes / 8) * sizeof(uint32_t);
    size_t metadata_size = dev_mem->max_align_tasks * sizeof(uint32_t);

    cudaMalloc(&dev_mem->d_align_unpacked_query, seq_unpacked_size);

    cudaMalloc(&dev_mem->d_align_unpacked_target, seq_unpacked_size);

    cudaMalloc(&dev_mem->d_align_packed_query, seq_packed_size);

    cudaMalloc(&dev_mem->d_align_packed_target, seq_packed_size);

    cudaMalloc(&dev_mem->d_align_query_offsets, metadata_size);
    cudaMalloc(&dev_mem->d_align_target_offsets, metadata_size);
    cudaMalloc(&dev_mem->d_align_query_lens, metadata_size);
    cudaMalloc(&dev_mem->d_align_target_lens, metadata_size);
    cudaMalloc(&dev_mem->d_align_flag, metadata_size);

    // global buffer (28 blocks * 32 threads/warp * max_query_len * 4)
    size_t global_buffer_size = 28 * (256 / 8) * dev_mem->max_align_query_len * 4;
    size_t global_buffer_bytes = global_buffer_size * sizeof(short2);
    cudaMalloc(&dev_mem->d_align_global_buffer, global_buffer_bytes);

    size_t short_task_batch_size = 7000; // For tasks with max(qlen, tlen) <= 1000bp
    size_t long_task_batch_size = 128;    // For tasks with max(qlen, tlen) > 1000bp
    size_t short_task_max_len = 1000;     // Max qlen or tlen for short tasks

    // KSW temp buffer (sized for short_task_batch_size concurrent tasks)
    size_t max_len = dev_mem->max_align_query_len;
    size_t H_size = max_len * sizeof(int32_t);
    size_t u8_arrays_size = (max_len + 1) * 7 * sizeof(int8_t);
    size_t seq_size = max_len * 2 * sizeof(uint8_t);
    size_t raw_size = H_size + u8_arrays_size + seq_size;
    dev_mem->align_ksw_temp_per_task = (raw_size + 7) & ~7ULL;
    size_t ksw_temp_bytes = short_task_batch_size * dev_mem->align_ksw_temp_per_task;
    cudaMalloc(&dev_mem->d_align_ksw_temp_buffer, ksw_temp_bytes);

    fprintf(stderr, " [Align] DP buffers: %.2f MB\n", (seq_unpacked_size *2 + seq_packed_size*2 + metadata_size*5 + global_buffer_bytes + ksw_temp_bytes) / (1024.0*1024.0));
    // Backtrack buffers(Strategy: Two-tier allocation for short vs long tasks)
    // ==================== Backtrack Buffers ====================
    // Purpose: Store information for CIGAR generation (alignment path reconstruction)
    //
    // For each alignment with qlen and tlen:
    // - Total antidiagonals: qlen + tlen - 1
    // - Cells per antidiagonal: n_col = min(bandwidth+1, min(qlen, tlen))
    //
    // backtrack_p[]: Direction bits for each DP cell
    //   - Size per task: (qlen + tlen - 1) × n_col bytes
    //   - Each byte stores 4 bits for direction (which cell we came from)
    //                     + 4 bits for state flags
    //
    // backtrack_off[], backtrack_off_end[]: Valid range for each antidiagonal
    //   - Size per task: (qlen + tlen - 1) × 2 integers
    //   - Tells us which cells in each antidiagonal are within the band
    //
    // backtrack_n_col[]: The n_col value for each task
    //   - Size per task: 1 integer
    //=============================================================
    // SHORT TASKS (max(qlen, tlen) <= 1000bp, ~90% of workload):
    //   - Allocate based on 1000bp max per sequence
    //   - Backtrack size per task: (1000+1000) × 752 = 1.5MB
    //   - With 32GB GPU memory, ~28GB available for backtrack
    //   - Theoretical max: 28GB / 1.5MB ≈ 18,000 tasks
    //   - Conservative allocation: 10,000 tasks (accounts for DP buffers, safety margin)
    //   - Memory: 10,000 × 1.5MB ≈ 15GB backtrack + other buffers ≈ 18GB total
    //
    // LONG TASKS (max(qlen, tlen) > 1000bp, ~10% of workload):
    //   - Allocate based on max length (50000bp)
    //   - Backtrack size per task: (50000+50000) × 752 = 75MB
    //   - Batch size: 200 tasks (original conservative value)
    //   - Memory: 200 × 75MB ≈ 14.3GB
    //
    // Processing flow in plalign.cu:
    //   1. Scan all tasks, classify by max(qlen, tlen)
    //   2. Process short tasks first (high throughput, 10,000 at a time)
    //   3. Process long tasks separately (low throughput, 200 at a time)
    //

   
    // Allocate buffers for SHORT tasks (most common case, optimized for throughput)
    // alloc_slots: backtrack buffer is slot-indexed (persistent kernel reuses slots).
    // Keep 7000 slots so both phases fit:
    //   Short: 7000 × 1.5MB = 10.5GB  (hardware runs ≤2560 concurrently)
    //   Long:  140 × 75MB  = 10.5GB  (same buffer, capped by plalign.cu)
    size_t alloc_slots = short_task_batch_size;  // 7000

    // Calculate max_antidiag based on SHORT task length (1000bp per sequence)
    size_t max_antidiag_short = 2 * short_task_max_len;  // 2000 antidiagonals for 1000+1000bp
    size_t max_n_col = 751 + 1;  // bandwidth + 1

    // Store configuration in deviceMemPtr for plalign.cu to use
    dev_mem->short_task_batch_size = short_task_batch_size;
    dev_mem->long_task_batch_size = long_task_batch_size;
    dev_mem->short_task_max_len = short_task_max_len;
    dev_mem->max_align_backtrack_size = max_antidiag_short * max_n_col;  // Optimized for short tasks
    dev_mem->max_align_cigar_len = 2 * short_task_max_len;  // Optimized for short tasks

    size_t bt_p_bytes       = alloc_slots * dev_mem->max_align_backtrack_size;
    size_t bt_off_bytes     = alloc_slots * max_antidiag_short * sizeof(int);
    size_t bt_off_end_bytes = alloc_slots * max_antidiag_short * sizeof(int);
    size_t bt_n_col_bytes   = alloc_slots * sizeof(int);

    // CIGAR buffer: task-indexed output, sized for max_align_tasks per batch.
    // With persistent kernel batch_size = max_align_tasks (short) or 2400 (long),
    // both require max_align_tasks × max_align_cigar_len × 4 = 120000×2000×4 = 960MB.
    // Same 960MB covers long tasks: 2400 × 100000 × 4 = 960MB.
    size_t cigar_buf_bytes = dev_mem->max_align_tasks * dev_mem->max_align_cigar_len * sizeof(uint32_t);
    size_t cigar_len_bytes = dev_mem->max_align_tasks * sizeof(int);

    cudaMalloc(&dev_mem->d_align_backtrack_p, bt_p_bytes);
    cudaMalloc(&dev_mem->d_align_backtrack_off, bt_off_bytes);
    cudaMalloc(&dev_mem->d_align_backtrack_off_end, bt_off_end_bytes);
    cudaMalloc(&dev_mem->d_align_backtrack_n_col, bt_n_col_bytes);
    cudaMalloc(&dev_mem->d_align_cigar_buffer, cigar_buf_bytes);
    cudaMalloc(&dev_mem->d_align_cigar_lengths, cigar_len_bytes);

    // P1: Compact CIGAR buffers
    // compact_cigar: worst-case same size as stride buffer (all tasks have max CIGAR length)
    // compact_offsets: (max_align_tasks + 1) entries so CUB ExclusiveSum output fits
    cudaMalloc(&dev_mem->d_align_compact_cigar, cigar_buf_bytes);
    cudaMalloc(&dev_mem->d_align_compact_offsets,
               (dev_mem->max_align_tasks + 1) * sizeof(uint32_t));

    // Query CUB DeviceScan::ExclusiveSum temp size (cast int* input, uint32_t* output)
    // Use ExclusiveSum on cigar_lengths (int) → compact_offsets (uint32_t).
    // CUB requires type match; we cast to (uint32_t*) for both since lengths are non-negative.
    dev_mem->d_align_cub_tmp = nullptr;
    dev_mem->align_cub_tmp_size = 0;
    {
        size_t tmp_bytes = 0;
        // Use (int*) cast for both pointers so CUB deduces type=int consistently.
        // d_align_compact_offsets is uint32_t* but same size as int; values are non-negative.
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                      dev_mem->d_align_cigar_lengths,
                                      (int*)dev_mem->d_align_compact_offsets,
                                      (int)dev_mem->max_align_tasks);
        dev_mem->align_cub_tmp_size = tmp_bytes;
        cudaMalloc(&dev_mem->d_align_cub_tmp, tmp_bytes);
    }

    // P2: GPU alignment statistics buffers (one entry per task slot)
    size_t stats_bytes = dev_mem->max_align_tasks * sizeof(int32_t);
    cudaMalloc(&dev_mem->d_align_blen,            stats_bytes);
    cudaMalloc(&dev_mem->d_align_mlen,            stats_bytes);
    cudaMalloc(&dev_mem->d_align_n_ambi,          stats_bytes);
    cudaMalloc(&dev_mem->d_align_dp_max,          stats_bytes);
    cudaMalloc(&dev_mem->d_align_gpu_stats_valid, stats_bytes);

    // Result structures
    cudaMalloc(&dev_mem->d_align_device_res, sizeof(gasal_res_t));
    cudaMalloc(&dev_mem->d_align_ez_array, sizeof(ksw_extz_t) * short_task_batch_size);
    cudaMalloc(&dev_mem->d_align_scores, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_query_ends, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_target_ends, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_mqe, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_mqe_t, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_mte, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_mte_q, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_task_to_align_id, dev_mem->max_align_tasks * sizeof(int32_t));
    cudaMalloc(&dev_mem->d_align_mat, 25 * sizeof(int8_t));

    // Persistent kernel: atomic task counter + concurrent block count
    // V100: 80 SMs x 32 blocks/SM (with 3072 bytes smem) = 2560 concurrent blocks
    // Use query to get actual SM count for portability
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks_per_sm = 32;  // limited by shared memory (3072 bytes/block, 98304/SM)
    dev_mem->n_align_concurrent_blocks = numSMs * max_blocks_per_sm;
    cudaMalloc(&dev_mem->d_align_task_counter, sizeof(int));
    cudaStreamCreate(&dev_mem->align_stream);

    // Calculate total memory allocated for alignment backtrack
    size_t bck_total = bt_p_bytes + bt_off_bytes + bt_off_end_bytes + bt_n_col_bytes +
                         cigar_buf_bytes + cigar_len_bytes +
                         dev_mem->max_align_tasks * sizeof(int32_t) * 4 +  // scores, ends, and task mapping
                         sizeof(gasal_res_t) + sizeof(ksw_extz_t) * short_task_batch_size + 25;  // 25 bytes for d_align_mat

    fprintf(stderr, " [Align] Total BackTrack buffers: %.2f GB\n", bck_total / (1024.0*1024.0*1024.0));

    // ========== GRAND TOTAL CALCULATION ==========
    size_t chain_bt_total = dev_mem->d_bt_max_total_n *
                            (4*sizeof(int64_t) + sizeof(int32_t) + sizeof(uint64_t) + 4*sizeof(int32_t))
                          + dev_mem->d_bt_max_n_reads * 6 * sizeof(int)
                          + dev_mem->d_bt_cub_tmp_size;

    size_t chain_total_all = chain_total +
                             (idx_total + cut_size + 2*long_seg_size + mid_seg_size + 2*sizeof(unsigned int)) +
                             long_total + chain_bt_total;

    size_t align_dp_total = (seq_unpacked_size*2 + seq_packed_size*2 +
                             metadata_size*5 + global_buffer_bytes + ksw_temp_bytes);
    size_t align_total_all = align_dp_total + bck_total;

    size_t grand_total = chain_total_all + align_total_all;

    fprintf(stderr, " [Chain] Total:       %8.2f MB (%.2f GB)\n",
            chain_total_all / (1024.0*1024.0), chain_total_all / (1024.0*1024.0*1024.0));
    fprintf(stderr, " [Align] Total:       %8.2f MB (%.2f GB)\n",
            align_total_all / (1024.0*1024.0), align_total_all / (1024.0*1024.0*1024.0));
    fprintf(stderr, "[Info] TurboMap Total GPU Memory: %.2f MB (%.2f GB)\n",
            grand_total / (1024.0*1024.0), grand_total / (1024.0*1024.0*1024.0));

    cudaCheck();
}

void plmem_free_device_mem(deviceMemPtr *dev_mem) {
    // chain buffer
    cudaFree(dev_mem->d_ax);
    cudaFree(dev_mem->d_ay);
    cudaFree(dev_mem->d_sid);
    cudaFree(dev_mem->d_xrev);
    cudaFree(dev_mem->d_yrev);
    cudaFree(dev_mem->d_range);
    cudaFree(dev_mem->d_f);
    cudaFree(dev_mem->d_p);

    cudaFree(dev_mem->d_start_idx);
    cudaFree(dev_mem->d_read_end_idx);
    cudaFree(dev_mem->d_cut_start_idx);

    cudaFree(dev_mem->d_cut);
    cudaFree(dev_mem->d_long_seg);
    cudaFree(dev_mem->d_long_seg_og);
    cudaFree(dev_mem->d_long_seg_count);
    cudaFree(dev_mem->d_mid_seg);
    cudaFree(dev_mem->d_mid_seg_count);

    cudaFree(dev_mem->d_ax_long);
    cudaFree(dev_mem->d_ay_long);
    cudaFree(dev_mem->d_sid_long);
    cudaFree(dev_mem->d_range_long);
    cudaFree(dev_mem->d_total_n_long);

    // Chain backtrack pre-allocated buffers
    cudaFree(dev_mem->d_bt_zx);
    cudaFree(dev_mem->d_bt_zy);
    cudaFree(dev_mem->d_bt_v);
    cudaFree(dev_mem->d_bt_p_abs);
    cudaFree(dev_mem->d_bt_t);
    cudaFree(dev_mem->d_bt_u);
    cudaFree(dev_mem->d_bt_ax_out);
    cudaFree(dev_mem->d_bt_ay_out);
    cudaFree(dev_mem->d_bt_xrev_out);
    cudaFree(dev_mem->d_bt_yrev_out);
    cudaFree(dev_mem->d_bt_n_a);
    cudaFree(dev_mem->d_bt_offset);
    cudaFree(dev_mem->d_bt_ofs_end);
    cudaFree(dev_mem->d_bt_num_elements);
    cudaFree(dev_mem->d_bt_n_v);
    cudaFree(dev_mem->d_bt_n_u);
    cudaFree(dev_mem->d_bt_cub_tmp);

    // Alignment buffers
    if (dev_mem->d_align_unpacked_query) cudaFree(dev_mem->d_align_unpacked_query);
    if (dev_mem->d_align_unpacked_target) cudaFree(dev_mem->d_align_unpacked_target);
    if (dev_mem->d_align_packed_query) cudaFree(dev_mem->d_align_packed_query);
    if (dev_mem->d_align_packed_target) cudaFree(dev_mem->d_align_packed_target);
    if (dev_mem->d_align_query_offsets) cudaFree(dev_mem->d_align_query_offsets);
    if (dev_mem->d_align_target_offsets) cudaFree(dev_mem->d_align_target_offsets);
    if (dev_mem->d_align_query_lens) cudaFree(dev_mem->d_align_query_lens);
    if (dev_mem->d_align_target_lens) cudaFree(dev_mem->d_align_target_lens);
    if (dev_mem->d_align_flag) cudaFree(dev_mem->d_align_flag);
    if (dev_mem->d_align_global_buffer) cudaFree(dev_mem->d_align_global_buffer);
    if (dev_mem->d_align_ksw_temp_buffer) cudaFree(dev_mem->d_align_ksw_temp_buffer);
    if (dev_mem->d_align_backtrack_p) cudaFree(dev_mem->d_align_backtrack_p);
    if (dev_mem->d_align_backtrack_off) cudaFree(dev_mem->d_align_backtrack_off);
    if (dev_mem->d_align_backtrack_off_end) cudaFree(dev_mem->d_align_backtrack_off_end);
    if (dev_mem->d_align_backtrack_n_col) cudaFree(dev_mem->d_align_backtrack_n_col);
    if (dev_mem->d_align_cigar_buffer) cudaFree(dev_mem->d_align_cigar_buffer);
    if (dev_mem->d_align_cigar_lengths) cudaFree(dev_mem->d_align_cigar_lengths);
    if (dev_mem->d_align_compact_cigar)   cudaFree(dev_mem->d_align_compact_cigar);
    if (dev_mem->d_align_compact_offsets) cudaFree(dev_mem->d_align_compact_offsets);
    if (dev_mem->d_align_cub_tmp)         cudaFree(dev_mem->d_align_cub_tmp);
    if (dev_mem->d_align_blen)            cudaFree(dev_mem->d_align_blen);
    if (dev_mem->d_align_mlen)            cudaFree(dev_mem->d_align_mlen);
    if (dev_mem->d_align_n_ambi)          cudaFree(dev_mem->d_align_n_ambi);
    if (dev_mem->d_align_dp_max)          cudaFree(dev_mem->d_align_dp_max);
    if (dev_mem->d_align_gpu_stats_valid) cudaFree(dev_mem->d_align_gpu_stats_valid);
    if (dev_mem->d_align_device_res) cudaFree(dev_mem->d_align_device_res);
    if (dev_mem->d_align_ez_array) cudaFree(dev_mem->d_align_ez_array);
    if (dev_mem->d_align_scores) cudaFree(dev_mem->d_align_scores);
    if (dev_mem->d_align_query_ends) cudaFree(dev_mem->d_align_query_ends);
    if (dev_mem->d_align_target_ends) cudaFree(dev_mem->d_align_target_ends);
    if (dev_mem->d_align_task_to_align_id) cudaFree(dev_mem->d_align_task_to_align_id);
    if (dev_mem->d_align_mat) cudaFree(dev_mem->d_align_mat);
    if (dev_mem->d_align_task_counter) cudaFree(dev_mem->d_align_task_counter);
    cudaStreamDestroy(dev_mem->align_stream);

    cudaCheck();
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

    /* If Use define max_total_n & max_read */
    // FIXME: this is limited by int32max
    cJSON *max_total_n_json = cJSON_GetObjectItem(json, "max_total_n"); 
    cJSON *max_read_json = cJSON_GetObjectItem(json, "max_read");
    cJSON *long_seg_buffer_size_json = cJSON_GetObjectItem(json, "long_seg_buffer_size");
    if (max_total_n_json && max_read_json){
        *max_total_n_ = (size_t) max_total_n_json->valuedouble;
        *max_read_ = max_read_json->valueint;
        *long_seg_buffer_size_ = long_seg_buffer_size_json->valueint;
        return;
    }

    /* Determine configuration smartly */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, CUDA_DEVICE);

    size_t avail_mem_per_stream = (prop.totalGlobalMem / *num_stream_ ) * 0.9;

    // memory per anchor = (ax + ay + range + f + p) + (start_idx + read_end_idx
    // + cut_start_idx) + cut + long_seg size: F1 = ax + ay + range + f + p; F2
    // = start_idx + read_end_idx + cut_start_idx; F3 = cut; F4 = long_seg
    int F1 = 8 + 8 + 4 + 4 + 2, F2 = 8 + 8 + 8, F3 = 8, F4 = 16;
    // TODO: define these data types

    // max iteration of each block, must be an integer
    // int max_it =
    //     range_kernel_config.anchor_per_block / range_kernel_config.blockdim;
    // int blockdim = range_kernel_config.blockdim;

    // g = max_grid_size
    // g * F2 + g * blockdim * max_it * F1 + max_cut * F3 + max_cut/2 * F4 <
    // mem_per_stream max_cut = g * max_it
    /*
    size_t cost_per_anchor = F1;
    size_t cost_per_grid = F2;
    size_t cost_per_cut = F3 + F4 / 2;
    */
    size_t avg_read_n = get_json_int(json, "avg_read_n");

    /**
     * Assume max_total_n = max_read * avg_read_n
     * max_grid = max_total_n / range.anchor_per_block + max_read
     *          = max_read *( avg_read_n / anchor_per_block + 1)
     * max_cut  = max_total_n / range.blockdim + max_read
     *          = max_read * ( avg_read_n / blockdim + 1)
     * total_mem = max_grid * cost_per_grid + max_cut * cost_per_cut + max_total_n * cost_per_anchor
     */

    float grid_cost_per_read =
        (avg_read_n / (float)range_kernel_config.anchor_per_block + 1) * F2;
    float cut_cost_per_read =
        (avg_read_n / (float)range_kernel_config.blockdim + 1) * (F3 + F4 / 2);
    *max_read_ = floor(avail_mem_per_stream /
                 (grid_cost_per_read + cut_cost_per_read + F1 * avg_read_n));
    *max_total_n_ = *max_read_ * avg_read_n;
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
                                max_range_grid, max_num_cut);
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
    g_current_dev_mem = &stream_setup.streams[0].dev_mem;
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
