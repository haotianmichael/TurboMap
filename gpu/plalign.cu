#include <algorithm>
#include "plalign.cuh"
#include "gasal_kernels.h"
#include "plmem.cuh"  // For deviceMemPtr
#include "plksw_kernel.cuh"
#include "plksw_shared_kernel.cuh"  // shared-memory long kernel (compile-time gated by USE_SHARED_LONG_KERNEL)
#include "plgrid_kernel.cuh"   // gridded traceback (compile-time gated by USE_GRIDDED_BT)
#include "pllog.h"             // PLOG_INFO macro (gated by PRINT)
// plksw2_kernel.cuh (CUDASW4-style column-parallel) no longer used; unified anti-diagonal kernel
#include <cub/device/device_scan.cuh>

/* Compile-time announcement: the user passes `make GRID=1` to enable.        */
/* If you don't see this message in the nvcc output, your build did not       */
/* recompile plalign.cu — try `make clean && make GRID=1`.                    */
#if USE_GRIDDED_BT
#pragma message ("plalign.cu: USE_GRIDDED_BT = 1  (gridded traceback path COMPILED IN)")
#else
#pragma message ("plalign.cu: USE_GRIDDED_BT = 0  (legacy bt_p path only)")
#endif

#ifndef USE_SHARED_LONG_KERNEL
#define USE_SHARED_LONG_KERNEL 0
#endif
#if USE_SHARED_LONG_KERNEL
#pragma message ("plalign.cu: USE_SHARED_LONG_KERNEL = 1  (shared-mem long kernel COMPILED IN)")
#else
#pragma message ("plalign.cu: USE_SHARED_LONG_KERNEL = 0")
#endif
// NVTX3 C API (nvtxRangePushA/nvtxRangePop) already available via cub/detail/nvtx.cuh


#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\

static align_config_t g_config = {
    .blocks = 28,
    .threads = 256,
    .slice_width = 3,
    .z_threshold = 400,
    .band_width = 751,
    .match_score = 2,
    .mismatch_score = 4,
    .gap_open = 4,
    .gap_extend = 2,
    .gap_open_long = 24,
    .gap_extend_long = 1
};

static bool g_subst_scores_uploaded = false;

static void ksw_gen_simple_mat(int m, int8_t *mat, int8_t a, int8_t b, int8_t sc_ambi)
{
	int i, j;
	a = a < 0? -a : a;
	b = b > 0? -b : b;
	sc_ambi = sc_ambi > 0? -sc_ambi : sc_ambi;
	for (i = 0; i < m - 1; ++i) {
		for (j = 0; j < m - 1; ++j)
			mat[i * m + j] = i == j? a : b;
		mat[i * m + m - 1] = sc_ambi;
	}
	for (j = 0; j < m; ++j)
		mat[(m - 1) * m + j] = sc_ambi;
}

void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOL, &(subst->gap_open_long), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtendL, &(subst->gap_extend_long), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = (subst->gap_open + subst->gap_extend);
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	// For AGAThA
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaSliceWidth, &(subst->slice_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaZThreshold, &(subst->z_threshold), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaBandWidth, &(subst->band_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}

// Set the device memory pointer for alignment operations
void gpu_align_copy_param() {

    // Upload substitution scores on first call
    if (!g_subst_scores_uploaded) {
        gasal_subst_scores subst;
        subst.match = g_config.match_score;
        subst.mismatch = g_config.mismatch_score;
        subst.gap_open = g_config.gap_open;
        subst.gap_extend = g_config.gap_extend;
        subst.gap_open_long = g_config.gap_open_long;
        subst.gap_extend_long = g_config.gap_extend_long;
        subst.slice_width = g_config.slice_width;
        subst.z_threshold = g_config.z_threshold;
        subst.band_width = g_config.band_width;
        gasal_copy_subst_scores(&subst);
        g_subst_scores_uploaded = true;
    }
}

// Kernel to initialize gasal_res_t structure on device
// This avoids cudaMemcpy host-to-device structure alignment issues
__global__ void init_gasal_res(gasal_res_t *res,
                                int32_t *aln_score, int32_t *query_batch_end, int32_t *target_batch_end,
                                int32_t *mqe, int32_t *mqe_t, int32_t *mte, int32_t *mte_q,
                                int32_t *zdropped) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        res->aln_score = aln_score;
        res->query_batch_end = query_batch_end;
        res->target_batch_end = target_batch_end;
        res->query_batch_start = NULL;
        res->target_batch_start = NULL;
        res->mqe = mqe;
        res->mqe_t = mqe_t;
        res->mte = mte;
        res->mte_q = mte_q;
        res->zdropped = zdropped;
        res->cigar = NULL;
        res->n_cigar_ops = NULL;
    }
}
// ============================================================
// P1: Compact CIGAR kernel
// Copies stride-layout CIGAR (d_cigar_buffer[task * max_len + i])
// into compact layout (d_compact_cigar[offsets[task] + i]).
// Launch: <<<batch_size, 256, 0, stream>>>
// ============================================================
__global__ void compact_cigar_kernel(
    const uint32_t * __restrict__ src,     // stride layout (task * max_len)
    uint32_t       * __restrict__ dst,     // compact layout (offsets[task])
    const uint32_t * __restrict__ offsets, // exclusive prefix-sum of lengths
    const int      * __restrict__ lengths, // n_cigar per task
    int              max_len               // stride width
) {
    int task_id = blockIdx.x;
    int n = lengths[task_id];
    if (n <= 0) return;
    uint32_t dst_base = offsets[task_id];
    uint32_t src_base = (uint32_t)task_id * (uint32_t)max_len;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dst[dst_base + i] = src[src_base + i];
}

// ============================================================
// P3/P2: gpu_fix_cigar device function (called by thread 0 only)
//
// Performs:
//   Pass 1 – left-align indels (shift indel earlier if sequence allows)
//   Pass 2 – consolidate runs like 5I6D7I → single I + single D
//   Pass 3a – squeeze zero-length ops + merge adjacent same-op
//   Pass 3b – SKIPPED (leading I/D removal adjusts coordinates; handled by CPU)
//
// Returns true if a leading I or D still exists (signals CPU fallback needed).
//
// Precision note: identical logic to CPU mm_fix_cigar; no precision difference.
// ============================================================
#define GPU_CIGAR_MATCH  0u
#define GPU_CIGAR_INS    1u
#define GPU_CIGAR_DEL    2u
#define GPU_CIGAR_N_SKIP 3u

__device__ static bool gpu_fix_cigar(
    uint32_t *cigar, int32_t *n_cigar_p,
    const uint8_t *qseq, const uint8_t *tseq
) {
    int32_t nc = *n_cigar_p;
    if (nc == 0) return false;
    if (nc == 1) {
        uint32_t op0 = cigar[0] & 0xfu;
        return (op0 == GPU_CIGAR_INS || op0 == GPU_CIGAR_DEL);
    }

    int toff = 0, qoff = 0, to_shrink = 0;

    // Pass 1: left-align indels
    for (int k = 0; k < nc; ++k) {
        uint32_t op  = cigar[k] & 0xfu;
        uint32_t len = cigar[k] >> 4;
        if (len == 0) to_shrink = 1;
        if (op == GPU_CIGAR_MATCH) {
            toff += (int)len; qoff += (int)len;
        } else if (op == GPU_CIGAR_INS || op == GPU_CIGAR_DEL) {
            if (k > 0 && k < nc - 1 &&
                (cigar[k-1] & 0xfu) == GPU_CIGAR_MATCH &&
                (cigar[k+1] & 0xfu) == GPU_CIGAR_MATCH)
            {
                int prev_len = (int)(cigar[k-1] >> 4);
                int l = 0;
                if (op == GPU_CIGAR_INS) {
                    for (l = 0; l < prev_len; ++l)
                        if (qseq[qoff - 1 - l] != qseq[qoff + (int)len - 1 - l]) break;
                } else {
                    for (l = 0; l < prev_len; ++l)
                        if (tseq[toff - 1 - l] != tseq[toff + (int)len - 1 - l]) break;
                }
                if (l > 0) {
                    cigar[k-1] -= (uint32_t)l << 4;
                    cigar[k+1] += (uint32_t)l << 4;
                    qoff -= l; toff -= l;
                }
                if (l == prev_len) to_shrink = 1;
            }
            if (op == GPU_CIGAR_INS) qoff += (int)len;
            else                     toff += (int)len;
        } else if (op == GPU_CIGAR_N_SKIP) {
            toff += (int)len;
        }
    }

    // Pass 2: consolidate I+D runs (e.g. 5I6D7I → 12I6D)
    for (int k = 0; k < nc - 2; ++k) {
        uint32_t op_k = cigar[k] & 0xfu;
        if (op_k > 0u && op_k + (cigar[k+1] & 0xfu) == 3u) {
            uint32_t s1 = 0, s2 = 0;
            int l;
            for (l = k; l < nc; ++l) {
                uint32_t op2 = cigar[l] & 0xfu;
                if (op2 == GPU_CIGAR_INS)      s1 += cigar[l] >> 4;
                else if (op2 == GPU_CIGAR_DEL) s2 += cigar[l] >> 4;
                else if (cigar[l] >> 4 == 0)   { /* zero-length: skip */ }
                else break;
            }
            if (s1 > 0 && s2 > 0 && l - k > 2) {
                cigar[k]   = (s1 << 4) | GPU_CIGAR_INS;
                cigar[k+1] = (s2 << 4) | GPU_CIGAR_DEL;
                for (int m = k + 2; m < l; ++m) cigar[m] &= 0xfu; // zero lengths
                to_shrink = 1;
            }
            k = l - 1; // outer loop will ++k → skip to l
        }
    }

    // Pass 3a: squeeze zero-length ops + merge adjacent same-op
    if (to_shrink) {
        // Squeeze zeros
        int32_t l = 0;
        for (int k = 0; k < nc; ++k)
            if (cigar[k] >> 4 != 0u) cigar[l++] = cigar[k];
        nc = l;
        // Merge adjacent same-op
        l = 0;
        for (int k = 0; k < nc; ++k) {
            if (k == nc - 1 || (cigar[k] & 0xfu) != (cigar[k+1] & 0xfu))
                cigar[l++] = cigar[k];
            else
                cigar[k+1] += cigar[k] >> 4 << 4; // accumulate length into next
        }
        nc = l;
    }

    *n_cigar_p = nc;
    // Pass 3b skipped: return whether leading I/D remains (CPU must handle it)
    return nc > 0 && ((cigar[0] & 0xfu) == GPU_CIGAR_INS || (cigar[0] & 0xfu) == GPU_CIGAR_DEL);
}

// ============================================================
// P2/P3: gpu_fix_cigar_and_stats kernel
//
// One block per task (blockIdx.x = batch slot / align_id).
// Thread 0 runs gpu_fix_cigar() then computes alignment stats.
// Other threads are unused (1 thread per block launch for simplicity).
//
// Writes: updated cigar_lengths, blen, mlen, n_ambi, dp_max, gpu_stats_valid.
//
// Precision notes vs CPU mm_update_extra:
//   - dp_max: uses integer log2 (31-__clz(1+len)) vs CPU float mg_log2 → ±1 in dp_max
//   - EQX mode: not handled here; caller checks gpu_stats_valid before using stats
//   - Leading I/D (pass 3b): sets gpu_stats_valid=0, CPU mm_update_extra handles it
// ============================================================
__global__ void gpu_fix_cigar_and_stats(
    uint32_t       *compact_cigar,         // in/out: compact CIGAR (writable)
    const uint32_t * __restrict__ offsets, // per-task start in compact_cigar
    int32_t        *cigar_lengths,         // in/out: n_cigar (updated by fix)
    const uint8_t  * __restrict__ d_query, // unpacked query sequences
    const uint8_t  * __restrict__ d_target,// unpacked target sequences
    const uint32_t * __restrict__ d_query_offsets,  // byte offsets into d_query
    const uint32_t * __restrict__ d_target_offsets, // byte offsets into d_target
    const int8_t   * __restrict__ d_mat,   // 5×5 scoring matrix
    int32_t         q_open,                // gap open penalty
    int32_t         e_ext,                 // gap extend penalty
    int             log_gap,               // 1 = use log-gap scoring
    int32_t        *d_blen,
    int32_t        *d_mlen,
    int32_t        *d_n_ambi,
    int32_t        *d_dp_max,
    int32_t        *d_gpu_stats_valid,
    int             batch_size
) {
    int task_id = blockIdx.x;
    if (task_id >= batch_size) return;

    // Only thread 0 does work (single-threaded sequential logic per task)
    if (threadIdx.x != 0) return;

    int32_t nc = cigar_lengths[task_id];
    if (nc <= 0) {
        d_blen[task_id]            = 0;
        d_mlen[task_id]            = 0;
        d_n_ambi[task_id]          = 0;
        d_dp_max[task_id]          = 0;
        d_gpu_stats_valid[task_id] = 1;
        return;
    }

    uint32_t cigar_base         = offsets[task_id];
    uint32_t *cigar             = compact_cigar + cigar_base;
    const uint8_t *qseq         = d_query  + d_query_offsets[task_id];
    const uint8_t *tseq         = d_target + d_target_offsets[task_id];

    // Pass 1/2/3a: fix CIGAR in-place
    bool has_leading_indel = gpu_fix_cigar(cigar, &nc, qseq, tseq);
    cigar_lengths[task_id] = nc;

    if (has_leading_indel) {
        // Pass 3b would adjust coordinates — CPU must handle this task
        d_gpu_stats_valid[task_id] = 0;
        return;
    }

    // Compute blen, mlen, n_ambi, dp_max via sequential scan
    int8_t mat[25];
    for (int i = 0; i < 25; i++) mat[i] = d_mat[i];

    int32_t blen = 0, mlen = 0, n_ambi = 0;
    double s = 0.0, max_s = 0.0;
    int toff = 0, qoff = 0;

    for (int k = 0; k < nc; ++k) {
        uint32_t op  = cigar[k] & 0xfu;
        uint32_t len = cigar[k] >> 4;

        if (op == GPU_CIGAR_MATCH) {
            int na = 0, nd = 0;
            for (uint32_t l = 0; l < len; ++l) {
                int cq = (int)qseq[qoff + l];
                int ct = (int)tseq[toff + l];
                if (ct > 3 || cq > 3) { na++; }
                else if (ct != cq)    { nd++; }
                s += (double)mat[ct * 5 + cq];
                if (s < 0.0)      s = 0.0;
                else if (s > max_s) max_s = s;
            }
            blen   += (int)len - na;
            mlen   += (int)len - (na + nd);
            n_ambi += na;
            toff   += (int)len;
            qoff   += (int)len;
        } else if (op == GPU_CIGAR_INS) {
            int na = 0;
            for (uint32_t l = 0; l < len; ++l)
                if (qseq[qoff + l] > 3) na++;
            blen   += (int)len - na;
            n_ambi += na;
            // log_gap: penalty = q + e * floor(log2(1+len))
            // Integer approximation: 31 - __clz(1u + len) = floor(log2(1+len))
            // Precision note: differs from CPU float mg_log2 at most by ±1 in integer part
            if (log_gap) s -= (double)q_open + (double)e_ext * (double)(31 - __clz(1u + len));
            else         s -= (double)q_open + (double)e_ext;
            if (s < 0.0) s = 0.0;
            qoff += (int)len;
        } else if (op == GPU_CIGAR_DEL) {
            int na = 0;
            for (uint32_t l = 0; l < len; ++l)
                if (tseq[toff + l] > 3) na++;
            blen   += (int)len - na;
            n_ambi += na;
            if (log_gap) s -= (double)q_open + (double)e_ext * (double)(31 - __clz(1u + len));
            else         s -= (double)q_open + (double)e_ext;
            if (s < 0.0) s = 0.0;
            toff += (int)len;
        } else if (op == GPU_CIGAR_N_SKIP) {
            toff += (int)len;
        }
    }

    d_blen[task_id]            = blen;
    d_mlen[task_id]            = mlen;
    d_n_ambi[task_id]          = n_ambi;
    d_dp_max[task_id]          = (int32_t)(max_s + 0.499);
    d_gpu_stats_valid[task_id] = 1;
}

extern "C" void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                            uint8_t *seq_buffer, uint32_t *cigar_buffer, int stream_id);
void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                            uint8_t *seq_buffer, uint32_t *cigar_buffer, int stream_id) {
    if (n_tasks <= 0) return;

    /* One-time announcement of compile-time gridded flag.  Lets the user      */
    /* verify whether the binary they're running has gridded traceback baked   */
    /* in.  Look for "[Info] Gridded traceback ..." in stderr.                  */
    static bool s_grid_announced = false;
    if (!s_grid_announced) {
        s_grid_announced = true;
#if USE_GRIDDED_BT
        PLOG_INFO(stderr, "[Info] Gridded traceback ENABLED (USE_GRIDDED_BT=1, G=%d)\n",
                GRID_BLOCK_SIZE);
#else
        PLOG_INFO(stderr, "[Info] Gridded traceback DISABLED (USE_GRIDDED_BT=0; legacy bt_p path)\n");
#endif
    }

    nvtxRangePushA("gpu_align_batch_execute");
    cudaSetDevice(0);
    gpu_align_copy_param();

    // Resolve per-stream device memory and CUDA stream (thread-safe, no globals)
    deviceMemPtr *dev_mem = gpu_get_dev_mem(stream_id);
    if (!dev_mem) {
        fprintf(stderr, "[ERROR] Invalid stream_id %d for alignment.\n", stream_id);
        return;
    }

    // Switch arena from chain phase to align phase
    plmem_phase_to_align(dev_mem);

    // ========== Three-Tier Batched Processing Setup ==========
    // Tasks are split into two runtime phases based on max sequence length:
    //   Tier 0 (short): max(qlen,tlen) ≤ short_task_max_len (1000 bp)
    //     - Fixed bt_p stride = max_antidiag_short × max_n_col_short (2MB/slot)
    //     - Up to n_concurrent_blocks concurrent slots
    //   Tier 1 (long): max(qlen,tlen) > short_task_max_len
    //     - Per-batch dynamic bt_p stride = actual_max_antidiag × actual_max_n_col
    //       so extra-long reads with large bandwidth (e.g. bw_long=7501) get
    //       a correctly-sized buffer with fewer concurrent slots
    //     - Separate bt_off buffer (max_antidiag_long stride) prevents the
    //       short-tier off arrays (stride 2000) from being overwritten
    //     - Concurrent slots = min(n_long_concurrent_slots,
    //                              bt_p_total / batch_max_backtrack_size)

    int kernel_blocks = 28;
    size_t short_task_max_len = dev_mem->short_task_max_len;      // 1000bp
    size_t short_batch_size = dev_mem->short_task_batch_size;      // 10,000

    // Classify tasks into short (tier-0) / long (tier-1) by max sequence length
    int *task_indices_short = (int*)malloc(n_tasks * sizeof(int));
    int *task_indices_long = (int*)malloc(n_tasks * sizeof(int));
    int n_short_tasks = 0;
    int n_long_tasks = 0;

    // Hard limits derived from GPU buffer allocation in setup_long_align_phase():
    //   ksw_temp_per_task  = f(max_align_query_len)   → max(qlen,tlen) must be ≤ limit
    //   bt_p antidiag buf  = 2 × max_align_query_len  → qlen+tlen must be ≤ 2×limit
    //   CIGAR buffer       = 2 × max_align_query_len  → qlen+tlen ≤ 2×limit-2 (guard)
    // Exceeding any of these causes silent memory corruption or wrong results.
    // Fatal-exit here so the problem is caught before any GPU work starts.
    const size_t gpu_max_one  = (size_t)dev_mem->max_align_query_len;        // 50000
    const size_t gpu_max_sum  = 2 * gpu_max_one - 2;  // 99998: tightest (CIGAR guard)

    for (int i = 0; i < n_tasks; i++) {
        int ql = tasks[i].qlen;
        int tl = tasks[i].tlen;
        size_t max_one = (ql > tl) ? (size_t)ql : (size_t)tl;
        size_t sum     = (size_t)ql + (size_t)tl;

        if (max_one > gpu_max_one || sum > gpu_max_sum) {
            fprintf(stderr,
                "[FATAL] GPU align task %d: qlen=%d tlen=%d exceeds GPU buffer limits "
                "(max single=%zu bp, max sum=%zu bp). "
                "GPU ksw_temp/bt_p/CIGAR buffers are sized for max %zu bp per sequence. "
                "Aborting to prevent memory corruption.\n",
                i, ql, tl, gpu_max_one, gpu_max_sum, gpu_max_one);
            exit(EXIT_FAILURE);
        }

        size_t max_seq_len = max_one;
        if (max_seq_len <= short_task_max_len) {
            task_indices_short[n_short_tasks++] = i;
        } else {
            task_indices_long[n_long_tasks++] = i;
        }
    }

    // Sort long tasks by DESCENDING estimated bt_stride = (qlen+tlen) × min(min(qlen,tlen), w+1).
    // Largest bt_stride first → first batch has the hardest tasks.
    // With dynamic per-batch batch_size = pool_cap × LATENCY_HIDE_FACTOR, the kernel is given
    // exactly as many tasks as it can process concurrently (× latency hide), so:
    //   - Large-bt_stride batches: small pool_cap → small batch (few tasks, runs fast)
    //   - Small-bt_stride batches: large pool_cap → large batch (many tasks, high utilisation)
    // Without this ordering, ONE large-bt_stride task would dominate ALL 5000+ tasks in the
    // same batch, collapsing pool_cap for everyone.
    std::sort(task_indices_long, task_indices_long + n_long_tasks,
        [&tasks](int a, int b) {
            int qa = tasks[a].qlen, ta = tasks[a].tlen, wa = tasks[a].w;
            int qb = tasks[b].qlen, tb = tasks[b].tlen, wb = tasks[b].w;
            int nca = (qa < ta) ? qa : ta;  if (wa >= 0 && wa + 1 < nca) nca = wa + 1;
            int ncb = (qb < tb) ? qb : tb;  if (wb >= 0 && wb + 1 < ncb) ncb = wb + 1;
            size_t sa = (size_t)(qa + ta) * (size_t)nca;  // estimated bt_stride (bytes / sizeof)
            size_t sb = (size_t)(qb + tb) * (size_t)ncb;
            return sa > sb;  // DESCENDING: largest bt_stride first
        });

    // Convenience tag used for all per-stream log lines in this function
    char stream_tag[32];
    snprintf(stream_tag, sizeof(stream_tag), "stream_%d", stream_id);

    PLOG_INFO(stderr, "[Info::%s] Alignment: %d short (max_len≤%zubp) + %d long (max_len>%zubp, dynamic bt)\n",
            stream_tag, n_short_tasks, short_task_max_len, n_long_tasks, short_task_max_len);

    int kernel_threads = 256;
    uint8_t *d_unpacked_query = dev_mem->d_align_unpacked_query;
    uint8_t *d_unpacked_target = dev_mem->d_align_unpacked_target;
    uint32_t *d_packed_query = dev_mem->d_align_packed_query;
    uint32_t *d_packed_target = dev_mem->d_align_packed_target;
    uint32_t *d_query_offsets = dev_mem->d_align_query_offsets;
    uint32_t *d_target_offsets = dev_mem->d_align_target_offsets;
    uint32_t *d_query_lens = dev_mem->d_align_query_lens;
    uint32_t *d_target_lens = dev_mem->d_align_target_lens;
    int32_t *d_flag = dev_mem->d_align_flag;
    int32_t *d_bw   = dev_mem->d_align_bw;
    void *d_ksw_temp_buffer = dev_mem->d_align_ksw_temp_buffer;
    size_t ksw_temp_per_task = dev_mem->align_ksw_temp_per_task;
    uint8_t *d_backtrack_p = dev_mem->d_align_backtrack_p;
    int *d_backtrack_off = dev_mem->d_align_backtrack_off;
    int *d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
    uint32_t *d_cigar_buffer = dev_mem->d_align_cigar_buffer;
    int *d_cigar_lengths = dev_mem->d_align_cigar_lengths;
    size_t max_backtrack_size = dev_mem->max_align_backtrack_size;
    size_t max_cigar_len = dev_mem->max_align_cigar_len;
    size_t max_query_len_limit = dev_mem->max_align_query_len;
    int8_t *d_mat = dev_mem->d_align_mat;
    void *device_res = dev_mem->d_align_device_res;
    int32_t *d_scores = dev_mem->d_align_scores;
    int32_t *d_query_ends = dev_mem->d_align_query_ends;
    int32_t *d_target_ends = dev_mem->d_align_target_ends;
    int32_t *d_mqe = dev_mem->d_align_mqe;
    int32_t *d_mqe_t = dev_mem->d_align_mqe_t;
    int32_t *d_mte = dev_mem->d_align_mte;
    int32_t *d_mte_q = dev_mem->d_align_mte_q;
    int32_t *d_zdropped = dev_mem->d_align_zdropped;
    int  *d_task_counter = dev_mem->d_align_task_counter;
    int   n_concurrent_blocks = dev_mem->n_align_concurrent_blocks;
    cudaStream_t align_stream = gpu_get_cudastream(stream_id);
    // P1/P2/P3 device buffers
    uint32_t *d_compact_cigar   = dev_mem->d_align_compact_cigar;
    uint32_t *d_compact_offsets = dev_mem->d_align_compact_offsets;
    void     *d_cub_tmp         = dev_mem->d_align_cub_tmp;
    size_t    cub_tmp_size      = dev_mem->align_cub_tmp_size;
    int32_t  *d_blen            = dev_mem->d_align_blen;
    int32_t  *d_mlen            = dev_mem->d_align_mlen;
    int32_t  *d_n_ambi          = dev_mem->d_align_n_ambi;
    int32_t  *d_dp_max          = dev_mem->d_align_dp_max;
    int32_t  *d_gpu_stats_valid = dev_mem->d_align_gpu_stats_valid;

    // Batch sizing
    size_t long_cigar_len = 2 * max_query_len_limit;  // 100000 for long tasks (50000bp)
    size_t short_batch_persistent = (size_t)dev_mem->max_align_tasks;
    size_t cigar_buf_total_tasks   = (size_t)dev_mem->max_align_tasks;
    size_t long_batch_persistent   = cigar_buf_total_tasks * max_cigar_len / long_cigar_len;

    int total_batches_short = n_short_tasks > 0 ? (int)((n_short_tasks + short_batch_persistent - 1) / short_batch_persistent) : 0;
    PLOG_INFO(stderr, "[Info::%s]   Tier-0 (short): %d batch(es) × up to %zu tasks  [fixed bt stride: %zu×%zu bytes]\n",
            stream_tag, total_batches_short, short_batch_persistent,
            2 * short_task_max_len, short_task_max_len + 1);
    // Tier-1 (long): batch count is determined dynamically per-batch (bt_stride-sorted tasks,
    // batch_size = pool_cap × LATENCY_HIDE_FACTOR).  Do not print an estimate here; the actual
    // count is shown at the end ("Alignment complete: X tasks in Y batches").
    PLOG_INFO(stderr, "[Info::%s]   Tier-1 (long):  %d tasks  [bt_stride-sorted; batch count determined dynamically]\n",
            stream_tag, n_long_tasks);

    size_t max_batch_size = (short_batch_persistent > long_batch_persistent) ?
                             short_batch_persistent : long_batch_persistent;

    // Use pre-allocated pinned host buffers from dev_mem (allocated once during init)
    uint32_t *h_compact_cigar   = dev_mem->h_align_compact_cigar;
    uint32_t *h_compact_offsets = dev_mem->h_align_compact_offsets;
    int      *h_cigar_lengths   = dev_mem->h_align_cigar_lengths;
    int32_t  *h_blen            = dev_mem->h_align_blen;
    int32_t  *h_mlen            = dev_mem->h_align_mlen;
    int32_t  *h_n_ambi          = dev_mem->h_align_n_ambi;
    int32_t  *h_dp_max          = dev_mem->h_align_dp_max;
    int32_t  *h_gpu_stats_valid = dev_mem->h_align_gpu_stats_valid;
    int32_t  *h_scores          = dev_mem->h_align_scores;
    int32_t  *h_query_ends      = dev_mem->h_align_query_ends;
    int32_t  *h_target_ends     = dev_mem->h_align_target_ends;
    int32_t  *h_mqe             = dev_mem->h_align_mqe;
    int32_t  *h_mqe_t           = dev_mem->h_align_mqe_t;
    int32_t  *h_mte             = dev_mem->h_align_mte;
    int32_t  *h_mte_q           = dev_mem->h_align_mte_q;
    int32_t  *h_zdropped        = dev_mem->h_align_zdropped;
    uint32_t *h_query_offsets   = dev_mem->h_align_query_offsets;
    uint32_t *h_target_offsets  = dev_mem->h_align_target_offsets;
    uint32_t *h_query_lens      = dev_mem->h_align_query_lens;
    uint32_t *h_target_lens     = dev_mem->h_align_target_lens;
    int32_t  *h_flag            = dev_mem->h_align_flag;
    int32_t  *h_bw              = dev_mem->h_align_bw;
    // h_task_to_align_id was always identity (i→i) — use i directly instead.

    int8_t h_scoring_matrix[25];
    ksw_gen_simple_mat(5, h_scoring_matrix, opt->a, opt->b, opt->sc_ambi);
    cudaError_t err;
    CHECKCUDAERROR(cudaMemcpyAsync(d_mat, h_scoring_matrix, 25 * sizeof(int8_t),
                                   cudaMemcpyHostToDevice, align_stream));

    // Initialize device result structure directly on device
    // Using a kernel avoids host-device structure alignment issues with cudaMemcpy
    // Performance impact: ~5-10 microseconds (negligible compared to alignment kernel runtime)
    init_gasal_res<<<1, 1, 0, align_stream>>>((gasal_res_t*)device_res, d_scores, d_query_ends, d_target_ends,
                              d_mqe, d_mqe_t, d_mte, d_mte_q, d_zdropped);
    CHECKCUDAERROR(cudaGetLastError());

    // ========== THREE-TIER BATCHED PROCESSING LOOP ==========
    // Phase 0 = short tasks (tier-0), phase 1 = long tasks (tier-1)
    // Tier-1 further adapts bt buffer size per batch (dynamic tier within the loop)

    int batch_num = 0;
    int total_tasks_processed = 0;

    for (int phase = 0; phase < 2; phase++) {
        // Select phase-specific parameters
        int *current_task_indices = (phase == 0) ? task_indices_short : task_indices_long;
        int n_tasks_in_phase = (phase == 0) ? n_short_tasks : n_long_tasks;
        // Use persistent-kernel batch sizes (CIGAR-buffer limited, not backtrack-limited)
        size_t current_batch_size = (phase == 0) ? short_batch_persistent : long_batch_persistent;
        const char *phase_name = (phase == 0) ? "Tier-0 Short" : "Tier-1 Long";

        // Dynamic backtrack buffer sizing based on phase
        // Short phase stride caps n_col at (short_task_max_len + 1) to match plmem alloc, allowing
        // any per-task bandwidth (n_col = min(qlen, tlen, w+1) is bounded by short_task_max_len).
        // Long phase: n_col and antidiag are computed PER BATCH below to handle varying bandwidths.
        size_t current_max_antidiag = 2 * short_task_max_len;          // used only for phase 0
        size_t current_max_n_col    = short_task_max_len + 1;          // used only for phase 0
        size_t current_max_backtrack_size = current_max_antidiag * current_max_n_col;  // phase 0
        size_t current_max_cigar_len = (phase == 0) ? (2 * short_task_max_len) : (2 * dev_mem->max_align_query_len);

        if (n_tasks_in_phase == 0) continue;  // Skip empty phase

        // ── Long-align arena transition ────────────────────────────────────────
        // When we enter phase 1 (long tasks), the short-align arena layout
        // wastes ~10 GB on large CIGAR/seq buffers we no longer need.
        // Transition to long-align layout: same 15 GB physical block, reset and
        // re-allocated so bt_p gets ~13 GB instead of ~5 GB.
        // We sync the stream first to ensure all short-phase work is flushed.
        if (phase == 1) {
            cudaStreamSynchronize(align_stream);

            plmem_phase_to_long_align(dev_mem);

            // ── Refresh all local GPU pointers from dev_mem ──────────────────
            d_unpacked_query  = dev_mem->d_align_unpacked_query;
            d_unpacked_target = dev_mem->d_align_unpacked_target;
            d_packed_query    = dev_mem->d_align_packed_query;
            d_packed_target   = dev_mem->d_align_packed_target;
            d_query_offsets   = dev_mem->d_align_query_offsets;
            d_target_offsets  = dev_mem->d_align_target_offsets;
            d_query_lens      = dev_mem->d_align_query_lens;
            d_target_lens     = dev_mem->d_align_target_lens;
            d_flag            = dev_mem->d_align_flag;
            d_bw              = dev_mem->d_align_bw;
            d_ksw_temp_buffer = dev_mem->d_align_ksw_temp_buffer;
            ksw_temp_per_task = dev_mem->align_ksw_temp_per_task;
            d_backtrack_p     = dev_mem->d_align_backtrack_p;  // now ~13 GB pool
            // d_backtrack_off / d_backtrack_off_end: stubs, not used for long tasks
            d_backtrack_off     = dev_mem->d_align_backtrack_off;
            d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
            d_cigar_buffer    = dev_mem->d_align_cigar_buffer;
            d_cigar_lengths   = dev_mem->d_align_cigar_lengths;
            max_cigar_len     = dev_mem->max_align_cigar_len;  // now 100,000
            d_mat             = dev_mem->d_align_mat;
            device_res        = dev_mem->d_align_device_res;
            d_scores          = dev_mem->d_align_scores;
            d_query_ends      = dev_mem->d_align_query_ends;
            d_target_ends     = dev_mem->d_align_target_ends;
            d_mqe             = dev_mem->d_align_mqe;
            d_mqe_t           = dev_mem->d_align_mqe_t;
            d_mte             = dev_mem->d_align_mte;
            d_mte_q           = dev_mem->d_align_mte_q;
            d_zdropped        = dev_mem->d_align_zdropped;
            d_task_counter    = dev_mem->d_align_task_counter;
            n_concurrent_blocks = dev_mem->n_align_concurrent_blocks;  // now n_long_cap (dynamic)
            d_compact_cigar   = dev_mem->d_align_compact_cigar;
            d_compact_offsets = dev_mem->d_align_compact_offsets;
            d_cub_tmp         = dev_mem->d_align_cub_tmp;
            cub_tmp_size      = dev_mem->align_cub_tmp_size;
            d_blen            = dev_mem->d_align_blen;
            d_mlen            = dev_mem->d_align_mlen;
            d_n_ambi          = dev_mem->d_align_n_ambi;
            d_dp_max          = dev_mem->d_align_dp_max;
            d_gpu_stats_valid = dev_mem->d_align_gpu_stats_valid;

            // Re-upload scoring matrix (d_align_mat is at a new arena address)
            int8_t h_scoring_matrix2[25];
            ksw_gen_simple_mat(5, h_scoring_matrix2, opt->a, opt->b, opt->sc_ambi);
            cudaMemcpyAsync(d_mat, h_scoring_matrix2, 25 * sizeof(int8_t),
                            cudaMemcpyHostToDevice, align_stream);

            // Re-init gasal_res (d_align_device_res is at a new arena address)
            init_gasal_res<<<1, 1, 0, align_stream>>>(
                (gasal_res_t*)device_res,
                d_scores, d_query_ends, d_target_ends,
                d_mqe, d_mqe_t, d_mte, d_mte_q, d_zdropped);

            // long_task_batch_size is set by setup_long_align_phase() via
            // compute_long_batch_size(): it is the CIGAR-buffer hard cap on batch size.
            // Actual per-batch batch_size is further constrained dynamically inside
            // the while loop below (pool_cap × LATENCY_HIDE_FACTOR).
            long_batch_persistent = (size_t)dev_mem->long_task_batch_size;
            current_batch_size    = long_batch_persistent;  // upper bound; overridden per-batch
            PLOG_INFO(stderr, "[Info::%s]   Tier-1 (long): CIGAR cap=%zu  bt_p=%.2f GB  slots=%d\n",
                    stream_tag, long_batch_persistent,
                    dev_mem->long_bt_p_pool_bytes / (1024.0*1024.0*1024.0),
                    dev_mem->n_long_concurrent_slots);
        }

        PLOG_INFO(stderr, "[Info::%s] === %s: %d tasks ===\n", stream_tag, phase_name, n_tasks_in_phase);

        // Problem: result buffers are 120,000 elements but we only clear phase_batch_size
        // This causes long phase to read stale data from short phase!
        // Solution: Clear ENTIRE physical buffer allocation at phase start

        // Clear ksw_temp_buffer (allocated for short_batch_size tasks)
        size_t ksw_temp_buffer_size = short_batch_size * ksw_temp_per_task;
        //cudaMemset(d_ksw_temp_buffer, 0, ksw_temp_buffer_size);

        // Both short (10000×2000) and long (200×100000) phases use same 15GB buffer with different strides
        // Must clear entire 15GB to prevent cross-phase contamination
        size_t full_backtrack_p_size = short_batch_size * (2 * short_task_max_len) * 752;  // Always 15GB
        size_t full_backtrack_off_size = short_batch_size * (2 * short_task_max_len) * sizeof(int);
        /*cudaMemset(d_backtrack_p, 0, full_backtrack_p_size);
        cudaMemset(d_backtrack_off, 0, full_backtrack_off_size);
        cudaMemset(d_backtrack_off_end, 0, full_backtrack_off_size);
        cudaMemset(d_backtrack_n_col, 0, short_batch_size * sizeof(int));
        cudaMemset(d_cigar_buffer, 0, short_batch_size * (2 * short_task_max_len) * sizeof(uint32_t));
        cudaMemset(d_cigar_lengths, 0, short_batch_size * sizeof(int));
        cudaMemset(d_ez_array, 0, short_batch_size * sizeof(ksw_extz_t));*.

        // These are shared across all phases and must be completely cleared
        size_t max_align_tasks = 120000;  // From plmem.cu
        /*cudaMemset(d_scores, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_query_ends, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_target_ends, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mqe, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mqe_t, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mte, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mte_q, 0, max_align_tasks * sizeof(int32_t));*/

        int tasks_processed_in_phase = 0;
        int phase_batch_num = 0;
        int total_phase_batches = (n_tasks_in_phase > 0) ?
            (int)((n_tasks_in_phase + (int)current_batch_size - 1) / (int)current_batch_size) : 0;

        // Repeat-suppression state for long-phase batch logging.
        // When consecutive batches have identical (bt_stride, slots, batch_size),
        // we print the first, accumulate the rest, and flush with "×N" when params change.
        size_t rep_bt_stride  = 0;
        int    rep_slots      = 0;
        int    rep_batch_size = 0;
        int    rep_count      = 0;   // number of suppressed identical batches after the first
        int    rep_done_start = 0;   // tasks_processed_in_phase when the group started

        while (tasks_processed_in_phase < n_tasks_in_phase) {
            int batch_start = tasks_processed_in_phase;

            // ── Dynamic batch sizing for long-task phase ─────────────────────────────
            // Tasks are sorted DESCENDING by estimated bt_stride = (qlen+tlen)×n_col.
            // The first task in this batch has the largest bt_stride → lowest pool_cap.
            // We want batch_size = pool_cap × LATENCY_HIDE_FACTOR so the GPU has
            // enough concurrent warps to hide global-memory latency without wasting
            // CIGAR buffer on tasks that won't improve concurrency further.
            //
            // pool_cap = bt_p_total / bt_stride_of_first_task (floor)
            // LATENCY_HIDE_FACTOR = 3: empirical — 3× physical occupancy is sufficient
            //   for L2-bandwidth-bound KSW bt_p accesses on A100 (tested range: 2–4×).
            if (phase == 1) {
                int tidx0  = current_task_indices[batch_start];
                int ql0    = tasks[tidx0].qlen;
                int tl0    = tasks[tidx0].tlen;
                int w0     = tasks[tidx0].w;
                int nc0    = (ql0 < tl0) ? ql0 : tl0;
                if (w0 >= 0 && w0 + 1 < nc0) nc0 = w0 + 1;
                size_t bt_stride0 = (size_t)(ql0 + tl0) * (size_t)nc0;

                // bt_p bytes available to the long phase (set at arena transition).
                // long_bt_p_pool_bytes holds either the dedicated cudaMalloc pool size
                // (if one was allocated in plmem_malloc_device_mem) or the arena bt_p
                // size (set by setup_long_align_phase when no dedicated pool exists).
                // Fallback: n_long_concurrent_slots × TYPICAL_BT_STRIDE (safe floor).
                const size_t TYPICAL_BT_STRIDE_FALLBACK = (size_t)6 << 20;
                size_t bt_p_avail = (dev_mem->long_bt_p_pool_bytes > 0)
                                    ? dev_mem->long_bt_p_pool_bytes
                                    : (size_t)dev_mem->n_long_concurrent_slots *
                                      TYPICAL_BT_STRIDE_FALLBACK;

                const size_t LATENCY_HIDE_FACTOR = 3;
                size_t pool_cap0 = (bt_stride0 > 0) ? (bt_p_avail / bt_stride0) : (size_t)256;
                if (pool_cap0 < 1) pool_cap0 = 1;

                size_t dyn_batch = pool_cap0 * LATENCY_HIDE_FACTOR;
                if (dyn_batch < 64)                          dyn_batch = 64;  // floor
                if (dyn_batch > (size_t)long_batch_persistent) dyn_batch = (size_t)long_batch_persistent;  // CIGAR cap

                current_batch_size = dyn_batch;
            }

            int batch_size = (tasks_processed_in_phase + (int)current_batch_size <= n_tasks_in_phase) ?
                             (int)current_batch_size : (n_tasks_in_phase - tasks_processed_in_phase);
            batch_num++;
            phase_batch_num++;

            /* ─── NVTX outer range for the entire batch ───
             * Label format: "<P>[B<N>|n=<size>]"  where P = S/L (short/long).
             * Slot count is appended later in the kernel-launch sub-range.
             * Open the batch range here; closed at the end of map_results.    */
            char nvtx_batch_label[64];
            snprintf(nvtx_batch_label, sizeof(nvtx_batch_label),
                     "%c[B%d|n=%d]", (phase == 0) ? 'S' : 'L',
                     phase_batch_num, batch_size);
            nvtxRangePushA(nvtx_batch_label);

            // Clear ONLY result buffers before each batch to prevent reading stale data
            // If a task fails (zdropped etc), kernel may not write to result buffers
            // Without clearing, we'd read previous batch's stale results
            // Note: backtrack buffers don't need clearing as they're fully written by kernel
            /*cudaMemset(d_scores, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_query_ends, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_target_ends, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mqe, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mqe_t, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mte, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mte_q, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_cigar_lengths, 0, batch_size * sizeof(int));*/

            // Pinned staging buffers pre-allocated once in plmem.cu — no per-batch calloc.
            uint8_t *h_unpacked_query  = dev_mem->h_align_unpacked_query;
            uint8_t *h_unpacked_target = dev_mem->h_align_unpacked_target;

            // Single pass: compute offsets, fill metadata, copy sequences, pad with N.
            size_t total_query_bytes = 0, total_target_bytes = 0;
            uint32_t max_query_len = 0;
            const uint8_t N_BASE = 4;

            nvtxRangePushA("host_prep");
            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];
                int qlen = tasks[task_idx].qlen;
                int tlen = tasks[task_idx].tlen;

                // Align to 8-byte boundary for AGATHA
                size_t qlen_aligned = ((qlen + 7) / 8) * 8;
                size_t tlen_aligned = ((tlen + 7) / 8) * 8;

                h_query_offsets[i]  = (uint32_t)total_query_bytes;
                h_target_offsets[i] = (uint32_t)total_target_bytes;
                h_query_lens[i]     = (uint32_t)qlen;
                h_target_lens[i]    = (uint32_t)tlen;
                h_flag[i]           = tasks[task_idx].flag;
                h_bw[i]             = tasks[task_idx].w;

                // Copy sequences into pinned staging buffer
                memcpy(h_unpacked_query  + total_query_bytes,
                       seq_buffer + tasks[task_idx].qseq_offset, qlen);
                memcpy(h_unpacked_target + total_target_bytes,
                       seq_buffer + tasks[task_idx].tseq_offset, tlen);

                // Pad tail with N (0x0F) up to 8-byte alignment
                for (int j = qlen; j < (int)qlen_aligned; j++)
                    h_unpacked_query[total_query_bytes + j] = N_BASE;
                for (int j = tlen; j < (int)tlen_aligned; j++)
                    h_unpacked_target[total_target_bytes + j] = N_BASE;

                total_query_bytes  += qlen_aligned;
                total_target_bytes += tlen_aligned;

                if ((uint32_t)qlen > max_query_len) max_query_len = qlen;
            }
            nvtxRangePop(); // host_prep

            // Copy batch data to GPU (async on align_stream so chain stream stays free)
            nvtxRangePushA("h2d");
            cudaMemcpyAsync(d_unpacked_query, h_unpacked_query,
                            total_query_bytes, cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_unpacked_target, h_unpacked_target,
                            total_target_bytes, cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_query_offsets, h_query_offsets,
                            batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_target_offsets, h_target_offsets,
                            batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_query_lens, h_query_lens,
                            batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_target_lens, h_target_lens,
                            batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_flag, h_flag,
                            batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, align_stream);
            cudaMemcpyAsync(d_bw, h_bw,
                            batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, align_stream);
            nvtxRangePop(); // h2d

            // Launch packing kernel
            nvtxRangePushA("pack_launch");
            int query_tasks_per_thread = (int)ceil((double)total_query_bytes /
                                                  (8 * kernel_threads * kernel_blocks));
            int target_tasks_per_thread = (int)ceil((double)total_target_bytes /
                                                   (8 * kernel_threads * kernel_blocks));

            gasal_pack_kernel<<<kernel_blocks, kernel_threads, 0, align_stream>>>(
                (uint32_t*)d_unpacked_query,
                (uint32_t*)d_unpacked_target,
                d_packed_query,
                d_packed_target,
                query_tasks_per_thread,
                target_tasks_per_thread,
                total_query_bytes / 4,
                total_target_bytes / 4
            );
            nvtxRangePop(); // pack_launch

            // ===== Persistent KSW Kernel (unified anti-diagonal, int32 dual-affine) =====
            // Both phases use the same fused persistent kernel with direct int32 H/E/F/E2/F2.
            // Anti-diagonal backtrack with banding for all task sizes.
            //
            // THREE-TIER buffer management:
            //   Tier-0 (short): bt_p stride fixed = max_antidiag_short × max_n_col_short
            //                   bt_off stride = max_antidiag_short (2000) — matches allocation
            //   Tier-1 (long):  bt_p from dedicated long pool (d_align_backtrack_p_long),
            //                   bt_p stride = actual_max_antidiag × actual_max_n_col (per batch)
            //                   bt_off stride = max_antidiag_long (100,000) — matches long alloc
            //                   Concurrent slots = min(n_long_slots, long_pool / bt_p_stride)

            // Short phase: bt_p pool is the shared arena allocation.
            size_t bt_p_total_bytes = (size_t)n_concurrent_blocks *
                                      dev_mem->max_align_backtrack_size;

            // Per-batch backtrack parameters (overridden for long phase below)
            size_t batch_max_backtrack_size = current_max_backtrack_size;
            size_t batch_max_antidiag       = current_max_antidiag;
            int   *d_bt_off_batch           = d_backtrack_off;
            int   *d_bt_off_end_batch       = d_backtrack_off_end;
            int    max_slots_cap            = n_concurrent_blocks;
            // bt_p pointer and total pool bytes for this batch.
            // Three possible sources (in priority order):
            //   1. Long-align arena (after plmem_phase_to_long_align):
            //        d_backtrack_p was refreshed to the ~13 GB pool;
            //        long_bt_p_pool_bytes reflects its size.
            //   2. Dedicated cudaMalloc pool (d_align_backtrack_p_long, single-stream):
            //        use that pointer and its size from long_bt_p_pool_bytes.
            //   3. Shared arena bt_p (fallback): d_backtrack_p points to ~5 GB pool.
            uint8_t *d_bt_p_batch = d_backtrack_p;  // default covers cases 1 and 3

            if (phase == 1) {
                // Cases 1 & 2 both set long_bt_p_pool_bytes > 0.
                // Case 1: d_backtrack_p already refreshed to the long arena pool.
                // Case 2: dedicated pool pointer differs from d_backtrack_p.
                if (dev_mem->d_align_backtrack_p_long != nullptr) {
                    // Dedicated cudaMalloc pool (case 2, single-stream).
                    d_bt_p_batch     = dev_mem->d_align_backtrack_p_long;
                    bt_p_total_bytes = dev_mem->long_bt_p_pool_bytes;
                } else if (dev_mem->long_bt_p_pool_bytes > 0) {
                    // Long-align arena (case 1): d_backtrack_p already refreshed.
                    bt_p_total_bytes = dev_mem->long_bt_p_pool_bytes;
                }
                // Case 3 (no pool): bt_p_total_bytes stays as computed above (~5 GB).
            }

            // Diagnostic variables for long-batch logging (populated in the phase==1 block below)
            int    diag_actual_max_qlen  = 0;
            int    diag_actual_max_tlen  = 0;
            int    diag_n_exceed         = 0;
            size_t diag_actual_antidiag  = 0;
            size_t diag_actual_n_col     = 0;
            int    batch_max_tlen        = 0;  /* used by shared-mem long kernel dispatch */

            if (phase == 1) nvtxRangePushA("long_setup");
            if (phase == 1) {
                // Scan this batch to find actual maximum n_col and antidiag needed.
                // n_col = min(min(qlen, tlen), w+1)  (mirrors kernel line 244-245)
                size_t actual_max_n_col    = 1;
                size_t actual_max_antidiag = 1;
                for (int i = 0; i < batch_size; i++) {
                    int tidx = current_task_indices[batch_start + i];
                    int ql   = tasks[tidx].qlen;
                    int tl   = tasks[tidx].tlen;
                    int w    = tasks[tidx].w;
                    int nc   = (ql < tl) ? ql : tl;
                    if (w >= 0 && w + 1 < nc) nc = w + 1;
                    size_t ad = (size_t)ql + (size_t)tl;
                    if ((size_t)nc > actual_max_n_col)    actual_max_n_col    = nc;
                    if (ad > actual_max_antidiag) actual_max_antidiag = ad;
                    if (ql > diag_actual_max_qlen) diag_actual_max_qlen = ql;
                    if (tl > diag_actual_max_tlen) diag_actual_max_tlen = tl;
                    if (tl > batch_max_tlen)       batch_max_tlen       = tl;
                    if (ql > (int)dev_mem->max_align_query_len ||
                        tl > (int)dev_mem->max_align_query_len)
                        diag_n_exceed++;
                }
                // The bt_p slot stride = actual_max_antidiag × actual_max_n_col.
                // We use the REAL per-batch values so the bt_p pool is redistributed
                // correctly among concurrent slots — no wasted space, no overflow.
                size_t max_antidiag_long = 2 * (size_t)dev_mem->max_align_query_len;
                if (actual_max_antidiag > max_antidiag_long)
                    actual_max_antidiag = max_antidiag_long;

                batch_max_backtrack_size = actual_max_antidiag * actual_max_n_col;
                diag_actual_antidiag = actual_max_antidiag;
                diag_actual_n_col    = actual_max_n_col;

                // IMPORTANT: batch_max_antidiag passed to the kernel is used as the
                // per-slot STRIDE in backtrack_off (off = backtrack_off + slot_id * max_antidiag).
                // This stride MUST match the allocation stride of d_align_backtrack_off_long,
                // which was allocated as n_long_concurrent_slots × max_antidiag_long × sizeof(int).
                // Therefore we always pass max_antidiag_long (not actual_max_antidiag) here.
                batch_max_antidiag = max_antidiag_long;

                // Use long-task bt_off buffers (stride = max_antidiag_long per slot)
                d_bt_off_batch     = dev_mem->d_align_backtrack_off_long;
                d_bt_off_end_batch = dev_mem->d_align_backtrack_off_end_long;
                // Concurrent slots bounded by the long-tier bt_off allocation
                max_slots_cap = dev_mem->n_long_concurrent_slots;

                // bt_p pool and total bytes were already selected above before the
                // scan loop; no further override needed here.
            }
            if (phase == 1) nvtxRangePop(); // long_setup

            // Per-batch path marker for the post-batch log (set by the dispatch).
            char batch_path_tag = 'L';   // L = legacy bt_p, G = gridded

            size_t max_slots_this_phase = (batch_max_backtrack_size > 0)
                ? (bt_p_total_bytes / batch_max_backtrack_size)
                : (size_t)n_concurrent_blocks;

            int phase_concurrent_slots = max_slots_cap;
            if ((size_t)phase_concurrent_slots > max_slots_this_phase)
                phase_concurrent_slots = (int)max_slots_this_phase;
            if (phase_concurrent_slots > batch_size)
                phase_concurrent_slots = batch_size;
            if (phase_concurrent_slots < 1) phase_concurrent_slots = 1;

            // Pre-batch long-phase log removed; info merged into post-batch line below.

            /* NVTX: kernel-launch range with slot count in label so a specific  */
            /* batch (e.g. slots=15 super-long) can be located in nsys easily.   */
            char nvtx_kernel_label[64];
            snprintf(nvtx_kernel_label, sizeof(nvtx_kernel_label),
                     "kernel:s=%d", phase_concurrent_slots);
            nvtxRangePushA(nvtx_kernel_label);

            // Reset atomic task counter to 0 before this batch (on align_stream for ordering)
            cudaMemsetAsync(d_task_counter, 0, sizeof(int), align_stream);

            {
                int parallel_threads = 32;   // one warp per block

#if USE_SHARED_LONG_KERNEL
                /* ─────────── Shared-memory long kernel dispatch ──────────────────── */
                /* Triggers when:                                                     */
                /*   - long phase (phase == 1)                                        */
                /*   - all tasks fit shared limit (max_tlen ≤ SHARED_KERNEL_TLEN_LIMIT)*/
                /*   - legacy bt_p budget is the bottleneck (phase_concurrent_slots   */
                /*     < 100), so we WON'T hurt regular-long batches that already     */
                /*     have many slots.                                               */
                /* Per-block shared mem = 6 × (max_tlen+1) bytes for delta arrays.    */
                bool use_shared = (phase == 1 &&
                                   batch_max_tlen > 0 &&
                                   batch_max_tlen <= SHARED_KERNEL_TLEN_LIMIT &&
                                   phase_concurrent_slots < 100);
                if (use_shared) {
                    /* One-shot driver opt-in for >48 KB shared per block.          */
                    static bool s_shared_attr_set = false;
                    if (!s_shared_attr_set) {
                        cudaFuncSetAttribute(ksw_long_shared_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            SHARED_KERNEL_MAX_BYTES);
                        s_shared_attr_set = true;
                    }
                    size_t shared_bytes =
                        (size_t)6 * (size_t)(batch_max_tlen + 1);

                    batch_path_tag = 'H';   /* H = sHared */

                    ksw_long_shared_kernel<<<phase_concurrent_slots, parallel_threads,
                                             shared_bytes, align_stream>>>(
                        d_task_counter,
                        d_packed_query,
                        d_packed_target,
                        d_query_lens,
                        d_target_lens,
                        d_query_offsets,
                        d_target_offsets,
                        (gasal_res_t*)device_res,
                        d_mat,
                        d_bt_p_batch,
                        d_bt_off_batch,
                        d_bt_off_end_batch,
                        (int)batch_max_backtrack_size,
                        (int)batch_max_antidiag,
                        d_ksw_temp_buffer,
                        d_flag,
                        d_bw,
                        ksw_temp_per_task,
                        batch_size,
                        5,
                        opt->zdrop,
                        opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len,
                        batch_max_tlen
                    );
                } else
#endif
#if USE_GRIDDED_BT
                /* ─────────── Gridded traceback dispatch (long phase only) ───────── */
                /* dblock + off arrays are PER-TASK (correctness — slot→task         */
                /* binding differs between forward and backtrack kernel launches).   */
                /* Pool layout for this batch:                                        */
                /*   [dblock: batch_size × dblock_per_task] [scratch: n_slots × spp] */
                /* n_slots is then determined by remaining pool for scratch, capped   */
                /* by batch_size and max_slots_cap.                                   */
                bool use_grid = false;
                size_t grid_dblock_per_task = 0, grid_scratch_per_slot = 0;
                size_t grid_dblock_total_bytes = 0;
                int grid_slots = 0;
                if (phase == 1 && cigar_buffer) {
                    const size_t G = (size_t)GRID_BLOCK_SIZE;
                    size_t batch_n_col = (size_t)diag_actual_n_col;
                    if (batch_n_col == 0) batch_n_col = 1;
                    size_t n_ckpt = ((size_t)batch_max_antidiag + G - 1) / G;
                    grid_dblock_per_task  =
                        n_ckpt * (size_t)GRID_NUM_DELTA_ARRAYS * batch_n_col;
                    grid_scratch_per_slot = G * batch_n_col;
                    grid_dblock_total_bytes = (size_t)batch_size * grid_dblock_per_task;

                    if (grid_dblock_total_bytes < (size_t)bt_p_total_bytes) {
                        size_t scratch_avail =
                            (size_t)bt_p_total_bytes - grid_dblock_total_bytes;
                        size_t n_slots_from_scratch =
                            (grid_scratch_per_slot > 0)
                            ? (scratch_avail / grid_scratch_per_slot) : 1;
                        grid_slots = (int)n_slots_from_scratch;
                        if (grid_slots > max_slots_cap) grid_slots = max_slots_cap;
                        if (grid_slots > batch_size)    grid_slots = batch_size;
                        if (grid_slots < 1)             grid_slots = 1;

                        /* Switch to gridded only when slot gain ≥ 2× to cover the   */
                        /* 2× compute overhead of replay.                            */
                        int legacy_slots = phase_concurrent_slots;
                        use_grid = (grid_slots >= 2 * legacy_slots);
                    }
                }
                if (use_grid) {
                    batch_path_tag = 'G';
                    size_t batch_n_col = (size_t)diag_actual_n_col;
                    if (batch_n_col == 0) batch_n_col = 1;
                    phase_concurrent_slots = grid_slots;   /* override for logging too */

                    /* Pool layout: [dblock: batch_size × dblock_per_task]           */
                    /*              [scratch: grid_slots × scratch_per_slot]         */
                    int8_t  *grid_dblock_pool  = (int8_t*)d_bt_p_batch;
                    uint8_t *grid_scratch_pool = (uint8_t*)d_bt_p_batch
                        + grid_dblock_total_bytes;

                    ksw_gridded_forward_kernel<<<grid_slots, parallel_threads,
                                                 0, align_stream>>>(
                        d_task_counter,
                        d_packed_query, d_packed_target,
                        d_query_lens,   d_target_lens,
                        d_query_offsets, d_target_offsets,
                        (gasal_res_t*)device_res, d_mat,
                        grid_dblock_pool, grid_dblock_per_task,
                        d_bt_off_batch, d_bt_off_end_batch,
                        (int)batch_max_antidiag, (int)batch_n_col,
                        d_ksw_temp_buffer, d_flag, d_bw,
                        ksw_temp_per_task, batch_size,
                        5, opt->zdrop, opt->end_bonus,
                        d_cigar_lengths
                    );

                    /* Reset task counter for the backtrack pass.                   */
                    cudaMemsetAsync(d_task_counter, 0, sizeof(int), align_stream);

                    ksw_gridded_backtrack_kernel<<<grid_slots, parallel_threads,
                                                   0, align_stream>>>(
                        d_task_counter,
                        d_packed_query, d_packed_target,
                        d_query_lens,   d_target_lens,
                        d_query_offsets, d_target_offsets,
                        (gasal_res_t*)device_res, d_mat,
                        grid_dblock_pool, grid_dblock_per_task,
                        grid_scratch_pool, grid_scratch_per_slot,
                        d_bt_off_batch, d_bt_off_end_batch,
                        (int)batch_max_antidiag, (int)batch_n_col,
                        d_ksw_temp_buffer, d_flag, d_bw,
                        ksw_temp_per_task, batch_size, 5,
                        d_cigar_buffer, d_cigar_lengths,
                        (int)current_max_cigar_len
                    );
                } else
#endif
                {
                    // Legacy per-cell bt_p path (always for short phase, also long
                    // phase when USE_GRIDDED_BT=0).
                    ksw_fused_persistent_kernel<<<phase_concurrent_slots, parallel_threads,
                                                  0, align_stream>>>(
                        d_task_counter,
                        d_packed_query,
                        d_packed_target,
                        d_query_lens,
                        d_target_lens,
                        d_query_offsets,
                        d_target_offsets,
                        (gasal_res_t*)device_res,
                        d_mat,
                        d_bt_p_batch,
                        d_bt_off_batch,
                        d_bt_off_end_batch,
                        (int)batch_max_backtrack_size,
                        (int)batch_max_antidiag,
                        d_ksw_temp_buffer,
                        d_flag,
                        d_bw,
                        ksw_temp_per_task,
                        batch_size,
                        5,              // m = alphabet size (ACGTN)
                        opt->zdrop,
                        opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len
                    );
                }
            }

            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel launch failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }
            nvtxRangePop(); // kernel:s=N

            // P1/P2/P3: GPU compaction + fix_cigar + stats before D2H
            // This eliminates the 960MB stride CIGAR D2H transfer and the CPU mm_fix_cigar/mm_update_extra loop.
            nvtxRangePushA("compact_launch");
            if (cigar_buffer) {
                // Step A: compute per-task compact offsets via exclusive prefix sum
                cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_tmp_size,
                                              d_cigar_lengths, (int*)d_compact_offsets,
                                              batch_size, align_stream);

                // Step B: scatter stride CIGAR → compact CIGAR
                compact_cigar_kernel<<<batch_size, 256, 0, align_stream>>>(
                    d_cigar_buffer, d_compact_cigar, d_compact_offsets,
                    d_cigar_lengths, (int)current_max_cigar_len
                );

                // Step C: fix CIGAR in-place + compute alignment stats (one thread per task block)
                // Sequences still valid in d_unpacked_query/target (same stream, not yet overwritten)
                gpu_fix_cigar_and_stats<<<batch_size, 1, 0, align_stream>>>(
                    d_compact_cigar, d_compact_offsets, d_cigar_lengths,
                    d_unpacked_query, d_unpacked_target,
                    d_query_offsets, d_target_offsets,
                    d_mat, opt->q, opt->e, !(opt->flag & MM_F_SR),
                    d_blen, d_mlen, d_n_ambi, d_dp_max, d_gpu_stats_valid,
                    batch_size
                );
            }
            nvtxRangePop(); // compact_launch

            // D2H Sync 1: small arrays — CIGAR lengths, scores, endpoints, GPU stats
            // The compact CIGAR bulk D2H happens after we know total_cigar_ops (see Sync 2 below).
            nvtxRangePushA("d2h_small");
            if (cigar_buffer) {
                cudaMemcpyAsync(h_cigar_lengths, d_cigar_lengths,
                                batch_size * sizeof(int), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_blen,   d_blen,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_mlen,   d_mlen,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_n_ambi, d_n_ambi, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_dp_max, d_dp_max, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_gpu_stats_valid, d_gpu_stats_valid,
                                batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            }
            // Copy score/endpoint results back (async, ordered after kernel via align_stream)
            cudaMemcpyAsync(h_scores,      d_scores,      batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_query_ends,  d_query_ends,  batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_target_ends, d_target_ends, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mqe,   d_mqe,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mqe_t, d_mqe_t, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mte,   d_mte,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mte_q, d_mte_q, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_zdropped, d_zdropped, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            nvtxRangePop(); // d2h_small

            // Sync 1: wait for small arrays (D2H above) to arrive on host
            // This is where the CPU actually waits for the GPU forward kernel +
            // compact + D2H to all complete.  Long bars here = GPU bottleneck.
            nvtxRangePushA("sync1_wait_gpu");
            cudaStreamSynchronize(align_stream);
            nvtxRangePop(); // sync1_wait_gpu
            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel execution failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }
            cudaCheck();

            // === DEBUG: validate kernel results before host processing ===
            if (phase == 0) {
                int bad_score = 0, bad_cigar = 0, bad_end = 0;
                for (int i = 0; i < batch_size; i++) {
                    if (h_scores[i] == (int32_t)(-0x40000000)) bad_score++;
                    if (h_query_ends[i] < -1 || h_target_ends[i] < -1) bad_end++;
                    if (cigar_buffer && (h_cigar_lengths[i] < 0 ||
                        h_cigar_lengths[i] > (int)current_max_cigar_len)) bad_cigar++;
                }
                if (bad_score || bad_cigar || bad_end) {
                    fprintf(stderr, "\n[DEBUG::%s] Phase0 batch %d: bad_score=%d bad_cigar=%d bad_end=%d (batch_size=%d)\n",
                            __func__, batch_num, bad_score, bad_cigar, bad_end, batch_size);
                    // Print first few tasks for diagnosis
                    for (int i = 0; i < batch_size && i < 5; i++) {
                        int tidx = current_task_indices[batch_start + i];
                        fprintf(stderr, "  task[%d→%d]: score=%d qend=%d tend=%d cigar_len=%d qlen=%d tlen=%d\n",
                                i, tidx, h_scores[i], h_query_ends[i], h_target_ends[i],
                                cigar_buffer ? h_cigar_lengths[i] : -1,
                                tasks[tidx].qlen, tasks[tidx].tlen);
                    }
                }
            }

            // Sync 2: D2H compact CIGAR (only actual data, no stride padding)
            // IMPORTANT: d_compact_offsets were computed BEFORE gpu_fix_cigar, which may
            // shrink CIGARs in-place. The data in d_compact_cigar is still at the ORIGINAL
            // offsets. We must copy d_compact_offsets from GPU rather than recomputing from
            // the updated h_cigar_lengths, which would produce wrong (shifted) offsets.
            nvtxRangePushA("d2h_cigar");
            int total_cigar_ops = 0;
            if (cigar_buffer) {
                // Validate cigar lengths
                for (int i = 0; i < batch_size; i++) {
                    int clen = h_cigar_lengths[i];
                    if (clen < 0 || clen > (int)current_max_cigar_len) {
                        fprintf(stderr, "\n[DEBUG::%s] CORRUPT cigar_length[%d]=%d (max=%zu), clamping to 0\n",
                                __func__, i, clen, current_max_cigar_len);
                        h_cigar_lengths[i] = 0;
                    }
                }

                // Copy the GPU-computed compact offsets (which match the actual data layout)
                cudaMemcpyAsync(h_compact_offsets, d_compact_offsets,
                                batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaStreamSynchronize(align_stream);

                // Compute total data extent: last task's offset + original (pre-fix) length
                // We need the original length to know the data extent, but we only have the
                // updated length. Use the GPU offsets: the extent is offset[last] + original_len[last].
                // Since we can't recover original_len, use offset[last] + max of updated lengths
                // as a safe upper bound. Or simply: last offset + current_max_cigar_len as safe bound.
                // Better approach: the total extent equals the sum of ORIGINAL lengths, which is
                // d_compact_offsets[batch_size-1] + original_length[batch_size-1].
                // Since original_length >= updated_length, we can use:
                //   total_extent = h_compact_offsets[batch_size-1] + current_max_cigar_len
                // But that's wasteful. Instead, the data we need for each task is at
                // h_compact_offsets[i] with h_cigar_lengths[i] (updated) entries.
                // Find the maximum extent needed:
                uint32_t max_extent = 0;
                for (int i = 0; i < batch_size; i++) {
                    uint32_t end = h_compact_offsets[i] + (uint32_t)h_cigar_lengths[i];
                    if (end > max_extent) max_extent = end;
                }
                total_cigar_ops = (int)max_extent;

                if (total_cigar_ops > 0) {
                    cudaMemcpyAsync(h_compact_cigar, d_compact_cigar,
                               total_cigar_ops * sizeof(uint32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaStreamSynchronize(align_stream);
                }
            }
            nvtxRangePop(); // d2h_cigar

            // Map results back to tasks
            nvtxRangePushA("map_results");
            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];  // Use task index from current phase
                int align_id = i;  // always identity; h_task_to_align_id removed

                tasks[task_idx].score = h_scores[align_id];
                // In approx_max mode (KSW_EZ_APPROX_MAX), max_q/max_t are not tracked and should be -1
                // h_query_ends/h_target_ends contain backtrack endpoints, not max score positions
                if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].max_q = -1;
                    tasks[task_idx].max_t = -1;
                } else {
                    tasks[task_idx].max_q = h_query_ends[align_id];
                    tasks[task_idx].max_t = h_target_ends[align_id];
                }
                tasks[task_idx].mqe = h_mqe[align_id];
                tasks[task_idx].mqe_t = h_mqe_t[align_id];
                tasks[task_idx].mte = h_mte[align_id];
                tasks[task_idx].mte_q = h_mte_q[align_id];

                // Copy compact CIGAR to output buffer (no stride, direct copy from compact layout)
                if (cigar_buffer) {
                    int n_cigar = h_cigar_lengths[align_id];
                    tasks[task_idx].n_cigar = n_cigar;
                    if (n_cigar > 0 && n_cigar <= tasks[task_idx].max_cigar) {
                        memcpy(cigar_buffer + tasks[task_idx].cigar_offset,
                               h_compact_cigar + h_compact_offsets[align_id],
                               n_cigar * sizeof(uint32_t));
                    } else if (n_cigar > tasks[task_idx].max_cigar) {
                        tasks[task_idx].n_cigar = 0;  // Reset to avoid corruption
                    }
                } else {
                    tasks[task_idx].n_cigar = 0;
                }

                // Store GPU-computed alignment stats (used in map.c to skip CPU mm_update_extra)
                tasks[task_idx].blen            = h_blen[align_id];
                tasks[task_idx].mlen            = h_mlen[align_id];
                tasks[task_idx].n_ambi          = h_n_ambi[align_id];
                tasks[task_idx].dp_max          = h_dp_max[align_id];
                tasks[task_idx].gpu_stats_valid = h_gpu_stats_valid[align_id];

                // Set completion flags (match CPU ksw2 semantics).
                // KSW_EZ_RIGHT: reach_end = query end reached (max_q == qlen-1).
                // Otherwise: reach_end = both ends reached.
                if (tasks[task_idx].flag & KSW_EZ_RIGHT) {
                    int mq = (tasks[task_idx].flag & KSW_EZ_APPROX_MAX)
                           ? h_query_ends[align_id]
                           : tasks[task_idx].max_q;
                    tasks[task_idx].reach_end = (mq == tasks[task_idx].qlen - 1);
                } else if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].reach_end = (h_query_ends[align_id] == tasks[task_idx].qlen - 1) &&
                                                (h_target_ends[align_id] == tasks[task_idx].tlen - 1);
                } else {
                    tasks[task_idx].reach_end = (tasks[task_idx].max_q == tasks[task_idx].qlen - 1) &&
                                                (tasks[task_idx].max_t == tasks[task_idx].tlen - 1);
                }
                tasks[task_idx].zdropped = h_zdropped[align_id] ? 1 : 0;
            }
            nvtxRangePop(); // map_results
            nvtxRangePop(); // outer batch (S/L[B<N>|n=...])

            // h_unpacked_query/target point to pinned dev_mem buffers — no free needed.

            tasks_processed_in_phase += batch_size;
            total_tasks_processed += batch_size;

            if (phase == 0) {
                // Short phase: one batch, single line is enough.
                PLOG_INFO(stderr, "[Info::%s] %s [%d/%d] tasks=%d slots=%d bt_stride=%zu  (%d/%d done)\n",
                        stream_tag, phase_name, phase_batch_num, total_phase_batches,
                        batch_size, phase_concurrent_slots, batch_max_backtrack_size,
                        tasks_processed_in_phase, n_tasks_in_phase);
            } else {
                // Long phase: merged one-line log with repeat suppression.
                // bt_stride in MB (2 decimal places) and n_col from diag vars.
                double bt_mb = batch_max_backtrack_size / (1024.0 * 1024.0);

                bool same = (batch_max_backtrack_size == rep_bt_stride &&
                             phase_concurrent_slots   == rep_slots      &&
                             batch_size               == rep_batch_size);
                if (same) {
                    // Suppress this line — just count it.
                    rep_count++;
                } else {
                    // Flush any accumulated repeats from the previous group.
                    if (rep_count > 0) {
                        PLOG_INFO(stderr,
                            "[Info::%s]   ... ×%d more identical batches"
                            "  (%d/%d done)\n",
                            stream_tag, rep_count,
                            tasks_processed_in_phase - batch_size, n_tasks_in_phase);
                        rep_count = 0;
                    }
                    // Print this batch.
                    PLOG_INFO(stderr,
                        "[Info::%s] Long [%d|%c] tasks=%d  n_col=%zu  bt=%.2fMB  slots=%d"
                        "  (%d/%d done)\n",
                        stream_tag, phase_batch_num, batch_path_tag,
                        batch_size, diag_actual_n_col, bt_mb, phase_concurrent_slots,
                        tasks_processed_in_phase, n_tasks_in_phase);
                    rep_bt_stride  = batch_max_backtrack_size;
                    rep_slots      = phase_concurrent_slots;
                    rep_batch_size = batch_size;
                    rep_done_start = tasks_processed_in_phase - batch_size;
                }
                // After the very last batch in this phase, flush any trailing repeats.
                if (tasks_processed_in_phase >= n_tasks_in_phase && rep_count > 0) {
                    PLOG_INFO(stderr,
                        "[Info::%s]   ... ×%d more identical batches"
                        "  (%d/%d done)\n",
                        stream_tag, rep_count,
                        tasks_processed_in_phase, n_tasks_in_phase);
                    rep_count = 0;
                }
            }
        }  // End of batch loop within phase
    }  // End of three-tier loop

    PLOG_INFO(stderr, "[Info::%s] Alignment complete: %d tasks in %d batches\n",
            stream_tag, n_tasks, batch_num);

    // Cleanup phase-specific arrays
    free(task_indices_short);
    free(task_indices_long);

    // Pinned host buffers are pre-allocated in dev_mem — no free needed here.

    // Switch arena back to chain phase for next batch
    plmem_phase_to_chain(dev_mem);
    nvtxRangePop(); // gpu_align_batch_execute
}