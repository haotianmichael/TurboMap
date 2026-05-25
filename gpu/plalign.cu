#include <algorithm>
#include <chrono>
#include <string>
#include "plalign.cuh"
#include "gasal_kernels.h"
#include "plmem.cuh"
#include "plksw_kernel.cuh"
#include "plksw_shared_kernel.cuh"
#include "pllog.h"
#include <cub/device/device_scan.cuh>
#include <cerrno>
#include <cstring>

// Log file for long-task bt_stride distribution (diagnostic).
// Opened on first batch, appended thereafter, never explicitly closed
// (flushed on exit via atexit).
static FILE *g_btdist_log = nullptr;
static int   g_btdist_batch_num = 0;

static FILE *btdist_log_open(void) {
    if (!g_btdist_log) {
        g_btdist_log = fopen("long_task_dist.log", "w");
        if (!g_btdist_log)
            fprintf(stderr, "[Warn] Cannot open long_task_dist.log: %s\n", strerror(errno));
    }
    return g_btdist_log;
}

#ifndef USE_SHARED_LONG_KERNEL
#define USE_SHARED_LONG_KERNEL 0
#endif

static double s_ksw_wall_total_sec = 0.0;
struct KswTimingPrinter {
    ~KswTimingPrinter() {
        if (s_ksw_wall_total_sec > 0.0)
            fprintf(stderr, "\n[KSW timing] total wall time: %.3f ms\n", s_ksw_wall_total_sec * 1000.0);
    }
} s_ksw_timing_printer;

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
    CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaSliceWidth, &(subst->slice_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaZThreshold, &(subst->z_threshold), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaBandWidth, &(subst->band_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    return;
}

void gpu_align_copy_param() {
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

__global__ void compact_cigar_kernel(
    const uint32_t * __restrict__ src,
    uint32_t       * __restrict__ dst,
    const uint32_t * __restrict__ offsets,
    const int      * __restrict__ lengths,
    int              max_len
) {
    int task_id = blockIdx.x;
    int n = lengths[task_id];
    if (n <= 0) return;
    uint32_t dst_base = offsets[task_id];
    uint32_t src_base = (uint32_t)task_id * (uint32_t)max_len;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dst[dst_base + i] = src[src_base + i];
}

#define GPU_CIGAR_MATCH  0u
#define GPU_CIGAR_INS    1u
#define GPU_CIGAR_DEL    2u
#define GPU_CIGAR_N_SKIP 3u

// Performs left-align, I+D consolidation, and zero-squeeze passes on CIGAR in-place.
// Returns true if a leading I or D remains (CPU must handle coordinate adjustment).
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
                for (int m = k + 2; m < l; ++m) cigar[m] &= 0xfu;
                to_shrink = 1;
            }
            k = l - 1;
        }
    }

    // Pass 3a: squeeze zero-length ops + merge adjacent same-op
    if (to_shrink) {
        int32_t l = 0;
        for (int k = 0; k < nc; ++k)
            if (cigar[k] >> 4 != 0u) cigar[l++] = cigar[k];
        nc = l;
        l = 0;
        for (int k = 0; k < nc; ++k) {
            if (k == nc - 1 || (cigar[k] & 0xfu) != (cigar[k+1] & 0xfu))
                cigar[l++] = cigar[k];
            else
                cigar[k+1] += cigar[k] >> 4 << 4;
        }
        nc = l;
    }

    *n_cigar_p = nc;
    // Pass 3b (leading I/D coordinate adjustment) is skipped; signal CPU fallback if needed.
    return nc > 0 && ((cigar[0] & 0xfu) == GPU_CIGAR_INS || (cigar[0] & 0xfu) == GPU_CIGAR_DEL);
}

// dp_max uses integer log2 (31-__clz(1+len)) vs CPU float mg_log2 — may differ by ±1.
// Sets gpu_stats_valid=0 for tasks with leading I/D (CPU mm_update_extra handles those).
__global__ void gpu_fix_cigar_and_stats(
    uint32_t       *compact_cigar,
    const uint32_t * __restrict__ offsets,
    int32_t        *cigar_lengths,
    const uint8_t  * __restrict__ d_query,
    const uint8_t  * __restrict__ d_target,
    const uint32_t * __restrict__ d_query_offsets,
    const uint32_t * __restrict__ d_target_offsets,
    const int8_t   * __restrict__ d_mat,
    int32_t         q_open,
    int32_t         e_ext,
    int             log_gap,
    int32_t        *d_blen,
    int32_t        *d_mlen,
    int32_t        *d_n_ambi,
    int32_t        *d_dp_max,
    int32_t        *d_gpu_stats_valid,
    int             batch_size
) {
    int task_id = blockIdx.x;
    if (task_id >= batch_size) return;
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

    uint32_t cigar_base  = offsets[task_id];
    uint32_t *cigar      = compact_cigar + cigar_base;
    const uint8_t *qseq  = d_query  + d_query_offsets[task_id];
    const uint8_t *tseq  = d_target + d_target_offsets[task_id];

    bool has_leading_indel = gpu_fix_cigar(cigar, &nc, qseq, tseq);
    cigar_lengths[task_id] = nc;

    if (has_leading_indel) {
        d_gpu_stats_valid[task_id] = 0;
        return;
    }

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
                if (s < 0.0)        s = 0.0;
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
            // Integer floor(log2(1+len)) via __clz; may differ from CPU float mg_log2 by ±1.
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

#if USE_SHARED_LONG_KERNEL
    static bool s_shared_announced = false;
    if (!s_shared_announced) {
        s_shared_announced = true;
        PLOG_INFO(stderr, "[Info] Shared-mem long kernel ENABLED (USE_SHARED_LONG_KERNEL=1)\n");
    }
#endif

    cudaSetDevice(0);
    gpu_align_copy_param();

    deviceMemPtr *dev_mem = gpu_get_dev_mem(stream_id);
    if (!dev_mem) {
        fprintf(stderr, "[ERROR] Invalid stream_id %d for alignment.\n", stream_id);
        return;
    }

    // Always reset arena to short-align layout before capturing local GPU pointers below.
    // If the previous call left dev_mem in long-align state (current_phase==GPU_PHASE_ALIGN),
    // plmem_phase_to_align() would be a no-op: the short-phase pointer captures at lines
    // ~491-531 would then point to long-phase arena locations (wrong size, wrong stride).
    // Forcing GPU_PHASE_CHAIN causes plmem_phase_to_align() to always call setup_align_phase().
    dev_mem->current_phase = GPU_PHASE_CHAIN;
    plmem_phase_to_align(dev_mem);

    int kernel_blocks = 28;
    size_t short_task_max_len = dev_mem->short_task_max_len;
    size_t short_batch_size   = dev_mem->short_task_batch_size;

    int *task_indices_short = (int*)malloc(n_tasks * sizeof(int));
    int *task_indices_long  = (int*)malloc(n_tasks * sizeof(int));
    int n_short_tasks = 0;
    int n_long_tasks  = 0;

    // Hard limits derived from GPU buffer allocation. Exceeding causes OOB or wrong results.
    const size_t gpu_max_one = (size_t)dev_mem->max_align_task_len;
    const size_t gpu_max_sum = 2 * gpu_max_one - 2;  // CIGAR buffer guard

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
        if (max_seq_len <= short_task_max_len)
            task_indices_short[n_short_tasks++] = i;
        else
            task_indices_long[n_long_tasks++] = i;
    }

    // Sort long tasks descending by estimated bt_stride = (qlen+tlen) × n_col.
    // Largest bt_stride first so per-batch dynamic sizing gives smallest batches to hardest tasks,
    // preventing one oversized task from collapsing pool_cap for an entire large batch.
    std::sort(task_indices_long, task_indices_long + n_long_tasks,
        [&tasks](int a, int b) {
            int qa = tasks[a].qlen, ta = tasks[a].tlen, wa = tasks[a].w;
            int qb = tasks[b].qlen, tb = tasks[b].tlen, wb = tasks[b].w;
            int nca = (qa < ta) ? qa : ta;  if (wa >= 0 && wa + 1 < nca) nca = wa + 1;
            int ncb = (qb < tb) ? qb : tb;  if (wb >= 0 && wb + 1 < ncb) ncb = wb + 1;
            size_t sa = (size_t)(qa + ta) * (size_t)nca;
            size_t sb = (size_t)(qb + tb) * (size_t)ncb;
            return sa > sb;
        });

    PLOG_INFO(stderr, "[Info::Align::Tasks]: %d short (max_len≤%zubp) + %d long (max_len>%zubp, dynamic bt)\n",
            n_short_tasks, short_task_max_len, n_long_tasks, short_task_max_len);

    // ---- bt_stride distribution statistics by task type (temporary diagnostic) ----
    {
        const size_t MB = 1024ULL * 1024ULL;
        const size_t bin_edges[] = {1*MB, 2*MB, 5*MB, 10*MB, 20*MB, 50*MB};
        const int n_bins = 7;
        const char *bin_labels[] = {"<1MB","1-2MB","2-5MB","5-10MB","10-20MB","20-50MB",">=50MB"};

        struct TypeStat {
            int    count;
            int    bin_count[7];
            size_t total_bt, max_bt, min_bt;
        } stat[3] = {};
        for (int t = 0; t < 3; t++) stat[t].min_bt = SIZE_MAX;

        for (int i = 0; i < n_long_tasks; i++) {
            int idx = task_indices_long[i];
            int ql = tasks[idx].qlen, tl = tasks[idx].tlen, w = tasks[idx].w;
            int nc = (ql < tl) ? ql : tl;
            if (w >= 0 && w + 1 < nc) nc = w + 1;
            size_t bt = (size_t)(ql + tl) * (size_t)nc;
            int tt = tasks[idx].task_type;  // 0=LEFT_EXT, 1=GAP_FILL, 2=RIGHT_EXT
            if (tt < 0 || tt > 2) tt = 1;
            stat[tt].count++;
            stat[tt].total_bt += bt;
            if (bt > stat[tt].max_bt) stat[tt].max_bt = bt;
            if (bt < stat[tt].min_bt) stat[tt].min_bt = bt;
            int b = n_bins - 1;
            for (int k = 0; k < n_bins - 1; k++) { if (bt < bin_edges[k]) { b = k; break; } }
            stat[tt].bin_count[b]++;
        }

        FILE *lf = btdist_log_open();
        g_btdist_batch_num++;
        if (lf) fprintf(lf, "=== Batch %d ===\n", g_btdist_batch_num);

        const char *type_names[] = {"LEFT_EXT", "GAP_FILL", "RIGHT_EXT"};
        for (int t = 0; t < 3; t++) {
            if (stat[t].count == 0) continue;
            double total_gb = stat[t].total_bt / (1024.0*1024.0*1024.0);
            double min_mb   = stat[t].min_bt == SIZE_MAX ? 0.0 : stat[t].min_bt / (1024.0*1024.0);
            double max_mb   = stat[t].max_bt / (1024.0*1024.0);
            if (lf) fprintf(lf, "[BT-DIST] %s  n=%d  total=%.2fGB  min=%.2fMB  max=%.2fMB\n",
                    type_names[t], stat[t].count, total_gb, min_mb, max_mb);
            for (int k = 0; k < n_bins; k++) {
                if (stat[t].bin_count[k] > 0) {
                    double pct = 100.0 * stat[t].bin_count[k] / stat[t].count;
                    if (lf) fprintf(lf, "[BT-DIST]   %8s : %6d (%5.1f%%)\n",
                            bin_labels[k], stat[t].bin_count[k], pct);
                }
            }
        }
        if (lf) fflush(lf);
    }
    // ---- end bt_stride distribution ----

    int kernel_threads = 256;
    uint8_t  *d_unpacked_query  = dev_mem->d_align_unpacked_query;
    uint8_t  *d_unpacked_target = dev_mem->d_align_unpacked_target;
    uint32_t *d_packed_query    = dev_mem->d_align_packed_query;
    uint32_t *d_packed_target   = dev_mem->d_align_packed_target;
    uint32_t *d_query_offsets   = dev_mem->d_align_query_offsets;
    uint32_t *d_target_offsets  = dev_mem->d_align_target_offsets;
    uint32_t *d_query_lens      = dev_mem->d_align_query_lens;
    uint32_t *d_target_lens     = dev_mem->d_align_target_lens;
    int32_t  *d_flag            = dev_mem->d_align_flag;
    int32_t  *d_bw              = dev_mem->d_align_bw;
    void     *d_ksw_temp_buffer = dev_mem->d_align_ksw_temp_buffer;
    size_t    ksw_temp_per_task = dev_mem->align_ksw_temp_per_task;
    uint8_t  *d_backtrack_p     = dev_mem->d_align_backtrack_p;
    int      *d_backtrack_off   = dev_mem->d_align_backtrack_off;
    int      *d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
    uint32_t *d_cigar_buffer    = dev_mem->d_align_cigar_buffer;
    int      *d_cigar_lengths   = dev_mem->d_align_cigar_lengths;
    size_t    max_cigar_len     = dev_mem->max_align_cigar_len;
    size_t    max_query_len_limit = dev_mem->max_align_task_len;
    int8_t   *d_mat             = dev_mem->d_align_mat;
    void     *device_res        = dev_mem->d_align_device_res;
    int32_t  *d_scores          = dev_mem->d_align_scores;
    int32_t  *d_query_ends      = dev_mem->d_align_query_ends;
    int32_t  *d_target_ends     = dev_mem->d_align_target_ends;
    int32_t  *d_mqe             = dev_mem->d_align_mqe;
    int32_t  *d_mqe_t           = dev_mem->d_align_mqe_t;
    int32_t  *d_mte             = dev_mem->d_align_mte;
    int32_t  *d_mte_q           = dev_mem->d_align_mte_q;
    int32_t  *d_zdropped        = dev_mem->d_align_zdropped;
    int      *d_task_counter    = dev_mem->d_align_task_counter;
    int       n_concurrent_blocks = dev_mem->n_align_concurrent_blocks;
    cudaStream_t align_stream   = gpu_get_cudastream(stream_id);
    uint32_t *d_compact_cigar   = dev_mem->d_align_compact_cigar;
    uint32_t *d_compact_offsets = dev_mem->d_align_compact_offsets;
    void     *d_cub_tmp         = dev_mem->d_align_cub_tmp;
    size_t    cub_tmp_size      = dev_mem->align_cub_tmp_size;
    int32_t  *d_blen            = dev_mem->d_align_blen;
    int32_t  *d_mlen            = dev_mem->d_align_mlen;
    int32_t  *d_n_ambi          = dev_mem->d_align_n_ambi;
    int32_t  *d_dp_max          = dev_mem->d_align_dp_max;
    int32_t  *d_gpu_stats_valid = dev_mem->d_align_gpu_stats_valid;

    size_t long_cigar_len       = 2 * max_query_len_limit;
    size_t short_batch_persistent = (size_t)dev_mem->max_align_tasks;
    size_t cigar_buf_total_tasks  = (size_t)dev_mem->max_align_tasks;
    size_t long_batch_persistent  = cigar_buf_total_tasks * max_cigar_len / long_cigar_len;

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

    int8_t h_scoring_matrix[25];
    ksw_gen_simple_mat(5, h_scoring_matrix, opt->a, opt->b, opt->sc_ambi);
    cudaError_t err;
    CHECKCUDAERROR(cudaMemcpyAsync(d_mat, h_scoring_matrix, 25 * sizeof(int8_t),
                                   cudaMemcpyHostToDevice, align_stream));

    init_gasal_res<<<1, 1, 0, align_stream>>>((gasal_res_t*)device_res, d_scores, d_query_ends, d_target_ends,
                              d_mqe, d_mqe_t, d_mte, d_mte_q, d_zdropped);
    CHECKCUDAERROR(cudaGetLastError());

    int batch_num = 0;
    int total_tasks_processed = 0;
    std::string deferred_short_log;

    for (int phase = 0; phase < 2; phase++) {
        int  *current_task_indices  = (phase == 0) ? task_indices_short : task_indices_long;
        int   n_tasks_in_phase      = (phase == 0) ? n_short_tasks : n_long_tasks;
        size_t current_batch_size   = (phase == 0) ? short_batch_persistent : long_batch_persistent;

        // Short phase: n_col bounded by short_task_max_len regardless of per-task bandwidth.
        // Long phase: n_col and antidiag are computed per-batch below.
        size_t current_max_antidiag      = 2 * short_task_max_len;
        size_t current_max_n_col         = short_task_max_len + 1;
        size_t current_max_backtrack_size = current_max_antidiag * current_max_n_col;
        size_t current_max_cigar_len     = (phase == 0)
            ? (2 * short_task_max_len)
            : (2 * dev_mem->max_align_task_len);

        if (n_tasks_in_phase == 0) continue;

        if (phase == 1) {
            cudaStreamSynchronize(align_stream);
            plmem_phase_to_long_align(dev_mem);

            // Refresh all local GPU pointers after arena transition
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
            d_backtrack_p     = dev_mem->d_align_backtrack_p;
            d_backtrack_off     = dev_mem->d_align_backtrack_off;
            d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
            d_cigar_buffer    = dev_mem->d_align_cigar_buffer;
            d_cigar_lengths   = dev_mem->d_align_cigar_lengths;
            max_cigar_len     = dev_mem->max_align_cigar_len;
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
            n_concurrent_blocks = dev_mem->n_align_concurrent_blocks;
            d_compact_cigar   = dev_mem->d_align_compact_cigar;
            d_compact_offsets = dev_mem->d_align_compact_offsets;
            d_cub_tmp         = dev_mem->d_align_cub_tmp;
            cub_tmp_size      = dev_mem->align_cub_tmp_size;
            d_blen            = dev_mem->d_align_blen;
            d_mlen            = dev_mem->d_align_mlen;
            d_n_ambi          = dev_mem->d_align_n_ambi;
            d_dp_max          = dev_mem->d_align_dp_max;
            d_gpu_stats_valid = dev_mem->d_align_gpu_stats_valid;

            // Re-upload scoring matrix and gasal_res (now at new arena addresses)
            int8_t h_scoring_matrix2[25];
            ksw_gen_simple_mat(5, h_scoring_matrix2, opt->a, opt->b, opt->sc_ambi);
            cudaMemcpyAsync(d_mat, h_scoring_matrix2, 25 * sizeof(int8_t),
                            cudaMemcpyHostToDevice, align_stream);
            init_gasal_res<<<1, 1, 0, align_stream>>>(
                (gasal_res_t*)device_res,
                d_scores, d_query_ends, d_target_ends,
                d_mqe, d_mqe_t, d_mte, d_mte_q, d_zdropped);

            long_batch_persistent = (size_t)dev_mem->long_task_batch_size;
            current_batch_size    = long_batch_persistent;
        }

        // ===== MULTI-CLASS LONG-PHASE DISPATCH =====
        // Splits long tasks into 3 bt_stride classes (A<5MB, B 5-20MB, C≥20MB),
        // allocates separate bt_p/bt_off/ksw_temp sub-pools for each class,
        // and launches 3 concurrent kernels so all 3456 GPU slots stay active
        // instead of collapsing to ~432 slots for the largest-stride class.
        if (phase == 1) {
            const size_t THRESH_B = (size_t)5  << 20;   // 5 MB
            const size_t THRESH_C = (size_t)20 << 20;   // 20 MB

            // --- Classify ---
            std::vector<int> idx_A, idx_B, idx_C;
            size_t stride_A_max = 8, stride_B_max = 8, stride_C_max = 8;
            for (int i = 0; i < n_long_tasks; i++) {
                int tidx = task_indices_long[i];
                int ql = tasks[tidx].qlen, tl = tasks[tidx].tlen, w = tasks[tidx].w;
                int nc = (ql < tl) ? ql : tl;
                if (w >= 0 && w + 1 < nc) nc = w + 1;
                size_t bt = ((size_t)(ql + tl) * (size_t)nc + 7) & ~(size_t)7;
                if (bt < THRESH_B) {
                    idx_A.push_back(tidx);
                    if (bt > stride_A_max) stride_A_max = bt;
                } else if (bt < THRESH_C) {
                    idx_B.push_back(tidx);
                    if (bt > stride_B_max) stride_B_max = bt;
                } else {
                    idx_C.push_back(tidx);
                    if (bt > stride_C_max) stride_C_max = bt;
                }
            }
            int nA = (int)idx_A.size(), nB = (int)idx_B.size(), nC = (int)idx_C.size();

            // --- bt_p pool (larger of arena vs dedicated) ---
            size_t bt_p_avail = dev_mem->long_arena_bt_p_bytes;
            uint8_t *bt_p_base = d_backtrack_p;
            {
                size_t dedi = (dev_mem->d_align_backtrack_p_long != nullptr)
                              ? dev_mem->long_bt_p_pool_bytes : 0;
                if (dedi > bt_p_avail) {
                    bt_p_avail = dedi;
                    bt_p_base  = dev_mem->d_align_backtrack_p_long;
                }
            }

            // Optimal slot allocation: minimize max(nX*strideX/sX) subject to
            // Σ(sX*strideX) ≤ bt_p_avail.  Closed-form solution:
            //   sX = nX*strideX / λ,  λ = Σ(nX*strideX²) / bt_p_avail
            // This equalises wall-clock time across classes and saturates BTP.
            const int TOTAL_SLOTS = dev_mem->n_long_concurrent_slots;

            double sum_stride2 = 0.0;
            if (nC > 0) sum_stride2 += (double)nC * (double)stride_C_max * (double)stride_C_max;
            if (nB > 0) sum_stride2 += (double)nB * (double)stride_B_max * (double)stride_B_max;
            if (nA > 0) sum_stride2 += (double)nA * (double)stride_A_max * (double)stride_A_max;

            double lambda_btp = (sum_stride2 > 0.0 && bt_p_avail > 0)
                ? sum_stride2 / (double)bt_p_avail
                : 1.0;

            // Integer truncation guarantees Σ(sX*strideX) ≤ bt_p_avail by construction.
            int sC = (nC > 0) ? (int)((double)nC * (double)stride_C_max / lambda_btp) : 0;
            int sB = (nB > 0) ? (int)((double)nB * (double)stride_B_max / lambda_btp) : 0;
            int sA = (nA > 0) ? (int)((double)nA * (double)stride_A_max / lambda_btp) : 0;
            // Ensure each active class gets at least 1 slot (may add ≤3 extra strides to BTP).
            if (sC < 1 && nC > 0) sC = 1;
            if (sB < 1 && nB > 0) sB = 1;
            if (sA < 1 && nA > 0) sA = 1;

            // Scale down proportionally if total exceeds GPU slot cap.
            {
                int total_want = sC + sB + sA;
                if (total_want > TOTAL_SLOTS) {
                    double scale = (double)TOTAL_SLOTS / total_want;
                    sC = (nC > 0) ? std::max(1, (int)(sC * scale)) : 0;
                    sB = (nB > 0) ? std::max(1, (int)(sB * scale)) : 0;
                    sA = (nA > 0) ? std::max(1, (int)(sA * scale)) : 0;
                    // Fine-trim to guarantee total ≤ TOTAL_SLOTS
                    while (sC + sB + sA > TOTAL_SLOTS) {
                        if      (sA > 1 && nA > 0) --sA;
                        else if (sB > 1 && nB > 0) --sB;
                        else if (sC > 1 && nC > 0) --sC;
                        else break;
                    }
                }
            }

            // Safety: clamp sub-pool pointers to stay within bt_p_avail.
            // With the optimal formula this should never trigger, but guards
            // against rounding (the min-1-slot overrides above).
            {
                size_t bt_used = (size_t)sC * stride_C_max
                               + (size_t)sB * stride_B_max
                               + (size_t)sA * stride_A_max;
                while (bt_used > bt_p_avail) {
                    if      (sA > 1 && nA > 0) { --sA; bt_used -= stride_A_max; }
                    else if (sB > 1 && nB > 0) { --sB; bt_used -= stride_B_max; }
                    else if (sC > 1 && nC > 0) { --sC; bt_used -= stride_C_max; }
                    else break;
                }
            }

            if (!deferred_short_log.empty()) {
                fprintf(stderr, "%s", deferred_short_log.c_str());
                deferred_short_log.clear();
            }
            PLOG_INFO(stderr,
                "[Info::MultiClass] nA=%d nB=%d nC=%d  "
                "sA=%d(%.1fMB) sB=%d(%.1fMB) sC=%d(%.1fMB)  "
                "bt_p=%.2fGB\n",
                nA, nB, nC,
                sA, stride_A_max / (1024.0*1024.0),
                sB, stride_B_max / (1024.0*1024.0),
                sC, stride_C_max / (1024.0*1024.0),
                bt_p_avail / (1024.0*1024.0*1024.0));

            // --- Sub-pool pointers ---
            // bt_p layout: [C slots][B slots][A slots]
            uint8_t *bt_p_C = bt_p_base;
            uint8_t *bt_p_B = bt_p_C + (size_t)sC * stride_C_max;
            uint8_t *bt_p_A = bt_p_B + (size_t)sB * stride_B_max;

            // bt_off layout: [C slots][B slots][A slots] × max_antidiag_long each
            size_t max_ad_long = 2 * dev_mem->max_align_task_len;
            int *bt_off_C     = dev_mem->d_align_backtrack_off_long;
            int *bt_off_B     = bt_off_C + (size_t)sC * max_ad_long;
            int *bt_off_A     = bt_off_B + (size_t)sB * max_ad_long;
            int *bt_off_end_C = dev_mem->d_align_backtrack_off_end_long;
            int *bt_off_end_B = bt_off_end_C + (size_t)sC * max_ad_long;
            int *bt_off_end_A = bt_off_end_B + (size_t)sB * max_ad_long;

            // ksw_temp layout: [C slots][B slots][A slots] × ksw_temp_per_task each
            void *ksw_temp_C = d_ksw_temp_buffer;
            void *ksw_temp_B = (char*)d_ksw_temp_buffer + (size_t)sC * ksw_temp_per_task;
            void *ksw_temp_A = (char*)ksw_temp_B + (size_t)sB * ksw_temp_per_task;

            // --- Static extra streams + counters (created once) ---
            static cudaStream_t s_stream_B = nullptr, s_stream_C = nullptr;
            static int *d_counter_B = nullptr, *d_counter_C = nullptr;
            if (!s_stream_B) {
                cudaStreamCreate(&s_stream_B);
                cudaStreamCreate(&s_stream_C);
                cudaMalloc(&d_counter_B, sizeof(int));
                cudaMalloc(&d_counter_C, sizeof(int));
            }

            // --- Batch loop: process all long tasks in CIGAR-buffer-sized windows ---
            int MAX_LONG_BATCH_SLOTS = (int)long_batch_persistent;
            int a_done = 0, b_done = 0, c_done = 0;
            int ml_batch_num = 0;

            while (a_done < nA || b_done < nB || c_done < nC) {
                int nA_rem = nA - a_done, nB_rem = nB - b_done, nC_rem = nC - c_done;
                int total_rem = nA_rem + nB_rem + nC_rem;
                int window = (total_rem < MAX_LONG_BATCH_SLOTS) ? total_rem : MAX_LONG_BATCH_SLOTS;

                // Proportional split of window across remaining classes
                int nA_b = (total_rem > 0) ? (int)((int64_t)nA_rem * window / total_rem) : 0;
                int nB_b = (total_rem > 0) ? (int)((int64_t)nB_rem * window / total_rem) : 0;
                int nC_b = window - nA_b - nB_b;
                if (nA_b > nA_rem) nA_b = nA_rem;
                if (nB_b > nB_rem) nB_b = nB_rem;
                if (nC_b > nC_rem) nC_b = nC_rem;
                // Fill remaining window slots C→B→A
                int rem = window - (nA_b + nB_b + nC_b);
                if (rem > 0 && nC_b < nC_rem) { int x = std::min(rem, nC_rem - nC_b); nC_b += x; rem -= x; }
                if (rem > 0 && nB_b < nB_rem) { int x = std::min(rem, nB_rem - nB_b); nB_b += x; rem -= x; }
                if (rem > 0 && nA_b < nA_rem) { int x = std::min(rem, nA_rem - nA_b); nA_b += x; rem -= x; }
                int batch_total = nA_b + nB_b + nC_b;
                if (batch_total == 0) break;

                ml_batch_num++;
                batch_num++;

                // --- Pack sequences H2D in [A|B|C] order ---
                uint8_t *h_uq = dev_mem->h_align_unpacked_query;
                uint8_t *h_ut = dev_mem->h_align_unpacked_target;
                size_t total_query_bytes = 0, total_target_bytes = 0;
                const uint8_t N_BASE = 4;

                auto mc_pack = [&](const std::vector<int>& vidx, int start, int count, int base_i) {
                    for (int i = 0; i < count; i++) {
                        int tidx = vidx[start + i];
                        int ql = tasks[tidx].qlen, tl = tasks[tidx].tlen;
                        size_t qa = ((size_t)(ql + 7) / 8) * 8;
                        size_t ta = ((size_t)(tl + 7) / 8) * 8;
                        h_query_offsets[base_i + i]  = (uint32_t)total_query_bytes;
                        h_target_offsets[base_i + i] = (uint32_t)total_target_bytes;
                        h_query_lens[base_i + i]     = (uint32_t)ql;
                        h_target_lens[base_i + i]    = (uint32_t)tl;
                        h_flag[base_i + i]           = tasks[tidx].flag;
                        h_bw[base_i + i]             = tasks[tidx].w;
                        memcpy(h_uq + total_query_bytes,
                               seq_buffer + tasks[tidx].qseq_offset, ql);
                        memcpy(h_ut + total_target_bytes,
                               seq_buffer + tasks[tidx].tseq_offset, tl);
                        for (int j = ql; j < (int)qa; j++) h_uq[total_query_bytes + j] = N_BASE;
                        for (int j = tl; j < (int)ta; j++) h_ut[total_target_bytes + j] = N_BASE;
                        total_query_bytes  += qa;
                        total_target_bytes += ta;
                    }
                };
                mc_pack(idx_A, a_done, nA_b, 0);
                mc_pack(idx_B, b_done, nB_b, nA_b);
                mc_pack(idx_C, c_done, nC_b, nA_b + nB_b);

                // H2D sequences + metadata (on align_stream)
                cudaMemcpyAsync(d_unpacked_query,  h_uq,
                                total_query_bytes,  cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_unpacked_target, h_ut,
                                total_target_bytes, cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_query_offsets, h_query_offsets,
                                batch_total * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_target_offsets, h_target_offsets,
                                batch_total * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_query_lens, h_query_lens,
                                batch_total * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_target_lens, h_target_lens,
                                batch_total * sizeof(uint32_t), cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_flag, h_flag,
                                batch_total * sizeof(int32_t), cudaMemcpyHostToDevice, align_stream);
                cudaMemcpyAsync(d_bw, h_bw,
                                batch_total * sizeof(int32_t), cudaMemcpyHostToDevice, align_stream);

                // Pack sequences on GPU
                {
                    int qt = (int)ceil(total_query_bytes  / (8.0 * kernel_threads * kernel_blocks));
                    int tt = (int)ceil(total_target_bytes / (8.0 * kernel_threads * kernel_blocks));
                    gasal_pack_kernel<<<kernel_blocks, kernel_threads, 0, align_stream>>>(
                        (uint32_t*)d_unpacked_query, (uint32_t*)d_unpacked_target,
                        d_packed_query, d_packed_target,
                        qt, tt,
                        total_query_bytes / 4, total_target_bytes / 4);
                }

                // Signal H2D+pack done → B and C streams can start
                cudaEvent_t ev_h2d;
                cudaEventCreateWithFlags(&ev_h2d, cudaEventDisableTiming);
                cudaEventRecord(ev_h2d, align_stream);
                if (nB_b > 0) cudaStreamWaitEvent(s_stream_B, ev_h2d, 0);
                if (nC_b > 0) cudaStreamWaitEvent(s_stream_C, ev_h2d, 0);

                // Reset task counters
                cudaMemsetAsync(d_task_counter, 0, sizeof(int), align_stream);
                if (nB_b > 0) cudaMemsetAsync(d_counter_B, 0, sizeof(int), s_stream_B);
                if (nC_b > 0) cudaMemsetAsync(d_counter_C, 0, sizeof(int), s_stream_C);

                auto t_start = std::chrono::steady_clock::now();
                const int par_threads = 32;

                // Effective slot counts (capped to actual task count)
                int sA_eff = (nA_b > 0) ? std::min(sA, nA_b) : 0;
                int sB_eff = (nB_b > 0) ? std::min(sB, nB_b) : 0;
                int sC_eff = (nC_b > 0) ? std::min(sC, nC_b) : 0;
                if (sA_eff < 1 && nA_b > 0) sA_eff = 1;
                if (sB_eff < 1 && nB_b > 0) sB_eff = 1;
                if (sC_eff < 1 && nC_b > 0) sC_eff = 1;

                // Launch class A on align_stream (task_id_base = 0)
                if (nA_b > 0) {
                    ksw_fused_persistent_kernel<<<sA_eff, par_threads, 0, align_stream>>>(
                        d_task_counter,
                        d_packed_query, d_packed_target,
                        d_query_lens, d_target_lens,
                        d_query_offsets, d_target_offsets,
                        (gasal_res_t*)device_res, d_mat,
                        bt_p_A, bt_off_A, bt_off_end_A,
                        (int)stride_A_max, (int)max_ad_long,
                        ksw_temp_A, d_flag, d_bw,
                        ksw_temp_per_task, nA_b, 5,
                        opt->zdrop, opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len,
                        0  // task_id_base
                    );
                }

                // Launch class B on s_stream_B (task_id_base = nA_b)
                if (nB_b > 0) {
                    ksw_fused_persistent_kernel<<<sB_eff, par_threads, 0, s_stream_B>>>(
                        d_counter_B,
                        d_packed_query, d_packed_target,
                        d_query_lens  + nA_b, d_target_lens  + nA_b,
                        d_query_offsets + nA_b, d_target_offsets + nA_b,
                        (gasal_res_t*)device_res, d_mat,
                        bt_p_B, bt_off_B, bt_off_end_B,
                        (int)stride_B_max, (int)max_ad_long,
                        ksw_temp_B, d_flag + nA_b, d_bw + nA_b,
                        ksw_temp_per_task, nB_b, 5,
                        opt->zdrop, opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len,
                        nA_b  // task_id_base
                    );
                }

                // Launch class C on s_stream_C (task_id_base = nA_b + nB_b)
                if (nC_b > 0) {
                    ksw_fused_persistent_kernel<<<sC_eff, par_threads, 0, s_stream_C>>>(
                        d_counter_C,
                        d_packed_query, d_packed_target,
                        d_query_lens   + nA_b + nB_b, d_target_lens   + nA_b + nB_b,
                        d_query_offsets + nA_b + nB_b, d_target_offsets + nA_b + nB_b,
                        (gasal_res_t*)device_res, d_mat,
                        bt_p_C, bt_off_C, bt_off_end_C,
                        (int)stride_C_max, (int)max_ad_long,
                        ksw_temp_C, d_flag + nA_b + nB_b, d_bw + nA_b + nB_b,
                        ksw_temp_per_task, nC_b, 5,
                        opt->zdrop, opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len,
                        nA_b + nB_b  // task_id_base
                    );
                }

                // Sync B and C back to align_stream
                if (nB_b > 0) {
                    cudaEvent_t ev_B;
                    cudaEventCreateWithFlags(&ev_B, cudaEventDisableTiming);
                    cudaEventRecord(ev_B, s_stream_B);
                    cudaStreamWaitEvent(align_stream, ev_B, 0);
                    cudaEventDestroy(ev_B);
                }
                if (nC_b > 0) {
                    cudaEvent_t ev_C;
                    cudaEventCreateWithFlags(&ev_C, cudaEventDisableTiming);
                    cudaEventRecord(ev_C, s_stream_C);
                    cudaStreamWaitEvent(align_stream, ev_C, 0);
                    cudaEventDestroy(ev_C);
                }
                cudaEventDestroy(ev_h2d);

                cudaError_t ml_kerr = cudaGetLastError();
                if (ml_kerr != cudaSuccess)
                    fprintf(stderr, "[ERROR] Multi-class KSW kernel launch: %s\n",
                            cudaGetErrorString(ml_kerr));

                // Post-process CIGAR for all batch_total tasks (on align_stream after sync)
                if (cigar_buffer) {
                    cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_tmp_size,
                                                  d_cigar_lengths, (int*)d_compact_offsets,
                                                  batch_total, align_stream);
                    compact_cigar_kernel<<<batch_total, 256, 0, align_stream>>>(
                        d_cigar_buffer, d_compact_cigar, d_compact_offsets,
                        d_cigar_lengths, (int)current_max_cigar_len);
                    gpu_fix_cigar_and_stats<<<batch_total, 1, 0, align_stream>>>(
                        d_compact_cigar, d_compact_offsets, d_cigar_lengths,
                        d_unpacked_query, d_unpacked_target,
                        d_query_offsets, d_target_offsets,
                        d_mat, opt->q, opt->e, !(opt->flag & MM_F_SR),
                        d_blen, d_mlen, d_n_ambi, d_dp_max, d_gpu_stats_valid,
                        batch_total);
                }

                cudaStreamSynchronize(align_stream);
                auto t_end = std::chrono::steady_clock::now();
                double ml_gpu_ms = std::chrono::duration<double>(t_end - t_start).count() * 1000.0;
                s_ksw_wall_total_sec += ml_gpu_ms / 1000.0;

                ml_kerr = cudaGetLastError();
                if (ml_kerr != cudaSuccess)
                    fprintf(stderr, "[ERROR] Multi-class kernel execution: %s\n",
                            cudaGetErrorString(ml_kerr));
                cudaCheck();

                // D2H results
                if (cigar_buffer) {
                    cudaMemcpyAsync(h_cigar_lengths, d_cigar_lengths,
                                    batch_total * sizeof(int), cudaMemcpyDeviceToHost, align_stream);
                    cudaMemcpyAsync(h_blen,   d_blen,   batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaMemcpyAsync(h_mlen,   d_mlen,   batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaMemcpyAsync(h_n_ambi, d_n_ambi, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaMemcpyAsync(h_dp_max, d_dp_max, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaMemcpyAsync(h_gpu_stats_valid, d_gpu_stats_valid,
                                    batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                }
                cudaMemcpyAsync(h_scores,      d_scores,      batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_query_ends,  d_query_ends,  batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_target_ends, d_target_ends, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_mqe,   d_mqe,   batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_mqe_t, d_mqe_t, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_mte,   d_mte,   batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_mte_q, d_mte_q, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaMemcpyAsync(h_zdropped, d_zdropped, batch_total * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaStreamSynchronize(align_stream);

                // D2H compact CIGAR
                int ml_total_cigar = 0;
                if (cigar_buffer) {
                    for (int i = 0; i < batch_total; i++) {
                        if (h_cigar_lengths[i] < 0 || h_cigar_lengths[i] > (int)current_max_cigar_len)
                            h_cigar_lengths[i] = 0;
                    }
                    cudaMemcpyAsync(h_compact_offsets, d_compact_offsets,
                                    batch_total * sizeof(uint32_t), cudaMemcpyDeviceToHost, align_stream);
                    cudaStreamSynchronize(align_stream);
                    uint32_t ml_max_ext = 0;
                    for (int i = 0; i < batch_total; i++) {
                        uint32_t end = h_compact_offsets[i] + (uint32_t)h_cigar_lengths[i];
                        if (end > ml_max_ext) ml_max_ext = end;
                    }
                    ml_total_cigar = (int)ml_max_ext;
                    if (ml_total_cigar > 0) {
                        cudaMemcpyAsync(h_compact_cigar, d_compact_cigar,
                                        ml_total_cigar * sizeof(uint32_t), cudaMemcpyDeviceToHost, align_stream);
                        cudaStreamSynchronize(align_stream);
                    }
                }

                // CPU result mapping — iterate [A|B|C] pack order
                auto mc_map = [&](const std::vector<int>& vidx, int start, int count, int base_i) {
                    for (int i = 0; i < count; i++) {
                        int tidx = vidx[start + i];
                        int gi   = base_i + i;

                        tasks[tidx].score = h_scores[gi];
                        if (tasks[tidx].flag & KSW_EZ_APPROX_MAX) {
                            tasks[tidx].max_q = -1;
                            tasks[tidx].max_t = -1;
                        } else {
                            tasks[tidx].max_q = h_query_ends[gi];
                            tasks[tidx].max_t = h_target_ends[gi];
                        }
                        tasks[tidx].mqe   = h_mqe[gi];
                        tasks[tidx].mqe_t = h_mqe_t[gi];
                        tasks[tidx].mte   = h_mte[gi];
                        tasks[tidx].mte_q = h_mte_q[gi];

                        if (cigar_buffer) {
                            int nc = h_cigar_lengths[gi];
                            tasks[tidx].n_cigar = nc;
                            if (nc > 0 && nc <= tasks[tidx].max_cigar) {
                                memcpy(cigar_buffer + tasks[tidx].cigar_offset,
                                       h_compact_cigar + h_compact_offsets[gi],
                                       nc * sizeof(uint32_t));
                            } else if (nc > tasks[tidx].max_cigar) {
                                tasks[tidx].n_cigar = 0;
                            }
                        } else {
                            tasks[tidx].n_cigar = 0;
                        }

                        tasks[tidx].blen            = h_blen[gi];
                        tasks[tidx].mlen            = h_mlen[gi];
                        tasks[tidx].n_ambi          = h_n_ambi[gi];
                        tasks[tidx].dp_max          = h_dp_max[gi];
                        tasks[tidx].gpu_stats_valid = h_gpu_stats_valid[gi];

                        if (tasks[tidx].flag & KSW_EZ_RIGHT) {
                            int mq = (tasks[tidx].flag & KSW_EZ_APPROX_MAX)
                                   ? h_query_ends[gi] : tasks[tidx].max_q;
                            tasks[tidx].reach_end = (mq == tasks[tidx].qlen - 1);
                        } else if (tasks[tidx].flag & KSW_EZ_APPROX_MAX) {
                            tasks[tidx].reach_end =
                                (h_query_ends[gi]  == tasks[tidx].qlen - 1) &&
                                (h_target_ends[gi] == tasks[tidx].tlen - 1);
                        } else {
                            tasks[tidx].reach_end =
                                (tasks[tidx].max_q == tasks[tidx].qlen - 1) &&
                                (tasks[tidx].max_t == tasks[tidx].tlen - 1);
                        }
                        tasks[tidx].zdropped = h_zdropped[gi] ? 1 : 0;
                    }
                };
                mc_map(idx_A, a_done, nA_b, 0);
                mc_map(idx_B, b_done, nB_b, nA_b);
                mc_map(idx_C, c_done, nC_b, nA_b + nB_b);

                PLOG_INFO(stderr,
                    "[Info::MLong %d]: A=%d B=%d C=%d  "
                    "sA=%d sB=%d sC=%d  (gpu_ms=%.1f)\n",
                    ml_batch_num, nA_b, nB_b, nC_b, sA_eff, sB_eff, sC_eff, ml_gpu_ms);

                a_done += nA_b; b_done += nB_b; c_done += nC_b;
                total_tasks_processed += batch_total;
            } // end multi-class batch loop

            continue;  // skip the while loop below; phase 0 uses it exclusively
        }
        // ===== END MULTI-CLASS LONG-PHASE =====

        int tasks_processed_in_phase = 0;
        int phase_batch_num = 0;

        size_t pool_cap0      = 0;
        size_t bt_stride0_val = 0;
        bool   long_config_logged = false;

        while (tasks_processed_in_phase < n_tasks_in_phase) {
            int batch_start = tasks_processed_in_phase;

            // Dynamic batch sizing for long phase:
            // Tasks sorted descending by bt_stride; first task has largest bt_stride → lowest pool_cap.
            // batch_size = pool_cap × LATENCY_HIDE_FACTOR (3) keeps GPU busy without wasting CIGAR buffer.
            if (phase == 1) {
                int tidx0 = current_task_indices[batch_start];
                int ql0   = tasks[tidx0].qlen;
                int tl0   = tasks[tidx0].tlen;
                int w0    = tasks[tidx0].w;
                int nc0   = (ql0 < tl0) ? ql0 : tl0;
                if (w0 >= 0 && w0 + 1 < nc0) nc0 = w0 + 1;
                size_t bt_stride0 = (size_t)(ql0 + tl0) * (size_t)nc0;

                // Use the LARGER of arena and dedicated bt_p pool to match the actual dispatch pointer.
                const size_t TYPICAL_BT_STRIDE_FALLBACK = (size_t)6 << 20;
                size_t arena_bt_p = dev_mem->long_arena_bt_p_bytes;
                size_t dedi_bt_p  = (dev_mem->d_align_backtrack_p_long != nullptr)
                                    ? dev_mem->long_bt_p_pool_bytes : 0;
                size_t bt_p_avail = (arena_bt_p > dedi_bt_p) ? arena_bt_p : dedi_bt_p;
                if (bt_p_avail == 0)
                    bt_p_avail = (size_t)dev_mem->n_long_concurrent_slots * TYPICAL_BT_STRIDE_FALLBACK;

                const size_t LATENCY_HIDE_FACTOR = 3;
                bt_stride0_val = bt_stride0;
                pool_cap0 = (bt_stride0 > 0) ? (bt_p_avail / bt_stride0) : (size_t)256;
                if (pool_cap0 < 1) pool_cap0 = 1;

                size_t dyn_batch = pool_cap0 * LATENCY_HIDE_FACTOR;
                if (dyn_batch < 64)                             dyn_batch = 64;
                if (dyn_batch > (size_t)long_batch_persistent) dyn_batch = (size_t)long_batch_persistent;

                current_batch_size = dyn_batch;

                if (!long_config_logged) {
                    PLOG_INFO(stderr,
                        "[Info::Align::LongConfig]: bt_p=%.2fGB  h2d_max=%zu"
                        "  gpu_max_slots=%d\n",
                        bt_p_avail / (1024.0*1024.0*1024.0),
                        (size_t)long_batch_persistent,
                        dev_mem->n_align_concurrent_blocks);
                    long_config_logged = true;
                }
            }

            int batch_size = (tasks_processed_in_phase + (int)current_batch_size <= n_tasks_in_phase)
                             ? (int)current_batch_size
                             : (n_tasks_in_phase - tasks_processed_in_phase);
            batch_num++;
            phase_batch_num++;

            uint8_t *h_unpacked_query  = dev_mem->h_align_unpacked_query;
            uint8_t *h_unpacked_target = dev_mem->h_align_unpacked_target;

            size_t total_query_bytes = 0, total_target_bytes = 0;
            uint32_t max_query_len = 0;
            const uint8_t N_BASE = 4;

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

                memcpy(h_unpacked_query  + total_query_bytes,
                       seq_buffer + tasks[task_idx].qseq_offset, qlen);
                memcpy(h_unpacked_target + total_target_bytes,
                       seq_buffer + tasks[task_idx].tseq_offset, tlen);

                for (int j = qlen; j < (int)qlen_aligned; j++)
                    h_unpacked_query[total_query_bytes + j] = N_BASE;
                for (int j = tlen; j < (int)tlen_aligned; j++)
                    h_unpacked_target[total_target_bytes + j] = N_BASE;

                total_query_bytes  += qlen_aligned;
                total_target_bytes += tlen_aligned;

                if ((uint32_t)qlen > max_query_len) max_query_len = qlen;
            }

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

            // Short phase: bt_p pool is the shared arena allocation.
            size_t bt_p_total_bytes = (size_t)n_concurrent_blocks *
                                      dev_mem->max_align_backtrack_size;

            size_t batch_max_backtrack_size = current_max_backtrack_size;
            size_t batch_max_antidiag       = current_max_antidiag;
            int   *d_bt_off_batch           = d_backtrack_off;
            int   *d_bt_off_end_batch       = d_backtrack_off_end;
            int    max_slots_cap            = n_concurrent_blocks;
            uint8_t *d_bt_p_batch           = d_backtrack_p;

            if (phase == 1) {
                // Pick the LARGER of dedicated and arena bt_p pools to avoid OOB on mismatched dispatch.
                size_t arena_bytes = dev_mem->long_arena_bt_p_bytes;
                size_t dedi_bytes  = (dev_mem->d_align_backtrack_p_long != nullptr)
                                     ? dev_mem->long_bt_p_pool_bytes : 0;
                if (dedi_bytes > arena_bytes) {
                    d_bt_p_batch     = dev_mem->d_align_backtrack_p_long;
                    bt_p_total_bytes = dedi_bytes;
                } else if (arena_bytes > 0) {
                    d_bt_p_batch     = d_backtrack_p;
                    bt_p_total_bytes = arena_bytes;
                }
            }

            int    diag_actual_max_qlen = 0;
            int    diag_actual_max_tlen = 0;
            size_t diag_actual_antidiag = 0;
            int    batch_max_tlen       = 0;

            if (phase == 1) {
                size_t actual_max_antidiag = 1;
                for (int i = 0; i < batch_size; i++) {
                    int tidx = current_task_indices[batch_start + i];
                    int ql   = tasks[tidx].qlen;
                    int tl   = tasks[tidx].tlen;
                    size_t ad = (size_t)ql + (size_t)tl;
                    if (ad > actual_max_antidiag) actual_max_antidiag = ad;
                    if (ql > diag_actual_max_qlen) diag_actual_max_qlen = ql;
                    if (tl > diag_actual_max_tlen) diag_actual_max_tlen = tl;
                    if (tl > batch_max_tlen)       batch_max_tlen       = tl;
                    if (ql > (int)dev_mem->max_align_task_len ||
                        tl > (int)dev_mem->max_align_task_len) {
                        fprintf(stderr,
                            "[FATAL] Long batch task: qlen=%d or tlen=%d exceeds "
                            "max_align_task_len=%zu. Increase max_align_task_len in "
                            "gpu_config.json.\n",
                            ql, tl, dev_mem->max_align_task_len);
                        exit(EXIT_FAILURE);
                    }
                }
                size_t max_antidiag_long = 2 * (size_t)dev_mem->max_align_task_len;
                if (actual_max_antidiag > max_antidiag_long) {
                    fprintf(stderr,
                        "[FATAL] Batch actual anti-diagonal count %zu exceeds bt_off_long "
                        "stride %zu (2 × max_align_task_len=%zu). This should not happen "
                        "if qlen/tlen checks passed — possible bug.\n",
                        actual_max_antidiag, max_antidiag_long, dev_mem->max_align_task_len);
                    exit(EXIT_FAILURE);
                }
                diag_actual_antidiag = actual_max_antidiag;

                // Per-slot bt_p stride: use the largest task's actual bt_stride (not the
                // independent max_antidiag × max_n_col product which overcounts).
                batch_max_backtrack_size = bt_stride0_val;

                // IMPORTANT: batch_max_antidiag is the per-slot stride for backtrack_off.
                // Must match the allocation stride (max_antidiag_long) of d_align_backtrack_off_long.
                batch_max_antidiag = max_antidiag_long;

                d_bt_off_batch     = dev_mem->d_align_backtrack_off_long;
                d_bt_off_end_batch = dev_mem->d_align_backtrack_off_end_long;
                max_slots_cap      = dev_mem->n_long_concurrent_slots;
            }

            // Safety guard: skip batch if largest task's bt_stride exceeds available pool.
            if (phase == 1 && bt_stride0_val > bt_p_total_bytes) {
                fprintf(stderr,
                    "[ERROR] Long batch [%d] largest task bt_stride=%.2f MB but "
                    "available pool is only %.2f MB.  Skipping batch (tasks will "
                    "fall back to CPU).\n",
                    phase_batch_num,
                    bt_stride0_val / (1024.0*1024.0),
                    bt_p_total_bytes / (1024.0*1024.0));
                for (int i = 0; i < batch_size; i++) {
                    int tidx = current_task_indices[batch_start + i];
                    tasks[tidx].zdropped = 1;
                    tasks[tidx].score    = KSW_NEG_INF;
                    tasks[tidx].n_cigar  = 0;
                }
                tasks_processed_in_phase += batch_size;
                total_tasks_processed    += batch_size;
                continue;
            }

            // Long phase: pool_cap0 = bt_p / bt_stride0 (largest task in batch).
            // Short phase: bt_p / current_max_backtrack_size (uniform allocation).
            size_t max_slots_this_phase = (phase == 1)
                ? pool_cap0
                : ((batch_max_backtrack_size > 0)
                   ? (bt_p_total_bytes / batch_max_backtrack_size)
                   : (size_t)n_concurrent_blocks);

            int phase_concurrent_slots = max_slots_cap;
            if ((size_t)phase_concurrent_slots > max_slots_this_phase)
                phase_concurrent_slots = (int)max_slots_this_phase;
            if (phase_concurrent_slots > batch_size)
                phase_concurrent_slots = batch_size;
            if (phase_concurrent_slots < 1) phase_concurrent_slots = 1;

            auto t_start = std::chrono::steady_clock::now();
            cudaMemsetAsync(d_task_counter, 0, sizeof(int), align_stream);

            {
                int parallel_threads = 32;

#if USE_SHARED_LONG_KERNEL
                /* Query device's per-block shared-mem opt-in cap once.
                 * V100 (CC 7.0) = 96 KB; A100 (CC 8.0) = 163 KB. */
                static int s_device_shared_cap = 0;
                if (s_device_shared_cap == 0) {
                    cudaDeviceGetAttribute(&s_device_shared_cap,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                    if (s_device_shared_cap <= 0) s_device_shared_cap = 48 * 1024;
                }
                /* Shared-mem long kernel dispatch (long phase only).
                 * Triggers when: phase==1, bt_p is the bottleneck (slots < 100),
                 * and batch max_tlen fits a shared variant.
                 * FULL  variant (6 deltas): needs 6*(max_tlen+1) ≤ shared cap.
                 * PARTIAL variant (3 hottest): needs 3*(max_tlen+1) ≤ shared cap. */
                int use_shared_variant = 0;
                size_t shared_bytes_full    = (size_t)6 * (size_t)(batch_max_tlen + 1);
                size_t shared_bytes_partial = (size_t)3 * (size_t)(batch_max_tlen + 1);
                if (phase == 1 &&
                    batch_max_tlen > 0 &&
                    phase_concurrent_slots < 100) {
                    if (shared_bytes_full <= (size_t)s_device_shared_cap)
                        use_shared_variant = 1;
                    else if (shared_bytes_partial <= (size_t)s_device_shared_cap)
                        use_shared_variant = 2;
                }
                if (use_shared_variant > 0) {
                    static bool s_shared_attr_set = false;
                    if (!s_shared_attr_set) {
                        cudaFuncSetAttribute(ksw_long_shared_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            s_device_shared_cap);
                        cudaFuncSetAttribute(ksw_long_shared3_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            s_device_shared_cap);
                        s_shared_attr_set = true;
                    }
                    if (use_shared_variant == 1) {
                        ksw_long_shared_kernel<<<phase_concurrent_slots, parallel_threads,
                                                 shared_bytes_full, align_stream>>>(
                            d_task_counter,
                            d_packed_query, d_packed_target,
                            d_query_lens, d_target_lens,
                            d_query_offsets, d_target_offsets,
                            (gasal_res_t*)device_res, d_mat,
                            d_bt_p_batch, d_bt_off_batch, d_bt_off_end_batch,
                            (int)batch_max_backtrack_size, (int)batch_max_antidiag,
                            d_ksw_temp_buffer, d_flag, d_bw,
                            ksw_temp_per_task, batch_size, 5,
                            opt->zdrop, opt->end_bonus,
                            cigar_buffer ? d_cigar_buffer  : NULL,
                            cigar_buffer ? d_cigar_lengths : NULL,
                            (int)current_max_cigar_len,
                            batch_max_tlen);
                    } else {
                        ksw_long_shared3_kernel<<<phase_concurrent_slots, parallel_threads,
                                                  shared_bytes_partial, align_stream>>>(
                            d_task_counter,
                            d_packed_query, d_packed_target,
                            d_query_lens, d_target_lens,
                            d_query_offsets, d_target_offsets,
                            (gasal_res_t*)device_res, d_mat,
                            d_bt_p_batch, d_bt_off_batch, d_bt_off_end_batch,
                            (int)batch_max_backtrack_size, (int)batch_max_antidiag,
                            d_ksw_temp_buffer, d_flag, d_bw,
                            ksw_temp_per_task, batch_size, 5,
                            opt->zdrop, opt->end_bonus,
                            cigar_buffer ? d_cigar_buffer  : NULL,
                            cigar_buffer ? d_cigar_lengths : NULL,
                            (int)current_max_cigar_len,
                            batch_max_tlen);
                    }
                } else
#endif
                {
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
                        5,
                        opt->zdrop,
                        opt->end_bonus,
                        cigar_buffer ? d_cigar_buffer  : NULL,
                        cigar_buffer ? d_cigar_lengths : NULL,
                        (int)current_max_cigar_len,
                        0  // task_id_base: short phase always starts at 0
                    );
                }
            }

            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel launch failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }

            if (cigar_buffer) {
                // Step A: exclusive prefix sum over cigar lengths → compact offsets
                cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_tmp_size,
                                              d_cigar_lengths, (int*)d_compact_offsets,
                                              batch_size, align_stream);

                // Step B: scatter stride CIGAR → compact CIGAR
                compact_cigar_kernel<<<batch_size, 256, 0, align_stream>>>(
                    d_cigar_buffer, d_compact_cigar, d_compact_offsets,
                    d_cigar_lengths, (int)current_max_cigar_len
                );

                // Step C: fix CIGAR in-place + compute alignment stats
                // Sequences still valid in d_unpacked_query/target (same stream, not overwritten)
                gpu_fix_cigar_and_stats<<<batch_size, 1, 0, align_stream>>>(
                    d_compact_cigar, d_compact_offsets, d_cigar_lengths,
                    d_unpacked_query, d_unpacked_target,
                    d_query_offsets, d_target_offsets,
                    d_mat, opt->q, opt->e, !(opt->flag & MM_F_SR),
                    d_blen, d_mlen, d_n_ambi, d_dp_max, d_gpu_stats_valid,
                    batch_size
                );
            }

            cudaStreamSynchronize(align_stream);
            auto t_end = std::chrono::steady_clock::now();
            double wall_sec = std::chrono::duration<double>(t_end - t_start).count();
            s_ksw_wall_total_sec += wall_sec;
            double gpu_ms = wall_sec * 1000.0;

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
            cudaMemcpyAsync(h_scores,      d_scores,      batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_query_ends,  d_query_ends,  batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_target_ends, d_target_ends, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mqe,   d_mqe,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mqe_t, d_mqe_t, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mte,   d_mte,   batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_mte_q, d_mte_q, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);
            cudaMemcpyAsync(h_zdropped, d_zdropped, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, align_stream);

            cudaStreamSynchronize(align_stream);
            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel execution failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }
            cudaCheck();

            // D2H compact CIGAR.
            // d_compact_offsets were computed BEFORE gpu_fix_cigar (which may shrink CIGARs in-place),
            // so the data for each task is still at offset[i] with h_cigar_lengths[i] entries.
            // Copy offsets from GPU to get the true layout, then transfer only the needed extent.
            int total_cigar_ops = 0;
            if (cigar_buffer) {
                for (int i = 0; i < batch_size; i++) {
                    int clen = h_cigar_lengths[i];
                    if (clen < 0 || clen > (int)current_max_cigar_len)
                        h_cigar_lengths[i] = 0;
                }

                cudaMemcpyAsync(h_compact_offsets, d_compact_offsets,
                                batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, align_stream);
                cudaStreamSynchronize(align_stream);

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

            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];

                tasks[task_idx].score = h_scores[i];
                if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].max_q = -1;
                    tasks[task_idx].max_t = -1;
                } else {
                    tasks[task_idx].max_q = h_query_ends[i];
                    tasks[task_idx].max_t = h_target_ends[i];
                }
                tasks[task_idx].mqe   = h_mqe[i];
                tasks[task_idx].mqe_t = h_mqe_t[i];
                tasks[task_idx].mte   = h_mte[i];
                tasks[task_idx].mte_q = h_mte_q[i];

                if (cigar_buffer) {
                    int n_cigar = h_cigar_lengths[i];
                    tasks[task_idx].n_cigar = n_cigar;
                    if (n_cigar > 0 && n_cigar <= tasks[task_idx].max_cigar) {
                        memcpy(cigar_buffer + tasks[task_idx].cigar_offset,
                               h_compact_cigar + h_compact_offsets[i],
                               n_cigar * sizeof(uint32_t));
                    } else if (n_cigar > tasks[task_idx].max_cigar) {
                        tasks[task_idx].n_cigar = 0;
                    }
                } else {
                    tasks[task_idx].n_cigar = 0;
                }

                tasks[task_idx].blen            = h_blen[i];
                tasks[task_idx].mlen            = h_mlen[i];
                tasks[task_idx].n_ambi          = h_n_ambi[i];
                tasks[task_idx].dp_max          = h_dp_max[i];
                tasks[task_idx].gpu_stats_valid = h_gpu_stats_valid[i];

                if (tasks[task_idx].flag & KSW_EZ_RIGHT) {
                    int mq = (tasks[task_idx].flag & KSW_EZ_APPROX_MAX)
                           ? h_query_ends[i]
                           : tasks[task_idx].max_q;
                    tasks[task_idx].reach_end = (mq == tasks[task_idx].qlen - 1);
                } else if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].reach_end =
                        (h_query_ends[i]  == tasks[task_idx].qlen - 1) &&
                        (h_target_ends[i] == tasks[task_idx].tlen - 1);
                } else {
                    tasks[task_idx].reach_end =
                        (tasks[task_idx].max_q == tasks[task_idx].qlen - 1) &&
                        (tasks[task_idx].max_t == tasks[task_idx].tlen - 1);
                }
                tasks[task_idx].zdropped = h_zdropped[i] ? 1 : 0;
            }

            tasks_processed_in_phase += batch_size;
            total_tasks_processed    += batch_size;

            if (phase == 0) {
                char tmp[256];
                snprintf(tmp, sizeof(tmp),
                        "[Info::Short %d]: %d tasks  slots=%d  bt_stride=%.2fMB  (gpu_ms=%.1f)\n",
                        phase_batch_num, batch_size, phase_concurrent_slots,
                        batch_max_backtrack_size / (1024.0 * 1024.0), gpu_ms);
                deferred_short_log += tmp;
            } else {
                if (!deferred_short_log.empty()) {
                    fprintf(stderr, "%s", deferred_short_log.c_str());
                    deferred_short_log.clear();
                }
                PLOG_INFO(stderr, "[Info::Long %d]: %d tasks  slots=%d  bt0=%.2fMB  (gpu_ms=%.1f)\n",
                        phase_batch_num, batch_size, phase_concurrent_slots,
                        batch_max_backtrack_size / (1024.0 * 1024.0), gpu_ms);
            }
        }
    }

    PLOG_INFO(stderr, "[Info] Alignment complete: %d tasks in %d batches\n",
            n_tasks, batch_num);

    free(task_indices_short);
    free(task_indices_long);

    plmem_phase_to_chain(dev_mem);
}
