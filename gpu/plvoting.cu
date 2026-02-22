/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining
 *
 * Full GPU pipeline (Genome-on-Diet style, no CPU DP):
 *
 *   [GPU] voting_bin_kernel_v2
 *         Upload uint64_t ax/ay → vote into histogram d_votes[]
 *   [GPU] mark_dilate_kernel
 *         d_keep_bin[i] = 1 if any bin in [i-gap, i+gap] has votes >= min_votes
 *   [GPU] seg_start_kernel
 *         d_seg_start[i] = 1 at the start of each winning run
 *   [GPU/CUB] cub::DeviceScan::ExclusiveSum(d_seg_start) → d_seg_id[]
 *         Assigns a 0-based segment ID to every bin
 *   [CPU] tiny sync: read n_segs (2 ints from device)
 *   [GPU] tag_mark_kernel
 *         For each anchor: d_mark[j]=1 if in winning bin, d_anchor_seg[j]=segment ID
 *   [GPU/CUB] cub::DeviceScan::ExclusiveSum(d_mark) → d_out_pos[]
 *         Output position in compacted b[] for each kept anchor
 *   [CPU] tiny sync: read n_b (2 ints from device)
 *   [GPU] scatter_compact_kernel
 *         Scatter kept ax/ay to d_bx[]/d_by[] at d_out_pos[j]
 *         atomicAdd d_seg_count[seg] and d_seg_qspan[seg]
 *   [GPU] build_u_kernel
 *         d_u[s] = (d_seg_qspan[s] << 32) | d_seg_count[s]
 *   [CPU] Download d_bx[], d_by[], d_u[] → append to b[], u_buf[]
 *
 * Each winning voting segment becomes one chain directly.
 * No mg_lchain_dp is called.  Output b[]/u[] goes to mm_gen_regs → GPU KSW.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <cub/cub.cuh>

#include "mmpriv.h"
#include "kalloc.h"
#include "hipify.cuh"
#include "plvoting.cuh"

/* =========================================================================
 * GPU kernels
 * ========================================================================= */

/*
 * voting_bin_kernel_v2
 *
 * Accepts the full uint64_t ax value; extracts ref_pos as the low 32 bits.
 * Avoids the extra int32_t upload needed by the original voting_bin_kernel.
 */
__global__ void voting_bin_kernel_v2(const uint64_t *d_ax, int64_t n,
                                     int32_t ref_min, int32_t bin_size,
                                     int32_t n_bins, int32_t *d_votes)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t ref_pos = (int32_t)d_ax[tid];   /* low 32 bits = ref position */
    int32_t bin = (ref_pos - ref_min) / bin_size;
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[bin], 1);
}

/*
 * mark_dilate_kernel
 *
 * For each bin i: d_keep_bin[i] = 1 if any bin in
 * [max(0,i-gap), min(n_bins-1,i+gap)] has votes >= min_votes.
 * This implements the Genome-on-Diet small-gap merge in a single pass.
 */
__global__ void mark_dilate_kernel(const int32_t *d_votes, int8_t *d_keep_bin,
                                   int32_t n_bins, int32_t min_votes,
                                   int32_t gap)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bins) return;
    int32_t lo = max(0, i - gap);
    int32_t hi = min(n_bins - 1, i + gap);
    int8_t keep = 0;
    for (int32_t k = lo; k <= hi; ++k) {
        if (d_votes[k] >= min_votes) { keep = 1; break; }
    }
    d_keep_bin[i] = keep;
}

/*
 * seg_start_kernel
 *
 * d_seg_start[i] = 1 iff bin i is the first bin of a winning run
 * (d_keep_bin[i]=1 and either i==0 or d_keep_bin[i-1]==0).
 * After an exclusive prefix sum, d_seg_id[i] gives the 0-based segment
 * index for bin i (valid only when d_keep_bin[i]=1).
 */
__global__ void seg_start_kernel(const int8_t *d_keep_bin,
                                 int32_t *d_seg_start,
                                 int32_t n_bins)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bins) return;
    d_seg_start[i] = (d_keep_bin[i] == 1 &&
                      (i == 0 || d_keep_bin[i - 1] == 0)) ? 1 : 0;
}

/*
 * tag_mark_kernel
 *
 * For each anchor j:
 *   - Compute its histogram bin from d_ax[j] low 32 bits.
 *   - d_mark[j] = d_keep_bin[bin]  (1 = keep, 0 = discard)
 *   - d_anchor_seg[j] = d_seg_id[bin]  (segment ID, only valid when d_mark=1)
 */
__global__ void tag_mark_kernel(const uint64_t *d_ax, int64_t n,
                                int32_t ref_min, int32_t bin_size,
                                int32_t n_bins,
                                const int8_t  *d_keep_bin,
                                const int32_t *d_seg_id,
                                int32_t *d_mark,
                                int32_t *d_anchor_seg)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int32_t ref_pos = (int32_t)d_ax[j];
    int32_t bin = (ref_pos - ref_min) / bin_size;
    if (bin < 0 || bin >= n_bins || d_keep_bin[bin] == 0) {
        d_mark[j]       = 0;
        d_anchor_seg[j] = 0;
    } else {
        d_mark[j]       = 1;
        d_anchor_seg[j] = d_seg_id[bin];
    }
}

/*
 * scatter_compact_kernel
 *
 * Scatters kept anchors into the compacted output arrays d_bx[]/d_by[],
 * and accumulates per-segment q_span sum and anchor count via atomicAdd.
 *
 * Because chr_a[] is sorted by ref_pos and segments are non-overlapping
 * ref_pos intervals, anchors from the same segment are contiguous in
 * chr_a[], so their output positions in d_bx[]/d_by[] are also contiguous.
 * This ensures each chain (segment) occupies a consecutive slice in b[].
 */
__global__ void scatter_compact_kernel(const uint64_t *d_ax, const uint64_t *d_ay,
                                       const int32_t  *d_mark,
                                       const int32_t  *d_out_pos,
                                       const int32_t  *d_anchor_seg,
                                       int64_t n,
                                       uint64_t *d_bx, uint64_t *d_by,
                                       int32_t  *d_seg_qspan,
                                       int32_t  *d_seg_count)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n || d_mark[j] == 0) return;
    int32_t out = d_out_pos[j];
    d_bx[out] = d_ax[j];
    d_by[out] = d_ay[j];
    int32_t seg    = d_anchor_seg[j];
    int32_t q_span = (int32_t)((d_ay[j] >> 32) & 0xffU); /* bits 32..39 of ay */
    atomicAdd(&d_seg_qspan[seg], q_span);
    atomicAdd(&d_seg_count[seg], 1);
}

/*
 * build_u_kernel
 *
 * Constructs the chain descriptor for each winning segment:
 *   u[s] = (sum_q_span << 32) | anchor_count
 * The score (sum_q_span) is used only for primary/secondary ranking.
 */
__global__ void build_u_kernel(const int32_t *d_seg_qspan,
                               const int32_t *d_seg_count,
                               int32_t n_segs,
                               uint64_t *d_u)
{
    int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_segs) return;
    d_u[s] = ((uint64_t)(uint32_t)d_seg_qspan[s] << 32)
           |  (uint64_t)(uint32_t)d_seg_count[s];
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

/* Run CUB exclusive sum: d_in → d_out, n elements (int32_t). */
static void cub_excl_sum(int32_t *d_in, int32_t *d_out, int n)
{
    void   *d_tmp  = nullptr;
    size_t  tmp_sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_sz, d_in, d_out, n);
    cudaMalloc(&d_tmp, tmp_sz);
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz, d_in, d_out, n);
    cudaFree(d_tmp);
}

/* Read n_segs or n_b: value = prefix[last] + flag[last]. */
static int32_t read_total(const int32_t *d_prefix, const int32_t *d_flag, int n)
{
    int32_t last_prefix, last_flag;
    cudaMemcpy(&last_prefix, d_prefix + (n - 1), sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_flag,   d_flag   + (n - 1), sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    return last_prefix + last_flag;
}

/* =========================================================================
 * Main exported function
 * ========================================================================= */

extern "C" {

void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

    /* Voting parameters */
    const int32_t max_dist       = (opt->max_gap > 0) ? opt->max_gap : 10000;
    const int32_t bin_size       = (max_dist / VOTING_BIN_DIVIDER > 0)
                                   ? (max_dist / VOTING_BIN_DIVIDER) : 1;
    const int32_t min_votes      = (opt->min_cnt > 1) ? opt->min_cnt : 2;
    const int32_t merge_gap_bins = (VOTING_LARGE_GAP + bin_size - 1) / bin_size;

    const int blk = 256;

    for (int ri = 0; ri < n_rechain; ++ri) {
        int idx          = rechain_indices[ri];
        chain_read_t *rd = &reads[idx];

        int64_t  n_a = rd->n;
        mm128_t *a   = rd->a;
        if (n_a == 0 || a == NULL) continue;

        /* Output buffers (CPU): worst-case n_a anchors, n_a chain descriptors */
        mm128_t  *b;
        uint64_t *u_buf;
        KMALLOC(km, b,     n_a);
        KMALLOC(km, u_buf, n_a);
        int64_t n_b = 0;
        int     n_u = 0;

        /* ------------------------------------------------------------------ *
         * Process each chromosome group on GPU.                              *
         * ------------------------------------------------------------------ */
        int64_t chr_start = 0;
        while (chr_start < n_a) {
            /* Delimit chromosome group */
            int32_t xrev    = (int32_t)(a[chr_start].x >> 32);
            int64_t chr_end = chr_start;
            while (chr_end < n_a && (int32_t)(a[chr_end].x >> 32) == xrev)
                ++chr_end;

            int64_t       chr_n = chr_end - chr_start;
            const mm128_t *chr_a = a + chr_start;

            int32_t ref_min  = (int32_t)chr_a[0].x;
            int32_t ref_max  = (int32_t)chr_a[chr_n - 1].x;
            int32_t ref_span = ref_max - ref_min + 1;
            int32_t n_bins   = (ref_span + bin_size - 1) / bin_size + 1;

            /* ---- Upload ax/ay as uint64_t ---- */
            uint64_t *h_ax = (uint64_t *)malloc(chr_n * sizeof(uint64_t));
            uint64_t *h_ay = (uint64_t *)malloc(chr_n * sizeof(uint64_t));
            for (int64_t j = 0; j < chr_n; ++j) {
                h_ax[j] = chr_a[j].x;
                h_ay[j] = chr_a[j].y;
            }
            uint64_t *d_ax_full, *d_ay_full;
            cudaMalloc(&d_ax_full, chr_n * sizeof(uint64_t));
            cudaMalloc(&d_ay_full, chr_n * sizeof(uint64_t));
            cudaMemcpy(d_ax_full, h_ax, chr_n * sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_ay_full, h_ay, chr_n * sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
            free(h_ax);
            free(h_ay);

            /* ---- Step 1: voting histogram ---- */
            int32_t *d_votes;
            cudaMalloc(&d_votes, n_bins * sizeof(int32_t));
            cudaMemset(d_votes, 0, n_bins * sizeof(int32_t));
            {
                int grd = (int)((chr_n + blk - 1) / blk);
                voting_bin_kernel_v2<<<grd, blk>>>(d_ax_full, chr_n,
                                                   ref_min, bin_size, n_bins,
                                                   d_votes);
            }

            /* ---- Step 2: mark + dilate ---- */
            int8_t *d_keep_bin;
            cudaMalloc(&d_keep_bin, n_bins * sizeof(int8_t));
            {
                int grd = (n_bins + blk - 1) / blk;
                mark_dilate_kernel<<<grd, blk>>>(d_votes, d_keep_bin,
                                                 n_bins, min_votes,
                                                 merge_gap_bins);
            }
            cudaFree(d_votes);

            /* ---- Step 3: segment-start flags → exclusive sum → seg IDs ---- */
            int32_t *d_seg_start, *d_seg_id;
            cudaMalloc(&d_seg_start, n_bins * sizeof(int32_t));
            cudaMalloc(&d_seg_id,    n_bins * sizeof(int32_t));
            {
                int grd = (n_bins + blk - 1) / blk;
                seg_start_kernel<<<grd, blk>>>(d_keep_bin, d_seg_start, n_bins);
            }
            cub_excl_sum(d_seg_start, d_seg_id, n_bins);
            int32_t n_segs = read_total(d_seg_id, d_seg_start, n_bins);
            cudaFree(d_seg_start);

            if (n_segs == 0) {
                cudaFree(d_keep_bin);
                cudaFree(d_seg_id);
                cudaFree(d_ax_full);
                cudaFree(d_ay_full);
                chr_start = chr_end;
                continue;
            }

            /* ---- Step 4: tag anchors (mark + segment ID) ---- */
            int32_t *d_mark, *d_anchor_seg;
            cudaMalloc(&d_mark,       chr_n * sizeof(int32_t));
            cudaMalloc(&d_anchor_seg, chr_n * sizeof(int32_t));
            {
                int grd = (int)((chr_n + blk - 1) / blk);
                tag_mark_kernel<<<grd, blk>>>(d_ax_full, chr_n,
                                              ref_min, bin_size, n_bins,
                                              d_keep_bin, d_seg_id,
                                              d_mark, d_anchor_seg);
            }
            cudaFree(d_keep_bin);
            cudaFree(d_seg_id);

            /* ---- Step 5: exclusive sum on d_mark → output positions ---- */
            int32_t *d_out_pos;
            cudaMalloc(&d_out_pos, chr_n * sizeof(int32_t));
            cub_excl_sum(d_mark, d_out_pos, (int)chr_n);
            int32_t n_b_chr = read_total(d_out_pos, d_mark, (int)chr_n);

            if (n_b_chr == 0) {
                cudaFree(d_mark);
                cudaFree(d_anchor_seg);
                cudaFree(d_out_pos);
                cudaFree(d_ax_full);
                cudaFree(d_ay_full);
                chr_start = chr_end;
                continue;
            }

            /* ---- Step 6: scatter anchors, accumulate per-segment stats ---- */
            uint64_t *d_bx, *d_by;
            int32_t  *d_seg_qspan, *d_seg_count;
            cudaMalloc(&d_bx,        n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_by,        n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_seg_qspan, n_segs  * sizeof(int32_t));
            cudaMalloc(&d_seg_count, n_segs  * sizeof(int32_t));
            cudaMemset(d_seg_qspan, 0, n_segs * sizeof(int32_t));
            cudaMemset(d_seg_count, 0, n_segs * sizeof(int32_t));
            {
                int grd = (int)((chr_n + blk - 1) / blk);
                scatter_compact_kernel<<<grd, blk>>>(
                    d_ax_full, d_ay_full,
                    d_mark, d_out_pos, d_anchor_seg,
                    chr_n,
                    d_bx, d_by,
                    d_seg_qspan, d_seg_count);
            }
            cudaFree(d_ax_full);
            cudaFree(d_ay_full);
            cudaFree(d_mark);
            cudaFree(d_out_pos);
            cudaFree(d_anchor_seg);

            /* ---- Step 7: build chain descriptors u[] ---- */
            uint64_t *d_u_gpu;
            cudaMalloc(&d_u_gpu, n_segs * sizeof(uint64_t));
            {
                int grd = (n_segs + blk - 1) / blk;
                build_u_kernel<<<grd, blk>>>(d_seg_qspan, d_seg_count,
                                             n_segs, d_u_gpu);
            }
            cudaFree(d_seg_qspan);
            cudaFree(d_seg_count);

            /* ---- Step 8: download results ---- */
            uint64_t *h_bx = (uint64_t *)malloc(n_b_chr * sizeof(uint64_t));
            uint64_t *h_by = (uint64_t *)malloc(n_b_chr * sizeof(uint64_t));
            uint64_t *h_u  = (uint64_t *)malloc(n_segs  * sizeof(uint64_t));
            cudaMemcpy(h_bx, d_bx,    n_b_chr * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_by, d_by,    n_b_chr * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_u,  d_u_gpu, n_segs  * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_bx);
            cudaFree(d_by);
            cudaFree(d_u_gpu);

            /* ---- Pack into CPU output buffers ---- */
            for (int32_t j = 0; j < n_b_chr; ++j) {
                b[n_b + j].x = h_bx[j];
                b[n_b + j].y = h_by[j];
            }
            for (int32_t s = 0; s < n_segs; ++s)
                u_buf[n_u + s] = h_u[s];
            n_b += n_b_chr;
            n_u += n_segs;

            free(h_bx);
            free(h_by);
            free(h_u);
            chr_start = chr_end;
        } /* end chromosome loop */

        /* Free the original (consumed) anchor array */
        kfree(km, a);
        rd->a = NULL;

        if (n_b == 0 || n_u == 0) {
            kfree(km, b);
            kfree(km, u_buf);
            rd->n   = 0;
            rd->n_u = 0;
            rd->u   = NULL;
            continue;
        }

        /*
         * Shrink b[] from worst-case n_a to actual n_b.
         *
         * CRITICAL: b was allocated with KMALLOC(km, b, n_a), but voting
         * filters out most anchors so n_b << n_a.  Leaving the full-size
         * block in the kalloc pool causes OOM during the subsequent
         * alignment phase when many reads have been re-chained.
         * For a batch with 200M total anchors and 805 re-chained reads,
         * the wasted (n_a - n_b) * 16 bytes can exceed 1.9 GB.
         */
        mm128_t *b_final;
        KMALLOC(km, b_final, n_b);
        memcpy(b_final, b, n_b * sizeof(mm128_t));
        kfree(km, b);

        /* Shrink u_buf to actual size */
        uint64_t *u_final;
        KMALLOC(km, u_final, n_u);
        memcpy(u_final, u_buf, n_u * sizeof(uint64_t));
        kfree(km, u_buf);

        rd->a   = b_final;
        rd->n   = n_b;
        rd->u   = u_final;
        rd->n_u = n_u;
    } /* end per-read loop */

#ifdef DEBUG_PRINT
    fprintf(stderr,
            "[Info::%s] voting re-chained %d reads "
            "(bin_size=%d, min_votes=%d, merge_gap_bins=%d)\n",
            __func__, n_rechain, bin_size, min_votes, merge_gap_bins);
#endif
}

} /* extern "C" */
