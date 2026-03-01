/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining
 *
 * Replaces mg_lchain_rmq with GPU voting.  Output is b[]/u[] anchor arrays
 * that feed directly into mm_gen_regs → mm_align1_batched (GPU KSW).
 *
 * Algorithm (Genome-on-Diet style)
 * ---------------------------------
 * For each chromosome/strand group of anchors:
 *   1. [GPU] voting_bin_kernel_v2  – histogram anchor ref_pos into bins
 *   2. [GPU] mark_dilate_kernel   – mark bins with votes >= min_cnt, merge gaps
 *   3. [GPU] seg_start_kernel     – detect start of each winning run
 *   4. [GPU] cub::ExclusiveSum    – assign segment IDs to bins
 *   5. [CPU] read n_segs
 *   6. [GPU] tag_mark_kernel      – per anchor: keep?, seg_id
 *   7. [GPU] cub::ExclusiveSum    – output positions in compacted array
 *   8. [CPU] read n_b_chr
 *   9. [GPU] scatter_compact_kernel – compact anchors into d_bx/d_by,
 *                                     atomicAdd per-segment anchor counts
 *  10. [CPU] download d_bx/d_by/d_seg_cnt, build b[] and u[]
 *
 * OUTPUT: rd->a = filtered anchor array b[]
 *         rd->u = chain descriptors u[] (one entry per winning segment,
 *                 score = sum of q_span, lower 32 bits = anchor count)
 *         rd->n / rd->n_u updated accordingly
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>

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
 * Accepts full uint64_t ax; extracts ref_pos as low 32 bits.
 */
__global__ void voting_bin_kernel_v2(const uint64_t *d_ax, int64_t n,
                                     int32_t ref_min, int32_t bin_size,
                                     int32_t n_bins, int32_t *d_votes)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t ref_pos = (int32_t)d_ax[tid];
    int32_t bin = (ref_pos - ref_min) / bin_size;
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[bin], 1);
}

/*
 * mark_dilate_kernel
 * d_keep_bin[i] = 1 if any bin in [max(0,i-gap), min(n_bins-1,i+gap)]
 * has votes >= min_votes.  Implements the Genome-on-Diet small-gap merge.
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
 * d_seg_start[i] = 1 iff bin i is the first bin of a winning run.
 * After exclusive prefix sum, d_seg_id[i] gives the 0-based segment
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
 * d_mark[j]       = d_keep_bin[bin]  (1=keep, 0=discard)
 * d_anchor_seg[j] = d_seg_id[bin]    (segment ID, valid when d_mark=1)
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
 * Scatters kept anchors into compacted d_bx[]/d_by[] and atomically
 * counts how many anchors land in each segment (d_seg_cnt[]).
 *
 * Caller must zero-initialise d_seg_cnt before launching.
 */
__global__ void scatter_compact_kernel(const uint64_t *d_ax, const uint64_t *d_ay,
                                       const int32_t  *d_mark,
                                       const int32_t  *d_out_pos,
                                       const int32_t  *d_anchor_seg,
                                       int64_t n,
                                       uint64_t *d_bx, uint64_t *d_by,
                                       int32_t  *d_seg_cnt)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n || d_mark[j] == 0) return;

    int32_t out   = d_out_pos[j];
    d_bx[out]     = d_ax[j];
    d_by[out]     = d_ay[j];

    int32_t seg   = d_anchor_seg[j];
    atomicAdd(&d_seg_cnt[seg], 1);
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

static void cub_excl_sum(int32_t *d_in, int32_t *d_out, int n)
{
    void   *d_tmp  = nullptr;
    size_t  tmp_sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_sz, d_in, d_out, n);
    cudaMalloc(&d_tmp, tmp_sz);
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz, d_in, d_out, n);
    cudaFree(d_tmp);
}

/* Total count = prefix_sum[last] + flag[last]. */
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

/*
 * plvoting_rechain_batch
 *
 * For each read in rechain_indices:
 *   1. Sorts anchors by ref_pos (via prepare_rechain_anchors).
 *   2. Runs the GPU voting pipeline per chromosome/strand group.
 *   3. Compacts winning anchors into b[] and builds u[] chain descriptors.
 *   4. Stores b[]/u[] in rd->a/rd->u so the read flows into the normal
 *      mm_gen_regs → mm_align1_batched (GPU KSW) path unchanged.
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

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

        /* Output buffers: worst-case n_a anchors, n_a chain entries */
        mm128_t  *b;
        uint64_t *u_buf;
        KMALLOC(km, b,     n_a);
        KMALLOC(km, u_buf, n_a);
        int64_t n_b = 0;
        int     n_u = 0;

        /* ------------------------------------------------------------------ *
         * Process each chromosome/strand group independently.                *
         * ------------------------------------------------------------------ */
        int64_t chr_start = 0;
        while (chr_start < n_a) {
            /* Delimit one chromosome/strand group */
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

            /* Upload ax/ay */
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

            /* Step 1: voting histogram */
            int32_t *d_votes;
            cudaMalloc(&d_votes, n_bins * sizeof(int32_t));
            cudaMemset(d_votes, 0, n_bins * sizeof(int32_t));
            {
                int grd = (int)((chr_n + blk - 1) / blk);
                voting_bin_kernel_v2<<<grd, blk>>>(d_ax_full, chr_n,
                                                   ref_min, bin_size, n_bins,
                                                   d_votes);
            }

            /* Step 2: mark + dilate */
            int8_t *d_keep_bin;
            cudaMalloc(&d_keep_bin, n_bins * sizeof(int8_t));
            {
                int grd = (n_bins + blk - 1) / blk;
                mark_dilate_kernel<<<grd, blk>>>(d_votes, d_keep_bin,
                                                 n_bins, min_votes,
                                                 merge_gap_bins);
            }
            cudaFree(d_votes);

            /* Step 3: segment-start flags → exclusive sum → seg IDs */
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

            /* Step 4: tag anchors (mark + segment ID) */
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

            /* Step 5: exclusive sum on d_mark → output positions */
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

            /* Step 6: scatter anchors + count anchors per segment */
            uint64_t *d_bx, *d_by;
            int32_t  *d_seg_cnt;
            cudaMalloc(&d_bx,      n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_by,      n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_seg_cnt, n_segs  * sizeof(int32_t));
            cudaMemset(d_seg_cnt, 0, n_segs * sizeof(int32_t));
            {
                int grd = (int)((chr_n + blk - 1) / blk);
                scatter_compact_kernel<<<grd, blk>>>(
                    d_ax_full, d_ay_full,
                    d_mark, d_out_pos, d_anchor_seg,
                    chr_n, d_bx, d_by, d_seg_cnt);
            }
            cudaFree(d_ax_full);
            cudaFree(d_ay_full);
            cudaFree(d_mark);
            cudaFree(d_out_pos);
            cudaFree(d_anchor_seg);

            /* Step 7: download compacted anchors and per-segment counts */
            uint64_t *h_bx = (uint64_t *)malloc(n_b_chr * sizeof(uint64_t));
            uint64_t *h_by = (uint64_t *)malloc(n_b_chr * sizeof(uint64_t));
            int32_t  *h_seg_cnt = (int32_t *)malloc(n_segs * sizeof(int32_t));
            cudaMemcpy(h_bx,      d_bx,      n_b_chr * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_by,      d_by,      n_b_chr * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_seg_cnt, d_seg_cnt, n_segs  * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_bx);
            cudaFree(d_by);
            cudaFree(d_seg_cnt);

            /* Step 8: append to master b[] and build u[] entries.
             *
             * b[] is ordered by ref_pos (preserved from the original sorted
             * a[]).  Segments are non-overlapping ref_pos ranges, so segment
             * s's anchors appear as a contiguous block in h_bx/h_by before
             * segment s+1's anchors.  This satisfies mm_gen_regs's assumption
             * that u[i]'s anchors are contiguous in a[].
             *
             * Score proxy = sum of per-anchor q_span (bits 32-39 of ay),
             * matching what compact_a / mg_chain_backtrack use. */
            int64_t anchor_off = 0;
            for (int32_t s = 0; s < n_segs; ++s) {
                int32_t cnt = h_seg_cnt[s];
                if (cnt == 0) continue;

                uint64_t score = 0;
                for (int32_t k = 0; k < cnt; ++k)
                    score += (h_by[anchor_off + k] >> 32) & 0xffU;

                u_buf[n_u++] = (score << 32) | (uint64_t)cnt;
                anchor_off  += cnt;
            }

            /* Copy compacted anchors into the master b[] array. */
            for (int64_t j = 0; j < n_b_chr; ++j) {
                b[n_b + j].x = h_bx[j];
                b[n_b + j].y = h_by[j];
            }
            n_b += n_b_chr;

            free(h_bx);
            free(h_by);
            free(h_seg_cnt);

            chr_start = chr_end;
        } /* end chromosome loop */

        /* Free the consumed anchor array */
        kfree(km, a);
        rd->a   = NULL;
        rd->n   = 0;
        rd->u   = NULL;  /* already freed by prepare_rechain_anchors */
        rd->n_u = 0;

        if (n_b == 0 || n_u == 0) {
            kfree(km, b);
            kfree(km, u_buf);
            continue;
        }

        /* Shrink to actual sizes and store */
        mm128_t *b_final;
        KMALLOC(km, b_final, n_b);
        memcpy(b_final, b, n_b * sizeof(mm128_t));
        kfree(km, b);

        uint64_t *u_final;
        KMALLOC(km, u_final, n_u);
        memcpy(u_final, u_buf, n_u * sizeof(uint64_t));
        kfree(km, u_buf);

        rd->a   = b_final;
        rd->n   = n_b;
        rd->u   = u_final;
        rd->n_u = n_u;

    } /* end per-read loop */
}

} /* extern "C" */
