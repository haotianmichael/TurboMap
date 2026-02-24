/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining
 *
 * Architecture (Genome-on-Diet style, corrected):
 *
 *   INPUT:  raw seed-hit anchors (mm128_t a[], grouped by chromosome/strand)
 *
 *   GPU pipeline per chromosome group:
 *     [GPU] voting_bin_kernel_v2   – histogram anchor ref_pos into bins
 *     [GPU] mark_dilate_kernel     – mark bins with votes >= min_cnt, merge gaps
 *     [GPU] seg_start_kernel       – detect start of each winning run of bins
 *     [GPU] cub::ExclusiveSum      – assign segment IDs to bins
 *     [CPU] read n_segs
 *     [GPU] tag_mark_kernel        – per anchor: mark=keep?, seg_id
 *     [GPU] cub::ExclusiveSum      – output positions in compacted array
 *     [CPU] read n_b (kept anchors)
 *     [GPU] scatter_compact_kernel – compacts anchors AND tracks per-segment
 *                                    rs_min, re_max, qs_min, qe_max via
 *                                    atomicMin / atomicMax
 *     [GPU] build_vt_kernel        – assembles one vt_t per winning segment
 *     [CPU] download d_vt[]
 *
 *   OUTPUT: vt_t regions stored in rd->vt_regions / rd->n_vt.
 *           rd->a, rd->u are left NULL; no b[]/u[] are produced.
 *
 * Critical safety properties:
 *   1. Each vt_t carries the correct rid/rev (from the chromosome group header).
 *   2. rs / re come from actual anchor ref_pos on rid  →  tseq[re-rs] is safe.
 *   3. Cross-chromosome isolation: the chromosome loop enforces same xrev.
 *   4. Voting regions are NEVER merged back into an anchor chain array;
 *      they go directly to mm_voting_align_regions() → KSW → CIGAR concat.
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
 *
 * Accepts the full uint64_t ax value; extracts ref_pos as the low 32 bits.
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
 * d_seg_start[i] = 1 iff bin i is the first bin of a winning run.
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
 *   d_mark[j]       = d_keep_bin[bin]   (1 = keep, 0 = discard)
 *   d_anchor_seg[j] = d_seg_id[bin]     (segment ID, valid when d_mark=1)
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
 * scatter_compact_kernel  (extended)
 *
 * Scatters kept anchors into compacted d_bx[]/d_by[] and simultaneously
 * accumulates per-segment coordinate bounds using atomicMin / atomicMax:
 *
 *   d_seg_rs_min[s]  = min ref_pos           in segment s
 *   d_seg_re_max[s]  = max (ref_pos + 1)     in segment s
 *   d_seg_qs_min[s]  = min (qpos+1 - q_span) in segment s  (query start)
 *   d_seg_qe_max[s]  = max (qpos + 1)        in segment s  (query end)
 *
 * These four arrays are used by build_vt_kernel to construct vt_t regions.
 * Initialise them to INT32_MAX / INT32_MIN before calling this kernel.
 */
__global__ void scatter_compact_kernel(const uint64_t *d_ax, const uint64_t *d_ay,
                                       const int32_t  *d_mark,
                                       const int32_t  *d_out_pos,
                                       const int32_t  *d_anchor_seg,
                                       int64_t n,
                                       uint64_t *d_bx, uint64_t *d_by,
                                       int32_t  *d_seg_rs_min,
                                       int32_t  *d_seg_re_max,
                                       int32_t  *d_seg_qs_min,
                                       int32_t  *d_seg_qe_max)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n || d_mark[j] == 0) return;

    int32_t out    = d_out_pos[j];
    d_bx[out]      = d_ax[j];
    d_by[out]      = d_ay[j];

    int32_t seg    = d_anchor_seg[j];

    /* Reference coordinates */
    int32_t ref_pos = (int32_t)d_ax[j];          /* low 32 bits = ref end (inclusive) */
    atomicMin(&d_seg_rs_min[seg], ref_pos);
    atomicMax(&d_seg_re_max[seg], ref_pos + 1);   /* exclusive end */

    /* Query coordinates */
    int32_t qpos   = (int32_t)d_ay[j];            /* low 32 bits = query end (inclusive) */
    int32_t q_span = (int32_t)((d_ay[j] >> 32) & 0xffU); /* bits 32..39 = minimizer span */
    int32_t qs     = qpos + 1 - q_span;           /* inclusive start */
    atomicMin(&d_seg_qs_min[seg], qs);
    atomicMax(&d_seg_qe_max[seg], qpos + 1);       /* exclusive end */
}

/*
 * build_vt_kernel
 *
 * Constructs one vt_t per winning segment from the per-segment coordinate
 * arrays.  rid and rev are uniform across all segments in this chromosome
 * group, so they are passed as scalars.
 *
 * Key safety invariant:
 *   vt.rs / vt.re come from actual anchor ref_pos values on chromosome rid.
 *   Therefore allocating tseq[vt.re - vt.rs] in mm_voting_align_regions()
 *   is always within the chromosome bounds and never causes a buffer overflow.
 */
__global__ void build_vt_kernel(const int32_t *d_seg_rs_min,
                                const int32_t *d_seg_re_max,
                                const int32_t *d_seg_qs_min,
                                const int32_t *d_seg_qe_max,
                                int32_t n_segs,
                                int32_t rid, int32_t rev,
                                vt_t *d_vt)
{
    int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_segs) return;
    d_vt[s].rid = rid;
    d_vt[s].rev = rev;
    d_vt[s].rs  = d_seg_rs_min[s];
    d_vt[s].re  = d_seg_re_max[s];
    d_vt[s].qs  = d_seg_qs_min[s];
    d_vt[s].qe  = d_seg_qe_max[s];
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

/* Read total count: value = prefix_sum[last] + flag[last]. */
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
 *   2. Runs the GPU voting pipeline per chromosome group.
 *   3. Collects vt_t regions (rid, rev, rs, re, qs, qe).
 *   4. Stores them in rd->vt_regions / rd->n_vt.
 *
 * Critically:
 *   - rd->a is freed (the consumed anchor array).
 *   - rd->u is left NULL, rd->n_u = 0.
 *   - NO b[]/u[] anchor-chain arrays are produced.
 *   - mm_voting_align_regions() in map.c will call KSW directly on each
 *     vt_t region; results are concatenated at the CIGAR level.
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    /* Sort anchors and release old chain descriptors */
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

        /* Worst-case: every anchor becomes its own region (unlikely). */
        vt_t *vt_buf;
        KMALLOC(km, vt_buf, n_a);
        int n_vt = 0;

        /* ------------------------------------------------------------------ *
         * Process each chromosome/strand group independently.                *
         * The grouping key is xrev = (int32_t)(a[j].x >> 32):               *
         *   bit 31 = strand, bits 30..0 ≈ rid.                               *
         * Using xrev ensures cross-chromosome isolation: uint64_t subtraction *
         * of targets with different high words will differ by ≥ 2^32, so     *
         * anchors from different chromosomes never end up in the same group.   *
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

            /* Extract rid and rev from the first anchor in this group.
             * Formula matches mmpriv.h: x = (rev<<63)|(rid<<33)|rpos */
            int32_t rid = (int32_t)(chr_a[0].x << 1 >> 33);
            int32_t rev = (int32_t)(chr_a[0].x >> 63) & 1;

            int32_t ref_min  = (int32_t)chr_a[0].x;
            int32_t ref_max  = (int32_t)chr_a[chr_n - 1].x;
            int32_t ref_span = ref_max - ref_min + 1;
            int32_t n_bins   = (ref_span + bin_size - 1) / bin_size + 1;

            /* ---- Upload ax/ay ---- */
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

            /* ---- Step 6: scatter anchors + accumulate per-segment ranges ----
             *
             * d_seg_rs_min / d_seg_re_max – reference coordinate bounds
             * d_seg_qs_min / d_seg_qe_max – query coordinate bounds
             *
             * Initialised to INT32_MAX / INT32_MIN so atomicMin/Max work
             * correctly even for the first anchor in each segment.
             */
            uint64_t *d_bx, *d_by;
            int32_t  *d_seg_rs_min, *d_seg_re_max;
            int32_t  *d_seg_qs_min, *d_seg_qe_max;
            cudaMalloc(&d_bx,         n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_by,         n_b_chr * sizeof(uint64_t));
            cudaMalloc(&d_seg_rs_min, n_segs  * sizeof(int32_t));
            cudaMalloc(&d_seg_re_max, n_segs  * sizeof(int32_t));
            cudaMalloc(&d_seg_qs_min, n_segs  * sizeof(int32_t));
            cudaMalloc(&d_seg_qe_max, n_segs  * sizeof(int32_t));

            /* Fill sentinel values for atomicMin / atomicMax.
             * Use a single host buffer + one bulk cudaMemcpy per array. */
            {
                int32_t *h_init = (int32_t *)malloc(n_segs * sizeof(int32_t));

                /* min arrays → INT32_MAX */
                for (int s = 0; s < n_segs; ++s) h_init[s] = INT32_MAX;
                cudaMemcpy(d_seg_rs_min, h_init, n_segs * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_seg_qs_min, h_init, n_segs * sizeof(int32_t),
                           cudaMemcpyHostToDevice);

                /* max arrays → INT32_MIN */
                for (int s = 0; s < n_segs; ++s) h_init[s] = INT32_MIN;
                cudaMemcpy(d_seg_re_max, h_init, n_segs * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_seg_qe_max, h_init, n_segs * sizeof(int32_t),
                           cudaMemcpyHostToDevice);

                free(h_init);
            }

            {
                int grd = (int)((chr_n + blk - 1) / blk);
                scatter_compact_kernel<<<grd, blk>>>(
                    d_ax_full, d_ay_full,
                    d_mark, d_out_pos, d_anchor_seg,
                    chr_n,
                    d_bx, d_by,
                    d_seg_rs_min, d_seg_re_max,
                    d_seg_qs_min, d_seg_qe_max);
            }
            /* d_bx / d_by are discarded: we no longer output anchor arrays */
            cudaFree(d_bx);
            cudaFree(d_by);
            cudaFree(d_ax_full);
            cudaFree(d_ay_full);
            cudaFree(d_mark);
            cudaFree(d_out_pos);
            cudaFree(d_anchor_seg);

            /* ---- Step 7: build vt_t regions ---- */
            vt_t *d_vt;
            cudaMalloc(&d_vt, n_segs * sizeof(vt_t));
            {
                int grd = (n_segs + blk - 1) / blk;
                build_vt_kernel<<<grd, blk>>>(
                    d_seg_rs_min, d_seg_re_max,
                    d_seg_qs_min, d_seg_qe_max,
                    n_segs, rid, rev, d_vt);
            }
            cudaFree(d_seg_rs_min);
            cudaFree(d_seg_re_max);
            cudaFree(d_seg_qs_min);
            cudaFree(d_seg_qe_max);

            /* ---- Step 8: download vt_t regions ---- */
            cudaMemcpy(vt_buf + n_vt, d_vt,
                       n_segs * sizeof(vt_t),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_vt);
            n_vt += n_segs;

            chr_start = chr_end;
        } /* end chromosome loop */

        /* Free the consumed anchor array */
        kfree(km, a);
        rd->a   = NULL;
        rd->n   = 0;
        rd->u   = NULL;   /* already freed by prepare_rechain_anchors */
        rd->n_u = 0;

        if (n_vt == 0) {
            kfree(km, vt_buf);
            rd->vt_regions = NULL;
            rd->n_vt       = 0;
            continue;
        }

        /* Shrink vt_buf from worst-case n_a to actual n_vt */
        vt_t *vt_final;
        KMALLOC(km, vt_final, n_vt);
        memcpy(vt_final, vt_buf, n_vt * sizeof(vt_t));
        kfree(km, vt_buf);

        rd->vt_regions = vt_final;
        rd->n_vt       = n_vt;

    } /* end per-read loop */

#ifdef DEBUG_PRINT
    fprintf(stderr,
            "[Info::%s] voting re-chained %d reads "
            "(bin_size=%d, min_votes=%d, merge_gap_bins=%d) "
            "→ vt_t regions only, no anchor arrays produced\n",
            __func__, n_rechain, bin_size, min_votes, merge_gap_bins);
#endif
}

} /* extern "C" */
