/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining (batch)
 *
 * Replaces mg_lchain_rmq with GPU voting.  ALL (read, chr-group) pairs
 * across the entire rechain batch are processed in a single GPU pipeline
 * pass with only two H↔D synchronisation points.
 *
 * Algorithm per (read, chr-group) – now executed in one fused batch:
 *   1. [GPU] voting_bin_kernel_batched   – histogram anchor ref_pos into bins
 *   2. [GPU] mark_dilate_kernel_batched  – keep bins with votes >= min_cnt
 *   3. [GPU] seg_start_kernel_batched    – detect first bin of each winning run
 *   4. [GPU] seg_incl_sum_kernel         – assign seg IDs; record n_segs/group
 *   5. [GPU] tag_mark_kernel_batched     – per anchor: keep?, seg_id
 *   6. [GPU] seg_excl_sum_kernel         – output positions; record n_compact/group
 *   7. [GPU] scatter_compact_kernel_batched
 *                                        – compact anchors, count per-seg
 *   8. [CPU] single D2H download: bx/by, n_segs[], n_compact[], seg_cnt[]
 *   9. [CPU] build per-read b[]/u[] with q_pos monotonicity enforcement
 *
 * No CUB / DeviceSegmentedScan dependency – segmented prefix sums are
 * implemented as simple sequential per-group kernels (one GPU thread per
 * group).  This is fast in practice because n_bins per group is small
 * (~50-200) and the kernel is I/O bound, not compute bound.
 *
 * OUTPUT: rd->a = filtered anchor array b[]
 *         rd->u = chain descriptors u[]
 *         rd->n / rd->n_u updated accordingly
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>

#include "mmpriv.h"
#include "kalloc.h"
#include "hipify.cuh"
#include "plvoting.cuh"

/* =========================================================================
 * Device helpers
 * ========================================================================= */

/*
 * find_group – binary search: return group g such that d_off[g] <= idx < d_off[g+1].
 * d_off has n_groups+1 elements; idx is always < d_off[n_groups].
 */
__device__ static inline int find_group(const int32_t *d_off, int n_groups,
                                         int32_t idx)
{
    int lo = 0, hi = n_groups - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (d_off[mid] <= idx) lo = mid;
        else                   hi = mid - 1;
    }
    return lo;
}

/* =========================================================================
 * GPU kernels – batched
 * ========================================================================= */

/*
 * Step 1: voting histogram for all groups.
 *
 * tid indexes into the flat anchor array [0, total_anchors).
 * Group membership is found via binary search on d_anchor_off[0..n_groups].
 */
__global__ void voting_bin_kernel_batched(
        const uint64_t *d_ax,
        const int32_t  *d_anchor_off,  /* [n_groups+1] */
        const int32_t  *d_bin_off,     /* [n_groups+1] */
        const int32_t  *d_ref_min,     /* [n_groups]   */
        int32_t         bin_size,
        int32_t        *d_votes,       /* flat [total_bins], zeroed by caller */
        int             n_groups,
        int             total_anchors)
{
    int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_anchors) return;

    int     g       = find_group(d_anchor_off, n_groups, tid);
    int32_t bin_off = d_bin_off[g];
    int32_t n_bins  = d_bin_off[g + 1] - bin_off;
    int32_t ref_pos = (int32_t)d_ax[tid];
    int32_t bin     = (ref_pos - d_ref_min[g]) / bin_size;
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[bin_off + bin], 1);
}

/*
 * Step 2: mark + dilate for all groups.
 * bid indexes into the flat bin array [0, total_bins).
 */
__global__ void mark_dilate_kernel_batched(
        const int32_t *d_votes,
        int8_t        *d_keep_bin,
        const int32_t *d_bin_off,   /* [n_groups+1] */
        int32_t        min_votes,
        int32_t        gap,
        int            n_groups,
        int            total_bins)
{
    int32_t bid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (bid >= total_bins) return;

    int     g       = find_group(d_bin_off, n_groups, bid);
    int32_t g_start = d_bin_off[g];
    int32_t n_bins  = d_bin_off[g + 1] - g_start;
    int32_t local_i = bid - g_start;

    int32_t lo  = max(0, local_i - gap);
    int32_t hi  = min(n_bins - 1, local_i + gap);
    int8_t  keep = 0;
    for (int32_t k = lo; k <= hi; ++k) {
        if (d_votes[g_start + k] >= min_votes) { keep = 1; break; }
    }
    d_keep_bin[bid] = keep;
}

/*
 * Step 3: segment-start flags for all groups.
 * A bin is a segment start if it is kept AND either it is the first bin
 * of its group or the previous bin is not kept.
 */
__global__ void seg_start_kernel_batched(
        const int8_t  *d_keep_bin,
        int32_t       *d_seg_start,
        const int32_t *d_bin_off,   /* [n_groups+1] */
        int            n_groups,
        int            total_bins)
{
    int32_t bid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (bid >= total_bins) return;

    int     g       = find_group(d_bin_off, n_groups, bid);
    int32_t local_i = bid - d_bin_off[g];

    d_seg_start[bid] = (d_keep_bin[bid] == 1 &&
                        (local_i == 0 || d_keep_bin[bid - 1] == 0)) ? 1 : 0;
}

/*
 * Step 4: segmented inclusive prefix sum (one GPU thread per group).
 * Produces:
 *   d_out[bin_off[g]+i] = sum(d_in[bin_off[g]..bin_off[g]+i])   ∀ i in group g
 *   d_nsegs[g]          = total segment-start count for group g
 */
__global__ void seg_incl_sum_kernel(
        const int32_t *d_in,
        int32_t       *d_out,
        const int32_t *d_off,    /* [n_groups+1] */
        int32_t       *d_nsegs,  /* [n_groups]   */
        int            n_groups)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    int     start   = d_off[g];
    int     n       = d_off[g + 1] - start;
    int32_t running = 0;
    for (int i = 0; i < n; ++i) {
        running     += d_in[start + i];
        d_out[start + i] = running;
    }
    d_nsegs[g] = running;  /* = inclusive sum total = number of segments */
}

/*
 * Step 5: tag each anchor: mark (keep?) and segment ID.
 * tid indexes into the flat anchor array.
 */
__global__ void tag_mark_kernel_batched(
        const uint64_t *d_ax,
        const int32_t  *d_anchor_off,  /* [n_groups+1] */
        const int32_t  *d_bin_off,     /* [n_groups+1] */
        const int32_t  *d_ref_min,     /* [n_groups]   */
        int32_t         bin_size,
        const int8_t   *d_keep_bin,
        const int32_t  *d_seg_id,      /* inclusive prefix sum of seg_start */
        int32_t        *d_mark,
        int32_t        *d_anchor_seg,
        int             n_groups,
        int             total_anchors)
{
    int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_anchors) return;

    int     g       = find_group(d_anchor_off, n_groups, tid);
    int32_t bin_off = d_bin_off[g];
    int32_t n_bins  = d_bin_off[g + 1] - bin_off;
    int32_t ref_pos = (int32_t)d_ax[tid];
    int32_t bin     = (ref_pos - d_ref_min[g]) / bin_size;

    if (bin < 0 || bin >= n_bins || d_keep_bin[bin_off + bin] == 0) {
        d_mark[tid]       = 0;
        d_anchor_seg[tid] = 0;
    } else {
        d_mark[tid]       = 1;
        /* d_seg_id is inclusive prefix sum of d_seg_start → each bin in a
         * segment carries the same running count.  Subtract 1 for 0-based ID. */
        d_anchor_seg[tid] = d_seg_id[bin_off + bin] - 1;
    }
}

/*
 * Step 6: segmented exclusive prefix sum (one GPU thread per group).
 * Produces:
 *   d_out[anchor_off[g]+i] = sum(d_in[anchor_off[g]..anchor_off[g]+i-1])
 *   d_ncompact[g]          = total kept-anchor count for group g
 */
__global__ void seg_excl_sum_kernel(
        const int32_t *d_in,
        int32_t       *d_out,
        const int32_t *d_off,       /* [n_groups+1] */
        int32_t       *d_ncompact,  /* [n_groups]   */
        int            n_groups)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    int     start   = d_off[g];
    int     n       = d_off[g + 1] - start;
    int32_t running = 0;
    for (int i = 0; i < n; ++i) {
        d_out[start + i]  = running;
        running          += d_in[start + i];
    }
    d_ncompact[g] = running;
}

/*
 * Step 7: scatter-compact for all groups.
 *
 * Group g's compacted anchors land at d_bx[anchor_off[g] + d_out_pos[tid]]
 * (worst-case output partitioning: each group has its own slice of d_bx/d_by).
 *
 * Per-segment anchor counts go to d_seg_cnt_flat[bin_off[g] + seg_id]
 * (at most n_bins_g segments per group, so bin_off fits).
 */
__global__ void scatter_compact_kernel_batched(
        const uint64_t *d_ax,
        const uint64_t *d_ay,
        const int32_t  *d_mark,
        const int32_t  *d_out_pos,
        const int32_t  *d_anchor_seg,
        const int32_t  *d_anchor_off,  /* [n_groups+1] */
        const int32_t  *d_bin_off,     /* [n_groups+1] */
        uint64_t       *d_bx,
        uint64_t       *d_by,
        int32_t        *d_seg_cnt_flat,
        int             n_groups,
        int             total_anchors)
{
    int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_anchors || d_mark[tid] == 0) return;

    int     g   = find_group(d_anchor_off, n_groups, tid);
    int32_t out = d_anchor_off[g] + d_out_pos[tid];
    d_bx[out] = d_ax[tid];
    d_by[out] = d_ay[tid];

    int32_t seg = d_anchor_seg[tid];
    atomicAdd(&d_seg_cnt_flat[d_bin_off[g] + seg], 1);
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
 *   2. Runs the GPU voting pipeline (all reads/groups in one pass).
 *   3. Compacts winning anchors into b[] and builds u[] chain descriptors.
 *   4. Stores b[]/u[] in rd->a/rd->u for downstream GPU KSW alignment.
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
    const int     blk            = 256;

    /* ====================================================================
     * Phase 1: enumerate all (read, chr-group) pairs → vg[] descriptors
     * ==================================================================== */

    struct VgDesc {
        int     ri;       /* index into rechain_indices            */
        int32_t a_start;  /* first anchor of this group in rd->a  */
        int32_t a_n;      /* number of anchors                     */
        int32_t ref_min;  /* smallest ref_pos in this group        */
        int32_t n_bins;   /* voting bin count                      */
    };

    int       vg_cap = 64, n_vg = 0;
    VgDesc   *vg     = (VgDesc *)malloc(vg_cap * sizeof(VgDesc));

    for (int ri = 0; ri < n_rechain; ++ri) {
        chain_read_t *rd = &reads[rechain_indices[ri]];
        int64_t n_a = rd->n;
        if (n_a == 0 || rd->a == NULL) continue;
        mm128_t *a = rd->a;

        int64_t chr_start = 0;
        while (chr_start < n_a) {
            int32_t xrev    = (int32_t)(a[chr_start].x >> 32);
            int64_t chr_end = chr_start;
            while (chr_end < n_a && (int32_t)(a[chr_end].x >> 32) == xrev)
                ++chr_end;

            int32_t ref_min  = (int32_t)a[chr_start].x;
            int32_t ref_max  = (int32_t)a[chr_end - 1].x;
            int32_t ref_span = ref_max - ref_min + 1;
            int32_t n_bins   = (ref_span + bin_size - 1) / bin_size + 1;

            if (n_vg == vg_cap) {
                vg_cap *= 2;
                vg      = (VgDesc *)realloc(vg, vg_cap * sizeof(VgDesc));
            }
            vg[n_vg].ri      = ri;
            vg[n_vg].a_start = (int32_t)chr_start;
            vg[n_vg].a_n     = (int32_t)(chr_end - chr_start);
            vg[n_vg].ref_min = ref_min;
            vg[n_vg].n_bins  = n_bins;
            ++n_vg;

            chr_start = chr_end;
        }
    }

    if (n_vg == 0) { free(vg); return; }

    /* ====================================================================
     * Phase 2: compute flat offsets for bins and anchors
     * ==================================================================== */

    int32_t *bin_off    = (int32_t *)malloc((n_vg + 1) * sizeof(int32_t));
    int32_t *anchor_off = (int32_t *)malloc((n_vg + 1) * sizeof(int32_t));
    int32_t *ref_min_h  = (int32_t *)malloc(n_vg       * sizeof(int32_t));

    bin_off[0] = anchor_off[0] = 0;
    for (int g = 0; g < n_vg; ++g) {
        bin_off[g + 1]    = bin_off[g]    + vg[g].n_bins;
        anchor_off[g + 1] = anchor_off[g] + vg[g].a_n;
        ref_min_h[g]      = vg[g].ref_min;
    }
    int total_bins    = bin_off[n_vg];
    int total_anchors = anchor_off[n_vg];

    /* ====================================================================
     * Phase 3: pack all anchors into flat host arrays, then H→D upload
     * ==================================================================== */

    uint64_t *h_ax = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    uint64_t *h_ay = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    for (int g = 0; g < n_vg; ++g) {
        chain_read_t *rd = &reads[rechain_indices[vg[g].ri]];
        mm128_t      *a  = rd->a + vg[g].a_start;
        int           off = anchor_off[g];
        for (int j = 0; j < vg[g].a_n; ++j) {
            h_ax[off + j] = a[j].x;
            h_ay[off + j] = a[j].y;
        }
    }

    uint64_t *d_ax, *d_ay;
    cudaMalloc(&d_ax, total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_ay, total_anchors * sizeof(uint64_t));
    cudaMemcpy(d_ax, h_ax, total_anchors * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, h_ay, total_anchors * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    free(h_ax);
    free(h_ay);

    /* Upload per-group metadata */
    int32_t *d_bin_off, *d_anchor_off_d, *d_ref_min;
    cudaMalloc(&d_bin_off,      (n_vg + 1) * sizeof(int32_t));
    cudaMalloc(&d_anchor_off_d, (n_vg + 1) * sizeof(int32_t));
    cudaMalloc(&d_ref_min,       n_vg      * sizeof(int32_t));
    cudaMemcpy(d_bin_off,      bin_off,    (n_vg + 1) * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor_off_d, anchor_off, (n_vg + 1) * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_min,      ref_min_h,   n_vg      * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    free(ref_min_h);

    /* ====================================================================
     * Phase 4: full GPU pipeline – zero intermediate D→H syncs
     * ==================================================================== */

    /* -- Bin-level arrays -- */
    int32_t *d_votes, *d_seg_start, *d_seg_id;
    int8_t  *d_keep_bin;
    cudaMalloc(&d_votes,     total_bins * sizeof(int32_t));
    cudaMalloc(&d_keep_bin,  total_bins * sizeof(int8_t));
    cudaMalloc(&d_seg_start, total_bins * sizeof(int32_t));
    cudaMalloc(&d_seg_id,    total_bins * sizeof(int32_t));
    cudaMemset(d_votes, 0, total_bins * sizeof(int32_t));

    /* -- Per-group result scalars -- */
    int32_t *d_nsegs, *d_ncompact;
    cudaMalloc(&d_nsegs,    n_vg * sizeof(int32_t));
    cudaMalloc(&d_ncompact, n_vg * sizeof(int32_t));

    /* -- Anchor-level arrays -- */
    int32_t *d_mark, *d_anchor_seg, *d_out_pos;
    cudaMalloc(&d_mark,       total_anchors * sizeof(int32_t));
    cudaMalloc(&d_anchor_seg, total_anchors * sizeof(int32_t));
    cudaMalloc(&d_out_pos,    total_anchors * sizeof(int32_t));

    /* -- Output: worst-case (all anchors kept per group, partition by group) -- */
    uint64_t *d_bx, *d_by;
    int32_t  *d_seg_cnt_flat;
    cudaMalloc(&d_bx,           total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_by,           total_anchors * sizeof(uint64_t));
    /* seg_cnt_flat: group g uses [bin_off[g], bin_off[g]+n_segs_g), fits in total_bins */
    cudaMalloc(&d_seg_cnt_flat, total_bins    * sizeof(int32_t));
    cudaMemset(d_seg_cnt_flat, 0, total_bins  * sizeof(int32_t));

    /* Step 1: voting histogram */
    {
        int grd = (total_anchors + blk - 1) / blk;
        voting_bin_kernel_batched<<<grd, blk>>>(
            d_ax, d_anchor_off_d, d_bin_off, d_ref_min,
            bin_size, d_votes, n_vg, total_anchors);
    }

    /* Step 2: mark + dilate */
    {
        int grd = (total_bins + blk - 1) / blk;
        mark_dilate_kernel_batched<<<grd, blk>>>(
            d_votes, d_keep_bin, d_bin_off,
            min_votes, merge_gap_bins, n_vg, total_bins);
    }
    cudaFree(d_votes);

    /* Step 3: segment-start flags */
    {
        int grd = (total_bins + blk - 1) / blk;
        seg_start_kernel_batched<<<grd, blk>>>(
            d_keep_bin, d_seg_start, d_bin_off, n_vg, total_bins);
    }

    /* Step 4: segmented inclusive prefix sum → seg IDs + per-group n_segs */
    {
        int grd = (n_vg + blk - 1) / blk;
        seg_incl_sum_kernel<<<grd, blk>>>(
            d_seg_start, d_seg_id, d_bin_off, d_nsegs, n_vg);
    }
    cudaFree(d_seg_start);

    /* Step 5: tag anchors (mark + segment ID) */
    {
        int grd = (total_anchors + blk - 1) / blk;
        tag_mark_kernel_batched<<<grd, blk>>>(
            d_ax, d_anchor_off_d, d_bin_off, d_ref_min,
            bin_size, d_keep_bin, d_seg_id,
            d_mark, d_anchor_seg, n_vg, total_anchors);
    }
    cudaFree(d_keep_bin);
    cudaFree(d_seg_id);

    /* Step 6: segmented exclusive prefix sum → output positions + n_compact */
    {
        int grd = (n_vg + blk - 1) / blk;
        seg_excl_sum_kernel<<<grd, blk>>>(
            d_mark, d_out_pos, d_anchor_off_d, d_ncompact, n_vg);
    }

    /* Step 7: scatter compact */
    {
        int grd = (total_anchors + blk - 1) / blk;
        scatter_compact_kernel_batched<<<grd, blk>>>(
            d_ax, d_ay,
            d_mark, d_out_pos, d_anchor_seg,
            d_anchor_off_d, d_bin_off,
            d_bx, d_by, d_seg_cnt_flat,
            n_vg, total_anchors);
    }
    cudaFree(d_ax);  cudaFree(d_ay);
    cudaFree(d_mark); cudaFree(d_out_pos); cudaFree(d_anchor_seg);
    cudaFree(d_bin_off); cudaFree(d_anchor_off_d); cudaFree(d_ref_min);

    /* ====================================================================
     * Phase 5: single D→H download
     * ==================================================================== */

    uint64_t *h_bx          = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    uint64_t *h_by          = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    int32_t  *h_nsegs       = (int32_t  *)malloc(n_vg          * sizeof(int32_t));
    int32_t  *h_ncompact    = (int32_t  *)malloc(n_vg          * sizeof(int32_t));
    int32_t  *h_seg_cnt_flat= (int32_t  *)malloc(total_bins    * sizeof(int32_t));

    cudaMemcpy(h_bx,           d_bx,           total_anchors * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_by,           d_by,           total_anchors * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nsegs,        d_nsegs,        n_vg          * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ncompact,     d_ncompact,     n_vg          * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_seg_cnt_flat, d_seg_cnt_flat, total_bins    * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_bx); cudaFree(d_by);
    cudaFree(d_nsegs); cudaFree(d_ncompact);
    cudaFree(d_seg_cnt_flat);

    /* ====================================================================
     * Phase 6: build per-read b[]/u[] output
     *
     * Groups in vg[] are ordered by (ri, chr-group).  We iterate in order,
     * accumulating output for each read.
     *
     * Anchors in h_bx/h_by[anchor_off[g]..anchor_off[g]+n_compact_g-1]
     * are in ref_pos order (inherited from the radix sort), so segments
     * are contiguous and appear in increasing ref_pos order.  The
     * segment-count array h_seg_cnt_flat[bin_off[g]+s] tells how many
     * anchors belong to segment s of group g.
     * ==================================================================== */

    {
        int g = 0;
        while (g < n_vg) {
            int           ri  = vg[g].ri;
            int           idx = rechain_indices[ri];
            chain_read_t *rd  = &reads[idx];

            /* Free old anchor array (data is now on GPU / in h_bx/h_by) */
            kfree(km, rd->a);
            rd->a   = NULL; rd->n   = 0;
            rd->u   = NULL; rd->n_u = 0;

            /* Worst-case output size = sum of a_n for all groups of this read */
            int read_n_a = 0;
            for (int gg = g; gg < n_vg && vg[gg].ri == ri; ++gg)
                read_n_a += vg[gg].a_n;

            mm128_t  *b;
            uint64_t *u_buf;
            KMALLOC(km, b,     read_n_a);
            KMALLOC(km, u_buf, read_n_a);
            int64_t n_b = 0;
            int     n_u = 0;

            /* Process each chr-group of this read */
            while (g < n_vg && vg[g].ri == ri) {
                int32_t n_segs_g    = h_nsegs[g];
                int32_t n_compact_g = h_ncompact[g];

                if (n_segs_g == 0 || n_compact_g == 0) { ++g; continue; }

                int32_t seg_base    = bin_off[g];    /* index into h_seg_cnt_flat */
                int64_t anchor_base = anchor_off[g]; /* base in h_bx/h_by         */
                int64_t anchor_cursor = anchor_base;

                /* ---------------------------------------------------------- *
                 * Step 8: build b[] and u[] with query_pos monotonicity.
                 *
                 * Within each segment anchors are in ref_pos order but may
                 * have non-monotone query_pos.  Classify each violation as:
                 *   Isolated  – bad anchor at k but k+1 is fine: discard k.
                 *   Sustained – bad at both k and k+1 (direction reversal):
                 *               close sub-chain, open new one at k.
                 * ---------------------------------------------------------- */
                for (int32_t s = 0; s < n_segs_g; ++s) {
                    int32_t cnt = h_seg_cnt_flat[seg_base + s];
                    if (cnt == 0) continue;

                    int32_t  last_q  = INT32_MIN;
                    uint64_t sub_sc  = 0;
                    int32_t  sub_cnt = 0;

                    for (int32_t k = 0; k < cnt; ++k) {
                        int32_t  qp  = (int32_t)(h_by[anchor_cursor + k]);
                        uint64_t qsp = (h_by[anchor_cursor + k] >> 32) & 0xffU;

                        if (qp > last_q) {
                            b[n_b].x = h_bx[anchor_cursor + k];
                            b[n_b].y = h_by[anchor_cursor + k];
                            ++n_b; sub_sc += qsp; ++sub_cnt; last_q = qp;
                        } else {
                            int8_t sustained =
                                (k + 1 < cnt) &&
                                ((int32_t)(h_by[anchor_cursor + k + 1]) <= last_q);
                            if (!sustained) {
                                /* isolated: silently discard */
                            } else {
                                /* sustained reversal: close current sub-chain */
                                if (sub_cnt > 0)
                                    u_buf[n_u++] = (sub_sc << 32) |
                                                   (uint64_t)sub_cnt;
                                b[n_b].x = h_bx[anchor_cursor + k];
                                b[n_b].y = h_by[anchor_cursor + k];
                                ++n_b; sub_sc = qsp; sub_cnt = 1; last_q = qp;
                            }
                        }
                    }
                    if (sub_cnt > 0)
                        u_buf[n_u++] = (sub_sc << 32) | (uint64_t)sub_cnt;

                    anchor_cursor += cnt;
                }
                ++g;
            } /* end chr-group loop for this read */

            if (n_b == 0 || n_u == 0) {
                kfree(km, b);
                kfree(km, u_buf);
                continue;
            }

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

    free(h_bx); free(h_by);
    free(h_nsegs); free(h_ncompact); free(h_seg_cnt_flat);
    free(bin_off); free(anchor_off);
    free(vg);
}

} /* extern "C" */
