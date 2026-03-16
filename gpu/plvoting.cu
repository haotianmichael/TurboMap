/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining (batch)
 *
 * Replaces mg_lchain_rmq with GPU voting.  ALL (read, chr-group) pairs
 * across the entire rechain batch are processed in one or more GPU mini-
 * batch passes, each capped at MEM_BUDGET_BYTES of GPU memory.
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
 *   8. [CPU] D2H download per mini-batch into pre-allocated full-batch host bufs
 *   9. [CPU] build per-read b[]/u[] with q_pos monotonicity enforcement
 *
 * n_bins per group is capped at the group's anchor count (a_n) to avoid
 * enormous bin arrays for sparse groups.  When the cap applies, the bin
 * size is scaled up so every anchor still maps into [0, n_bins) -- no
 * anchor is ever cut off.
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

/* 8 GB GPU memory budget per mini-batch for voting arrays */
#define MEM_BUDGET_BYTES (8ULL << 30)

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
 * Each group uses its own per-group bin size (d_bin_size[g]) so that
 * no anchor falls outside [0, n_bins_g).
 */
__global__ void voting_bin_kernel_batched(
        const uint64_t *d_ax,
        const int32_t  *d_anchor_off,  /* [n_groups+1] */
        const int32_t  *d_bin_off,     /* [n_groups+1] */
        const int32_t  *d_ref_min,     /* [n_groups]   */
        const int32_t  *d_bin_size,    /* [n_groups]   per-group bin size */
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
    int32_t bsz     = d_bin_size[g];
    int32_t bin     = (ref_pos - d_ref_min[g]) / bsz;
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
        const int32_t  *d_bin_size,    /* [n_groups]   per-group bin size */
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
    int32_t bsz     = d_bin_size[g];
    int32_t bin     = (ref_pos - d_ref_min[g]) / bsz;

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
 * run_voting_minibatch – executes steps 1-7 for one mini-batch, then D2H
 * =========================================================================
 *
 * mb_n            : number of groups in this mini-batch
 * mb_total_anchors: total anchors across mini-batch groups
 * mb_total_bins   : total bins across mini-batch groups (after cap)
 * mb_h_ax/ay      : packed host anchor arrays for this mini-batch
 * mb_bin_off[]    : local bin offsets  [mb_n+1], starting from 0
 * mb_anchor_off[] : local anchor offsets [mb_n+1], starting from 0
 * mb_ref_min[]    : per-group ref_min [mb_n]
 * mb_bin_size_h[] : per-group effective bin size [mb_n]
 *
 * Results written into caller's full-batch host buffers at base offsets:
 *   h_bx/by        + base_anchor
 *   h_nsegs/ncompact + mb_g   (group index in the global vg[] array)
 *   h_seg_cnt_flat  + base_bin
 */
static void run_voting_minibatch(
        int             mb_n,
        int32_t         mb_total_anchors,
        int32_t         mb_total_bins,
        const uint64_t *mb_h_ax,
        const uint64_t *mb_h_ay,
        const int32_t  *mb_bin_off,
        const int32_t  *mb_anchor_off,
        const int32_t  *mb_ref_min,
        const int32_t  *mb_bin_size_h,
        int32_t         min_votes,
        int32_t         merge_gap_bins,
        cudaStream_t    stream,
        /* output slices */
        uint64_t       *h_bx_base,
        uint64_t       *h_by_base,
        int32_t        *h_nsegs_base,
        int32_t        *h_ncompact_base,
        int32_t        *h_seg_cnt_flat_base)
{
    const int blk = 256;

    /* ---- upload anchors + metadata ---- */
    uint64_t *d_ax = NULL, *d_ay = NULL;
    int32_t  *d_bin_off_d = NULL, *d_anchor_off_d = NULL, *d_ref_min_d = NULL, *d_bin_size_d = NULL;

    cudaError_t e;
    e = cudaMalloc(&d_ax,           (size_t)mb_total_anchors * sizeof(uint64_t));
    e = cudaMalloc(&d_ay,           (size_t)mb_total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_bin_off_d,    (size_t)(mb_n + 1)       * sizeof(int32_t));
    cudaMalloc(&d_anchor_off_d, (size_t)(mb_n + 1)       * sizeof(int32_t));
    cudaMalloc(&d_ref_min_d,    (size_t)mb_n             * sizeof(int32_t));
    cudaMalloc(&d_bin_size_d,   (size_t)mb_n             * sizeof(int32_t));

    cudaMemcpyAsync(d_ax,           mb_h_ax,         (size_t)mb_total_anchors * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ay,           mb_h_ay,         (size_t)mb_total_anchors * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bin_off_d,    mb_bin_off,      (size_t)(mb_n + 1)       * sizeof(int32_t),  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_anchor_off_d, mb_anchor_off,   (size_t)(mb_n + 1)       * sizeof(int32_t),  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ref_min_d,    mb_ref_min,      (size_t)mb_n             * sizeof(int32_t),  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bin_size_d,   mb_bin_size_h,   (size_t)mb_n             * sizeof(int32_t),  cudaMemcpyHostToDevice, stream);

    /* ---- bin-level arrays ---- */
    int32_t *d_votes, *d_seg_start, *d_seg_id;
    int8_t  *d_keep_bin;
    cudaMalloc(&d_votes,     (size_t)mb_total_bins * sizeof(int32_t));
    cudaMalloc(&d_keep_bin,  (size_t)mb_total_bins * sizeof(int8_t));
    cudaMalloc(&d_seg_start, (size_t)mb_total_bins * sizeof(int32_t));
    cudaMalloc(&d_seg_id,    (size_t)mb_total_bins * sizeof(int32_t));
    cudaMemsetAsync(d_votes, 0,   (size_t)mb_total_bins * sizeof(int32_t), stream);

    /* ---- per-group scalars ---- */
    int32_t *d_nsegs_d, *d_ncompact_d;
    cudaMalloc(&d_nsegs_d,    (size_t)mb_n * sizeof(int32_t));
    cudaMalloc(&d_ncompact_d, (size_t)mb_n * sizeof(int32_t));

    /* ---- anchor-level arrays ---- */
    int32_t *d_mark, *d_anchor_seg, *d_out_pos;
    cudaMalloc(&d_mark,       (size_t)mb_total_anchors * sizeof(int32_t));
    cudaMalloc(&d_anchor_seg, (size_t)mb_total_anchors * sizeof(int32_t));
    cudaMalloc(&d_out_pos,    (size_t)mb_total_anchors * sizeof(int32_t));

    /* ---- output arrays ---- */
    uint64_t *d_bx, *d_by;
    int32_t  *d_seg_cnt_flat_d;
    cudaMalloc(&d_bx,              (size_t)mb_total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_by,              (size_t)mb_total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_seg_cnt_flat_d,  (size_t)mb_total_bins    * sizeof(int32_t));
    cudaMemsetAsync(d_seg_cnt_flat_d, 0,(size_t)mb_total_bins    * sizeof(int32_t), stream);

    /* ---- step 1: voting histogram ---- */
    {
        int grd = (mb_total_anchors + blk - 1) / blk;
        voting_bin_kernel_batched<<<grd, blk, 0, stream>>>(
            d_ax, d_anchor_off_d, d_bin_off_d, d_ref_min_d, d_bin_size_d,
            d_votes, mb_n, mb_total_anchors);
    }

    /* ---- step 2: mark + dilate ---- */
    {
        int grd = (mb_total_bins + blk - 1) / blk;
        mark_dilate_kernel_batched<<<grd, blk, 0, stream>>>(
            d_votes, d_keep_bin, d_bin_off_d,
            min_votes, merge_gap_bins, mb_n, mb_total_bins);
    }
    cudaFree(d_votes);

    /* ---- step 3: segment-start flags ---- */
    {
        int grd = (mb_total_bins + blk - 1) / blk;
        seg_start_kernel_batched<<<grd, blk, 0, stream>>>(
            d_keep_bin, d_seg_start, d_bin_off_d, mb_n, mb_total_bins);
    }

    /* ---- step 4: segmented inclusive prefix sum → seg IDs + n_segs ---- */
    {
        int grd = (mb_n + blk - 1) / blk;
        seg_incl_sum_kernel<<<grd, blk, 0, stream>>>(
            d_seg_start, d_seg_id, d_bin_off_d, d_nsegs_d, mb_n);
    }
    cudaFree(d_seg_start);

    /* ---- step 5: tag anchors (mark + segment ID) ---- */
    {
        int grd = (mb_total_anchors + blk - 1) / blk;
        tag_mark_kernel_batched<<<grd, blk, 0, stream>>>(
            d_ax, d_anchor_off_d, d_bin_off_d, d_ref_min_d, d_bin_size_d,
            d_keep_bin, d_seg_id,
            d_mark, d_anchor_seg, mb_n, mb_total_anchors);
    }
    cudaFree(d_keep_bin);
    cudaFree(d_seg_id);

    /* ---- step 6: segmented exclusive prefix sum → output positions + n_compact ---- */
    {
        int grd = (mb_n + blk - 1) / blk;
        seg_excl_sum_kernel<<<grd, blk, 0, stream>>>(
            d_mark, d_out_pos, d_anchor_off_d, d_ncompact_d, mb_n);
    }

    /* ---- step 7: scatter compact ---- */
    {
        int grd = (mb_total_anchors + blk - 1) / blk;
        scatter_compact_kernel_batched<<<grd, blk, 0, stream>>>(
            d_ax, d_ay,
            d_mark, d_out_pos, d_anchor_seg,
            d_anchor_off_d, d_bin_off_d,
            d_bx, d_by, d_seg_cnt_flat_d,
            mb_n, mb_total_anchors);
    }
    cudaFree(d_ax);  cudaFree(d_ay);
    cudaFree(d_mark); cudaFree(d_out_pos); cudaFree(d_anchor_seg);
    cudaFree(d_bin_off_d); cudaFree(d_anchor_off_d);
    cudaFree(d_ref_min_d); cudaFree(d_bin_size_d);

    /* ---- D2H download into caller's full-batch host arrays ---- */
    {
        cudaError_t err_k = cudaGetLastError();
        if (err_k != cudaSuccess) {
            fprintf(stderr, "[DEBUG] run_voting_minibatch: error AFTER kernels (before D2H): %s\n",
                    cudaGetErrorString(err_k));
        }
    }
    cudaMemcpyAsync(h_bx_base,            d_bx,           (size_t)mb_total_anchors * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_by_base,            d_by,           (size_t)mb_total_anchors * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_nsegs_base,         d_nsegs_d,      (size_t)mb_n             * sizeof(int32_t),  cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_ncompact_base,      d_ncompact_d,   (size_t)mb_n             * sizeof(int32_t),  cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_seg_cnt_flat_base,  d_seg_cnt_flat_d,(size_t)mb_total_bins   * sizeof(int32_t),  cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    {
        cudaError_t err_d2h = cudaGetLastError();
        if (err_d2h != cudaSuccess) {
            fprintf(stderr, "[DEBUG] run_voting_minibatch: error AFTER D2H sync: %s\n",
                    cudaGetErrorString(err_d2h));
        } else {
            fprintf(stderr, "[DEBUG] run_voting_minibatch: completed OK\n");
        }
    }

    cudaFree(d_bx); cudaFree(d_by);
    cudaFree(d_nsegs_d); cudaFree(d_ncompact_d);
    cudaFree(d_seg_cnt_flat_d);
}

/* =========================================================================
 * Main exported function
 * ========================================================================= */

extern "C" {

/*
 * plvoting_rechain_batch
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km,
                            cudaStream_t stream)
{
    if (n_rechain == 0) return;

    fprintf(stderr, "[DEBUG] plvoting_rechain_batch: n_rechain=%d, stream=%p\n",
            n_rechain, (void*)stream);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[DEBUG] plvoting_rechain_batch: STICKY error on entry: %s\n",
                    cudaGetErrorString(err));
    }

    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

    const int32_t max_dist       = (opt->max_gap > 0) ? opt->max_gap : 10000;
    const int32_t bin_size       = (max_dist / VOTING_BIN_DIVIDER > 0)
                                   ? (max_dist / VOTING_BIN_DIVIDER) : 1;
    const int32_t min_votes      = (opt->min_cnt > 1) ? opt->min_cnt : 2;
    const int32_t merge_gap_bins = (VOTING_LARGE_GAP + bin_size - 1) / bin_size;

    /* ====================================================================
     * Phase 1: enumerate all (read, chr-group) pairs → vg[] descriptors
     * ==================================================================== */

    struct VgDesc {
        int     ri;           /* index into rechain_indices            */
        int32_t a_start;      /* first anchor of this group in rd->a  */
        int32_t a_n;          /* number of anchors                     */
        int32_t ref_min;      /* smallest ref_pos in this group        */
        int32_t n_bins;       /* voting bin count (capped at a_n)      */
        int32_t eff_bin_size; /* effective bin size (>= bin_size)      */
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

            int32_t a_n     = (int32_t)(chr_end - chr_start);
            int32_t ref_min = (int32_t)a[chr_start].x;
            int32_t ref_max = (int32_t)a[chr_end - 1].x;
            /* ref_span as used in original n_bins formula (+1 so ref_max maps
             * to bin <= n_bins-1 under integer division)                      */
            int32_t ref_span = ref_max - ref_min + 1;

            /* Compute n_bins and eff_bin_size.
             *
             * Principle: n_bins = min(original_n_bins, a_n) with eff_bin_size
             * scaled up when we apply the cap so that every anchor still maps
             * into [0, n_bins).  No anchor is ever cut off.
             *
             * When capping: eff_bin_size = floor((ref_span-1)/a_n) + 1
             * guarantees (ref_max - ref_min) / eff_bin_size < a_n.
             */
            int32_t n_bins_g, bsz_g;
            if (a_n <= 1) {
                n_bins_g = 1;
                bsz_g    = bin_size;
            } else {
                n_bins_g = (ref_span + bin_size - 1) / bin_size + 1;
                if (n_bins_g > a_n) {
                    n_bins_g = a_n;
                    bsz_g    = (ref_span - 1) / a_n + 1;
                } else {
                    bsz_g = bin_size;
                }
            }

            if (n_vg == vg_cap) {
                vg_cap *= 2;
                vg      = (VgDesc *)realloc(vg, vg_cap * sizeof(VgDesc));
            }
            vg[n_vg].ri           = ri;
            vg[n_vg].a_start      = (int32_t)chr_start;
            vg[n_vg].a_n          = a_n;
            vg[n_vg].ref_min      = ref_min;
            vg[n_vg].n_bins       = n_bins_g;
            vg[n_vg].eff_bin_size = bsz_g;
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

    bin_off[0] = anchor_off[0] = 0;
    for (int g = 0; g < n_vg; ++g) {
        bin_off[g + 1]    = bin_off[g]    + vg[g].n_bins;
        anchor_off[g + 1] = anchor_off[g] + vg[g].a_n;
    }
    int32_t total_bins    = bin_off[n_vg];
    int32_t total_anchors = anchor_off[n_vg];

    /* ====================================================================
     * Phase 3: pre-allocate full-batch host output buffers
     * ==================================================================== */

    uint64_t *h_bx           = (uint64_t *)malloc((size_t)total_anchors * sizeof(uint64_t));
    uint64_t *h_by           = (uint64_t *)malloc((size_t)total_anchors * sizeof(uint64_t));
    int32_t  *h_nsegs        = (int32_t  *)malloc((size_t)n_vg          * sizeof(int32_t));
    int32_t  *h_ncompact     = (int32_t  *)malloc((size_t)n_vg          * sizeof(int32_t));
    int32_t  *h_seg_cnt_flat = (int32_t  *)malloc((size_t)total_bins    * sizeof(int32_t));

    /* ====================================================================
     * Phase 4: mini-batch GPU pipeline
     *
     * Groups are processed in slices; each slice's peak GPU memory usage
     * is bounded by MEM_BUDGET_BYTES.
     *
     * Peak cost per mini-batch (bytes):
     *   anchors : mb_total_anchors * (8+8 upload + 4+4+4 work + 8+8 output) = * 44
     *   bins    : mb_total_bins    * (4+1+4+4 work + 4 seg_cnt)             = * 17
     * ==================================================================== */

    int mb_g = 0;
    while (mb_g < n_vg) {

        /* determine mini-batch end */
        int     mb_end = mb_g;
        int64_t mb_na  = 0, mb_nb = 0;
        while (mb_end < n_vg) {
            int64_t na   = vg[mb_end].a_n;
            int64_t nb   = vg[mb_end].n_bins;
            size_t  cost = (size_t)(mb_na + na) * 44 +
                           (size_t)(mb_nb + nb) * 17;
            if (cost > MEM_BUDGET_BYTES && mb_end > mb_g) break;
            mb_na += na;
            mb_nb += nb;
            ++mb_end;
        }

        int     mb_n              = mb_end - mb_g;
        int32_t mb_total_anchors  = (int32_t)mb_na;
        int32_t mb_total_bins     = (int32_t)mb_nb;
        int32_t base_anchor       = anchor_off[mb_g];
        int32_t base_bin          = bin_off[mb_g];

        /* build local (0-based) offset arrays for this mini-batch */
        int32_t *mb_bin_off    = (int32_t *)malloc((size_t)(mb_n + 1) * sizeof(int32_t));
        int32_t *mb_anchor_off = (int32_t *)malloc((size_t)(mb_n + 1) * sizeof(int32_t));
        int32_t *mb_ref_min    = (int32_t *)malloc((size_t)mb_n       * sizeof(int32_t));
        int32_t *mb_bin_size_h = (int32_t *)malloc((size_t)mb_n       * sizeof(int32_t));

        mb_bin_off[0] = mb_anchor_off[0] = 0;
        for (int i = 0; i < mb_n; ++i) {
            int g = mb_g + i;
            mb_bin_off[i + 1]    = mb_bin_off[i]    + vg[g].n_bins;
            mb_anchor_off[i + 1] = mb_anchor_off[i] + vg[g].a_n;
            mb_ref_min[i]        = vg[g].ref_min;
            mb_bin_size_h[i]     = vg[g].eff_bin_size;
        }

        /* pack mini-batch anchors into flat host arrays */
        uint64_t *mb_h_ax = (uint64_t *)malloc((size_t)mb_total_anchors * sizeof(uint64_t));
        uint64_t *mb_h_ay = (uint64_t *)malloc((size_t)mb_total_anchors * sizeof(uint64_t));
        for (int i = 0; i < mb_n; ++i) {
            int           g   = mb_g + i;
            chain_read_t *rd  = &reads[rechain_indices[vg[g].ri]];
            mm128_t      *a   = rd->a + vg[g].a_start;
            int           off = mb_anchor_off[i];
            for (int j = 0; j < vg[g].a_n; ++j) {
                mb_h_ax[off + j] = a[j].x;
                mb_h_ay[off + j] = a[j].y;
            }
        }

        /* run GPU pipeline and write results into global host buffers */
        run_voting_minibatch(
            mb_n, mb_total_anchors, mb_total_bins,
            mb_h_ax, mb_h_ay,
            mb_bin_off, mb_anchor_off, mb_ref_min, mb_bin_size_h,
            min_votes, merge_gap_bins,
            stream,
            h_bx           + base_anchor,
            h_by           + base_anchor,
            h_nsegs        + mb_g,
            h_ncompact     + mb_g,
            h_seg_cnt_flat + base_bin);

        free(mb_h_ax); free(mb_h_ay);
        free(mb_bin_off); free(mb_anchor_off);
        free(mb_ref_min); free(mb_bin_size_h);

        mb_g = mb_end;
    }

    /* ====================================================================
     * Phase 5: build per-read b[]/u[] output
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

            /* Free old anchor array (data is now in h_bx/h_by) */
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

                int32_t seg_base      = bin_off[g];     /* index into h_seg_cnt_flat */
                int64_t anchor_base   = anchor_off[g];  /* base in h_bx/h_by         */
                int64_t anchor_cursor = anchor_base;

                /* ---------------------------------------------------------- *
                 * Build b[] and u[] with query_pos monotonicity.
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
