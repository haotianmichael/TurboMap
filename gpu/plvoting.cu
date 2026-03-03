/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining (batched)
 *
 * All (read, chr/strand) groups across the entire n_rechain batch are now
 * processed in a SINGLE set of GPU kernel launches instead of one set per
 * (read, chr-group).  This collapses O(reads × chr_groups) kernel launches
 * and synchronisation points down to a fixed handful regardless of batch size.
 *
 * Batch pipeline  (one launch per step for all groups combined):
 *   1. voting_bin_kernel_batch      – histogram anchor ref_pos into bins
 *   2. mark_dilate_kernel_batch     – mark winning bins, merge small gaps
 *   3. seg_start_kernel_batch       – detect start of each winning run
 *   4. CUB DeviceSegmentedScan      – inclusive sum of seg_start → seg_id
 *                                    (segmented: one segment per vg_t group)
 *   5. extract_nsegs_kernel         – gather last seg_id element per group
 *      ↕  [sync 1] D2H copy of h_n_segs, compute seg_cnt offsets, H2D upload
 *   6. tag_mark_kernel_batch        – per anchor: keep?, local seg_id
 *   7. CUB DeviceSegmentedScan      – exclusive sum of d_mark → out_pos
 *                                    (segmented: one segment per vg_t group)
 *   8. scatter_compact_kernel_batch – scatter kept anchors into d_bx/d_by,
 *                                    count anchors per segment and per group
 *      ↕  [sync 2] D2H h_bx, h_by, h_seg_cnt, h_n_b_chr
 *   CPU: per-group query_pos monotonicity enforcement, build b[]/u[]
 *
 * OUTPUT: rd->a = filtered anchor array b[]
 *         rd->u = chain descriptors u[] (score<<32 | anchor_count)
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
 * Per-group metadata
 * ========================================================================= */

/*
 * vg_t  –  describes one (read, chr/strand) group of anchors.
 *
 * All offsets are into the global flattened device arrays; they are filled
 * on the CPU before the GPU phase begins.
 */
typedef struct {
    int64_t anchor_off;  /* first anchor index in global d_ax/d_ay        */
    int32_t n_anchors;   /* number of anchors in this group                */
    int64_t bin_off;     /* first bin index in global d_votes / d_keep_bin */
    int32_t n_bins;      /* number of bins in this group                   */
    int32_t ref_min;     /* minimum ref_pos (for bin index computation)    */
    int32_t read_ri;     /* index into rechain_indices[] for this group    */
} vg_t;

/* =========================================================================
 * GPU kernels  (batch-aware: one thread per anchor or bin, all groups)
 * ========================================================================= */

/*
 * voting_bin_kernel_batch
 *
 * Each anchor thread computes its group from d_ax_grp[], then maps ref_pos
 * into the group's local bin range and increments d_votes[].
 */
__global__ void voting_bin_kernel_batch(const uint64_t *d_ax,
                                        int64_t         total_n,
                                        const int32_t  *d_ax_grp,
                                        const vg_t     *d_groups,
                                        int32_t         bin_size,
                                        int32_t        *d_votes)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total_n) return;

    int32_t g       = d_ax_grp[j];
    int32_t ref_pos = (int32_t)d_ax[j];
    int32_t bin     = (ref_pos - d_groups[g].ref_min) / bin_size;
    int32_t n_bins  = d_groups[g].n_bins;
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[d_groups[g].bin_off + bin], 1);
}

/*
 * mark_dilate_kernel_batch
 *
 * Each bin thread checks a window of ±gap bins within its group (group
 * boundaries are not crossed) and sets d_keep_bin[i] accordingly.
 */
__global__ void mark_dilate_kernel_batch(const int32_t *d_votes,
                                         int8_t        *d_keep_bin,
                                         int32_t        total_bins,
                                         const int32_t *d_bin_grp,
                                         const vg_t    *d_groups,
                                         int32_t        min_votes,
                                         int32_t        gap)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bins) return;

    int32_t g         = d_bin_grp[i];
    int32_t bin_start = (int32_t)d_groups[g].bin_off;
    int32_t n_bins    = d_groups[g].n_bins;
    int32_t local_i   = i - bin_start;

    int32_t lo = bin_start + max(0,           local_i - gap);
    int32_t hi = bin_start + min(n_bins - 1,  local_i + gap);

    int8_t keep = 0;
    for (int32_t k = lo; k <= hi; ++k) {
        if (d_votes[k] >= min_votes) { keep = 1; break; }
    }
    d_keep_bin[i] = keep;
}

/*
 * seg_start_kernel_batch
 *
 * A bin is a segment start if it is kept AND either it is the first bin of
 * its group or the preceding bin is not kept.
 */
__global__ void seg_start_kernel_batch(const int8_t  *d_keep_bin,
                                       int32_t       *d_seg_start,
                                       int32_t        total_bins,
                                       const int32_t *d_bin_grp,
                                       const vg_t    *d_groups)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bins) return;

    int32_t g         = d_bin_grp[i];
    int32_t bin_start = (int32_t)d_groups[g].bin_off;
    int is_group_first = (i == bin_start);

    d_seg_start[i] = (d_keep_bin[i] == 1 &&
                      (is_group_first || d_keep_bin[i - 1] == 0)) ? 1 : 0;
}

/*
 * extract_nsegs_kernel
 *
 * After the segmented inclusive sum of d_seg_start → d_seg_id, the last
 * element of each group's bin range holds that group's segment count.
 */
__global__ void extract_nsegs_kernel(const int32_t *d_seg_id,
                                     const vg_t    *d_groups,
                                     int32_t       *d_n_segs,
                                     int32_t        n_groups)
{
    int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    int32_t last_bin = (int32_t)d_groups[g].bin_off + d_groups[g].n_bins - 1;
    d_n_segs[g] = d_seg_id[last_bin];
}

/*
 * tag_mark_kernel_batch
 *
 * Per anchor: determine d_mark (keep/discard) and d_anchor_seg (local
 * 0-based segment ID within the group).
 */
__global__ void tag_mark_kernel_batch(const uint64_t *d_ax,
                                      int64_t         total_n,
                                      const int32_t  *d_ax_grp,
                                      const vg_t     *d_groups,
                                      int32_t         bin_size,
                                      const int8_t   *d_keep_bin,
                                      const int32_t  *d_seg_id,
                                      int32_t        *d_mark,
                                      int32_t        *d_anchor_seg)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total_n) return;

    int32_t g       = d_ax_grp[j];
    int32_t ref_pos = (int32_t)d_ax[j];
    int32_t bin     = (ref_pos - d_groups[g].ref_min) / bin_size;
    int32_t n_bins  = d_groups[g].n_bins;
    int32_t bin_off = (int32_t)d_groups[g].bin_off;

    if (bin < 0 || bin >= n_bins || d_keep_bin[bin_off + bin] == 0) {
        d_mark[j]       = 0;
        d_anchor_seg[j] = 0;
    } else {
        d_mark[j]       = 1;
        /* Inclusive prefix sum value at bin (1-indexed) → subtract 1 for
         * the 0-based local segment ID within this group. */
        d_anchor_seg[j] = d_seg_id[bin_off + bin] - 1;
    }
}

/*
 * scatter_compact_kernel_batch
 *
 * Scatters kept anchors into d_bx/d_by using the per-group-local output
 * positions from the segmented exclusive sum.  Also maintains:
 *   d_seg_cnt[seg_cnt_off[g] + local_seg] – anchors per segment
 *   d_n_b_chr[g]                           – total kept anchors per group
 *
 * d_out_pos[] contains group-local output positions (0-based within each
 * group).  The global write position is computed as bx_off[g] + local_pos,
 * where d_bx_off[g] is supplied by the caller.
 */
__global__ void scatter_compact_kernel_batch(const uint64_t *d_ax,
                                             const uint64_t *d_ay,
                                             const int32_t  *d_mark,
                                             const int32_t  *d_out_pos,
                                             const int32_t  *d_anchor_seg,
                                             int64_t         total_n,
                                             const int32_t  *d_ax_grp,
                                             const int32_t  *d_bx_off,
                                             const int32_t  *d_seg_cnt_off,
                                             uint64_t       *d_bx,
                                             uint64_t       *d_by,
                                             int32_t        *d_seg_cnt,
                                             int32_t        *d_n_b_chr)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total_n || d_mark[j] == 0) return;

    int32_t g          = d_ax_grp[j];
    int32_t global_out = d_bx_off[g] + d_out_pos[j];

    d_bx[global_out] = d_ax[j];
    d_by[global_out] = d_ay[j];

    atomicAdd(&d_seg_cnt[d_seg_cnt_off[g] + d_anchor_seg[j]], 1);
    atomicAdd(&d_n_b_chr[g], 1);
}

/* =========================================================================
 * CUB helpers  (segmented)
 * ========================================================================= */

/*
 * Segmented inclusive prefix sum.
 * d_in  → d_out, with n_seg independent segments defined by begin/end offsets.
 */
static void cub_seg_incl_sum(int32_t *d_in, int32_t *d_out,
                             int n_items, int n_seg,
                             int32_t *d_begins, int32_t *d_ends)
{
    void  *d_tmp  = nullptr;
    size_t tmp_sz = 0;
    cub::DeviceSegmentedScan::InclusiveSum(nullptr, tmp_sz,
        d_in, d_out, n_items, n_seg, d_begins, d_ends);
    cudaMalloc(&d_tmp, tmp_sz);
    cub::DeviceSegmentedScan::InclusiveSum(d_tmp, tmp_sz,
        d_in, d_out, n_items, n_seg, d_begins, d_ends);
    cudaFree(d_tmp);
}

/*
 * Segmented exclusive prefix sum.
 */
static void cub_seg_excl_sum(int32_t *d_in, int32_t *d_out,
                             int n_items, int n_seg,
                             int32_t *d_begins, int32_t *d_ends)
{
    void  *d_tmp  = nullptr;
    size_t tmp_sz = 0;
    cub::DeviceSegmentedScan::ExclusiveSum(nullptr, tmp_sz,
        d_in, d_out, n_items, n_seg, d_begins, d_ends);
    cudaMalloc(&d_tmp, tmp_sz);
    cub::DeviceSegmentedScan::ExclusiveSum(d_tmp, tmp_sz,
        d_in, d_out, n_items, n_seg, d_begins, d_ends);
    cudaFree(d_tmp);
}

/* =========================================================================
 * Main exported function
 * ========================================================================= */

extern "C" {

/*
 * plvoting_rechain_batch
 *
 * Processes all n_rechain reads in a single batched GPU pipeline.
 * Replaces the previous nested (per-read × per-chr-group) loop that
 * issued O(reads × chr_groups) kernel launches and synchronisations.
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    /* ---------------------------------------------------------------
     * Phase 1: flatten each read's chains into a single sorted anchor
     * array (prepare_rechain_anchors frees u[], sorts by ref_pos).
     * --------------------------------------------------------------- */
    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

    const int32_t max_dist       = (opt->max_gap > 0) ? opt->max_gap : 10000;
    const int32_t bin_size       = (max_dist / VOTING_BIN_DIVIDER > 0)
                                   ? (max_dist / VOTING_BIN_DIVIDER) : 1;
    const int32_t min_votes      = (opt->min_cnt > 1) ? opt->min_cnt : 2;
    const int32_t merge_gap_bins = (VOTING_LARGE_GAP + bin_size - 1) / bin_size;

    const int blk = 256;

    /* ---------------------------------------------------------------
     * Phase 2: enumerate all (read, chr/strand) groups on CPU.
     *
     * We walk each read's anchor array (sorted by ref_pos) to find
     * contiguous chr/strand blocks, record their metadata in vg_t,
     * and accumulate the global array sizes needed for device buffers.
     * --------------------------------------------------------------- */

    /* First pass: count groups to size the vg_t array. */
    int n_groups = 0;
    for (int ri = 0; ri < n_rechain; ++ri) {
        chain_read_t *rd = &reads[rechain_indices[ri]];
        if (rd->n == 0 || rd->a == NULL) continue;
        int64_t  n_a = rd->n;
        mm128_t *a   = rd->a;
        int64_t  pos = 0;
        while (pos < n_a) {
            int32_t xrev = (int32_t)(a[pos].x >> 32);
            int64_t end  = pos;
            while (end < n_a && (int32_t)(a[end].x >> 32) == xrev) ++end;
            ++n_groups;
            pos = end;
        }
    }

    if (n_groups == 0) return;

    /* Allocate host metadata arrays. */
    vg_t    *h_groups    = (vg_t    *)malloc(n_groups * sizeof(vg_t));
    int32_t *h_ax_grp    = NULL;  /* filled after sizing total_anchors */
    int32_t *h_bin_grp   = NULL;  /* filled after sizing total_bins    */

    /* Second pass: fill vg_t, compute totals. */
    int64_t total_anchors = 0;
    int32_t total_bins    = 0;
    int32_t g_idx         = 0;

    for (int ri = 0; ri < n_rechain; ++ri) {
        chain_read_t *rd = &reads[rechain_indices[ri]];
        if (rd->n == 0 || rd->a == NULL) continue;
        int64_t  n_a = rd->n;
        mm128_t *a   = rd->a;
        int64_t  pos = 0;
        while (pos < n_a) {
            int32_t xrev    = (int32_t)(a[pos].x >> 32);
            int64_t end     = pos;
            while (end < n_a && (int32_t)(a[end].x >> 32) == xrev) ++end;

            int64_t chr_n    = end - pos;
            int32_t ref_min  = (int32_t)a[pos].x;
            int32_t ref_max  = (int32_t)a[end - 1].x;
            int32_t ref_span = ref_max - ref_min + 1;
            int32_t nb       = (ref_span + bin_size - 1) / bin_size + 1;

            h_groups[g_idx].anchor_off = total_anchors;
            h_groups[g_idx].n_anchors  = (int32_t)chr_n;
            h_groups[g_idx].bin_off    = total_bins;
            h_groups[g_idx].n_bins     = nb;
            h_groups[g_idx].ref_min    = ref_min;
            h_groups[g_idx].read_ri    = ri;

            total_anchors += chr_n;
            total_bins    += nb;
            ++g_idx;
            pos = end;
        }
    }
    /* (g_idx == n_groups is guaranteed by the two-pass design) */

    /* Build per-anchor and per-bin group-ID arrays. */
    h_ax_grp  = (int32_t *)malloc(total_anchors * sizeof(int32_t));
    h_bin_grp = (int32_t *)malloc(total_bins    * sizeof(int32_t));

    for (int g = 0; g < n_groups; ++g) {
        int64_t a0 = h_groups[g].anchor_off;
        for (int32_t k = 0; k < h_groups[g].n_anchors; ++k)
            h_ax_grp[a0 + k] = g;
        int64_t b0 = h_groups[g].bin_off;
        for (int32_t k = 0; k < h_groups[g].n_bins; ++k)
            h_bin_grp[b0 + k] = g;
    }

    /* Build CUB segmented-scan offset arrays (int32_t, host side). */
    int32_t *h_bin_begins = (int32_t *)malloc(n_groups * sizeof(int32_t));
    int32_t *h_bin_ends   = (int32_t *)malloc(n_groups * sizeof(int32_t));
    int32_t *h_anc_begins = (int32_t *)malloc(n_groups * sizeof(int32_t));
    int32_t *h_anc_ends   = (int32_t *)malloc(n_groups * sizeof(int32_t));
    for (int g = 0; g < n_groups; ++g) {
        h_bin_begins[g] = (int32_t)h_groups[g].bin_off;
        h_bin_ends[g]   = (int32_t)h_groups[g].bin_off + h_groups[g].n_bins;
        h_anc_begins[g] = (int32_t)h_groups[g].anchor_off;
        h_anc_ends[g]   = (int32_t)h_groups[g].anchor_off + h_groups[g].n_anchors;
    }

    /* ---------------------------------------------------------------
     * Phase 3: Flatten anchor data into host arrays for one H2D copy.
     * --------------------------------------------------------------- */
    uint64_t *h_ax_all = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    uint64_t *h_ay_all = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    {
        int64_t dst = 0;
        for (int ri = 0; ri < n_rechain; ++ri) {
            chain_read_t *rd = &reads[rechain_indices[ri]];
            if (rd->n == 0 || rd->a == NULL) continue;
            for (int64_t k = 0; k < rd->n; ++k) {
                h_ax_all[dst] = rd->a[k].x;
                h_ay_all[dst] = rd->a[k].y;
                ++dst;
            }
        }
    }

    /* ---------------------------------------------------------------
     * Phase 4: Allocate device buffers (one allocation per array).
     *
     * d_seg_cnt is pre-sized to total_bins (worst case: every bin is a
     * segment start, so total_segs <= total_bins).
     * --------------------------------------------------------------- */
    uint64_t *d_ax, *d_ay;
    int32_t  *d_votes, *d_ax_grp, *d_bin_grp_dev;
    int8_t   *d_keep_bin;
    int32_t  *d_seg_start, *d_seg_id;
    int32_t  *d_mark, *d_out_pos, *d_anchor_seg;
    uint64_t *d_bx, *d_by;
    int32_t  *d_seg_cnt, *d_n_b_chr, *d_n_segs;
    vg_t     *d_groups;
    int32_t  *d_bin_begins, *d_bin_ends, *d_anc_begins, *d_anc_ends;
    int32_t  *d_seg_cnt_off, *d_bx_off;

    cudaMalloc(&d_ax,         total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_ay,         total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_votes,      total_bins    * sizeof(int32_t));
    cudaMalloc(&d_ax_grp,     total_anchors * sizeof(int32_t));
    cudaMalloc(&d_bin_grp_dev,total_bins    * sizeof(int32_t));
    cudaMalloc(&d_keep_bin,   total_bins    * sizeof(int8_t));
    cudaMalloc(&d_seg_start,  total_bins    * sizeof(int32_t));
    cudaMalloc(&d_seg_id,     total_bins    * sizeof(int32_t));
    cudaMalloc(&d_mark,       total_anchors * sizeof(int32_t));
    cudaMalloc(&d_out_pos,    total_anchors * sizeof(int32_t));
    cudaMalloc(&d_anchor_seg, total_anchors * sizeof(int32_t));
    cudaMalloc(&d_bx,         total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_by,         total_anchors * sizeof(uint64_t));
    cudaMalloc(&d_seg_cnt,    total_bins    * sizeof(int32_t));   /* worst-case */
    cudaMalloc(&d_n_b_chr,    n_groups      * sizeof(int32_t));
    cudaMalloc(&d_n_segs,     n_groups      * sizeof(int32_t));
    cudaMalloc(&d_groups,     n_groups      * sizeof(vg_t));
    cudaMalloc(&d_bin_begins, n_groups      * sizeof(int32_t));
    cudaMalloc(&d_bin_ends,   n_groups      * sizeof(int32_t));
    cudaMalloc(&d_anc_begins, n_groups      * sizeof(int32_t));
    cudaMalloc(&d_anc_ends,   n_groups      * sizeof(int32_t));
    cudaMalloc(&d_seg_cnt_off,n_groups      * sizeof(int32_t));
    cudaMalloc(&d_bx_off,     n_groups      * sizeof(int32_t));

    cudaMemset(d_votes,   0, total_bins    * sizeof(int32_t));
    cudaMemset(d_seg_cnt, 0, total_bins    * sizeof(int32_t));
    cudaMemset(d_n_b_chr, 0, n_groups      * sizeof(int32_t));

    /* ---------------------------------------------------------------
     * Phase 5: Upload all data in one pass.
     * --------------------------------------------------------------- */
    cudaMemcpy(d_ax,          h_ax_all,     total_anchors * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay,          h_ay_all,     total_anchors * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax_grp,      h_ax_grp,     total_anchors * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_grp_dev, h_bin_grp,    total_bins    * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_groups,      h_groups,     n_groups      * sizeof(vg_t),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_begins,  h_bin_begins, n_groups      * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_ends,    h_bin_ends,   n_groups      * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_anc_begins,  h_anc_begins, n_groups      * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_anc_ends,    h_anc_ends,   n_groups      * sizeof(int32_t),  cudaMemcpyHostToDevice);

    free(h_ax_all);   free(h_ay_all);
    free(h_ax_grp);   free(h_bin_grp);
    free(h_bin_begins); free(h_bin_ends);
    free(h_anc_begins); free(h_anc_ends);

    /* ---------------------------------------------------------------
     * Phase 6: Batch GPU pipeline.
     * --------------------------------------------------------------- */

    /* Step 1: voting histogram */
    {
        int grd = (int)((total_anchors + blk - 1) / blk);
        voting_bin_kernel_batch<<<grd, blk>>>(d_ax, total_anchors, d_ax_grp,
                                              d_groups, bin_size, d_votes);
    }

    /* Step 2: mark + dilate */
    {
        int grd = (total_bins + blk - 1) / blk;
        mark_dilate_kernel_batch<<<grd, blk>>>(d_votes, d_keep_bin,
                                               total_bins, d_bin_grp_dev,
                                               d_groups, min_votes,
                                               merge_gap_bins);
    }
    cudaFree(d_votes);  /* no longer needed */

    /* Step 3: segment-start flags */
    {
        int grd = (total_bins + blk - 1) / blk;
        seg_start_kernel_batch<<<grd, blk>>>(d_keep_bin, d_seg_start,
                                             total_bins, d_bin_grp_dev,
                                             d_groups);
    }

    /* Step 4: segmented inclusive sum → per-group seg IDs */
    cub_seg_incl_sum(d_seg_start, d_seg_id,
                     (int)total_bins, n_groups,
                     d_bin_begins, d_bin_ends);
    cudaFree(d_seg_start);

    /* Step 5: extract n_segs per group, then sync to CPU.
     *
     * [Sync 1]  This is the only mandatory CPU–GPU synchronisation point
     * before the final download.  We need n_segs[g] to compute
     * seg_cnt_off[] and bx_off[] which are required by scatter_compact. */
    {
        int grd = (n_groups + blk - 1) / blk;
        extract_nsegs_kernel<<<grd, blk>>>(d_seg_id, d_groups,
                                           d_n_segs, n_groups);
    }
    int32_t *h_n_segs = (int32_t *)malloc(n_groups * sizeof(int32_t));
    cudaMemcpy(h_n_segs, d_n_segs, n_groups * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_n_segs);

    /* Compute seg_cnt_off[] on CPU, upload to device.
     *
     * d_bx_off[g] = anchor_off[g]: the segmented exclusive sum produces
     * group-local output positions 0..n_b_chr[g]-1; adding anchor_off[g]
     * places them in non-overlapping slots of d_bx/d_by so groups never
     * overwrite each other. */
    int32_t *h_seg_cnt_off = (int32_t *)malloc(n_groups * sizeof(int32_t));
    int32_t *h_bx_off_arr  = (int32_t *)malloc(n_groups * sizeof(int32_t));
    {
        int32_t seg_acc = 0;
        for (int g = 0; g < n_groups; ++g) {
            h_seg_cnt_off[g] = seg_acc;
            h_bx_off_arr[g]  = (int32_t)h_groups[g].anchor_off;
            seg_acc += h_n_segs[g];
        }
    }
    cudaMemcpy(d_seg_cnt_off, h_seg_cnt_off, n_groups * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bx_off, h_bx_off_arr, n_groups * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    free(h_bx_off_arr);

    /* Step 6: tag anchors (mark + local segment ID) */
    {
        int grd = (int)((total_anchors + blk - 1) / blk);
        tag_mark_kernel_batch<<<grd, blk>>>(d_ax, total_anchors, d_ax_grp,
                                            d_groups, bin_size,
                                            d_keep_bin, d_seg_id,
                                            d_mark, d_anchor_seg);
    }
    cudaFree(d_keep_bin);
    cudaFree(d_seg_id);

    /* Step 7: segmented exclusive sum → group-local compaction positions */
    cub_seg_excl_sum(d_mark, d_out_pos,
                     (int)total_anchors, n_groups,
                     d_anc_begins, d_anc_ends);
    cudaFree(d_anc_begins);
    cudaFree(d_anc_ends);
    cudaFree(d_bin_begins);
    cudaFree(d_bin_ends);

    /* Step 8: scatter compaction */
    {
        int grd = (int)((total_anchors + blk - 1) / blk);
        scatter_compact_kernel_batch<<<grd, blk>>>(
            d_ax, d_ay,
            d_mark, d_out_pos, d_anchor_seg,
            total_anchors, d_ax_grp,
            d_bx_off, d_seg_cnt_off,
            d_bx, d_by, d_seg_cnt, d_n_b_chr);
    }
    cudaFree(d_ax);        cudaFree(d_ay);
    cudaFree(d_mark);      cudaFree(d_out_pos);
    cudaFree(d_anchor_seg);cudaFree(d_ax_grp);
    cudaFree(d_bin_grp_dev);
    cudaFree(d_groups);
    cudaFree(d_seg_cnt_off);
    cudaFree(d_bx_off);

    /* ---------------------------------------------------------------
     * Phase 7: Download results in one pass.  [Sync 2]
     * --------------------------------------------------------------- */
    int32_t  total_segs = 0;
    for (int g = 0; g < n_groups; ++g) total_segs += h_n_segs[g];

    uint64_t *h_bx      = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    uint64_t *h_by      = (uint64_t *)malloc(total_anchors * sizeof(uint64_t));
    int32_t  *h_seg_cnt = (int32_t  *)malloc((total_segs > 0 ? total_segs : 1) * sizeof(int32_t));
    int32_t  *h_n_b_chr = (int32_t  *)malloc(n_groups * sizeof(int32_t));

    cudaMemcpy(h_bx,      d_bx,      total_anchors * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_by,      d_by,      total_anchors * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (total_segs > 0)
        cudaMemcpy(h_seg_cnt, d_seg_cnt, total_segs * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_b_chr, d_n_b_chr, n_groups * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_bx);   cudaFree(d_by);
    cudaFree(d_seg_cnt); cudaFree(d_n_b_chr);

    /* ---------------------------------------------------------------
     * Phase 8: Allocate per-read output buffers and run CPU
     * post-processing (query_pos monotonicity enforcement).
     *
     * For each read we need b[] (filtered anchors) and u_buf[] (chain
     * descriptors).  We iterate over the groups that belong to each
     * read and accumulate into those buffers.
     * --------------------------------------------------------------- */

    /* Build output buffers per read (index by ri). */
    mm128_t  **b_bufs   = (mm128_t  **)calloc(n_rechain, sizeof(mm128_t *));
    uint64_t **u_bufs   = (uint64_t **)calloc(n_rechain, sizeof(uint64_t *));
    int64_t   *n_b_arr  = (int64_t  *)calloc(n_rechain, sizeof(int64_t));
    int       *n_u_arr  = (int      *)calloc(n_rechain, sizeof(int));

    for (int ri = 0; ri < n_rechain; ++ri) {
        chain_read_t *rd = &reads[rechain_indices[ri]];
        int64_t n_a = rd->n;
        if (n_a == 0) continue;
        KMALLOC(km, b_bufs[ri], n_a);
        KMALLOC(km, u_bufs[ri], n_a);
    }

    /* Per-group CPU Step 8: monotonicity enforcement.
     *
     * Group g's kept anchors live at h_bx[anchor_off..anchor_off+n_b_chr-1]
     * because scatter used d_bx_off[g]=anchor_off[g] as the base. */
    for (int g = 0; g < n_groups; ++g) {
        int     ri       = h_groups[g].read_ri;
        int32_t n_segs_g = h_n_segs[g];
        if (n_segs_g == 0) continue;

        int32_t  bx_base   = (int32_t)h_groups[g].anchor_off; /* base in h_bx/h_by */
        int32_t  sc_base   = h_seg_cnt_off[g];     /* start in h_seg_cnt */
        int64_t *n_b       = &n_b_arr[ri];
        int     *n_u       = &n_u_arr[ri];
        mm128_t  *b        = b_bufs[ri];
        uint64_t *u_buf    = u_bufs[ri];

        int32_t anchor_off = 0;
        for (int32_t s = 0; s < n_segs_g; ++s) {
            int32_t cnt = h_seg_cnt[sc_base + s];
            if (cnt == 0) continue;

            /* ay layout: flags<<40 | q_span<<32 | q_pos
             *   q_pos  = (int32_t)(ay)         [low 32 bits]
             *   q_span = (ay >> 32) & 0xff      [bits 32-39]  */
            int32_t  last_q  = INT32_MIN;
            uint64_t sub_sc  = 0;
            int32_t  sub_cnt = 0;

            for (int32_t k = 0; k < cnt; ++k) {
                int32_t  qp  = (int32_t)(h_by[bx_base + anchor_off + k]);
                uint64_t qsp = (h_by[bx_base + anchor_off + k] >> 32) & 0xffU;

                if (qp > last_q) {
                    b[*n_b].x = h_bx[bx_base + anchor_off + k];
                    b[*n_b].y = h_by[bx_base + anchor_off + k];
                    ++(*n_b); sub_sc += qsp; ++sub_cnt; last_q = qp;
                } else {
                    int8_t sustained =
                        (k + 1 < cnt) &&
                        ((int32_t)(h_by[bx_base + anchor_off + k + 1]) <= last_q);
                    if (!sustained) {
                        /* Isolated bad anchor – discard silently */
                    } else {
                        /* Sustained reversal – close sub-chain, open new */
                        if (sub_cnt > 0)
                            u_buf[(*n_u)++] = (sub_sc << 32) | (uint64_t)sub_cnt;
                        b[*n_b].x = h_bx[bx_base + anchor_off + k];
                        b[*n_b].y = h_by[bx_base + anchor_off + k];
                        ++(*n_b); sub_sc = qsp; sub_cnt = 1; last_q = qp;
                    }
                }
            }

            if (sub_cnt > 0)
                u_buf[(*n_u)++] = (sub_sc << 32) | (uint64_t)sub_cnt;

            anchor_off += cnt;
        }
    }

    free(h_bx);  free(h_by);
    free(h_seg_cnt);  free(h_n_b_chr);
    free(h_n_segs);   free(h_seg_cnt_off);
    free(h_groups);

    /* ---------------------------------------------------------------
     * Phase 9: Store results back into each read.
     * --------------------------------------------------------------- */
    for (int ri = 0; ri < n_rechain; ++ri) {
        chain_read_t *rd = &reads[rechain_indices[ri]];

        /* Free the original (pre-voting) anchor array. */
        kfree(km, rd->a);
        rd->a   = NULL;
        rd->n   = 0;
        rd->u   = NULL;
        rd->n_u = 0;

        int64_t n_b = n_b_arr[ri];
        int     n_u = n_u_arr[ri];

        if (n_b == 0 || n_u == 0) {
            if (b_bufs[ri]) kfree(km, b_bufs[ri]);
            if (u_bufs[ri]) kfree(km, u_bufs[ri]);
            continue;
        }

        mm128_t *b_final;
        KMALLOC(km, b_final, n_b);
        memcpy(b_final, b_bufs[ri], n_b * sizeof(mm128_t));
        kfree(km, b_bufs[ri]);

        uint64_t *u_final;
        KMALLOC(km, u_final, n_u);
        memcpy(u_final, u_bufs[ri], n_u * sizeof(uint64_t));
        kfree(km, u_bufs[ri]);

        rd->a   = b_final;
        rd->n   = n_b;
        rd->u   = u_final;
        rd->n_u = n_u;
    }

    free(b_bufs);
    free(u_bufs);
    free(n_b_arr);
    free(n_u_arr);
}

} /* extern "C" */
