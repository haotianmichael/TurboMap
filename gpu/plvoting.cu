/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining
 *
 * Replaces the RMQ-based long-join re-chaining (mg_lchain_rmq) with pure
 * GPU voting.  The CPU DP step (mg_lchain_dp) is intentionally absent:
 * each winning voting segment is treated as one chain directly, so the
 * output u[]/a[] can be handed straight to mm_gen_regs → GPU KSW.
 *
 * Algorithm (Genome-on-Diet style)
 * ---------------------------------
 * 1. GPU histogram: atomically bin anchors by reference position.
 * 2. CPU: scan histogram → find "winning" bins (≥ min_votes votes).
 *    Adjacent winning runs separated by ≤ 50 kb are merged (small gap).
 *    Runs separated by > 50 kb remain as separate chains (large gap).
 * 3. CPU: two-pointer filter → keep only anchors inside winning segments.
 *    Simultaneously build u[]: one entry per winning segment.
 *      u[i] = (sum_q_span << 32) | n_anchors_in_segment
 *    The sum of q_span values serves as a proxy chain score used only
 *    for primary/secondary ranking downstream.
 * 4. Write b[] (filtered anchors, compacted by segment) and u[] back to
 *    the chain_read_t.  No mg_lchain_dp is called.
 */

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "mmpriv.h"
#include "kalloc.h"
#include "hipify.cuh"
#include "plvoting.cuh"

/* =========================================================================
 * GPU kernel
 * ========================================================================= */

/**
 * voting_bin_kernel
 *
 * Each thread handles one anchor.  It computes the bin index from the anchor's
 * reference position (lower 32 bits of a[i].x) and atomically increments that
 * bin's vote counter.
 *
 * The kernel is chromosome-agnostic: the caller invokes it once per chromosome
 * group so that ref_min is meaningful.
 */
__global__ void voting_bin_kernel(const int32_t *d_ref_pos,
                                  int64_t n,
                                  int32_t ref_min,
                                  int32_t bin_size,
                                  int32_t n_bins,
                                  int32_t *d_votes)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int32_t bin = (d_ref_pos[tid] - ref_min) / bin_size;
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[bin], 1);
}

/* =========================================================================
 * Host helpers
 * ========================================================================= */

/**
 * find_winning_segments
 *
 * Scans the vote histogram and identifies winning reference intervals.
 * A bin is winning if votes[bin] >= min_votes.  Adjacent winning bins (or
 * bins separated by ≤ merge_gap_bins empty bins) are merged into one segment
 * (Genome-on-Diet: small gaps ≤ 50 kb → concatenated CIGAR).
 * Gaps > merge_gap_bins break segments (large gaps → separate alignments).
 *
 * Returns number of winning segments found.
 */
static int find_winning_segments(const int32_t *votes, int32_t n_bins,
                                 int32_t ref_min,  int32_t bin_size,
                                 int32_t min_votes, int32_t merge_gap_bins,
                                 int32_t *seg_start, int32_t *seg_end,
                                 int max_segs)
{
    int n_segs = 0;
    int32_t i = 0;

    while (i < n_bins && n_segs < max_segs) {
        if (votes[i] < min_votes) { ++i; continue; }

        int32_t win_start = i;
        int32_t win_end   = i;
        int32_t gap       = 0;

        while (i < n_bins) {
            if (votes[i] >= min_votes) {
                win_end = i;
                gap = 0;
            } else {
                if (++gap > merge_gap_bins) break;
            }
            ++i;
        }

        seg_start[n_segs] = ref_min + win_start * bin_size;
        seg_end[n_segs]   = ref_min + (win_end + 1) * bin_size;
        ++n_segs;
    }

    return n_segs;
}

/* =========================================================================
 * Main exported function
 * ========================================================================= */

extern "C" {

/**
 * plvoting_rechain_batch
 *
 * For each read in rechain_indices:
 *  1. Flatten + sort anchors via prepare_rechain_anchors().
 *  2. GPU voting histogram per chromosome group.
 *  3. CPU: find winning segments, two-pointer filter anchors into b[].
 *     Build u[] simultaneously: one entry per winning segment,
 *     score = sum of per-anchor q_span values.
 *  4. Write b[] and u[] directly back to reads[] — no mg_lchain_dp.
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

    /* Voting parameters */
    const int32_t max_dist = (opt->max_gap > 0) ? opt->max_gap : 10000;
    const int32_t bin_size = (max_dist / VOTING_BIN_DIVIDER > 0)
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

        /*
         * b[]   : filtered anchors, grouped by winning segment (compacted).
         * u_buf[]: chain descriptors, one per winning segment.
         *          u[i] = (sum_q_span << 32) | n_anchors_in_segment
         *
         * Upper bound: at most n_a anchors and n_a segments (degenerate case
         * where every anchor is its own segment).
         */
        mm128_t  *b;
        uint64_t *u_buf;
        KMALLOC(km, b,     n_a);
        KMALLOC(km, u_buf, n_a);
        int64_t  n_b   = 0;
        int      n_u   = 0;

        /* ------------------------------------------------------------------ *
         * Process each chromosome group independently.                        *
         * ------------------------------------------------------------------ */
        int64_t chr_start = 0;
        while (chr_start < n_a) {
            /* Delimit this chromosome group */
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

            /* ---- GPU: upload ref positions and run voting ---- */
            int32_t *h_ref_pos = (int32_t *)malloc(chr_n * sizeof(int32_t));
            for (int64_t j = 0; j < chr_n; ++j)
                h_ref_pos[j] = (int32_t)chr_a[j].x;

            int32_t *d_ref_pos = NULL, *d_votes = NULL;
            cudaMalloc(&d_ref_pos, chr_n * sizeof(int32_t));
            cudaMalloc(&d_votes,   n_bins * sizeof(int32_t));
            cudaMemset(d_votes, 0, n_bins * sizeof(int32_t));
            cudaMemcpy(d_ref_pos, h_ref_pos, chr_n * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
            free(h_ref_pos);

            int grd = (int)((chr_n + blk - 1) / blk);
            voting_bin_kernel<<<grd, blk>>>(d_ref_pos, chr_n,
                                            ref_min, bin_size, n_bins,
                                            d_votes);

            int32_t *h_votes = (int32_t *)malloc(n_bins * sizeof(int32_t));
            cudaMemcpy(h_votes, d_votes, n_bins * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_ref_pos);
            cudaFree(d_votes);

            /* ---- CPU: find winning segments ---- */
            int      max_segs  = n_bins + 1;
            int32_t *seg_start = (int32_t *)malloc(max_segs * sizeof(int32_t));
            int32_t *seg_end   = (int32_t *)malloc(max_segs * sizeof(int32_t));

            int n_segs = find_winning_segments(h_votes, n_bins,
                                               ref_min, bin_size,
                                               min_votes, merge_gap_bins,
                                               seg_start, seg_end, max_segs);
            free(h_votes);

            if (n_segs == 0) {
                /* No winning segments: drop all anchors from this chromosome.
                 * (Keeping all would make one large noisy chain.) */
                free(seg_start);
                free(seg_end);
                chr_start = chr_end;
                continue;
            }

            /*
             * Two-pointer filter + u[] construction.
             *
             * Both chr_a[] (sorted by ref_pos) and seg_*[] (sorted by
             * seg_start) are monotonically increasing → O(chr_n + n_segs).
             *
             * For each segment we open a new chain entry in u_buf[]:
             *   - record start offset in b[] as seg_b_start
             *   - accumulate score = sum of q_span (bits 32..39 of a[i].y)
             *   - on segment boundary: write u_buf[n_u++]
             */
            int s = 0;          /* current segment index */
            int64_t seg_b_start = n_b;   /* start of current segment in b[] */
            uint64_t seg_score  = 0;
            int32_t  cur_seg    = -1;    /* segment index being filled */

            for (int64_t j = 0; j < chr_n; ++j) {
                int32_t rpos = (int32_t)chr_a[j].x;

                /* Advance segment pointer past segments ending before rpos */
                while (s < n_segs && seg_end[s] <= rpos) {
                    /* Close the segment we were filling, if any */
                    if (cur_seg == s && n_b > seg_b_start) {
                        u_buf[n_u++] = (seg_score << 32) | (uint64_t)(n_b - seg_b_start);
                        seg_b_start  = n_b;
                        seg_score    = 0;
                    }
                    ++s;
                    cur_seg = -1;
                }

                if (s >= n_segs) break;

                if (rpos >= seg_start[s]) {
                    /* Anchor falls in the current winning segment */
                    if (cur_seg != s) {
                        /* Entering a new segment: close previous if open */
                        if (cur_seg >= 0 && n_b > seg_b_start) {
                            u_buf[n_u++] = (seg_score << 32) | (uint64_t)(n_b - seg_b_start);
                            seg_b_start  = n_b;
                            seg_score    = 0;
                        }
                        cur_seg = s;
                    }
                    /* Accumulate q_span (bits 32..39 of a[i].y) as score proxy */
                    seg_score += (uint32_t)(chr_a[j].y >> 32) & 0xffU;
                    b[n_b++]   = chr_a[j];
                }
            }

            /* Close the last open segment */
            if (cur_seg >= 0 && n_b > seg_b_start)
                u_buf[n_u++] = (seg_score << 32) | (uint64_t)(n_b - seg_b_start);

            free(seg_start);
            free(seg_end);
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

        /* Shrink u_buf to actual size */
        uint64_t *u_final;
        KMALLOC(km, u_final, n_u);
        memcpy(u_final, u_buf, n_u * sizeof(uint64_t));
        kfree(km, u_buf);

        /* Write results: b[] is already compacted by segment, ready for
         * mm_gen_regs.  No mg_lchain_dp needed. */
        rd->a   = b;
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
