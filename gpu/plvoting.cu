/*
 * plvoting.cu  --  GPU-accelerated location-voting re-chaining
 *
 * Replaces the RMQ-based long-join re-chaining (mg_lchain_rmq) with a hybrid
 * GPU-voting + CPU-DP pipeline.  See plvoting.cuh for the algorithm description.
 *
 * Design notes
 * ------------
 * The GPU side does only the *voting histogram* – a single embarrassingly parallel
 * pass that is a natural fit for atomics-heavy GPU workloads.  The subsequent DP
 * (mg_lchain_dp) runs on the CPU but operates on a *filtered* anchor set that is
 * typically much smaller than the full set, so it is fast in practice.
 *
 * Using bw_long in the CPU DP instead of a hard-coded bandwidth reproduces the
 * key semantic of the original mg_lchain_rmq call in post_chaining_helper.
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
 * The kernel is chromosome-agnostic: the caller is responsible for invoking it
 * once per chromosome group so that ref_min is meaningful.
 *
 * @param d_ref_pos  Lower 32 bits of a[i].x for each anchor in this chr group
 * @param n          Number of anchors in this chr group
 * @param ref_min    Minimum reference position (= d_ref_pos[0] since sorted)
 * @param bin_size   Width of each bin in reference bases
 * @param n_bins     Total number of bins (pre-computed by host)
 * @param d_votes    Output: per-bin vote counts (caller zero-initialises)
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
    /* Guard: clamp to valid range (should not happen with correct n_bins, but
     * be defensive against off-by-one at the upper boundary). */
    if (bin >= 0 && bin < n_bins)
        atomicAdd(&d_votes[bin], 1);
}

/* =========================================================================
 * Host helpers
 * ========================================================================= */

/**
 * find_winning_segments
 *
 * Scans the vote histogram and identifies "winning" reference intervals.
 * A bin is winning if votes[bin] >= min_votes.  Adjacent winning bins (or bins
 * separated by a gap of <= merge_gap_bins empty bins) are merged into a single
 * segment (implements the small-gap merging from the Genome-on-Diet paper).
 *
 * @param votes          Vote counts per bin
 * @param n_bins         Number of bins
 * @param ref_min        Reference position of the left edge of bin 0
 * @param bin_size       Bin width in reference bases
 * @param min_votes      Minimum votes for a bin to be "winning"
 * @param merge_gap_bins Maximum gap (in bins) between winning runs to still merge
 * @param seg_start      Output: reference start of each winning segment
 * @param seg_end        Output: reference end   of each winning segment
 * @param max_segs       Capacity of the output arrays
 * @return               Number of winning segments found
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
        /* Skip non-winning bins */
        if (votes[i] < min_votes) { ++i; continue; }

        /* Start of a winning run */
        int32_t win_start = i;
        int32_t win_end   = i;
        int32_t gap       = 0;

        /* Extend, bridging gaps of up to merge_gap_bins */
        while (i < n_bins) {
            if (votes[i] >= min_votes) {
                win_end = i;
                gap = 0;
            } else {
                if (++gap > merge_gap_bins) break;
            }
            ++i;
        }

        /* Record segment: ref positions covered by bins [win_start, win_end] */
        seg_start[n_segs] = ref_min + win_start * bin_size;
        /* +1 to win_end so we include all anchors at the right edge of the bin */
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
 *  1. Run GPU voting histogram over anchors (grouped by chromosome).
 *  2. Identify winning reference segments on the CPU.
 *  3. Filter the anchor array to only anchors in winning segments.
 *  4. Run mg_lchain_dp with bw_long bandwidth on the filtered anchors.
 *  5. Write updated chain data back into reads[].
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km)
{
    if (n_rechain == 0) return;

    /* Consolidate and sort each read's anchors before voting.
     * prepare_rechain_anchors() flattens all chain anchors into a single
     * sorted array and resets n_u / u so the read is ready for re-chaining. */
    for (int i = 0; i < n_rechain; ++i)
        prepare_rechain_anchors(&reads[rechain_indices[i]], km);

    /* --- Voting parameters ---
     *
     * bin_size: one-tenth of max_dist gives ~10 voting windows per max-gap
     *           interval, giving fine enough resolution to detect seed clusters
     *           without wasting too many bins.
     *
     * min_votes: require at least min_cnt anchors in a bin for it to be
     *            "winning" – the same threshold mg_lchain_dp uses for a valid
     *            chain.
     *
     * merge_gap_bins: number of consecutive empty bins below the vote threshold
     *                 that are still bridged (implements the Genome-on-Diet 50 kb
     *                 large-gap boundary). */
    const int32_t max_dist = (opt->max_gap > 0) ? opt->max_gap : 10000;
    const int32_t bin_size = (max_dist / VOTING_BIN_DIVIDER > 0)
                             ? (max_dist / VOTING_BIN_DIVIDER) : 1;
    const int32_t min_votes      = (opt->min_cnt > 1) ? opt->min_cnt : 2;
    const int32_t merge_gap_bins = (VOTING_LARGE_GAP + bin_size - 1) / bin_size;

    /* GPU kernel launch parameters (constant for all reads) */
    const int blk = 256;

    for (int ri = 0; ri < n_rechain; ++ri) {
        int idx          = rechain_indices[ri];
        chain_read_t *rd = &reads[idx];

        int64_t  n_a = rd->n;
        mm128_t *a   = rd->a;

        if (n_a == 0 || a == NULL) continue;

        /* Allocate output buffer for filtered anchors (worst case: keep all) */
        mm128_t *b;
        KMALLOC(km, b, n_a);
        int64_t n_b = 0;

        /* ------------------------------------------------------------------ *
         * Process each chromosome group independently.                        *
         * Anchors are sorted by a[i].x = (xrev << 32 | ref_pos), so           *
         * consecutive anchors with the same xrev form one chr group.          *
         * ------------------------------------------------------------------ */
        int64_t chr_start = 0;
        while (chr_start < n_a) {
            /* Find end of this chromosome group */
            int32_t xrev = (int32_t)(a[chr_start].x >> 32);
            int64_t chr_end = chr_start;
            while (chr_end < n_a && (int32_t)(a[chr_end].x >> 32) == xrev)
                ++chr_end;

            int64_t       chr_n = chr_end - chr_start;
            const mm128_t *chr_a = a + chr_start;

            /* Reference range for this chromosome group (sorted, so min/max
             * are at the endpoints) */
            int32_t ref_min  = (int32_t)chr_a[0].x;
            int32_t ref_max  = (int32_t)chr_a[chr_n - 1].x;
            int32_t ref_span = ref_max - ref_min + 1;
            int32_t n_bins   = (ref_span + bin_size - 1) / bin_size + 1;

            /* ---- GPU: upload reference positions and run voting ---- */
            int32_t *h_ref_pos = (int32_t *)malloc(chr_n * sizeof(int32_t));
            for (int64_t j = 0; j < chr_n; ++j)
                h_ref_pos[j] = (int32_t)chr_a[j].x;

            int32_t *d_ref_pos = NULL;
            int32_t *d_votes   = NULL;
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

            /* Download vote histogram */
            int32_t *h_votes = (int32_t *)malloc(n_bins * sizeof(int32_t));
            cudaMemcpy(h_votes, d_votes, n_bins * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_ref_pos);
            cudaFree(d_votes);

            /* ---- CPU: find winning segments ---- */
            int      max_segs  = n_bins + 1; /* upper bound */
            int32_t *seg_start = (int32_t *)malloc(max_segs * sizeof(int32_t));
            int32_t *seg_end   = (int32_t *)malloc(max_segs * sizeof(int32_t));

            int n_segs = find_winning_segments(h_votes, n_bins,
                                               ref_min, bin_size,
                                               min_votes, merge_gap_bins,
                                               seg_start, seg_end, max_segs);
            free(h_votes);

            if (n_segs == 0) {
                /* Conservative fallback: no winning segments found for this
                 * chromosome, keep all its anchors so they can still form
                 * chains in the DP step. */
                for (int64_t j = 0; j < chr_n; ++j)
                    b[n_b++] = chr_a[j];
            } else {
                /* ---- CPU: filter anchors to winning segments ----
                 *
                 * Both the anchor array (sorted by ref_pos within chromosome)
                 * and the segment list (sorted by seg_start) are monotonically
                 * increasing, so a two-pointer scan is O(chr_n + n_segs).   */
                int s = 0;
                for (int64_t j = 0; j < chr_n; ++j) {
                    int32_t rpos = (int32_t)chr_a[j].x;
                    /* Advance the segment pointer past segments that end before
                     * this anchor */
                    while (s < n_segs && seg_end[s] <= rpos) ++s;
                    /* Keep the anchor if it falls within the current segment */
                    if (s < n_segs && rpos >= seg_start[s])
                        b[n_b++] = chr_a[j];
                }
            }

            free(seg_start);
            free(seg_end);
            chr_start = chr_end;
        } /* end chromosome loop */

        /* Free the original (now consumed) anchor array */
        kfree(km, a);
        rd->a = NULL;

        if (n_b == 0) {
            /* Voting retained nothing – discard this read's chains */
            kfree(km, b);
            rd->n   = 0;
            rd->n_u = 0;
            rd->u   = NULL;
            continue;
        }

        /* ---- CPU: DP chaining on filtered anchors with wide bandwidth ----
         *
         * Use opt->bw_long (the key difference vs. the narrow-bw DP that the
         * GPU forward pass already ran) so that we can bridge the larger indels
         * that motivated the long-join re-chaining step in the first place.
         *
         * mg_lchain_dp takes ownership of b[] and will free it internally. */
        int      n_u_new = 0;
        uint64_t *u_new  = NULL;

        mm128_t *new_a = mg_lchain_dp(
            opt->max_gap,           /* max_dist_x */
            opt->max_gap,           /* max_dist_y */
            opt->bw_long,           /* bandwidth – wider than the initial pass */
            opt->max_chain_skip,    /* max_skip */
            opt->max_chain_iter,    /* max_iter */
            opt->min_cnt,           /* min_cnt */
            opt->min_chain_score,   /* min_sc */
            misc.chn_pen_gap,       /* gap penalty */
            misc.chn_pen_skip,      /* skip penalty */
            misc.is_cdna,           /* is_cdna (0 for long-join path) */
            1,                      /* n_segs = 1 */
            n_b,                    /* number of filtered anchors */
            b,                      /* filtered anchors (owned by mg_lchain_dp) */
            &n_u_new,
            &u_new,
            km);

        /* Write results back into the read struct */
        rd->a   = new_a;
        rd->n   = n_b;   /* mg_lchain_dp shrinks this via compact_a; keep n_b
                          * as a safe upper bound – downstream code uses n_u. */
        rd->u   = u_new;
        rd->n_u = n_u_new;
    } /* end per-read loop */

#ifdef DEBUG_PRINT
    fprintf(stderr,
            "[Info::%s] voting re-chained %d reads "
            "(bin_size=%d, min_votes=%d, merge_gap_bins=%d)\n",
            __func__, n_rechain, bin_size, min_votes, merge_gap_bins);
#endif
}

} /* extern "C" */
