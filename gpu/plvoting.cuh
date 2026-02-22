#ifndef _PLVOTING_CUH_
#define _PLVOTING_CUH_

/*
 * plvoting.cuh  --  GPU-accelerated location-voting re-chaining
 *
 * Implements the Genome-on-Diet inspired "location voting" heuristic to replace
 * RMQ-based long-join re-chaining in post_chaining_helper.
 *
 * Algorithm overview (per read):
 *   1. [GPU] Voting kernel: bin each anchor by its reference position and
 *      atomically accumulate vote counts per bin.
 *   2. [CPU] Find "winning" bins (vote_count >= min_cnt).  Merge consecutive
 *      winning bins separated by a gap <= VOTING_LARGE_GAP bases (the 50 kb
 *      threshold from the Genome-on-Diet paper).
 *   3. [CPU] Filter the anchor array: keep only anchors whose reference
 *      position falls inside a winning segment.
 *   4. [CPU] Run mg_lchain_dp with bw_long bandwidth on the filtered anchors.
 *
 * The key accuracy trade-off: anchors in sparse, low-vote regions are dropped.
 * Chains that span only such regions will be missed, but high-coverage regions
 * (which are the important ones for long reads) are retained.
 *
 * The GPU voting step replaces the KRMQ tree entirely: instead of O(n log n)
 * tree construction and O(log n) range-maximum queries, we do a single O(n)
 * histogram pass on GPU, followed by O(B) CPU post-processing (B = #bins).
 */

#include "plmem.cuh"
#include "mmpriv.h"
#include "plutils.h"

/* Width of each voting bin expressed as a divisor of max_dist.
 * bin_size = max_dist / VOTING_BIN_DIVIDER
 * Smaller values give finer resolution at the cost of more bins. */
#define VOTING_BIN_DIVIDER  10

/* Reference-gap threshold (bases) below which two winning segments are merged
 * into a single chain candidate (analogous to the 50 kb threshold in the
 * Genome-on-Diet pseudocode). */
#define VOTING_LARGE_GAP    50000

#ifdef __cplusplus
extern "C" {
#endif

/**
 * plvoting_rechain_batch - voting-based GPU re-chaining (replaces gpu_rechain_batch)
 *
 * For each read in rechain_indices:
 *   - Anchors have already been consolidated and sorted by prepare_rechain_anchors()
 *   - Runs GPU voting + CPU filter + CPU mg_lchain_dp(bw_long)
 *   - Writes updated a[], u[], n, n_u back into reads[]
 *
 * @param mi               Reference index (for penalty parameters)
 * @param opt              Mapping options (bw_long, max_gap, etc.)
 * @param reads            Array of all reads
 * @param rechain_indices  Indices into reads[] that need re-chaining
 * @param n_rechain        Number of reads to re-chain
 * @param misc             Chaining misc parameters (chn_pen_gap, chn_pen_skip, ...)
 * @param km               kalloc memory pool
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km);

#ifdef __cplusplus
}
#endif

#endif /* _PLVOTING_CUH_ */
