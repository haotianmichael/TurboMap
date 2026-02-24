#ifndef _PLVOTING_CUH_
#define _PLVOTING_CUH_

/*
 * plvoting.cuh  --  GPU-accelerated location-voting re-chaining
 *
 * Implements the Genome-on-Diet inspired "location voting" heuristic as a
 * CORRECT replacement for RMQ-based long-join re-chaining.
 *
 * Correct architecture (operates on raw seed hits, not chain arrays):
 *
 *   seed hits (mm128_t a[], already DP-chained)
 *       ↓  prepare_rechain_anchors() – flatten + sort by ref_pos
 *       ↓  GPU voting per chromosome group
 *       ↓  vt_t regions (rid + ref_range + qry_range)
 *       ↓  mm_voting_align_regions() – direct KSW per region (NOT mm_align1_batched)
 *       ↓  CIGAR concatenation via mm_append_cigar()
 *
 * Critical invariants:
 *   1. Voting output is vt_t[] (regions), NOT mm128_t anchor arrays.
 *      Voting results are NEVER re-merged into an anchor chain and NEVER
 *      passed to mm_align1_batched (which assumes same-chromosome colinear
 *      anchors and would overflow on mixed-chromosome input).
 *   2. Each vt_t carries its own rid/rev; tseq buffer = tseq[re-rs] is
 *      always safe because rs/re come from actual anchor positions on rid.
 *   3. Cross-chromosome safety: grouping by xrev = (int32_t)(x >> 32) keeps
 *      anchors from different chromosomes in separate groups.
 *
 * Width of each voting bin expressed as a divisor of max_dist.
 * bin_size = max_dist / VOTING_BIN_DIVIDER
 */
#define VOTING_BIN_DIVIDER  10

/* Reference-gap threshold (bases) below which two winning segments are merged
 * into a single chain candidate (analogous to the 50 kb threshold in the
 * Genome-on-Diet pseudocode). */
#define VOTING_LARGE_GAP    50000

#include "plmem.cuh"
#include "mmpriv.h"
#include "plutils.h"   /* chain_read_t, vt_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * plvoting_rechain_batch - voting-based GPU re-chaining (seed-hit layer)
 *
 * For each read in rechain_indices:
 *   - Runs the full GPU voting pipeline to identify high-coverage ref regions.
 *   - Outputs vt_t regions into reads[i].vt_regions / reads[i].n_vt.
 *   - Does NOT produce anchor arrays (b[]) or chain descriptors (u[]).
 *   - Does NOT call mm_align1_batched; alignment is done by
 *     mm_voting_align_regions() in map.c via direct KSW calls.
 *
 * @param mi               Reference index
 * @param opt              Mapping options (max_gap, min_cnt, etc.)
 * @param reads            Array of all reads (chain_read_t)
 * @param rechain_indices  Indices into reads[] that need re-chaining
 * @param n_rechain        Number of reads to re-chain
 * @param misc             Chaining misc parameters (kept for API compatibility)
 * @param km               kalloc memory pool
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km);

#ifdef __cplusplus
}
#endif

#endif /* _PLVOTING_CUH_ */
