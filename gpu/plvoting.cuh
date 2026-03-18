#ifndef _PLVOTING_CUH_
#define _PLVOTING_CUH_

/*
 * plvoting.cuh  --  GPU-accelerated location-voting re-chaining
 *
 * Implements the Genome-on-Diet inspired "location voting" heuristic as a
 * CORRECT replacement for RMQ-based long-join re-chaining.
 *
 * Architecture:
 *
 *   seed hits (mm128_t a[], already DP-chained)
 *       ↓  prepare_rechain_anchors() – flatten + sort by ref_pos
 *       ↓  GPU voting per chromosome/strand group
 *       ↓  filtered anchor array b[] + chain descriptors u[]
 *       ↓  mm_gen_regs → mm_align1_batched (GPU KSW)   ← normal path, unchanged
 *
 * Voting output feeds directly into the existing GPU alignment pipeline.
 * Cross-chromosome safety: each chromosome group is processed independently;
 * anchors from different chromosomes never mix within a single u[] chain entry.
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
#include "plutils.h"   /* chain_read_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * plvoting_rechain_batch - voting-based GPU re-chaining
 *
 * For each read in rechain_indices runs the GPU voting pipeline and stores
 * the filtered anchor array (b[]) in rd->a and chain descriptors (u[]) in
 * rd->u.  The read then flows through the normal mm_gen_regs →
 * mm_align1_batched (GPU KSW) path without any special handling.
 */
void plvoting_rechain_batch(const mm_idx_t *mi, const mm_mapopt_t *opt,
                            chain_read_t *reads, int *rechain_indices,
                            int n_rechain, Misc misc, void *km,
                            cudaStream_t stream, deviceMemPtr *dev_mem);

#ifdef __cplusplus
}
#endif

#endif /* _PLVOTING_CUH_ */
