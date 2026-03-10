#ifndef _PLBACKTRACK_CUH_
#define _PLBACKTRACK_CUH_

#include "plmem.cuh"
#include "mmpriv.h"
#include "plutils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU kernel wrapper for chain backtracking
 *
 * Performs chain backtracking on the GPU for all reads in a batch.
 * Filters anchors, sorts by score, backtrack, and generates chain regions.
 *
 * @param n_reads Number of reads to backtrack
 * @param total_n Total number of anchors across all reads
 * @param dev_mem Device memory (d_ax/d_ay/d_f/d_p must contain concatenated data)
 * @param reads Array of chain reads
 * @param misc Chaining parameters
 * @param km Memory pool
 * @param stream CUDA stream for async execution
 */
void plbacktrack_gpu(int n_reads, size_t total_n, deviceMemPtr *dev_mem,
                     chain_read_t *reads, Misc misc,
                     void* km, cudaStream_t stream);

/**
 * @brief Initialize GPU backtracking memory
 *
 * @param dev_mem Device memory structure to initialize
 * @param max_anchors Maximum number of anchors per stream
 */
void plbacktrack_init_memory(deviceMemPtr *dev_mem, size_t max_anchors);

/**
 * @brief Free GPU backtracking memory
 *
 * @param dev_mem Device memory structure to free
 */
void plbacktrack_free_memory(deviceMemPtr *dev_mem);

#ifdef __cplusplus
}
#endif

#endif // _PLBACKTRACK_CUH_
