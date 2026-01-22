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
 * This function performs chain backtracking on the GPU, converting
 * forward chaining scores into actual chains. It filters anchors,
 * sorts them by score, performs backtracking, and generates final
 * chain regions.
 *
 * @param host_mem Host memory structure containing scores and predecessors
 * @param dev_mem Device memory structure
 * @param reads Array of chain reads
 * @param misc Chaining parameters
 * @param km Memory pool
 * @param stream CUDA stream for async execution
 */
void plbacktrack_gpu(hostMemPtr *host_mem, deviceMemPtr *dev_mem,
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
