#ifndef _PLBACKTRACK_CUH_
#define _PLBACKTRACK_CUH_

#include <stdint.h>
#include <cuda_runtime.h>
#include "plutils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU-based chain backtracking using subwarp parallelization
 *
 * @param n          number of anchors
 * @param f          anchor scores (device memory)
 * @param p_rel      relative predecessor indices (device memory, uint16_t)
 * @param v          peak scores (device memory, output)
 * @param t          visited markers (device memory, temp)
 * @param min_cnt    minimum chain length
 * @param min_sc     minimum chain score
 * @param max_drop   maximum score drop threshold
 * @param n_u        output: number of chains (device memory)
 * @param n_v        output: number of vertices in chains (device memory)
 * @param u          output: chain info (device memory)
 * @param stream     CUDA stream for async execution
 */
void plbacktrack_gpu_async(
    int64_t n,
    const int32_t *d_f,
    const int64_t *d_p_rel,
    int32_t *d_v,
    int32_t *d_t,
    int32_t min_cnt,
    int32_t min_sc,
    int32_t max_drop,
    int32_t *d_n_u,
    int32_t *d_n_v,
    uint64_t *d_u,
    cudaStream_t *stream
);

/**
 * @brief Allocate device memory for backtracking
 */
void plbacktrack_alloc_device_mem(
    int64_t max_n,
    int32_t **d_f,
    int64_t **d_p_rel,
    int32_t **d_v,
    int32_t **d_t,
    int32_t **d_n_u,
    int32_t **d_n_v,
    uint64_t **d_u,
    void **d_temp_storage,
    size_t *temp_storage_bytes
);

/**
 * @brief Free device memory for backtracking
 */
void plbacktrack_free_device_mem(
    int32_t *d_f,
    int64_t *d_p_rel,
    int32_t *d_v,
    int32_t *d_t,
    int32_t *d_n_u,
    int32_t *d_n_v,
    uint64_t *d_u,
    void *d_temp_storage
);

#ifdef __cplusplus
}
#endif

#endif // _PLBACKTRACK_CUH_