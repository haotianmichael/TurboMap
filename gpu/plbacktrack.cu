#include "plbacktrack.cuh"
#include "hipify.cuh"
#include <stdio.h>
#include <algorithm>

// Subwarp size for chain backtracking
#define SUBWARP_SIZE 8
#define WARP_SIZE 32
#define MAX_CHAINS_PER_BLOCK 128

// Structure for sorted anchor indices
struct anchor_score_t {
    int32_t score;
    int64_t idx;
};

// Device function: compare for sorting (descending order)
__device__ __forceinline__ bool cmp_anchor_score(const anchor_score_t &a, const anchor_score_t &b) {
    return a.score > b.score;
}

/**
 * @brief Convert relative predecessor index to absolute index
 */
__device__ __forceinline__ int64_t p_rel2abs(uint16_t rel, int64_t idx) {
    return (rel == 0) ? -1 : (idx - rel);
}

/**
 * @brief Find the optimal end point of a chain (GPU version of mg_chain_bk_end)
 *
 * This function backtracks along the predecessor chain starting from anchor k,
 * finding the position with maximum score or stopping when score drops too much.
 *
 * @param max_drop  maximum allowed score drop
 * @param z_score   score of starting anchor
 * @param z_idx     index of starting anchor
 * @param f         anchor scores array
 * @param p_rel     relative predecessor array (0 means no predecessor, otherwise i-p[i])
 * @param t         temporary marker array (0=unvisited, 1=in_chain, 2=temp)
 * @return          index of optimal end position
 */
__device__ int64_t gpu_chain_bk_end(
    int32_t max_drop,
    int32_t z_score,
    int64_t z_idx,
    const int32_t *f,
    const uint16_t *p_rel,
    int32_t *t
) {
    int64_t i = z_idx;
    int64_t end_i = -1;
    int64_t max_i = i;
    int32_t max_s = 0;

    if (i < 0 || t[i] != 0) return i;

    // Backtrack along predecessor chain
    do {
        t[i] = 2;  // Mark as temporarily visited
        end_i = i = p_rel2abs(p_rel[i], i);

        int32_t s = (i < 0) ? z_score : (z_score - f[i]);
        if (s > max_s) {
            max_s = s;
            max_i = i;
        } else if (max_s - s > max_drop) {
            break;
        }
    } while (i >= 0 && t[i] == 0);

    // Reset temporary markers
    for (i = z_idx; i >= 0 && i != end_i; i = p_rel2abs(p_rel[i], i)) {
        t[i] = 0;
    }

    return max_i;
}

/**
 * @brief Backtrack a single chain using a subwarp
 *
 * Uses cooperative parallelism within a subwarp:
 * - Lane 0 (leader) does the main backtracking
 * - Other lanes can assist with validation or counting
 *
 * @param subwarp_id    ID of this subwarp within the block
 * @param lane_id       Lane ID within the subwarp (0 to SUBWARP_SIZE-1)
 * @param k             Index into sorted anchor array
 * @param z             Sorted anchor array (score, index)
 * @param n             Total number of anchors
 * @param f             Anchor scores
 * @param p             Predecessor indices
 * @param v             Output: vertex indices in chains
 * @param t             Temporary marker array
 * @param min_cnt       Minimum chain length
 * @param min_sc        Minimum chain score
 * @param max_drop      Maximum score drop
 * @param chain_start   Output: start index in v array for this chain
 * @param chain_len     Output: length of this chain
 * @param chain_score   Output: score of this chain
 * @return              true if valid chain found
 */
__device__ bool gpu_backtrack_chain_subwarp(
    int subwarp_id,
    int lane_id,
    int64_t k,
    const anchor_score_t *z,
    int64_t n,
    const int32_t *f,
    const uint16_t *p,
    int32_t *v,
    int32_t *t,
    int32_t min_cnt,
    int32_t min_sc,
    int32_t max_drop,
    int64_t *chain_start,
    int32_t *chain_len,
    int32_t *chain_score
) {
    // Only lane 0 does the actual backtracking
    if (lane_id == 0) {
        int64_t anchor_idx = z[k].idx;

        // Check if this anchor is already in a chain
        if (t[anchor_idx] != 0) {
            *chain_len = 0;
            return false;
        }

        // Find optimal end point
        int64_t end_i = gpu_chain_bk_end(max_drop, z[k].score, anchor_idx, f, p, t);

        // Backtrack and collect vertices
        int64_t n_v = 0;
        int64_t i;
        for (i = anchor_idx; i != end_i; i = p[i]) {
            n_v++;
            t[i] = 1;  // Mark as in chain
        }

        // Calculate chain score
        int32_t sc = (i < 0) ? z[k].score : (z[k].score - f[i]);

        // Check if chain meets quality criteria
        if (sc >= min_sc && n_v >= min_cnt) {
            *chain_len = n_v;
            *chain_score = sc;

            // Collect vertices into output array (reverse order)
            int64_t v_idx = *chain_start;
            for (i = anchor_idx; i != end_i; i = p[i]) {
                v[v_idx++] = i;
            }
            return true;
        } else {
            // Invalid chain, reset markers
            for (i = anchor_idx; i != end_i; i = p[i]) {
                t[i] = 0;
            }
            *chain_len = 0;
            return false;
        }
    }

    // Other lanes wait
    return false;
}

/**
 * @brief GPU kernel for chain backtracking using subwarp parallelization
 *
 * Each block processes one read's anchors.
 * Within each block, multiple subwarps process different chain candidates in parallel.
 *
 * This kernel uses a two-pass approach:
 * Pass 1: Count chain lengths to allocate v array space
 * Pass 2: Write actual vertex data
 *
 * @param n          number of anchors
 * @param f          anchor scores
 * @param p          predecessor indices
 * @param v          output: vertex indices
 * @param t          temporary marker array
 * @param min_cnt    minimum chain length
 * @param min_sc     minimum chain score
 * @param max_drop   maximum score drop
 * @param d_n_u      output: number of valid chains
 * @param d_n_v      output: total vertices in all chains
 * @param d_u        output: chain metadata (score << 32 | length)
 * @param pass       0=count pass, 1=write pass
 */
__global__ void chain_backtrack_kernel_pass(
    int64_t n,
    const int32_t *f,
    const uint16_t *p_rel,
    int32_t *v,
    int32_t *t,
    int32_t min_cnt,
    int32_t min_sc,
    int32_t max_drop,
    int32_t *d_n_u,
    int32_t *d_n_v,
    uint64_t *d_u,
    int pass
) {
    // Shared memory for sorted anchors
    __shared__ anchor_score_t s_sorted[MAX_CHAINS_PER_BLOCK];
    __shared__ int64_t s_n_candidates;
    __shared__ int64_t s_v_offsets[MAX_CHAINS_PER_BLOCK + 1];  // Prefix sum of chain lengths
    __shared__ int32_t s_chain_lens[MAX_CHAINS_PER_BLOCK];
    __shared__ int32_t s_chain_scores[MAX_CHAINS_PER_BLOCK];
    __shared__ int64_t s_chain_anchor_idx[MAX_CHAINS_PER_BLOCK];
    __shared__ int64_t s_chain_end_idx[MAX_CHAINS_PER_BLOCK];
    __shared__ int32_t s_n_chains;

    int tid = threadIdx.x;
    int lane_id = tid % SUBWARP_SIZE;
    int subwarp_id = tid / SUBWARP_SIZE;
    int num_subwarps = blockDim.x / SUBWARP_SIZE;

    // Initialize shared counters
    if (tid == 0) {
        s_n_candidates = 0;
        s_n_chains = 0;
    }
    __syncthreads();

    // Step 1: Collect high-scoring anchors (score >= min_sc)
    for (int64_t i = tid; i < n; i += blockDim.x) {
        if (f[i] >= min_sc) {
            int64_t pos = atomicAdd((unsigned long long*)&s_n_candidates, 1ULL);
            if (pos < MAX_CHAINS_PER_BLOCK) {
                s_sorted[pos].score = f[i];
                s_sorted[pos].idx = i;
            }
        }
    }
    __syncthreads();

    // Step 2: Sort candidates by score (descending) using bitonic sort
    int64_t n_cand = min(s_n_candidates, (int64_t)MAX_CHAINS_PER_BLOCK);

    // Bitonic sort for better parallelism
    for (int64_t k = 2; k <= n_cand; k *= 2) {
        for (int64_t j = k / 2; j > 0; j /= 2) {
            for (int64_t i = tid; i < n_cand; i += blockDim.x) {
                int64_t ixj = i ^ j;
                if (ixj > i) {
                    if (ixj < n_cand) {
                        bool should_swap = ((i & k) == 0) ?
                            (s_sorted[i].score < s_sorted[ixj].score) :
                            (s_sorted[i].score > s_sorted[ixj].score);

                        if (should_swap) {
                            anchor_score_t temp = s_sorted[i];
                            s_sorted[i] = s_sorted[ixj];
                            s_sorted[ixj] = temp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Initialize visited markers
    for (int64_t i = tid; i < n; i += blockDim.x) {
        t[i] = 0;
    }
    __syncthreads();

    // Step 3: Process chains - subwarp parallel backtracking
    for (int64_t k = subwarp_id; k < n_cand; k += num_subwarps) {
        if (lane_id == 0) {
            int64_t anchor_idx = s_sorted[k].idx;

            // Check if this anchor is already in a chain
            if (t[anchor_idx] == 0) {
                // Find optimal end point
                int64_t end_i = gpu_chain_bk_end(max_drop, s_sorted[k].score, anchor_idx, f, p_rel, t);

                // Count chain length and mark vertices
                int32_t n_v = 0;
                int64_t i;
                for (i = anchor_idx; i != end_i; i = p_rel2abs(p_rel[i], i)) {
                    n_v++;
                    t[i] = 1;  // Mark as in chain
                }

                // Calculate chain score
                int32_t sc = (i < 0) ? s_sorted[k].score : (s_sorted[k].score - f[i]);

                // Check if chain meets quality criteria
                if (sc >= min_sc && n_v >= min_cnt) {
                    int chain_idx = atomicAdd(&s_n_chains, 1);
                    if (chain_idx < MAX_CHAINS_PER_BLOCK) {
                        s_chain_lens[chain_idx] = n_v;
                        s_chain_scores[chain_idx] = sc;
                        s_chain_anchor_idx[chain_idx] = anchor_idx;
                        s_chain_end_idx[chain_idx] = end_i;
                    }
                } else {
                    // Invalid chain, reset markers
                    for (i = anchor_idx; i != end_i; i = p_rel2abs(p_rel[i], i)) {
                        t[i] = 0;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Step 4: Compute prefix sum of chain lengths
    if (tid < s_n_chains + 1) {
        if (tid == 0) {
            s_v_offsets[0] = 0;
        } else if (tid <= s_n_chains) {
            s_v_offsets[tid] = s_chain_lens[tid - 1];
        }
    }
    __syncthreads();

    // Simple prefix sum (can be optimized with parallel scan)
    for (int stride = 1; stride <= s_n_chains; stride *= 2) {
        int64_t temp = 0;
        if (tid <= s_n_chains && tid >= stride) {
            temp = s_v_offsets[tid] + s_v_offsets[tid - stride];
        }
        __syncthreads();
        if (tid <= s_n_chains && tid >= stride) {
            s_v_offsets[tid] = temp;
        }
        __syncthreads();
    }

    // Step 5: Write results (only in write pass or if single-pass)
    if (pass == 1 || pass == 0) {
        // Write chain metadata
        for (int i = tid; i < s_n_chains; i += blockDim.x) {
            d_u[i] = ((uint64_t)s_chain_scores[i] << 32) | (uint64_t)s_chain_lens[i];
        }

        // Write vertex data - each subwarp handles one chain
        for (int chain_idx = subwarp_id; chain_idx < s_n_chains; chain_idx += num_subwarps) {
            int64_t v_start = s_v_offsets[chain_idx];
            int32_t chain_len = s_chain_lens[chain_idx];
            int64_t anchor_idx = s_chain_anchor_idx[chain_idx];
            int64_t end_idx = s_chain_end_idx[chain_idx];

            // Lane 0 writes the vertices
            if (lane_id == 0) {
                int64_t v_pos = v_start;
                for (int64_t i = anchor_idx; i != end_idx; i = p_rel2abs(p_rel[i], i)) {
                    v[v_pos++] = i;
                }
            }
        }
    }

    // Write totals
    if (tid == 0) {
        *d_n_u = s_n_chains;
        *d_n_v = (s_n_chains > 0) ? s_v_offsets[s_n_chains] : 0;
    }
}

/**
 * @brief Async GPU backtracking wrapper
 */
void plbacktrack_gpu_async(
    int64_t n,
    const int32_t *d_f,
    const uint16_t *d_p_rel,
    int32_t *d_v,
    int32_t *d_t,
    int32_t min_cnt,
    int32_t min_sc,
    int32_t max_drop,
    int32_t *d_n_u,
    int32_t *d_n_v,
    uint64_t *d_u,
    cudaStream_t *stream
) {
    if (n == 0) return;

    // Configure kernel launch parameters
    // Use 256 threads per block = 32 subwarps of size 8
    int block_size = 256;
    int grid_size = 1;  // One block per read for simplicity

    // Launch kernel (single-pass version, pass=0)
    if (stream) {
        chain_backtrack_kernel_pass<<<grid_size, block_size, 0, *stream>>>(
            n, d_f, d_p_rel, d_v, d_t, min_cnt, min_sc, max_drop,
            d_n_u, d_n_v, d_u, 0
        );
    } else {
        chain_backtrack_kernel_pass<<<grid_size, block_size>>>(
            n, d_f, d_p_rel, d_v, d_t, min_cnt, min_sc, max_drop,
            d_n_u, d_n_v, d_u, 0
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Chain backtrack kernel failed: %s\n",
                cudaGetErrorString(err));
    }
}

/**
 * @brief Allocate device memory for backtracking
 */
void plbacktrack_alloc_device_mem(
    int64_t max_n,
    int32_t **d_f,
    uint16_t **d_p_rel,
    int32_t **d_v,
    int32_t **d_t,
    int32_t **d_n_u,
    int32_t **d_n_v,
    uint64_t **d_u,
    void **d_temp_storage,
    size_t *temp_storage_bytes
) {
    cudaMalloc(d_f, max_n * sizeof(int32_t));
    cudaMalloc(d_p_rel, max_n * sizeof(uint16_t));
    cudaMalloc(d_v, max_n * sizeof(int32_t));
    cudaMalloc(d_t, max_n * sizeof(int32_t));
    cudaMalloc(d_n_u, sizeof(int32_t));
    cudaMalloc(d_n_v, sizeof(int32_t));
    cudaMalloc(d_u, max_n * sizeof(uint64_t));  // Conservative estimate

    *d_temp_storage = NULL;
    *temp_storage_bytes = 0;
}

/**
 * @brief Free device memory for backtracking
 */
void plbacktrack_free_device_mem(
    int32_t *d_f,
    uint16_t *d_p_rel,
    int32_t *d_v,
    int32_t *d_t,
    int32_t *d_n_u,
    int32_t *d_n_v,
    uint64_t *d_u,
    void *d_temp_storage
) {
    if (d_f) cudaFree(d_f);
    if (d_p_rel) cudaFree(d_p_rel);
    if (d_v) cudaFree(d_v);
    if (d_t) cudaFree(d_t);
    if (d_n_u) cudaFree(d_n_u);
    if (d_n_v) cudaFree(d_n_v);
    if (d_u) cudaFree(d_u);
    if (d_temp_storage) cudaFree(d_temp_storage);
}