#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "plbacktrack.cuh"
#include "hipify.cuh"
#include "mmpriv.h"

// Kernel configuration constants
#define BLOCK_NUM_SHORT 256
#define THREAD_NUM_SHORT 256

// Device helper functions
__device__ static inline uint64_t hash64(uint64_t key)
{
    key = (~key + (key << 21));
    key = key ^ key >> 24;
    key = ((key + (key << 3)) + (key << 8));
    key = key ^ key >> 14;
    key = ((key + (key << 2)) + (key << 4));
    key = key ^ key >> 28;
    key = (key + (key << 31));
    return key;
}

__device__ static inline uint32_t __ac_Wang_hash(uint32_t key) {
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}

// Note: mm_cal_fuzzy_len and mm_reg_set_coor removed
// These are not needed since region generation happens on CPU

__device__ static int64_t mg_chain_bk_end(int32_t max_drop, const int64_t *z_x,
                                          const int64_t *z_y, const int32_t *f,
                                          const int64_t *p, int32_t *t, int64_t k)
{
    int64_t i = z_y[k], end_i = -1, max_i = i;
    int32_t max_s = 0;
    if (i < 0 || t[i] != 0) return i;
    do {
        int32_t s;
        t[i] = 2;
        end_i = i = p[i];  // p[i] is absolute index (converted by expand_p_to_int64)
        s = i < 0? (int32_t)z_x[k] : (int32_t)z_x[k] - f[i];
        if (s > max_s)
            max_s = s, max_i = i;
        else if (max_s - s > max_drop)
            break;
    } while (i >= 0 && t[i] == 0);
    for (i = z_y[k]; i >= 0 && i != end_i; i = p[i])  // p[i] is absolute index
        t[i] = 0;
    return max_i;
}

// Kernel 1: Filter anchors by score and populate z arrays
//
// Optimization: one block per task.
//   - Coalesced reads: all blockDim.x threads read g_sc[ofs + tid..+blockDim.x-1]
//   - Parallel t[] zero: blockDim.x threads cover blockDim.x elements per step
//   - Shared-memory atomic for the output index k: smem atomics serialize
//     within the SM in ~5 cycles vs serial k++ which cannot overlap iterations.
//     Output order before the CUB sort is arbitrary – the sort fixes it anyway.
__global__ void mm_filter_anchors(int* n_a, int* offset, int32_t min_sc,
                                   int32_t* g_sc, int64_t* g_zx, int64_t* g_zy,
                                   int* ofs_end, int32_t* g_t, int* num_elements,
                                   int* g_n_v, int n_task)
{
    __shared__ int s_k;  // shared output index counter

    int task_id = blockIdx.x;
    if (task_id >= n_task) return;

    int ofs = offset[task_id];
    int n   = n_a[task_id];
    int32_t* f   = &g_sc[ofs];
    int64_t* z_x = &g_zx[ofs];
    int64_t* z_y = &g_zy[ofs];
    int32_t* t   = &g_t[ofs];

    if (threadIdx.x == 0) s_k = 0;
    __syncthreads();

    // Parallel t[] zero – coalesced writes, blockDim.x elements per step
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        t[j] = 0;
    __syncthreads();

    // Filter g_sc by min_sc; use smem atomic to claim output slot k
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (f[i] >= min_sc) {
            int k = atomicAdd(&s_k, 1);   // shared-mem atomic – very fast
            z_x[k] = (int64_t)f[i];
            z_y[k] = (int64_t)i;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n_z = s_k;
        num_elements[task_id] = n_z;
        ofs_end[task_id]      = ofs + n_z;
        if (n_z <= 0) g_n_v[task_id] = 0;
    }
}

// Kernel 2: Backtrack filtered anchors in parallel
__global__ void mm_chain_backtrack_parallel(int* n_a, int32_t* g_ax, int32_t* g_ay, int32_t* g_xrev, int32_t* g_yrev,
                                            int32_t* g_sc, int64_t *g_p, uint64_t* g_u,
                                            int64_t* g_zx, int64_t* g_zy, int32_t* g_t,
                                            int64_t* g_v, int* offset, int32_t min_cnt,
                                            int32_t min_sc, int32_t max_drop,
                                            int n_task, int* g_n_v, int* g_n_u,
                                            int* num_elements, int* ofs_end)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for(int job_idx = bid; job_idx < n_task; job_idx += gridDim.x) {
        int ofs = offset[job_idx];
        int n_z = num_elements[job_idx];

        if (n_z <= 0) {
            if (tid == 0) {
                g_n_v[job_idx] = 0;
                g_n_u[job_idx] = 0;
            }
            continue;
        }

        uint64_t* u = &g_u[ofs];
        int32_t* t = &g_t[ofs];
        int64_t* v = &g_v[ofs];
        int n = n_a[job_idx];

        int64_t* z_x = &g_zx[ofs];
        int64_t* z_y = &g_zy[ofs];
        int32_t* ax = &g_ax[ofs];
        int32_t* ay = &g_ay[ofs];
        int32_t* xrev = &g_xrev[ofs];
        int32_t* yrev = &g_yrev[ofs];
        int32_t* f = &g_sc[ofs];
        int64_t* p = &g_p[ofs];

        int n_v = 0, n_u = 0;

        if (tid == 0) {
            // Backtrack to populate u[]
            // Note: z arrays should already be sorted by calling code using CUB
            // Note: p[] array contains absolute indices (converted by expand_p_to_int64)
            for (int k = n_z - 1; k >= 0; --k) {
                if (t[z_y[k]] == 0) {
                    int64_t n_v0 = n_v, end_i;
                    int32_t sc;
                    end_i = mg_chain_bk_end(max_drop, z_x, z_y, f, p, t, k);
                    for (int64_t i = z_y[k]; i != end_i; i = p[i])  // p[i] is absolute index
                        v[n_v++] = i, t[i] = 1;
                    sc = end_i < 0? (int32_t)z_x[k] : (int32_t)z_x[k] - f[end_i];
                    if (sc >= min_sc && n_v > n_v0 && n_v - n_v0 >= min_cnt)
                        u[n_u++] = (uint64_t)sc << 32 | (n_v - n_v0);
                    else n_v = n_v0;
                }
            }

            g_n_v[job_idx] = n_v;
            g_n_u[job_idx] = n_u;
        }
        __syncthreads();

        n_v = g_n_v[job_idx];
        n_u = g_n_u[job_idx];

        // Reuse allocated space for compact_a operation
        int64_t *b_x = z_x;
        int64_t *b_y = z_y;
        int64_t *w_x = p;
        int64_t *w_y = v;

        // Compact anchors (parallel version) - reconstruct full 64-bit values
        for (int i = 0, k = 0; i < n_u; ++i) {
            int32_t k0 = k, ni = (int32_t)u[i];
            for(int j = tid; j < ni; j += blockDim.x){
                int v_idx = v[k0 + (ni-j-1)];
                // Reconstruct complete 64-bit anchor values
                b_x[k + j] = ((uint64_t)xrev[v_idx] << 32) | (uint32_t)ax[v_idx];
                b_y[k + j] = ((uint64_t)yrev[v_idx] << 32) | (uint32_t)ay[v_idx];
            }
            k += ni;
            __syncthreads();
        }

        if(tid == 0) {
            // Prepare to sort by target position
            for (int i = 0, k = 0; i < n_u; ++i) {
                w_x[i] = b_x[k];
                w_y[i] = (uint64_t)k<<32|i;
                k += (int32_t)u[i];
            }
            // NOTE: Sorting will be done OUTSIDE this kernel from host code
            // Cannot call thrust::sort from device code (kernel)
            ofs_end[job_idx] = ofs + n_u;
        }
        __syncthreads();
    }
}

// Kernel 3: Set chain anchor information
// Note: Sorting is done separately using CUB
__global__ void mm_set_chain(int* g_na, int n_task, int32_t* g_ax, int32_t* g_ay, int32_t* g_xrev, int32_t* g_yrev,
                             int* offset, int* ofs_end, int32_t* g_sc, int64_t *g_p,
                             uint64_t* g_u, int64_t* g_zx, int64_t* g_zy,
                             int32_t* g_t, int64_t* g_v, int* g_n_v)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    for(int job_idx = bid; job_idx < n_task; job_idx += gridDim.x) {
        int ofs = offset[job_idx];
        int n_u = ofs_end[job_idx] - ofs;

        int64_t *b_x = &g_zx[ofs];
        int64_t *b_y = &g_zy[ofs];
        int64_t *w_x = &g_p[ofs];
        int64_t *w_y = &g_v[ofs];
        // Reuse g_p for u2 since w_x is no longer needed after sorting
        int64_t *u2 = &g_p[ofs];
        uint64_t* u = &g_u[ofs];
        int32_t* ax = &g_ax[ofs];
        int32_t* ay = &g_ay[ofs];
        int32_t* xrev = &g_xrev[ofs];
        int32_t* yrev = &g_yrev[ofs];
        int n_a = g_na[job_idx];
        int n_v = g_n_v[job_idx];  // CRITICAL: Number of anchors in b[] array

        // w_x and w_y should already be sorted by calling code using CUB

        // Copy sorted anchors back - decompose 64-bit values into ax/ay/xrev/yrev
        // k tracks the write position (sum of all previous chain lengths)
        __shared__ int k_shared;

        for (int i = 0; i < n_u; ++i) {
            int32_t j = (int32_t)w_y[i], n = (int32_t)u[j];
            if(tid == 0) {
                u2[i] = u[j];
                // Compute k for this chain (sum of lengths of all previous sorted chains)
                if (i == 0) {
                    k_shared = 0;
                } else {
                    // k was already set at end of previous iteration
                }
            }
            __syncthreads();

            int k = k_shared;  // All threads read the same k value
            uint32_t b_offset = (uint32_t)(w_y[i]>>32);

            for(int x = tid; x < n; x += blockDim.x){
                uint32_t b_idx = b_offset + x;

                // CRITICAL bounds check: b[] array has n_v elements, not n_a!
                // n_v is the number of anchors after backtracking
                // If b_idx >= n_v, we'd be reading uninitialized/garbage data!
                if (b_idx >= n_v) {
                    continue;
                }

                uint64_t b_x_val = b_x[b_idx];
                uint64_t b_y_val = b_y[b_idx];

                int out_idx = k + x;
                if (out_idx >= n_a) {
                    continue;
                }

                ax[out_idx] = (int32_t)b_x_val;  // Low 32 bits
                xrev[out_idx] = (int32_t)(b_x_val >> 32);  // High 32 bits
                ay[out_idx] = (int32_t)b_y_val;  // Low 32 bits
                yrev[out_idx] = (int32_t)(b_y_val >> 32);  // High 32 bits
            }

            // Update k for next iteration
            if(tid == 0) {
                k_shared += n;
            }
            __syncthreads();
        }
        __syncthreads();

        // Clear remaining anchors
        for(int x = tid; x < n_u; x += blockDim.x){
            u[x] = u2[x];
        }
        __syncthreads();
    }
}

// Note: mm_gen_regs kernels removed - region generation happens on CPU
// This matches the CPU version behavior in lchain.c where mg_chain_backtrack
// returns the compacted anchor array and u metadata, then the CPU generates regs

// Helper to expand uint16_t predecessor to int64_t and convert to absolute index
// This matches CPU version's p_rel2idx function
//
// Optimization: one block per task → all threads in a block access consecutive
// p_rel[ofs+i..ofs+i+255] → coalesced 16-bit reads; same for int64_t writes.
// Old: one thread per task with serial inner loop → uncoalesced + no ILP.
__global__ void expand_p_to_int64(uint16_t* p_rel, int64_t* p_expanded, int* offset, int* n_a, int n_task)
{
    int task_id = blockIdx.x;
    if (task_id >= n_task) return;

    int ofs = offset[task_id];
    int n   = n_a[task_id];

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint16_t r = p_rel[ofs + i];
        p_expanded[ofs + i] = (r == 0) ? -1 : (int64_t)(i - (int)r);
    }
}

// Gather kernel: extract ay values needed for needs_rmq_rechain check.
// For each read, extracts ay_out[offset] (first anchor) and
// ay_out[offset + chain0_len - 1] (last anchor of first chain).
__global__ void gather_rechain_ay_kernel(
    const int32_t  *d_ay_out,
    const int      *d_offset,
    const int      *d_n_u,
    const uint64_t *d_u,
    int32_t        *d_ay_first,
    int32_t        *d_ay_last,
    int             n_reads)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_reads) return;

    int n_u = d_n_u[i];
    if (n_u <= 0) {
        d_ay_first[i] = 0;
        d_ay_last[i]  = 0;
        return;
    }

    int ofs = d_offset[i];
    d_ay_first[i] = d_ay_out[ofs];

    int chain0_len = (int32_t)d_u[ofs];  // low 32 bits of u[0]
    if (chain0_len > 0)
        d_ay_last[i] = d_ay_out[ofs + chain0_len - 1];
    else
        d_ay_last[i] = d_ay_out[ofs];
}

// Host function to orchestrate backtracking.
// Uses pre-allocated device buffers from dev_mem->d_bt_* — no cudaMalloc in hot path.
void plbacktrack_gpu(int n_reads, size_t total_n, deviceMemPtr *dev_mem,
                     chain_read_t *reads, Misc misc,
                     void* km, cudaStream_t stream)
{
    if (n_reads == 0) return;

    assert(total_n     <= dev_mem->d_bt_max_total_n);
    assert((size_t)n_reads <= dev_mem->d_bt_max_n_reads);

    int min_sc  = misc.min_score;
    int min_cnt = misc.min_cnt;
    int max_drop = misc.bw;

    // Use pre-allocated device buffers (no cudaMalloc/cudaFree in hot path)
    int64_t  *d_zx          = dev_mem->d_bt_zx;
    int64_t  *d_zy          = dev_mem->d_bt_zy;
    int64_t  *d_v           = dev_mem->d_bt_v;
    int64_t  *d_p_abs       = dev_mem->d_bt_p_abs;
    int32_t  *d_t           = dev_mem->d_bt_t;
    uint64_t *d_u           = dev_mem->d_bt_u;
    int32_t  *d_ax_out      = dev_mem->d_bt_ax_out;
    int32_t  *d_ay_out      = dev_mem->d_bt_ay_out;
    int32_t  *d_xrev_out    = dev_mem->d_bt_xrev_out;
    int32_t  *d_yrev_out    = dev_mem->d_bt_yrev_out;
    int      *d_n_a         = dev_mem->d_bt_n_a;
    int      *d_offset      = dev_mem->d_bt_offset;
    int      *d_ofs_end     = dev_mem->d_bt_ofs_end;
    int      *d_num_elements= dev_mem->d_bt_num_elements;
    int      *d_n_v         = dev_mem->d_bt_n_v;
    int      *d_n_u         = dev_mem->d_bt_n_u;
    void     *d_cub_tmp     = dev_mem->d_bt_cub_tmp;
    size_t    cub_tmp_size  = dev_mem->d_bt_cub_tmp_size;

    // Zero output buffers (async, same stream — no sync needed before kernels)
    cudaMemsetAsync(d_ax_out,   0, sizeof(int32_t) * total_n, stream);
    cudaMemsetAsync(d_ay_out,   0, sizeof(int32_t) * total_n, stream);
    cudaMemsetAsync(d_xrev_out, 0, sizeof(int32_t) * total_n, stream);
    cudaMemsetAsync(d_yrev_out, 0, sizeof(int32_t) * total_n, stream);

    // Build per-read offset and anchor-count arrays on host, then H2D
    int *h_offset = (int*)malloc(sizeof(int) * n_reads);
    int *h_n_a    = (int*)malloc(sizeof(int) * n_reads);
    int ofs = 0;
    for (int i = 0; i < n_reads; i++) {
        h_offset[i] = ofs;
        h_n_a[i]    = reads[i].n;
        ofs += reads[i].n;
    }
    cudaMemcpyAsync(d_offset, h_offset, sizeof(int) * n_reads, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_n_a,    h_n_a,    sizeof(int) * n_reads, cudaMemcpyHostToDevice, stream);

    // Expand uint16_t predecessors → int64_t (reads from d_bt_p_in)
    expand_p_to_int64<<<n_reads, THREAD_NUM_SHORT, 0, stream>>>(
        dev_mem->d_bt_p_in, d_p_abs, d_offset, d_n_a, n_reads);
    cudaCheck();

    // Step 1: Filter anchors by score (reads from d_bt_f_in)
    mm_filter_anchors<<<n_reads, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, d_offset, min_sc, dev_mem->d_bt_f_in, d_zx, d_zy,
        d_ofs_end, d_t, d_num_elements, d_n_v, n_reads);
    cudaCheck();

    // Step 2: Segmented sort by score (ascending) using pre-allocated CUB temp
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_cub_tmp, cub_tmp_size,
        d_zx, d_zx, d_zy, d_zy,
        (int)total_n, n_reads, d_offset, d_ofs_end,
        0, (int)(sizeof(int64_t) * 8), stream);
    cudaCheck();

    // Step 3: Backtrack — compute u chains and compacted anchor arrays
    //         Reads from d_bt_*_in (separate from d_ax used by chain forward pass)
    mm_chain_backtrack_parallel<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, dev_mem->d_bt_ax_in, dev_mem->d_bt_ay_in,
        dev_mem->d_bt_xrev_in, dev_mem->d_bt_yrev_in,
        dev_mem->d_bt_f_in, d_p_abs, d_u,
        d_zx, d_zy, d_t, d_v, d_offset, min_cnt, min_sc, max_drop,
        n_reads, d_n_v, d_n_u, d_num_elements, d_ofs_end);

    // D2H: n_u per read (needed for per-read w-array sort)
    int *h_n_u = (int*)malloc(sizeof(int) * n_reads);
    cudaMemcpyAsync(h_n_u, d_n_u, sizeof(int) * n_reads,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Step 4: Sort w arrays (d_p_abs = w_x, d_v = w_y) per-read by target position
    for (int i = 0; i < n_reads; i++) {
        if (h_n_u[i] > 1) {
            thrust::sort_by_key(thrust::cuda::par.on(stream),
                                d_p_abs + h_offset[i],
                                d_p_abs + h_offset[i] + h_n_u[i],
                                d_v     + h_offset[i]);
        }
    }

    // Step 5: Write compacted anchor output (reads d_bt_f_in for scores)
    mm_set_chain<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, n_reads, d_ax_out, d_ay_out, d_xrev_out, d_yrev_out, d_offset, d_ofs_end,
        dev_mem->d_bt_f_in, d_p_abs, d_u, d_zx, d_zy, d_t, d_v, d_n_v);

    // D2H: only u array (per-read chain metadata) — anchor arrays stay on GPU
    uint64_t *h_u_all = (uint64_t*)malloc(sizeof(uint64_t) * total_n);
    cudaMemcpyAsync(h_u_all, d_u, sizeof(uint64_t) * total_n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Save per-read offset/n_u on host for later use by finish_backtrack
    // (needed for deferred D2H of non-rechain reads and for voting)
    dev_mem->bt_n_reads = n_reads;
    dev_mem->bt_total_n = total_n;

    // Update read structures: set n_u and u[] from host data.
    // Anchor arrays (a[]) are NOT updated yet — they stay on GPU.
    // reads[i].n holds the compacted anchor count (sum of u[j] chain lengths).
    for (int i = 0; i < n_reads; i++) {
        reads[i].n_u = h_n_u[i];
        if (h_n_u[i] > 0) {
            KMALLOC(km, reads[i].u, h_n_u[i]);
            memcpy(reads[i].u, &h_u_all[h_offset[i]], sizeof(uint64_t) * h_n_u[i]);

            int new_n = 0;
            for (int j = 0; j < h_n_u[i]; j++)
                new_n += (int32_t)reads[i].u[j];
            reads[i].n = new_n;
            // reads[i].a is stale (old anchors) — will be rebuilt by
            // plbacktrack_d2h_read() after voting decides which reads need it.
        } else {
            reads[i].u = NULL;
            reads[i].n = 0;
            // Free stale a[] — no compacted anchors for this read
            kfree(km, reads[i].a);
            reads[i].a = NULL;
        }
    }

    // Store h_offset for deferred D2H (freed by plbacktrack_d2h_finish)
    dev_mem->bt_h_offset = h_offset;
    dev_mem->bt_h_n_u    = h_n_u;

    free(h_u_all);
    free(h_n_a);
    // h_offset and h_n_u are NOT freed here — owned by dev_mem until d2h_finish.
    // No cudaFree — all device buffers are pre-allocated and reused.
}

/**
 * plbacktrack_d2h_read: Deferred D2H for a single read's anchor data.
 * Called after voting decides this read does NOT need GPU re-chaining,
 * so we must pull its anchor arrays from GPU to rebuild reads[i].a.
 */
void plbacktrack_d2h_read(deviceMemPtr *dev_mem, chain_read_t *read,
                           int read_idx, void *km, cudaStream_t stream)
{
    int *h_offset = (int*)dev_mem->bt_h_offset;
    int *h_n_u    = (int*)dev_mem->bt_h_n_u;
    int n_u = h_n_u[read_idx];
    if (n_u <= 0) {
        // n_u == 0 reads already had a[] freed in plbacktrack_gpu
        return;
    }

    int ofs = h_offset[read_idx];
    int new_n = read->n;  // already set by plbacktrack_gpu

    int32_t *h_ax   = (int32_t*)malloc(sizeof(int32_t) * new_n);
    int32_t *h_ay   = (int32_t*)malloc(sizeof(int32_t) * new_n);
    int32_t *h_xrev = (int32_t*)malloc(sizeof(int32_t) * new_n);
    int32_t *h_yrev = (int32_t*)malloc(sizeof(int32_t) * new_n);
    cudaMemcpyAsync(h_ax,   dev_mem->d_bt_ax_out   + ofs, sizeof(int32_t) * new_n, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_ay,   dev_mem->d_bt_ay_out   + ofs, sizeof(int32_t) * new_n, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_xrev, dev_mem->d_bt_xrev_out + ofs, sizeof(int32_t) * new_n, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_yrev, dev_mem->d_bt_yrev_out + ofs, sizeof(int32_t) * new_n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    mm128_t *old_a = read->a;
    mm128_t *new_a;
    KMALLOC(km, new_a, new_n);
    for (int j = 0; j < new_n; j++) {
        new_a[j].x = ((uint64_t)h_xrev[j] << 32) | (uint32_t)h_ax[j];
        new_a[j].y = ((uint64_t)h_yrev[j] << 32) | (uint32_t)h_ay[j];
    }
    kfree(km, old_a);
    read->a = new_a;

    free(h_ax); free(h_ay); free(h_xrev); free(h_yrev);
}

/**
 * plbacktrack_d2h_finish: Free deferred metadata. Call once after all
 * per-read D2H is complete.
 */
void plbacktrack_d2h_finish(deviceMemPtr *dev_mem)
{
    free(dev_mem->bt_h_offset);
    free(dev_mem->bt_h_n_u);
    dev_mem->bt_h_offset = NULL;
    dev_mem->bt_h_n_u    = NULL;
}

/**
 * plbacktrack_gather_rechain_ay: Gather ay values for needs_rmq_rechain check.
 * Uses a GPU kernel to extract the first and last-of-chain-0 ay values,
 * then D2H into caller-owned host arrays.
 */
void plbacktrack_gather_rechain_ay(deviceMemPtr *dev_mem, int n_reads,
                                    cudaStream_t stream,
                                    int32_t **h_ay_first_out, int32_t **h_ay_last_out)
{
    // Use d_vt_ref_min and d_vt_bin_size as temporary device storage
    // (they have d_bt_max_n_reads capacity and aren't used until voting starts)
    int32_t *d_ay_first = dev_mem->d_vt_ref_min;
    int32_t *d_ay_last  = dev_mem->d_vt_bin_size;

    int blk = 256;
    int grd = (n_reads + blk - 1) / blk;
    gather_rechain_ay_kernel<<<grd, blk, 0, stream>>>(
        dev_mem->d_bt_ay_out,
        dev_mem->d_bt_offset,  // d_offset is still on device from plbacktrack_gpu
        dev_mem->d_bt_n_u,
        dev_mem->d_bt_u,
        d_ay_first, d_ay_last,
        n_reads);

    int32_t *h_ay_first = (int32_t *)malloc(sizeof(int32_t) * n_reads);
    int32_t *h_ay_last  = (int32_t *)malloc(sizeof(int32_t) * n_reads);
    cudaMemcpyAsync(h_ay_first, d_ay_first, sizeof(int32_t) * n_reads,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_ay_last,  d_ay_last,  sizeof(int32_t) * n_reads,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    *h_ay_first_out = h_ay_first;
    *h_ay_last_out  = h_ay_last;
}

void plbacktrack_init_memory(deviceMemPtr *dev_mem, size_t max_anchors)
{
    // Backtrack buffers are now pre-allocated in plmem_malloc_device_mem (d_bt_*).
    (void)dev_mem; (void)max_anchors;
}

void plbacktrack_free_memory(deviceMemPtr *dev_mem)
{
    // Backtrack buffers are freed in plmem_free_device_mem.
    (void)dev_mem;
}
