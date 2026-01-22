#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <cub/cub.cuh>
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
        end_i = i = (p[i] > 0)? i - p[i] : p[i];
        s = i < 0? (int32_t)z_x[k] : (int32_t)z_x[k] - f[i];
        if (s > max_s)
            max_s = s, max_i = i;
        else if (max_s - s > max_drop)
            break;
    } while (i >= 0 && t[i] == 0);
    for (i = z_y[k]; i >= 0 && i != end_i && p[i] > 0; i = i - p[i])
        t[i] = 0;
    return max_i;
}

// Kernel 1: Filter anchors by score and populate z arrays
__global__ void mm_filter_anchors(int* n_a, int* offset, int32_t min_sc,
                                   int32_t* g_sc, int64_t* g_zx, int64_t* g_zy,
                                   int* ofs_end, int32_t* g_t, int* num_elements,
                                   int* g_n_v, int n_task)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for(int job_idx = id; job_idx < n_task; job_idx += gridDim.x * blockDim.x) {
        int ofs = offset[job_idx];
        int n = n_a[job_idx];
        int32_t* f = &g_sc[ofs];
        int64_t* z_x = &g_zx[ofs];
        int64_t* z_y = &g_zy[ofs];
        int32_t* t = &g_t[ofs];

        int64_t i, k, n_z = 0;

        // Initialize t (visited) array
        for(int j = 0; j < n; j++) t[j] = 0;

        // Populate z[] - filter anchors by score
        for (i = 0, k = 0; i < n; ++i) {
            if (f[i] >= min_sc) {
                ++n_z;
                z_x[k] = (int64_t)f[i];
                z_y[k++] = i;
            }
        }

        num_elements[job_idx] = n_z;
        ofs_end[job_idx] = ofs + n_z;

        if(n_z <= 0) {
            g_n_v[job_idx] = 0;
        }
    }
}

// Kernel 2: Backtrack filtered anchors in parallel
__global__ void mm_chain_backtrack_parallel(int* n_a, mm128_t* g_a, int32_t* g_sc,
                                            int64_t *g_p, uint64_t* g_u,
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
        mm128_t* a = &g_a[ofs];
        int32_t* f = &g_sc[ofs];
        int64_t* p = &g_p[ofs];

        int n_v = 0, n_u = 0;

        if (tid == 0) {
            // Backtrack to populate u[]
            // Note: z arrays should already be sorted by calling code using CUB
            for (int k = n_z - 1; k >= 0; --k) {
                if (t[z_y[k]] == 0) {
                    int64_t n_v0 = n_v, end_i;
                    int32_t sc;
                    end_i = mg_chain_bk_end(max_drop, z_x, z_y, f, p, t, k);
                    for (int64_t i = z_y[k]; i != end_i; i = (p[i]<0)? p[i] : i - p[i])
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

        // Compact anchors (parallel version)
        for (int i = 0, k = 0; i < n_u; ++i) {
            int32_t k0 = k, ni = (int32_t)u[i];
            for(int j = tid; j < ni; j += blockDim.x){
                b_x[k + j] = a[v[k0 + (ni-j-1)]].x;
                b_y[k + j] = a[v[k0 + (ni-j-1)]].y;
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
            ofs_end[job_idx] = ofs + n_u;
        }
        __syncthreads();
    }
}

// Kernel 3: Set chain anchor information
// Note: Sorting is done separately using CUB
__global__ void mm_set_chain(int* g_na, int n_task, mm128_t* g_a, int* offset,
                             int* ofs_end, int32_t* g_sc, int64_t *g_p,
                             uint64_t* g_u, int64_t* g_zx, int64_t* g_zy,
                             int32_t* g_t, int64_t* g_v)
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
        int64_t *u2 = (int64_t*)(&g_t[ofs]);
        uint64_t* u = &g_u[ofs];
        mm128_t* a = &g_a[ofs];
        int n_a = g_na[job_idx];

        // w_x and w_y should already be sorted by calling code using CUB

        // Copy sorted anchors back
        for (int i = 0, k = 0; i < n_u; ++i) {
            int32_t j = (int32_t)w_y[i], n = (int32_t)u[j];
            if(tid == 0) u2[i] = u[j];

            for(int x = tid; x < n; x += blockDim.x){
                a[k+x].x = b_x[(w_y[i]>>32)+x];
                a[k+x].y = b_y[(w_y[i]>>32)+x];
            }
            k += n;
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

// Helper to convert relative predecessor to absolute
__global__ void convert_p_rel_to_abs(uint16_t* p_rel, int64_t* p_abs, int* offset, int* n_a, int n_task)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int job_idx = id; job_idx < n_task; job_idx += gridDim.x * blockDim.x) {
        int ofs = offset[job_idx];
        int n = n_a[job_idx];
        for (int i = 0; i < n; ++i) {
            if (p_rel[ofs + i] == 0)
                p_abs[ofs + i] = -1;
            else
                p_abs[ofs + i] = i - p_rel[ofs + i];
        }
    }
}

// Host function to orchestrate backtracking
void plbacktrack_gpu(hostMemPtr *host_mem, deviceMemPtr *dev_mem,
                     chain_read_t *reads, Misc misc,
                     void* km, cudaStream_t stream)
{
    int n_reads = host_mem->size;
    if (n_reads == 0) return;

    size_t total_n = host_mem->total_n;
    int min_sc = misc.min_score;
    int min_cnt = misc.min_cnt;
    int max_drop = misc.bw;
    int is_qstrand = 0;
    uint32_t hash = 11; // Default seed hash

    // Allocate temporary buffers for backtracking (like new GPU version)
    int64_t *d_zx, *d_zy, *d_v, *d_p_abs;
    int32_t *d_t;
    uint64_t *d_u;
    mm128_t *d_a;
    int *d_n_a, *d_offset, *d_ofs_end, *d_num_elements, *d_n_v, *d_n_u, *d_qlen;
    void *d_temp_storage = nullptr;

    cudaMalloc(&d_zx, sizeof(int64_t) * total_n);
    cudaMalloc(&d_zy, sizeof(int64_t) * total_n);
    cudaMalloc(&d_v, sizeof(int64_t) * total_n);
    cudaMalloc(&d_p_abs, sizeof(int64_t) * total_n);
    cudaMalloc(&d_t, sizeof(int32_t) * total_n);
    cudaMalloc(&d_u, sizeof(uint64_t) * total_n);
    cudaMalloc(&d_a, sizeof(mm128_t) * total_n);
    cudaMalloc(&d_n_a, sizeof(int) * n_reads);
    cudaMalloc(&d_offset, sizeof(int) * n_reads);
    cudaMalloc(&d_ofs_end, sizeof(int) * n_reads);
    cudaMalloc(&d_num_elements, sizeof(int) * n_reads);
    cudaMalloc(&d_n_v, sizeof(int) * n_reads);
    cudaMalloc(&d_n_u, sizeof(int) * n_reads);
    cudaMalloc(&d_qlen, sizeof(int) * n_reads);

    // Copy input data to device
    // Convert anchors from host to device format
    size_t ofs = 0;
    for (int i = 0; i < n_reads; i++) {
        cudaMemcpy(&d_a[ofs], reads[i].a, sizeof(mm128_t) * reads[i].n, cudaMemcpyHostToDevice);
        ofs += reads[i].n;
    }

    // Set up offset array
    int *h_offset = (int*)malloc(sizeof(int) * n_reads);
    int *h_n_a = (int*)malloc(sizeof(int) * n_reads);
    int *h_qlen = (int*)malloc(sizeof(int) * n_reads);
    ofs = 0;
    for (int i = 0; i < n_reads; i++) {
        h_offset[i] = ofs;
        h_n_a[i] = reads[i].n;
        h_qlen[i] = reads[i].qlens[0];  // Assuming n_seg == 1
        ofs += reads[i].n;
    }
    cudaMemcpy(d_offset, h_offset, sizeof(int) * n_reads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_a, h_n_a, sizeof(int) * n_reads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qlen, h_qlen, sizeof(int) * n_reads, cudaMemcpyHostToDevice);

    // Copy scores and predecessors from device memory (already on GPU)
    cudaMemcpy(d_p_abs, dev_mem->d_f, sizeof(int32_t) * total_n, cudaMemcpyDeviceToDevice);

    // Convert relative predecessors to absolute
    convert_p_rel_to_abs<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        dev_mem->d_p, d_p_abs, d_offset, d_n_a, n_reads);

    // Step 1: Filter anchors by score
    mm_filter_anchors<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, d_offset, min_sc, dev_mem->d_f, d_zx, d_zy,
        d_ofs_end, d_t, d_num_elements, d_n_v, n_reads);

    cudaStreamSynchronize(stream);

    // Step 2: Sort z arrays by score using CUB (per-read segmented sort)
    size_t temp_storage_bytes = 0;

    // Determine temporary storage requirements
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_zx, d_zx, d_zy, d_zy,
        total_n, n_reads, d_offset, d_ofs_end);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort descending by score
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_zx, d_zx, d_zy, d_zy,
        total_n, n_reads, d_offset, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Step 3: Backtrack
    mm_chain_backtrack_parallel<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, d_a, dev_mem->d_f, d_p_abs, d_u,
        d_zx, d_zy, d_t, d_v, d_offset, min_cnt, min_sc, max_drop,
        n_reads, d_n_v, d_n_u, d_num_elements, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Step 4: Sort by target position using CUB
    // Reuse temp storage
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_p_abs, d_p_abs, d_v, d_v,
        total_n, n_reads, d_offset, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Step 5: Set chain information
    mm_set_chain<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, n_reads, d_a, d_offset, d_ofs_end, dev_mem->d_f, d_p_abs,
        d_u, d_zx, d_zy, d_t, d_v);

    cudaStreamSynchronize(stream);

    // Note: Steps 6-8 (gen_regs) are not needed here
    // The CPU side will generate regions from the u array
    // This matches the CPU version in lchain.c

    // Copy results back to host
    int *h_n_u = (int*)malloc(sizeof(int) * n_reads);
    cudaMemcpy(h_n_u, d_n_u, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);

    // Update read structures
    for (int i = 0; i < n_reads; i++) {
        reads[i].n_u = h_n_u[i];
        if (h_n_u[i] > 0) {
            // Allocate and copy u array
            KMALLOC(km, reads[i].u, h_n_u[i]);
            cudaMemcpy(reads[i].u, &d_u[h_offset[i]], sizeof(uint64_t) * h_n_u[i], cudaMemcpyDeviceToHost);

            // Update anchor array
            cudaMemcpy(reads[i].a, &d_a[h_offset[i]], sizeof(mm128_t) * reads[i].n, cudaMemcpyDeviceToHost);
        }
    }

    // Cleanup
    free(h_offset);
    free(h_n_a);
    free(h_qlen);
    free(h_n_u);

    // Free device buffers
    cudaFree(d_zx);
    cudaFree(d_zy);
    cudaFree(d_v);
    cudaFree(d_p_abs);
    cudaFree(d_t);
    cudaFree(d_u);
    cudaFree(d_a);
    cudaFree(d_n_a);
    cudaFree(d_offset);
    cudaFree(d_ofs_end);
    cudaFree(d_num_elements);
    cudaFree(d_n_v);
    cudaFree(d_n_u);
    cudaFree(d_qlen);
    cudaFree(d_temp_storage);
}

void plbacktrack_init_memory(deviceMemPtr *dev_mem, size_t max_anchors)
{
    // Memory is allocated per-call in plbacktrack_gpu
    // This can be optimized to use pre-allocated buffers
}

void plbacktrack_free_memory(deviceMemPtr *dev_mem)
{
    // Memory is freed per-call in plbacktrack_gpu
    // This can be optimized if using pre-allocated buffers
}
