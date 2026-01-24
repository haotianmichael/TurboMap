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
        // Reuse g_p for u2 since w_x is no longer needed after sorting
        int64_t *u2 = &g_p[ofs];
        uint64_t* u = &g_u[ofs];
        int32_t* ax = &g_ax[ofs];
        int32_t* ay = &g_ay[ofs];
        int32_t* xrev = &g_xrev[ofs];
        int32_t* yrev = &g_yrev[ofs];
        int n_a = g_na[job_idx];

        // w_x and w_y should already be sorted by calling code using CUB

        // Debug: Print first few w values for first job
        if (job_idx < 2 && tid == 0 && n_u > 0) {
            printf("[GPU-SETCHAIN] job %d: n_u=%d, n_a=%d\n", job_idx, n_u, n_a);
            for (int dbg = 0; dbg < min(3, n_u); dbg++) {
                printf("[GPU-SETCHAIN]   w[%d]: w_x=0x%lx, w_y=0x%lx (j=%d, k_offset=%u)\n",
                       dbg, w_x[dbg], w_y[dbg], (int32_t)w_y[dbg], (uint32_t)(w_y[dbg]>>32));
                int32_t j_dbg = (int32_t)w_y[dbg];
                if (j_dbg >= 0 && j_dbg < n_u) {
                    printf("[GPU-SETCHAIN]     u[%d]=0x%lx (chain_len=%d)\n",
                           j_dbg, u[j_dbg], (int32_t)u[j_dbg]);
                }
            }
        }
        __syncthreads();

        // Copy sorted anchors back - decompose 64-bit values into ax/ay/xrev/yrev
        // CRITICAL: k must be shared across all threads, so declare in shared memory or compute per-iteration
        for (int i = 0; i < n_u; ++i) {
            int32_t j = (int32_t)w_y[i], n = (int32_t)u[j];
            if(tid == 0) u2[i] = u[j];

            // Validate indices
            if (tid == 0 && (j < 0 || j >= n_u || n <= 0 || n > 100000)) {
                printf("[GPU-SETCHAIN-ERROR] job %d, i=%d: invalid j=%d or n=%d (n_u=%d)\n",
                       job_idx, i, j, n, n_u);
            }

            // Compute k for this iteration (sum of all previous chain lengths)
            int k = 0;
            for (int ii = 0; ii < i; ++ii) {
                k += (int32_t)u[(int32_t)w_y[ii]];
            }

            // Validate k and b array indices
            uint32_t b_offset = (uint32_t)(w_y[i]>>32);
            if (tid == 0 && job_idx < 2 && i < 3) {
                printf("[GPU-SETCHAIN]   Processing chain %d: j=%d, n=%d, k=%d, b_offset=%u\n",
                       i, j, n, k, b_offset);
            }

            for(int x = tid; x < n; x += blockDim.x){
                uint32_t b_idx = b_offset + x;

                // Bounds check
                if (b_idx >= n_a) {
                    if (job_idx < 2 && i < 3 && x == 0) {
                        printf("[GPU-SETCHAIN-ERROR] Out of bounds: b_idx=%u >= n_a=%d\n", b_idx, n_a);
                    }
                    continue;
                }

                uint64_t b_x_val = b_x[b_idx];
                uint64_t b_y_val = b_y[b_idx];

                int out_idx = k + x;
                if (out_idx >= n_a) {
                    if (job_idx < 2 && i < 3 && x == 0) {
                        printf("[GPU-SETCHAIN-ERROR] Output out of bounds: out_idx=%d >= n_a=%d\n", out_idx, n_a);
                    }
                    continue;
                }

                ax[out_idx] = (int32_t)b_x_val;  // Low 32 bits
                xrev[out_idx] = (int32_t)(b_x_val >> 32);  // High 32 bits
                ay[out_idx] = (int32_t)b_y_val;  // Low 32 bits
                yrev[out_idx] = (int32_t)(b_y_val >> 32);  // High 32 bits
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

// Helper to expand uint16_t predecessor to int64_t (keep relative distance semantics)
__global__ void expand_p_to_int64(uint16_t* p_rel, int64_t* p_expanded, int* offset, int* n_a, int n_task)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int job_idx = id; job_idx < n_task; job_idx += gridDim.x * blockDim.x) {
        int ofs = offset[job_idx];
        int n = n_a[job_idx];
        for (int i = 0; i < n; ++i) {
            // Keep relative distance: 0 means no predecessor (-1), others stay as-is
            p_expanded[ofs + i] = (p_rel[ofs + i] == 0) ? -1 : (int64_t)p_rel[ofs + i];
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

    // Allocate temporary buffers + output buffers for compacted anchors
    int64_t *d_zx, *d_zy, *d_v, *d_p_abs;
    int32_t *d_t, *d_ax_out, *d_ay_out, *d_xrev_out, *d_yrev_out;
    uint64_t *d_u;
    int *d_n_a, *d_offset, *d_ofs_end, *d_num_elements, *d_n_v, *d_n_u;
    void *d_temp_storage = nullptr;

    cudaMalloc(&d_zx, sizeof(int64_t) * total_n);
    cudaMalloc(&d_zy, sizeof(int64_t) * total_n);
    cudaMalloc(&d_v, sizeof(int64_t) * total_n);
    cudaMalloc(&d_p_abs, sizeof(int64_t) * total_n);
    cudaMalloc(&d_t, sizeof(int32_t) * total_n);
    cudaMalloc(&d_u, sizeof(uint64_t) * total_n);
    cudaMalloc(&d_ax_out, sizeof(int32_t) * total_n);
    cudaMalloc(&d_ay_out, sizeof(int32_t) * total_n);
    cudaMalloc(&d_xrev_out, sizeof(int32_t) * total_n);
    cudaMalloc(&d_yrev_out, sizeof(int32_t) * total_n);

    // Initialize output buffers to zero to avoid reading garbage data
    cudaMemset(d_ax_out, 0, sizeof(int32_t) * total_n);
    cudaMemset(d_ay_out, 0, sizeof(int32_t) * total_n);
    cudaMemset(d_xrev_out, 0, sizeof(int32_t) * total_n);
    cudaMemset(d_yrev_out, 0, sizeof(int32_t) * total_n);

    cudaMalloc(&d_n_a, sizeof(int) * n_reads);
    cudaMalloc(&d_offset, sizeof(int) * n_reads);
    cudaMalloc(&d_ofs_end, sizeof(int) * n_reads);
    cudaMalloc(&d_num_elements, sizeof(int) * n_reads);
    cudaMalloc(&d_n_v, sizeof(int) * n_reads);
    cudaMalloc(&d_n_u, sizeof(int) * n_reads);

    // Set up offset array
    int *h_offset = (int*)malloc(sizeof(int) * n_reads);
    int *h_n_a = (int*)malloc(sizeof(int) * n_reads);
    int ofs = 0;
    for (int i = 0; i < n_reads; i++) {
        h_offset[i] = ofs;
        h_n_a[i] = reads[i].n;
        ofs += reads[i].n;
    }
    cudaMemcpy(d_offset, h_offset, sizeof(int) * n_reads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_a, h_n_a, sizeof(int) * n_reads, cudaMemcpyHostToDevice);

    // Input validation removed to reduce debug output

    // Expand uint16_t predecessors to int64_t (keep relative distance semantics)
    expand_p_to_int64<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
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
        d_n_a, dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_xrev, dev_mem->d_yrev, dev_mem->d_f, d_p_abs, d_u,
        d_zx, d_zy, d_t, d_v, d_offset, min_cnt, min_sc, max_drop,
        n_reads, d_n_v, d_n_u, d_num_elements, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Backtracking statistics removed to reduce debug output

    // Step 4: Sort by target position using CUB
    // Note: Backtrack kernel already updated ofs_end to ofs+n_u for each read
    // w_x and w_y data are in [offset[i], ofs_end[i]) for each read
    // Use the already-updated ofs_end array directly for segmented sort
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_p_abs, d_p_abs, d_v, d_v,
        total_n, n_reads, d_offset, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Get n_u and ofs_end for diagnostics
    int *h_n_u_check = (int*)malloc(sizeof(int) * n_reads);
    int *h_ofs_end_check = (int*)malloc(sizeof(int) * n_reads);
    cudaMemcpy(h_n_u_check, d_n_u, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ofs_end_check, d_ofs_end, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);

    // Validate input to mm_set_chain - check g_zx and g_zy before kernel runs
    // Use first read's offset to check the correct data
    int first_read_offset = h_offset[0];
    int num_to_check_input = min(100, h_n_a[0]);

    fprintf(stderr, "[DEBUG-GPU-LAYOUT] First read: offset=%d, n_a=%d, n_u=%d, ofs_end=%d\n",
            first_read_offset, h_n_a[0], h_n_u_check[0], h_ofs_end_check[0]);

    int64_t *h_test_zx = (int64_t*)malloc(sizeof(int64_t) * num_to_check_input);
    int64_t *h_test_zy = (int64_t*)malloc(sizeof(int64_t) * num_to_check_input);

    // Read from the correct offset for first read
    cudaMemcpy(h_test_zx, d_zx + first_read_offset, sizeof(int64_t) * num_to_check_input, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test_zy, d_zy + first_read_offset, sizeof(int64_t) * num_to_check_input, cudaMemcpyDeviceToHost);

    fprintf(stderr, "[DEBUG-GPU-INPUT] First read (offset=%d): First 10 input b_x/b_y values before mm_set_chain:\n", first_read_offset);
    for (int i = 0; i < min(10, num_to_check_input); i++) {
        uint32_t b_y_low = (uint32_t)h_test_zy[i];
        uint32_t b_y_high = (uint32_t)(h_test_zy[i] >> 32);
        uint32_t qpos = b_y_low;
        uint32_t qspan = b_y_high & 0xff;
        fprintf(stderr, "[DEBUG-GPU-INPUT]   b[%d]: b_x=0x%lx, b_y=0x%lx (qpos=%u, qspan=%u)\n",
                i, h_test_zx[i], h_test_zy[i], qpos, qspan);
    }
    free(h_test_zx);
    free(h_test_zy);
    free(h_n_u_check);
    free(h_ofs_end_check);

    // Read a few output values BEFORE kernel to verify they're zeros
    int32_t test_before[5];
    cudaMemcpy(test_before, d_ay_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    fprintf(stderr, "[DEBUG-BEFORE-KERNEL] First 5 ay_out values before mm_set_chain: %d %d %d %d %d\n",
            test_before[0], test_before[1], test_before[2], test_before[3], test_before[4]);

    // Step 5: Set chain information (write to output buffers)
    mm_set_chain<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, n_reads, d_ax_out, d_ay_out, d_xrev_out, d_yrev_out, d_offset, d_ofs_end,
        dev_mem->d_f, d_p_abs, d_u, d_zx, d_zy, d_t, d_v);

    cudaStreamSynchronize(stream);

    // Read a few output values AFTER kernel to verify they changed
    int32_t test_after[5];
    cudaMemcpy(test_after, d_ay_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    fprintf(stderr, "[DEBUG-AFTER-KERNEL] First 5 ay_out values after mm_set_chain: %d %d %d %d %d\n",
            test_after[0], test_after[1], test_after[2], test_after[3], test_after[4]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] mm_set_chain kernel failed: %s\n", cudaGetErrorString(err));
    }
    // Force flush GPU printf buffer
    cudaDeviceSynchronize();

    // Validate output buffers - check first read's output
    // Reuse first_read_offset from earlier validation
    int num_to_check = min(100, h_n_a[0]);  // Don't read beyond first read's anchors

    int32_t *h_test_ay = (int32_t*)malloc(sizeof(int32_t) * num_to_check);
    int32_t *h_test_yrev = (int32_t*)malloc(sizeof(int32_t) * num_to_check);
    int32_t *h_test_ax = (int32_t*)malloc(sizeof(int32_t) * num_to_check);

    // Read from the correct offset for first read
    cudaMemcpy(h_test_ay, d_ay_out + first_read_offset, sizeof(int32_t) * num_to_check, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test_yrev, d_yrev_out + first_read_offset, sizeof(int32_t) * num_to_check, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test_ax, d_ax_out + first_read_offset, sizeof(int32_t) * num_to_check, cudaMemcpyDeviceToHost);

    fprintf(stderr, "[DEBUG-GPU-OUTPUT] First read (offset=%d): First 10 output anchors from GPU:\n", first_read_offset);
    for (int i = 0; i < min(10, num_to_check); i++) {
        uint64_t y_val = ((uint64_t)(uint32_t)h_test_yrev[i] << 32) | (uint32_t)h_test_ay[i];
        uint32_t qpos = (uint32_t)h_test_ay[i];
        uint32_t qspan = (uint32_t)h_test_yrev[i];
        fprintf(stderr, "[DEBUG-GPU-OUTPUT]   anchor[%d]: ax=%d, ay=%d, yrev=%d (qpos=%u, qspan=%u, y=0x%lx)\n",
                i, h_test_ax[i], h_test_ay[i], h_test_yrev[i], qpos, qspan, y_val);
    }
    free(h_test_ay);
    free(h_test_yrev);
    free(h_test_ax);

    // Note: Steps 6-8 (gen_regs) are not needed here
    // The CPU side will generate regions from the u array
    // This matches the CPU version in lchain.c

    // Copy results back to host
    int *h_n_u = (int*)malloc(sizeof(int) * n_reads);
    cudaMemcpy(h_n_u, d_n_u, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);

    // Allocate temporary host buffers for compacted anchors
    int32_t *h_ax = (int32_t*)malloc(sizeof(int32_t) * total_n);
    int32_t *h_ay = (int32_t*)malloc(sizeof(int32_t) * total_n);
    int32_t *h_xrev = (int32_t*)malloc(sizeof(int32_t) * total_n);
    int32_t *h_yrev = (int32_t*)malloc(sizeof(int32_t) * total_n);

    // Copy compacted anchors from output buffers
    cudaMemcpy(h_ax, d_ax_out, sizeof(int32_t) * total_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ay, d_ay_out, sizeof(int32_t) * total_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xrev, d_xrev_out, sizeof(int32_t) * total_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_yrev, d_yrev_out, sizeof(int32_t) * total_n, cudaMemcpyDeviceToHost);

    // Update read structures
    for (int i = 0; i < n_reads; i++) {
        reads[i].n_u = h_n_u[i];
        if (h_n_u[i] > 0) {
            // Allocate and copy u array
            KMALLOC(km, reads[i].u, h_n_u[i]);
            cudaMemcpy(reads[i].u, &d_u[h_offset[i]], sizeof(uint64_t) * h_n_u[i], cudaMemcpyDeviceToHost);

            // Debug: Validate u array
            if (i < 3) {
                fprintf(stderr, "[DEBUG] Read %d: n_u=%d, u[0]=%lu (chain_len=%d)\n",
                        i, h_n_u[i], reads[i].u[0], (int32_t)reads[i].u[0]);
            }

            // Calculate new anchor count (sum of chain lengths)
            int new_n = 0;
            for (int j = 0; j < h_n_u[i]; j++) {
                new_n += (int32_t)reads[i].u[j];
            }

            // Debug: Check if new_n matches expected count
            if (i < 3) {
                fprintf(stderr, "[DEBUG-NEWN] Read %d: new_n=%d (from u array sum)\n", i, new_n);
                // Also check what we'll actually copy
                int actual_copy_count = 0;
                for (int j = 0; j < new_n && (h_offset[i] + j) < total_n; j++) {
                    actual_copy_count++;
                }
                if (actual_copy_count != new_n) {
                    fprintf(stderr, "[ERROR-NEWN] Read %d: will only copy %d anchors (new_n=%d, h_offset=%d, total_n=%d)\n",
                            i, actual_copy_count, new_n, h_offset[i], total_n);
                }
            }

            // Allocate new array for compacted anchors (like compact_a does)
            mm128_t *old_a = reads[i].a;
            mm128_t *new_a;
            KMALLOC(km, new_a, new_n);

            // Reconstruct complete mm128_t from ax/xrev (x field) and ay/yrev (y field)
            // Copy to new array to avoid stale data in old oversized array
            uint32_t max_qpos_found = 0;  // FIX: Use uint32_t instead of int
            uint32_t min_qpos_found = UINT32_MAX;
            int invalid_count = 0;
            for (int j = 0; j < new_n; j++) {
                int idx = h_offset[i] + j;
                new_a[j].x = ((uint64_t)h_xrev[idx] << 32) | (uint32_t)h_ax[idx];
                new_a[j].y = ((uint64_t)h_yrev[idx] << 32) | (uint32_t)h_ay[idx];

                // Validate qpos and q_span
                uint32_t qpos = (uint32_t)new_a[j].y;
                uint32_t q_span = (uint32_t)(new_a[j].y >> 32) & 0xff;

                if (qpos > max_qpos_found) max_qpos_found = qpos;
                if (qpos < min_qpos_found) min_qpos_found = qpos;

                // Check for obviously invalid values
                if (qpos > 1000000 || q_span > 255 || q_span == 0) {
                    if (i < 3 && invalid_count < 5) {
                        fprintf(stderr, "[DEBUG-ANCHOR] Read %d, anchor %d: ax=%d, ay=%d, xrev=%d, yrev=%d\n",
                                i, j, h_ax[idx], h_ay[idx], h_xrev[idx], h_yrev[idx]);
                        fprintf(stderr, "[DEBUG-ANCHOR]   Reconstructed: x=0x%lx, y=0x%lx (qpos=%u, q_span=%u)\n",
                                new_a[j].x, new_a[j].y, qpos, q_span);
                    }
                    invalid_count++;
                }
            }

            // Print diagnostics for first few reads or suspicious data
            if (i < 5 || invalid_count > 0) {
                fprintf(stderr, "[DEBUG-BACKTRACK] Read %d: n_u=%d, new_n=%d, qpos range [%d, %d], invalid=%d\n",
                        i, h_n_u[i], new_n, min_qpos_found, max_qpos_found, invalid_count);
            }

            // Free old oversized array and update pointer to new right-sized array
            kfree(km, old_a);
            reads[i].a = new_a;

            // Update anchor count
            reads[i].n = new_n;

            // Verify u array cumulative count
            if (i < 5) {
                int cumulative = 0;
                for (int j = 0; j < h_n_u[i]; j++) {
                    cumulative += (int32_t)reads[i].u[j];
                }
                if (cumulative != new_n) {
                    fprintf(stderr, "[ERROR] Read %d: u array mismatch, cumulative=%d != new_n=%d\n",
                            i, cumulative, new_n);
                }
            }
        } else {
            // No chains found for this read
            reads[i].u = NULL;
            reads[i].n = 0;
            reads[i].a = NULL;
            if (i < 3) {
                fprintf(stderr, "[DEBUG] Read %d: No chains (n_u=0), set u/a to NULL\n", i);
            }
        }
    }

    // Backtracking statistics removed to reduce debug output

    free(h_ax);
    free(h_ay);
    free(h_xrev);
    free(h_yrev);

    // Cleanup
    free(h_offset);
    free(h_n_a);
    free(h_n_u);

    // Free device buffers
    cudaFree(d_zx);
    cudaFree(d_zy);
    cudaFree(d_v);
    cudaFree(d_p_abs);
    cudaFree(d_t);
    cudaFree(d_u);
    cudaFree(d_ax_out);
    cudaFree(d_ay_out);
    cudaFree(d_xrev_out);
    cudaFree(d_yrev_out);
    cudaFree(d_n_a);
    cudaFree(d_offset);
    cudaFree(d_ofs_end);
    cudaFree(d_num_elements);
    cudaFree(d_n_v);
    cudaFree(d_n_u);
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
