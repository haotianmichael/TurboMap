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

        // Copy sorted anchors back - decompose 64-bit values into ax/ay/xrev/yrev
        for (int i = 0, k = 0; i < n_u; ++i) {
            int32_t j = (int32_t)w_y[i], n = (int32_t)u[j];
            if(tid == 0) u2[i] = u[j];

            for(int x = tid; x < n; x += blockDim.x){
                uint64_t b_x_val = b_x[(w_y[i]>>32)+x];
                uint64_t b_y_val = b_y[(w_y[i]>>32)+x];
                ax[k+x] = (int32_t)b_x_val;  // Low 32 bits
                xrev[k+x] = (int32_t)(b_x_val >> 32);  // High 32 bits
                ay[k+x] = (int32_t)b_y_val;  // Low 32 bits
                yrev[k+x] = (int32_t)(b_y_val >> 32);  // High 32 bits
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

    // Debug: Check input anchor data from device memory
    if (n_reads > 0 && reads[0].n > 0) {
        int32_t h_check_input_ax[3], h_check_input_ay[3], h_check_input_xrev[3], h_check_input_yrev[3];
        int check_n = min(3, (int)reads[0].n);
        cudaMemcpy(h_check_input_ax, dev_mem->d_ax, sizeof(int32_t) * check_n, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_check_input_ay, dev_mem->d_ay, sizeof(int32_t) * check_n, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_check_input_xrev, dev_mem->d_xrev, sizeof(int32_t) * check_n, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_check_input_yrev, dev_mem->d_yrev, sizeof(int32_t) * check_n, cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DEBUG] Input anchors from device memory (first read, first %d anchors):\n", check_n);
        for (int i = 0; i < check_n; i++) {
            uint64_t x = ((uint64_t)h_check_input_xrev[i] << 32) | (uint32_t)h_check_input_ax[i];
            uint64_t y = ((uint64_t)h_check_input_yrev[i] << 32) | (uint32_t)h_check_input_ay[i];
            uint32_t q_pos = (uint32_t)y;
            uint32_t q_span = (uint32_t)(y >> 32) & 0xff;
            uint32_t seg_id = (uint32_t)(y >> 48) & 0xff;
            fprintf(stderr, "[DEBUG]   Input anchor %d: x=0x%lx, y=0x%lx (qpos=%u, qspan=%u, seg=%u)\n",
                    i, x, y, q_pos, q_span, seg_id);
        }
    }

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

    // Debug: Check results after backtracking
    int *h_n_u_debug = (int*)malloc(sizeof(int) * n_reads);
    int *h_n_v_debug = (int*)malloc(sizeof(int) * n_reads);
    int *h_n_z_debug = (int*)malloc(sizeof(int) * n_reads);
    cudaMemcpy(h_n_u_debug, d_n_u, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_v_debug, d_n_v, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_z_debug, d_num_elements, sizeof(int) * n_reads, cudaMemcpyDeviceToHost);

    fprintf(stderr, "[DEBUG] plbacktrack_gpu: n_reads=%d, total_n=%zu\n", n_reads, total_n);
    int total_chains = 0, total_filtered = 0;
    for (int i = 0; i < min(5, n_reads); i++) {
        fprintf(stderr, "[DEBUG]   Read %d: n_a=%d, n_z=%d, n_u=%d, n_v=%d\n",
                i, reads[i].n, h_n_z_debug[i], h_n_u_debug[i], h_n_v_debug[i]);
        total_chains += h_n_u_debug[i];
        total_filtered += h_n_z_debug[i];
    }
    fprintf(stderr, "[DEBUG] First 5 reads: total_chains=%d, total_filtered=%d\n", total_chains, total_filtered);
    free(h_n_u_debug);
    free(h_n_v_debug);
    free(h_n_z_debug);

    // Step 4: Sort by target position using CUB
    // Reuse temp storage
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_p_abs, d_p_abs, d_v, d_v,
        total_n, n_reads, d_offset, d_ofs_end);

    cudaStreamSynchronize(stream);

    // Step 5: Set chain information (write to output buffers)
    mm_set_chain<<<BLOCK_NUM_SHORT, THREAD_NUM_SHORT, 0, stream>>>(
        d_n_a, n_reads, d_ax_out, d_ay_out, d_xrev_out, d_yrev_out, d_offset, d_ofs_end,
        dev_mem->d_f, d_p_abs, d_u, d_zx, d_zy, d_t, d_v);

    cudaStreamSynchronize(stream);

    // Debug: Check if output buffers have data
    int32_t h_check_ax[5], h_check_ay[5], h_check_xrev[5], h_check_yrev[5];
    cudaMemcpy(h_check_ax, d_ax_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check_ay, d_ay_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check_xrev, d_xrev_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check_yrev, d_yrev_out, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
    fprintf(stderr, "[DEBUG] Output buffers after mm_set_chain:\n");
    for (int i = 0; i < 5; i++) {
        uint64_t x_val = ((uint64_t)h_check_xrev[i] << 32) | (uint32_t)h_check_ax[i];
        uint64_t y_val = ((uint64_t)h_check_yrev[i] << 32) | (uint32_t)h_check_ay[i];
        fprintf(stderr, "[DEBUG]   [%d]: x=0x%lx, y=0x%lx\n", i, x_val, y_val);
    }

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

    // Debug: Check first few anchors after copy
    fprintf(stderr, "[DEBUG] First 5 compacted anchors:\n");
    for (int i = 0; i < min(5, (int)total_n); i++) {
        uint64_t x_full = ((uint64_t)h_xrev[i] << 32) | (uint32_t)h_ax[i];
        uint64_t y_full = ((uint64_t)h_yrev[i] << 32) | (uint32_t)h_ay[i];
        fprintf(stderr, "[DEBUG]   Anchor %d: x=0x%lx (ax=0x%x, xrev=0x%x), y=0x%lx (ay=0x%x, yrev=0x%x)\n",
                i, x_full, h_ax[i], h_xrev[i], y_full, h_ay[i], h_yrev[i]);
    }

    // Update read structures
    for (int i = 0; i < n_reads; i++) {
        reads[i].n_u = h_n_u[i];
        if (h_n_u[i] > 0) {
            // Allocate and copy u array
            KMALLOC(km, reads[i].u, h_n_u[i]);
            cudaMemcpy(reads[i].u, &d_u[h_offset[i]], sizeof(uint64_t) * h_n_u[i], cudaMemcpyDeviceToHost);

            // Calculate new anchor count (sum of chain lengths)
            int new_n = 0;
            for (int j = 0; j < h_n_u[i]; j++) {
                new_n += (int32_t)reads[i].u[j];
            }

            // Reconstruct complete mm128_t from ax/xrev (x field) and ay/yrev (y field)
            // mm_set_chain writes to input offset position, so we use h_offset[i]
            int max_qpos_found = -1;
            for (int j = 0; j < new_n; j++) {
                int idx = h_offset[i] + j;
                reads[i].a[j].x = ((uint64_t)h_xrev[idx] << 32) | (uint32_t)h_ax[idx];
                reads[i].a[j].y = ((uint64_t)h_yrev[idx] << 32) | (uint32_t)h_ay[idx];

                // Validate qpos
                uint32_t qpos = (uint32_t)reads[i].a[j].y;
                if (qpos > max_qpos_found) max_qpos_found = qpos;
            }

            // Debug: Check if any qpos exceeds expected range
            if (i < 3 && max_qpos_found > 200000) {  // Suspicious if > 200k
                fprintf(stderr, "[DEBUG] WARNING: Read %d has suspicious max qpos=%u, new_n=%d, n_u=%d\n",
                        i, max_qpos_found, new_n, h_n_u[i]);
                fprintf(stderr, "[DEBUG]   Input n_a=%d, offset=%d\n", h_n_a[i], h_offset[i]);
                // Print first few anchors
                for (int j = 0; j < min(3, new_n); j++) {
                    uint32_t qp = (uint32_t)reads[i].a[j].y;
                    uint32_t qs = (uint32_t)(reads[i].a[j].y >> 32) & 0xff;
                    fprintf(stderr, "[DEBUG]   Anchor %d: qpos=%u, qspan=%u\n", j, qp, qs);
                }
            }

            // Update anchor count
            reads[i].n = new_n;

            // Debug: Print first read's chain info
            if (i == 0) {
                fprintf(stderr, "[DEBUG] Read 0 after reconstruction: n_u=%d, new_n=%d\n", h_n_u[i], new_n);
                fprintf(stderr, "[DEBUG]   u array (first 3 chains):\n");
                for (int j = 0; j < min(3, h_n_u[i]); j++) {
                    fprintf(stderr, "[DEBUG]     Chain %d: score=%u, len=%u\n",
                            j, (uint32_t)(reads[i].u[j] >> 32), (uint32_t)reads[i].u[j]);
                }
                fprintf(stderr, "[DEBUG]   First 3 reconstructed anchors (with field breakdown):\n");
                for (int j = 0; j < min(3, new_n); j++) {
                    uint64_t x = reads[i].a[j].x;
                    uint64_t y = reads[i].a[j].y;
                    uint32_t q_pos = (uint32_t)y;
                    uint32_t q_span = (uint32_t)(y >> 32) & 0xff;
                    uint32_t seg_id = (uint32_t)(y >> 48) & 0xff;
                    uint32_t t_pos = (uint32_t)x;
                    uint32_t rid_rev = (uint32_t)(x >> 32);
                    fprintf(stderr, "[DEBUG]     Anchor %d: x=0x%lx (tid_rev=0x%x, tpos=%u), y=0x%lx (qpos=%u, qspan=%u, seg=%u)\n",
                            j, x, rid_rev, t_pos, y, q_pos, q_span, seg_id);
                }
            }
        }
    }

    // Debug: Count reads with chains
    int reads_with_chains = 0;
    for (int i = 0; i < n_reads; i++) {
        if (h_n_u[i] > 0) reads_with_chains++;
    }
    fprintf(stderr, "[DEBUG] Backtracking complete. Reads with chains: %d/%d\n",
            reads_with_chains, n_reads);

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
