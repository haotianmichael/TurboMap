#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plscore.cuh"
#include "cuda_utils.cuh"

/* 

Parallel chaining helper functions with CUDA

*/

__constant__ Misc misc;
__constant__ int long_seg_cutoff;
__constant__ int mid_seg_cutoff;
__device__ unsigned curr_long_segid;

/* arithmetic functions begin */

// Accurate float log2 — matches CPU mg_log2().  Safe for x >= 2 (caller ensures dd+1 >= 2).
__device__ static inline float cuda_mg_log2(float x)
{
    union { float f; uint32_t i; } z = { x };
    float log_2 = ((z.i >> 23) & 255) - 128;
    z.i &= ~(255 << 23);
    z.i += 127 << 23;
    log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
    return log_2;
}

__device__ int32_t original_comput_sc(const int32_t ai_x, const int32_t ai_y, const int32_t aj_x, const int32_t aj_y,
                                const int8_t sidi,  const int8_t sidj,
                                int32_t max_dist_x, int32_t max_dist_y,
                                int32_t bw, float chn_pen_gap,
                                float chn_pen_skip, int is_cdna, int n_seg, int32_t q_span) {
    int32_t dq = ai_y - aj_y, dr, dd, dg, sc;
    if (dq <= 0 || dq > max_dist_x) return INT32_MIN;
    dr = ai_x - aj_x;
    if (sidi == sidj && (dr == 0 || dq > max_dist_y)) return INT32_MIN;
    dd = dr > dq ? dr - dq : dq - dr;
    if (sidi == sidj && dd > bw) return INT32_MIN;
    if (n_seg > 1 && !is_cdna && sidi == sidj && dr > max_dist_y)
        return INT32_MIN;  // nseg = 1 by default
    dg = dr < dq ? dr : dq;
    sc = q_span < dg ? q_span : dg;
    if (dd || dg > q_span) {
        float lin_pen, log_pen;
        lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
        log_pen =
            dd >= 1 ? cuda_mg_log2(dd + 1) : 0.0f;  // mg_log2() only works for dd>=2
        if (is_cdna || sidi != sidj) {
            if (sidi != sidj && dr == 0)
                ++sc;  // possibly due to overlapping paired ends; give a minor
                       // bonus
            else if (dr > dq || sidi != sidj)
                sc -=
                    (int)(lin_pen < log_pen ? lin_pen
                                            : log_pen);  // deletion or jump
                                                         // between paired ends
            else
                sc -= (int)(lin_pen + .5f * log_pen);
        } else
            sc -= (int)(lin_pen + .5f * log_pen);
    }
    return sc;
}

inline __device__ int32_t comput_sc(const int32_t ai_x, const int32_t ai_y, const int32_t aj_x, const int32_t aj_y,
                                const int32_t sidi,  const int32_t sidj,
                                const int32_t xrev_i, const int32_t xrev_j,
                                const int32_t max_dist_x, const int32_t max_dist_y,
                                const int32_t bw, const float chn_pen_gap,
                                const float chn_pen_skip, const int is_cdna, const int n_seg,
                                const int32_t q_span) {
    if (xrev_i != xrev_j) return INT32_MIN;  // different chromosome or strand
    const bool is_same_sid = sidi == sidj;
    const int32_t dq = ai_y - aj_y, dr = ai_x - aj_x;
    const int32_t dd = __sad(dr, dq, 0);

    if (dq <= 0 || dq > max_dist_x ||
        (is_same_sid && (dr == 0 ||
                        dq > max_dist_y ||
                        dd > bw ||
                        (n_seg > 1 && !is_cdna && dr > max_dist_y))))
        return INT32_MIN;

    const int32_t dg = dr < dq ? dr : dq;
    int32_t sc = q_span < dg ? q_span : dg;

    if (dd || dg > q_span) {
        // Use float throughout to match CPU comput_sc precision
        float log_pen = dd >= 1 ? cuda_mg_log2((float)(dd + 1)) : 0.0f;
        float lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
        bool minorBonus = is_cdna && !is_same_sid && dr == 0;
        bool majorAdjustment = (is_cdna && dr > dq) || !is_same_sid;
        sc += minorBonus;
        sc -= (!minorBonus && majorAdjustment) ? (int)(lin_pen < log_pen ? lin_pen : log_pen) : 0;
        sc -= (!minorBonus && !majorAdjustment) ? (int)(lin_pen + 0.5f * log_pen) : 0;
    }
    return sc;
}


/* arithmetic functions end */

inline __device__ void compute_sc_seg_one_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const int32_t* range,
                    const int32_t* xrev,
                    const size_t start_idx, const size_t end_idx,
                    int32_t* f, uint16_t* p
){
    const Misc blk_misc = misc;
    const int32_t q_span = blk_misc.q_span;
    int tid = threadIdx.x;
    // init f and p
    for (size_t i=start_idx+tid; i < end_idx; i += blockDim.x) {
        f[i] = q_span;
        p[i] = 0;
    }
    __syncwarp();
    for (size_t i=start_idx; i < end_idx; i++) {
        // Cap at segment boundary to prevent writes beyond end_idx (data race with adjacent segments).
        int32_t max_j = (int32_t)(end_idx - i - 1);
        int32_t range_i = range[i] < max_j ? range[i] : max_j;
        for (int32_t j = tid; j < range_i; j += blockDim.x) {
            int32_t sc = comput_sc(
                                anchors_x[i+j+1],
                                anchors_y[i+j+1],
                                anchors_x[i],
                                anchors_y[i],
                                sid [i+j+1],
                                sid [i],
                                xrev[i+j+1],
                                xrev[i],
                                blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
                                blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg, q_span);
            if (sc == INT32_MIN) continue;
            sc += f[i];
            if (sc >= f[i+j+1] && sc != q_span) {
                f[i+j+1] = sc;
                p[i+j+1] = j+1;
            }
        }
        __syncwarp();
    }
}


inline __device__ void compute_sc_seg_multi_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const int32_t* range,
                    const int32_t* xrev,
                    const size_t start_idx, const size_t end_idx,
                    int32_t* f, uint16_t* p
){
    const Misc blk_misc = misc;
    const int32_t q_span = blk_misc.q_span;
    int tid = threadIdx.x;
    // init f and p
    for (size_t i=start_idx+tid; i < end_idx; i += blockDim.x) {
        f[i] = q_span;
        p[i] = 0;
    }
    __syncthreads();
    for (size_t i=start_idx; i < end_idx; i++) {
        // Cap at segment boundary to prevent writes beyond end_idx (data race with adjacent segments).
        int32_t max_j = (int32_t)(end_idx - i - 1);
        int32_t range_i = range[i] < max_j ? range[i] : max_j;
        for (int32_t j = tid; j < range_i; j += blockDim.x) {
            int32_t sc = comput_sc(
                                anchors_x[i+j+1],
                                anchors_y[i+j+1],
                                anchors_x[i],
                                anchors_y[i],
                                sid [i+j+1],
                                sid [i],
                                xrev[i+j+1],
                                xrev[i],
                                blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
                                blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg, q_span);
            if (sc == INT32_MIN) continue;
            sc += f[i];
            if (sc >= f[i+j+1] && sc != q_span) {
                f[i+j+1] = sc;
                p[i+j+1] = j+1;
            }
        }
        __syncthreads();
    }
}

/* kernels begin */


template <size_t short_block_size>
__launch_bounds__(short_block_size)
__global__ void score_generation_short(
                                /* Input: Anchor & Range Inputs */
                                int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                int32_t* xrev,
                                /* Input: Segmentations */
                                size_t *seg_start_arr,
                                /* Output: Score and Previous Anchor */
                                int32_t* f, uint16_t* p,
                                /* Sizes*/
                                size_t total_n, size_t seg_count,
                                /* Output: Long segs */
                                int32_t* a_x_long, int32_t* a_y_long, int8_t* sid_long, int32_t* range_long, int32_t* xrev_long,
                                size_t* total_n_long, size_t buffer_size_long
                                , seg_t* long_seg, seg_t* long_seg_og, unsigned int *long_seg_count
                                ,seg_t *mid_seg, unsigned int *mid_seg_count){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t long_seg_start_idx;
    __shared__ size_t long_seg_start_idx_shared;

    for(int segid = bid; segid < seg_count; segid += gridDim.x){
        size_t start_idx = seg_start_arr[segid];
        if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
        size_t end_idx = SIZE_MAX;
        int end_segid = segid + 1;
        while (true) {
            if (end_segid >= seg_count) {
                end_idx = total_n;
                break;
            }
            if (seg_start_arr[end_segid] != SIZE_MAX) {
                end_idx = seg_start_arr[end_segid];
                break;
            }
            ++end_segid;
        }
        if (end_segid > segid + long_seg_cutoff) {
            if (tid == 0) {
                /* Allocate space in long seg buffer */
                long_seg_start_idx = atomicAdd((unsigned long long int*)total_n_long, (unsigned long long int)end_idx - start_idx);
                if (long_seg_start_idx + (end_idx - start_idx) >= buffer_size_long){ // long segement buffer is full
                /* rollback total_n_long — CUDA lacks atomicSub for ull, use negative add */
                    atomicAdd((unsigned long long int*)total_n_long, (unsigned long long int)(start_idx - end_idx));
                    long_seg_start_idx = SIZE_MAX;
                    // fallback to mid kernel
                    int mid_seg_idx = atomicAdd((unsigned long long int*)mid_seg_count, 1);
                    mid_seg[mid_seg_idx].start_idx = start_idx;
                    mid_seg[mid_seg_idx].end_idx = end_idx;
                } else {
                    int long_seg_idx = atomicAdd((unsigned long long int*)long_seg_count, 1);
                    long_seg[long_seg_idx].start_idx = long_seg_start_idx;
                    long_seg[long_seg_idx].end_idx = long_seg_start_idx + (end_idx - start_idx);
                    long_seg_og[long_seg_idx].start_idx = start_idx;
                    long_seg_og[long_seg_idx].end_idx = end_idx;
                }
            }
            // broadcast long_seg_start_idx to all threads in warp
            if (tid == 0) long_seg_start_idx_shared = long_seg_start_idx;
            __syncwarp();
            long_seg_start_idx = long_seg_start_idx_shared;
            __syncwarp();
            if (long_seg_start_idx == SIZE_MAX)
                continue;  // failed to allocate long_seg buffer
            for (uint64_t idx = tid; idx < end_idx - start_idx; idx += blockDim.x){
                a_x_long[long_seg_start_idx + idx] = anchors_x[start_idx + idx];
                a_y_long[long_seg_start_idx + idx] = anchors_y[start_idx + idx];
                sid_long[long_seg_start_idx + idx] = sid[start_idx + idx];
                range_long[long_seg_start_idx + idx] = range[start_idx + idx];
                xrev_long[long_seg_start_idx + idx] = xrev[start_idx + idx];
            }
            continue;
        } else if (end_segid > segid + mid_seg_cutoff) {
            if (tid == 0) {
                int mid_seg_idx = atomicAdd(mid_seg_count, 1);
                mid_seg[mid_seg_idx].start_idx = start_idx;
                mid_seg[mid_seg_idx].end_idx = end_idx;
            }
            continue;
        }
        compute_sc_seg_one_wf(anchors_x, anchors_y, sid, range, xrev, start_idx, end_idx, f, p);
    }
}


template <size_t mid_block_size>
__launch_bounds__(mid_block_size)
__global__ void score_generation_mid(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                int32_t* xrev,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    for(int segid = bid; segid < *long_seg_count; segid += gridDim.x){
        seg_t seg = long_seg[segid];
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, xrev, seg.start_idx, seg.end_idx, f, p);
    }
}

template <size_t long_block_size>
__launch_bounds__(long_block_size)
__global__ void score_generation_long(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                int32_t* xrev,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    for(int segid = bid; segid < *long_seg_count; segid += gridDim.x){
        seg_t seg = long_seg[segid];
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, xrev, seg.start_idx, seg.end_idx, f, p);
    }
}

// FIXME: merge together
template <size_t long_block_size>
__launch_bounds__(long_block_size)
__global__ void score_generation_long_map(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                int32_t* xrev,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p, unsigned int* map){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    unsigned int seg_count = 0;

    // #ifdef DEBUG_CHECK
    // auto start = clock64();
    // #endif

    __shared__ unsigned int segid;
    if (tid == 0 && bid == 0) {
        // init the first batch as the size of the grid
        curr_long_segid = gridDim.x;
    }
    if (tid == 0) {
        segid = bid;
    }

    __syncthreads();
    while (segid < *long_seg_count) {
        seg_t seg = long_seg[map[segid]]; // sorted
        // seg_t seg = long_seg[segid]; // unsorted
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, xrev, seg.start_idx, seg.end_idx, f, p);
        seg_count++;
        if (tid == 0) segid = atomicAdd(&curr_long_segid, 1);
        __syncthreads();
    }
}

__global__ void score_generation_naive(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                        int32_t* xrev,
                        size_t *seg_start_arr,
                        int32_t* f, uint16_t* p, size_t total_n, size_t seg_count) {

    // NOTE: each block deal with one batch 
    // the number of threads in a block is fixed, so we need to calculate iter
    // n = end_idx_arr - start_idx_arr
    // iter = (range[i] - 1) / num_threads + 1

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for (int segid = bid; segid < seg_count; segid += gridDim.x){
        /* calculate the segement for current block */
        size_t start_idx = seg_start_arr[segid];
        if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
        size_t end_idx = SIZE_MAX;
        int end_segid = segid + 1;
        while (true) {
            if (end_segid >= seg_count) {
                end_idx = total_n;
                break;
            }
            if (seg_start_arr[end_segid] != SIZE_MAX) {
                end_idx = seg_start_arr[end_segid];
                break;
            }
            ++end_segid;
        }
        // assert(end_idx <= total_n);
        compute_sc_seg_one_wf(anchors_x, anchors_y, sid, range, xrev, start_idx, end_idx, f, p);
    }
}

/* kernels end */

/* host functions begin */
score_kernel_config_t score_kernel_config;

void plscore_upload_misc(Misc input_misc) {
    cudaMemcpyToSymbol(misc, &input_misc, sizeof(Misc));
    cudaMemcpyToSymbol(long_seg_cutoff, &score_kernel_config.long_seg_cutoff, sizeof(int));
    cudaMemcpyToSymbol(mid_seg_cutoff, &score_kernel_config.mid_seg_cutoff, sizeof(int));
    cudaCheck();
}

void plscore_async_short_mid_forward_dp(deviceMemPtr* dev_mem, cudaStream_t* stream) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    size_t buffer_size_long = dev_mem->buffer_size_long;
    dim3 shortDimGrid(score_kernel_config.short_griddim, 1, 1);
    dim3 midDimGrid(score_kernel_config.mid_griddim, 1, 1);
    dim3 shortDimBlock(score_kernel_config.short_blockdim, 1, 1);

    // Run kernel;
    cudaMemsetAsync(dev_mem->d_mid_seg_count, 0, sizeof(unsigned int),
                    *stream);

    if (score_kernel_config.short_blockdim == 32 ){
    score_generation_short<32><<<shortDimGrid, dim3(32, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev,
        dev_mem->d_cut, dev_mem->d_f, dev_mem->d_p, total_n, cut_num,
        dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_range_long, dev_mem->d_xrev_long,
        dev_mem->d_total_n_long, buffer_size_long,
        dev_mem->d_long_seg, dev_mem->d_long_seg_og, dev_mem->d_long_seg_count,
        dev_mem->d_mid_seg, dev_mem->d_mid_seg_count);
    } else if (score_kernel_config.short_blockdim == 64) {
        score_generation_short<64><<<shortDimGrid, dim3(64, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev,
        dev_mem->d_cut, dev_mem->d_f, dev_mem->d_p, total_n, cut_num,
        dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_range_long, dev_mem->d_xrev_long,
        dev_mem->d_total_n_long, buffer_size_long,
        dev_mem->d_long_seg, dev_mem->d_long_seg_og, dev_mem->d_long_seg_count,
        dev_mem->d_mid_seg, dev_mem->d_mid_seg_count);
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported warpsize: %d. mm2-gb only supports device "
                "with a warpsize of 32 / 64. ",
                score_kernel_config.short_blockdim);
        exit(1);
    }
    cudaCheck();


    if (score_kernel_config.mid_blockdim == 128){
    score_generation_mid<128><<<midDimGrid, dim3(128, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev, dev_mem->d_mid_seg,
        dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p);
    } else if (score_kernel_config.mid_blockdim == 256){
        score_generation_mid<256><<<midDimGrid, dim3(256, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev, dev_mem->d_mid_seg,
        dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p);
    } else if (score_kernel_config.mid_blockdim == 512){
        score_generation_mid<512><<<midDimGrid, dim3(512, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev, dev_mem->d_mid_seg,
        dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p);
    } else if (score_kernel_config.mid_blockdim == 1024){
        score_generation_mid<1024><<<midDimGrid, dim3(1024, 1, 1), 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev, dev_mem->d_mid_seg,
        dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p);
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported mid_blockdim: %d. mm2-gb only supports a "
                "blockdim of 128/256/512/1024 for mid kernel \n\n"
                "Please adjust score_kernel:mid_blockdim in gpu config file. ",
                score_kernel_config.mid_blockdim);
        exit(1);
    }
    cudaCheck();

    cudaCheck();
}

void plscore_async_long_forward_dp(deviceMemPtr* dev_mem, cudaStream_t* stream) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    size_t buffer_size_long = dev_mem->buffer_size_long;
    dim3 longDimGrid(score_kernel_config.long_griddim, 1, 1);

    if (score_kernel_config.long_blockdim == 1024){
    score_generation_long_map<1024><<<longDimGrid, dim3(1024, 1, 1), 0, *stream>>>(
        dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_range_long, dev_mem->d_xrev_long,
        dev_mem->d_long_seg, dev_mem->d_long_seg_count, dev_mem->d_f_long, dev_mem->d_p_long, dev_mem->d_map);
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported MaxThreadsPerBlock: %d. mm2-gb only supports a blockdim of 1024 for long kernel ",
                score_kernel_config.long_blockdim);
        exit(1);
    }

    cudaCheck();

    cudaCheck();
}

void plscore_async_naive_forward_dp(deviceMemPtr* dev_mem,
                                    cudaStream_t* stream) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    dim3 DimBlock(score_kernel_config.long_blockdim, 1, 1);
    dim3 longDimGrid(score_kernel_config.long_griddim, 1, 1);
    dim3 shortDimGrid(score_kernel_config.short_griddim, 1, 1);

    // Run kernel
    // printf("Grid Dim, %d\n", DimGrid.x);
    score_generation_naive<<<shortDimGrid, DimBlock, 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_range, dev_mem->d_xrev, dev_mem->d_cut,
        dev_mem->d_f, dev_mem->d_p, total_n, cut_num);
    cudaCheck();
}

