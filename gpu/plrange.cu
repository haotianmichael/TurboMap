#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plrange.cuh"
#include "cuda_utils.cuh"


/* CUDA kernel for range selection using forward chaining */

/* kernels begin */
__constant__ int d_max_dist_x;
__constant__ int d_max_iter;

inline __device__ int64_t range_binary_search(const int32_t* ax, const int32_t* rev, int64_t i, int64_t st_end){
    int64_t st_high = st_end, st_low=i;
    while (st_high != st_low) {
        int64_t mid = (st_high + st_low -1) / 2+1;
        if (rev[i] != rev[mid] || ax[mid] > ax[i] + d_max_dist_x) {
            st_high = mid -1;
        } else {
            st_low = mid;
        }
    }
    return st_high;
}


/**
 * Forward Range Selection Kernel using global memory and binary range search. 
 * cut reads into segements where successor range = 0. 
*/
__global__ void range_selection_kernel_binary(const int32_t* ax, const int32_t* rev, size_t *start_idx_arr, size_t *read_end_idx_arr, 
    int32_t *range, size_t* cut, size_t* cut_start_idx, size_t total_n, range_kernel_config_t config){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + config.anchor_per_block;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;
    size_t cut_idx = cut_start_idx[bid];
    if(tid == 0 && (bid == 0 || read_end_idx_arr[bid-1] != read_end_idx)){
        cut[cut_idx] = start_idx;
    }
    cut_idx++;
    int range_op[3] = {16, 512, 5000};  // Range Options
    range_op[2] = d_max_iter;
    for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
        size_t st_max = i + d_max_iter;
        st_max = st_max < read_end_idx ? st_max : read_end_idx -1;
        size_t st;
        for (int j=0; j<3; ++j){
            st = i + range_op[j];
            st = st <= st_max ? st : st_max;
            assert(st < total_n);
            assert(i < total_n);
            if (st > i && (rev[st] != rev[i] || ax[st] > ax[i] + d_max_dist_x)){
                break;
            }
        }
        st = range_binary_search(ax, rev, i, st);
        range[i] = st - i;

        if (st == i) {
            atomicMin((unsigned long long*)(cut + cut_idx), (unsigned long long)(i + 1));
        }
        cut_idx++;
    }
}

/**
 * Forward Range Selection Kernel using global memory and linear range search.
 * cut reads into segements where successor range = 0.
 */
__global__ void range_selection_kernel_naive(const int32_t* ax, const int32_t* rev, size_t *start_idx_arr, size_t *read_end_idx_arr, 
    int32_t *range, size_t* cut, size_t* cut_start_idx, size_t total_n, range_kernel_config_t config){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + config.anchor_per_block;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;
    assert(end_idx == (bid +1 < gridDim.x) ? start_idx_arr[bid+1]: total_n);
    // if(end_idx_ref != end_idx){
    //     if (tid == 0){
    //         int grimdim = gridDim.x;
    //     printf("start idx %d anchor_per_block %d read_end_idx %d, next start idx %d gridDim %d bid %d\n", 
    //     start_idx, config.anchor_per_block, read_end_idx, start_idx_arr[bid+1], grimdim, bid);
    //     }
    // }
    // __syncthreads();
    
    size_t cut_idx = cut_start_idx[bid];
    if(tid == 0 && (bid == 0 || read_end_idx_arr[bid-1] != read_end_idx)){
        cut[cut_idx] = start_idx;
    }
    cut_idx++;
    for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x){
        size_t st = i + d_max_iter;
        st = i + d_max_iter < read_end_idx ? st : read_end_idx -1;
        assert(st < total_n);
        assert(i < total_n);
        while (st > i && 
                (rev[i] != rev[st] // NOTE: different prefix cannot become predecessor 
                || ax[st] > ax[i] + d_max_dist_x)) { // NOTE: same prefix compare the value
            --st;
        }
        range[i] = st - i;

        if (st == i) {
            atomicMin((unsigned long long*)(cut + cut_idx), (unsigned long long)(i + 1));
        }
        cut_idx++;
    }
}

/* kernels end */

#ifdef __cplusplus
extern "C" {
#endif

/* host functions begin */
range_kernel_config_t range_kernel_config;

void plrange_upload_misc(Misc misc){
    cudaCheck();
    cudaMemcpyToSymbol(d_max_dist_x, &misc.max_dist_x, sizeof(int));
    cudaMemcpyToSymbol(d_max_iter, &misc.max_iter, sizeof(int));
    cudaCheck();
}

void plrange_async_range_selection(deviceMemPtr* dev_mem, cudaStream_t* stream) {
    size_t total_n = dev_mem->total_n, cut_num = dev_mem->num_cut;
    int griddim = dev_mem->griddim;
    if (griddim == 0 || total_n == 0) return;
    dim3 DimBlock(range_kernel_config.blockdim, 1, 1);
    dim3 DimGrid(griddim, 1, 1);

    // Run kernel
    range_selection_kernel_binary<<<DimGrid, DimBlock, 0, *stream>>>(
        dev_mem->d_ax, dev_mem->d_xrev, dev_mem->d_start_idx, dev_mem->d_read_end_idx,
        dev_mem->d_range, dev_mem->d_cut, dev_mem->d_cut_start_idx, total_n, range_kernel_config);
    cudaCheck();
}

void plrange_sync_range_selection(deviceMemPtr *dev_mem, Misc misc) {
    size_t total_n = dev_mem->total_n, cut_num = dev_mem->num_cut;
    int griddim = dev_mem->griddim;
    if (griddim == 0 || total_n == 0) return;
    dim3 DimBlock(range_kernel_config.blockdim, 1, 1);
    dim3 DimGrid(griddim,1,1);

    plrange_upload_misc(misc);

    // Run kernel
    range_selection_kernel_binary<<<DimGrid, DimBlock>>>(
        dev_mem->d_ax, dev_mem->d_xrev, dev_mem->d_start_idx, dev_mem->d_read_end_idx,
        dev_mem->d_range, dev_mem->d_cut, dev_mem->d_cut_start_idx, total_n, range_kernel_config);
    cudaCheck();
    cudaDeviceSynchronize();
    cudaCheck();
}

#ifdef __cplusplus
}
#endif

/* host functions end */
