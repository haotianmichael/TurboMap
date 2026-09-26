#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#include <cuda.h>

#define cudaCheck() {                                                   \
    cudaError_t err = cudaGetLastError();                               \
    if (cudaSuccess != err) {                                           \
        fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__,\
                __func__, cudaGetErrorString(err));                     \
        fflush(stderr);                                                 \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

#endif // __CUDA_UTILS_CUH__
