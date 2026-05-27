#pragma once
/* NVTX range markers for nsys profiling.
 * Compile with NVTX=1 to enable: make NVTX=1
 * Link flag -lnvToolsExt is added automatically by gpu.mk when NVTX=1. */
#ifdef NVTX_ENABLE
#include <nvToolsExt.h>
#define NVTX_PUSH(name)  nvtxRangePushA(name)
#define NVTX_POP()       nvtxRangePop()
#define NVTX_MARK(name)  nvtxMarkA(name)
#else
#define NVTX_PUSH(name)  ((void)0)
#define NVTX_POP()       ((void)0)
#define NVTX_MARK(name)  ((void)0)
#endif
