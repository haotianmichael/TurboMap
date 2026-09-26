GPUARCH			?= sm_70
CONFIG			+= $(if $(MAX_MICRO_BATCH),-DMICRO_BATCH=\($(MAX_MICRO_BATCH)\))
# Enable verbose [Info...] logging from plmem.cu / plalign.cu:  make PRINT=1
# See gpu/pllog.h.  Default OFF.
CONFIG			+= $(if $(PRINT),-DPRINT=1)
# Enable NVTX range markers for nsys profiling: make NVTX=1
# nvtx3 is header-only (bundled with CUDA via CUB) — no link flag needed.
CONFIG			+= $(if $(NVTX),-DNVTX_ENABLE)
# Kernel DP path selection (mutually exclusive, compile-time only — the chosen
# path is the ONLY one compiled into the binary, so register/resource pressure
# stays scoped to that path).  Both use ksw_fused_persistent_kernel (GMEM):
#   default        HALF2 — 2 cells per lane via FP16x2 (SK deltas packed in half2)
#   make DPX=1     DPX   — 2 cells per lane via Hopper int16x2 + __viaddmax_s16x2
#                          (fused add+max in one instruction).  Requires sm_90+.
CONFIG			+= $(if $(filter-out 0,$(DPX)),-DDPX_KSW,-DHALF2_KSW)
# make RAW=1: compile out the Z-drop split re-extension (single forward pass).
# NOTE: RAW output SAM is not correct.
CONFIG			+= $(if $(filter-out 0,$(RAW)),-DRAW_EXT_ONLY)
# make KERNEL_ONLY=1: disable Z-drop split re-extension and batch-level scheduling.
CONFIG			+= $(if $(filter-out 0,$(KERNEL_ONLY)),-DRAW_EXT_ONLY -DNO_BATCH_SCHED)

###################################################
############  	CPU Compile 	###################
###################################################
CU_SRC			= $(wildcard gpu/*.cu)
CU_OBJS			= $(CU_SRC:%.cu=%.o)
CU_PTX			= $(CU_SRC:%.cu=%.ptx)
C_SRC			= $(wildcard gpu/*.c)
OBJS			+= $(C_SRC:%.c=%.o)
INCLUDES		+= -I gpu

###################################################
############  	CUDA Compile (NVIDIA only) ##########
###################################################
COMPUTE_ARCH    = $(GPUARCH:sm_%=compute_%)
GPU_CC 			= nvcc
GPU_FLAGS		= -rdc=true -gencode arch=$(COMPUTE_ARCH),code=$(GPUARCH) -diag-suppress=177 -diag-suppress=1650 -maxrregcount=255
CUDANALYZEFLAG	= -Xptxas -v
CUDATESTFLAG	= -G

ifeq ($(DEBUG),analyze)
	GPU_FLAGS	+= $(CUDANALYZEFLAG)
endif
ifeq ($(DEBUG),verbose)
	GPU_FLAGS	+= $(CUDANALYZEFLAG)
	GPU_FLAGS	+= $(CUDATESTFLAG)
endif

%.o: %.cu
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

%.ptx: %.cu
	$(GPU_CC) -ptx -src-in-ptx $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

%.as: %.o
	cuobjdump -all $< > $@

cleangpu:
	rm -f $(CU_OBJS) $(CU_PTX)

cudep: gpu/.depend

gpu/.depend: $(CU_SRC)
	rm -f gpu/.depend
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -MM $^ > $@

include gpu/.depend
