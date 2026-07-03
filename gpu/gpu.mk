GPUARCH			?= sm_70
CONFIG			+= $(if $(MAX_MICRO_BATCH),-DMICRO_BATCH=\($(MAX_MICRO_BATCH)\))
# Enable verbose [Info...] logging from plmem.cu / plalign.cu:  make PRINT=1
# See gpu/pllog.h.  Default OFF.
CONFIG			+= $(if $(PRINT),-DPRINT=1)
# Enable NVTX range markers for nsys profiling: make NVTX=1
# nvtx3 is header-only (bundled with CUDA via CUB) — no link flag needed.
CONFIG			+= $(if $(NVTX),-DNVTX_ENABLE)
# Use the cp.async double-buffered long-task kernel (ksw_double_buffer_kernel):
# make SHARED=1.  Default OFF → legacy ksw_fused_persistent_kernel.
CONFIG			+= $(if $(SHARED),-DSHARED)
# half2 2-cell-per-lane DP path in plksw_kernel (Suzuki-Kasahara deltas packed
# in FP16x2 — each warp lane computes 2 adjacent cells per step, stride 64).
# make HALF2=1.  Default OFF (legacy int8 scalar, 1 cell per lane).
# FP16 represents small integers (≤2048) exactly, so for normal DNA scoring
# penalties the half2 path matches the int8 path bit-for-bit.
CONFIG			+= $(if $(filter-out 0,$(HALF2)),-DHALF2_KSW)
# Raw extension benchmark: compile out the Z-drop split re-extension so each read
# is aligned in a single forward pass (G3SA-comparable). make RAW=1.  RAW=0 or
# unset = OFF (full pipeline with exact Z-drop handling). NOTE: RAW output SAM is
# not correct. (filter-out 0 so that RAW=0 means OFF, not "non-empty -> ON".)
CONFIG			+= $(if $(filter-out 0,$(RAW)),-DRAW_EXT_ONLY)
# Ablation "kernel-only" build: disable BOTH task-level scheduling (Z-drop split
# re-extension) AND batch-level scheduling (Class A/B/C analytical partitioning +
# concurrent-C overlap), keeping only the kernel-level optimizations. Use this to
# isolate the kernel contribution.  make KERNEL_ONLY=1
#   full         = (default)       kernel + batch + task
#   raw          = RAW=1           kernel + batch          (task off)
#   kernel-only  = KERNEL_ONLY=1   kernel                  (task + batch off)
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
