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
