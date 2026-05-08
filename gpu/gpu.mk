GPUARCH			?= sm_70
CONFIG			+= $(if $(MAX_MICRO_BATCH),-DMICRO_BATCH=\($(MAX_MICRO_BATCH)\))
# Enable verbose [Info...] logging from plmem.cu / plalign.cu:  make PRINT=1
# See gpu/pllog.h.  Default OFF.
CONFIG			+= $(if $(PRINT),-DPRINT=1)

# Enable shared-memory long-task kernel (V100 super-long bottleneck batches): make SHARED=1
# See gpu/plksw_shared_kernel.cuh.  Default OFF (legacy global-mem kernel).
CONFIG			+= $(if $(SHARED),-DUSE_SHARED_LONG_KERNEL=1)

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
GPU_FLAGS		= -rdc=true -gencode arch=$(COMPUTE_ARCH),code=$(GPUARCH) -diag-suppress=177 -diag-suppress=1650
CUDANALYZEFLAG	= -Xptxas -v
CUDATESTFLAG	= -G
CUDADEBUGFLAG	= -maxrregcount=128

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
