GPU				?= 		AMD
CONFIG			+= $(if $(MAX_MICRO_BATCH),-DMICRO_BATCH=\($(MAX_MICRO_BATCH)\))
# Enable gridded traceback for the long-task tier:  make GRID=1
# See gpu/plgrid_config.h for the design.  Default is OFF (legacy path).
CONFIG			+= $(if $(GRID),-DUSE_GRIDDED_BT=1)

ifeq ($(GPU), AMD)
    GPUARCH    ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
else
    GPUARCH    ?= sm_86
endif

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
############  	CUDA Compile 	###################
###################################################
COMPUTE_ARCH    = $(GPUARCH:sm_%=compute_%)
NVCC 			= nvcc
CUDAFLAGS		= -rdc=true -gencode arch=$(COMPUTE_ARCH),code=$(GPUARCH) -diag-suppress=177 -diag-suppress=1650 # supress unused variable / func warning
CUDANALYZEFLAG	= -Xptxas -v
CUDATESTFLAG	= -G
CUDADEBUGFLAG	= -maxrregcount=128 
###################################################
############	HIP Compile		###################
###################################################
HIPCC			= hipcc
HIPFLAGS		= -DUSEHIP --offload-arch=$(GPUARCH)
HIPANALYZEFLAG  = -Rpass-analysis=kernel-resource-usage
HIPTESTFLAGS	= -G -ggdb
HIPLIBS			= -L${ROCM_PATH}/lib -lroctx64 -lroctracer64

###################################################
############	DEBUG Options	###################
###################################################
ifeq ($(GPU), AMD)
	GPU_CC 		= $(HIPCC)
	GPU_FLAGS	= $(HIPFLAGS)
	GPU_TESTFL	= $(HIPTESTFLAGS)
	GPU_ANALYZE	= $(HIPANALYZEFLAG)
	LIBS		+= $(HIPLIBS)
else
	GPU_CC 		= $(NVCC)
	GPU_FLAGS	= $(CUDAFLAGS)
	GPU_ANALYZE = $(CUDANALYZEFLAG)
	GPU_TESTFL	= $(CUDATESTFLAG)
	LIBS		+= -lnvToolsExt
endif

ifeq ($(DEBUG),analyze)
	GPU_FLAGS	+= $(GPU_ANALYZE)
endif
ifeq ($(DEBUG),verbose)
	GPU_FLAGS	+= $(GPU_ANALYZE)
	GPU_FLAGS	+= $(GPU_TESTFL)
endif
ifeq ($(GPU), NV)
	GPU_FLAGS	+= $(CUDADEBUGFLAG)
endif


%.o: %.cu
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

%.ptx: %.cu
	$(GPU_CC) -ptx -src-in-ptx $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

%.as: %.o
	cuobjdump -all $< > $@

cleangpu: 
	rm -f $(CU_OBJS) $(CU_PTX)

# profile:CFLAGS += -pg -g3
# profile:all
# 	perf record --call-graph=dwarf -e cycles:u time ./minimap2 -a test/MT-human.fa test/MT-orang.fa > test.sam

cudep: gpu/.depend

gpu/.depend: $(CU_SRC)
	rm -f gpu/.depend
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS)  $(CPPFLAGS) $(INCLUDES) -MM $^ > $@

include gpu/.depend
