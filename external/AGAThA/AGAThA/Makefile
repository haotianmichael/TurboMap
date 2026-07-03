GPU_SM_ARCH=sm_70
#GPU_SM_ARCH=sm_75
MAX_QUERY_LEN=10000
N_CODE=0x4E
N_PENALTY=1

GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/usr/local/cuda/bin/nvcc
CC=g++
SRC_DIR=./src/
OBJ_DIR=./obj/
LIB_DIR=./lib/
INCLUDE_DIR=./include/

SOURCES=  args_parser.cpp host_batch.cpp ctors.cpp interfaces.cpp res.cpp gasal_align.cu 
LOBJS=$(patsubst %,%o,$(SOURCES))

LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib
YELLOW=\033[1;33m
NC=\033[0m # No Color

# 分开设置主机和设备调试标志
HOST_DEBUG_FLAGS=-g -O0 -fno-eliminate-unused-debug-types -fno-omit-frame-pointer
DEVICE_DEBUG_FLAGS=-g -G -O0 -lineinfo

ifeq ($(GPU_SM_ARCH),)
error1:
	@echo "Must specify GPU architecture as sm_xx"
endif
ifeq ($(MAX_QUERY_LEN),)
error2:
	@echo "Must specify maximum sequence length"
endif

ifeq ($(N_CODE),)
error3:
	@echo "Must specify the code for 'N'"
endif

# 更新CUDA编译规则，使用设备调试标志
ifeq ($(N_PENALTY),)
%.cuo: %.cu
	$(NVCC) -c $(DEVICE_DEBUG_FLAGS) -std=c++11 \
        -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE) \
        --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) \
        --keep --keep-dir $(OBJ_DIR) --keep-device-functions \
        -Xptxas -v -Xptxas -allow-expensive-optimizations=false \
        --default-stream per-thread \
        --generate-line-info \
        $< -o $(OBJ_DIR)$@
else
%.cuo: %.cu
	$(NVCC) -c $(DEVICE_DEBUG_FLAGS) -std=c++11 \
        -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE) \
        --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) \
        --keep --keep-dir $(OBJ_DIR) --keep-device-functions \
        -Xptxas -v -Xptxas -allow-expensive-optimizations=false \
        --default-stream per-thread \
        --generate-line-info \
        -DN_PENALTY=$(N_PENALTY) \
        $< -o $(OBJ_DIR)$@
endif

# 更新C++编译规则，使用主机调试标志
ifeq ($(N_PENALTY),)
%.cppo: %.cpp
	$(CC) -c $(HOST_DEBUG_FLAGS) -std=c++11 -Wall \
        -DMAX_QUERY_LEN=$(MAX_QUERY_LEN) -DN_CODE=$(N_CODE) \
        -Werror $< -o $(OBJ_DIR)$@
else
%.cppo: %.cpp
	$(CC) -c $(HOST_DEBUG_FLAGS) -std=c++11 -Wall \
        -DMAX_QUERY_LEN=$(MAX_QUERY_LEN) -DN_CODE=$(N_CODE) \
        -DN_PENALTY=$(N_PENALTY) \
        -Werror $< -o $(OBJ_DIR)$@
endif

all: clean makedir libgasal.a

makedir:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(INCLUDE_DIR)
	@cp $(SRC_DIR)/*.h $(INCLUDE_DIR)
	@sed -i "s/MAX_QUERY_LEN=[0-9]\{1,9\}/MAX_QUERY_LEN=$(MAX_QUERY_LEN)/" ./test_prog/Makefile

# 添加清理调试文件的规则
clean_debug:
	rm -f $(OBJ_DIR)*.cudafe* $(OBJ_DIR)*.fatbin* $(OBJ_DIR)*.module_id $(OBJ_DIR)*.ii $(OBJ_DIR)*.cpp* $(OBJ_DIR)*.cubin $(OBJ_DIR)*.ptx

ifeq ($(N_PENALTY),)
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
	@echo ""
	@echo -e "${YELLOW}WARNING:${NC}\"N_PENALTY\" is not defined"
else
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
endif

clean: clean_debug
	rm -f -r $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR)  *~ *.exe *.cppo *.cuo *.txt *~

gasal_align.cuo: gasal.h gasal_kernels.h