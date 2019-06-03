# Makefile based off of CS 179 lab assignments

# Product Names
CUDA_OBJ = cuda.o

# Input Names
SVD_CUDA_FILES = svd/gpu_svd.cu
SVD_CPP_FILES = run_svd.cpp svd/svd.cpp

KNN_CPP_FILES = run_knn.cpp knn/knn.cpp

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
CUDA_LIB_PATH = $(CUDA_PATH)/lib

NVCC = $(CUDA_BIN_PATH)/nvcc

ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS =
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# CUDA Object Files
SVD_CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(SVD_CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lsndfile -lcusolver

# C++ Object Files
OBJ_SVD = $(notdir $(addsuffix .o, $(SVD_CPP_FILES)))
OBJ_KNN = $(notdir $(addsuffix .o, $(KNN_CPP_FILES)))

all: run_svd run_knn

run_knn: $(OBJ_KNN)
	$(GPP) $(FLAGS) -o $@ $(INCLUDE) $^ $(LIBS)

run_svd: $(OBJ_SVD) $(CUDA_OBJ) $(SVD_CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o $@ $(INCLUDE) $^ $(LIBS)

# Compile C++ Source Files
svd.cpp.o: svd/svd.cpp svd/svd.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

run_svd.cpp.o: run_svd.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

knn.cpp.o: knn/knn.cpp knn/knn.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

run_knn.cpp.o: run_knn.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

# Compile CUDA Source Files
%_svd.cu.o: svd/%_svd.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(SVD_CUDA_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(SVD_CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^

clean:
	rm -f run_svd run_knn *.o svd/*.o knn/*.o
