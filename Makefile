N_ERR_COMPUTING_IN_DEVICE?=1500
FLOAT_TYPE?=double
COMMON_CC_FLAGS=-DFLOAT_TYPE=$(FLOAT_TYPE) -lm -O3
COMMON_CPP_FLAGS=-std=c++17
OPENACC_CC_FLAGS=$(COMMON_CC_FLAGS) -acc -Minfo=accel
CUDA_CC_FLAGS=-std=c++17 -O3
CUDA_LINK_FLAGS=-lm -lcuda -lcudart -lcublas -Xcompiler -fopenmp
GPU_CC_FLAGS=$(OPENACC_CC_FLAGS) -ta=tesla -DTARGET_DEVICE=GPU -Mcudalib=cublas
CPU_CC_FLAGS=$(OPENACC_CC_FLAGS) -ta=multicore -DTARGET_DEVICE=CPU -Mcudalib=cublas

all: build

rebuild: clean build

build: openacc cuda_naive cuda_without_sync cuda_once_mem_alloc cuda_cub_one_block cuda_cub_partial_errors
 
openacc: src/heat_equation_solver_openacc.c heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	pgcc $(GPU_CC_FLAGS) -std=c11 -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o -o $@_gpu.out
	pgcc $(CPU_CC_FLAGS) -std=c11 -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o -o $@_cpu.out

cuda_naive: src/heat_equation_solver_cuda_naive.cu cublas_utils.o cuda_utils.o heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	nvcc $(CUDA_CC_FLAGS) -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o cublas_utils.o cuda_utils.o $(CUDA_LINK_FLAGS) -o $@.out

cuda_without_sync: src/heat_equation_solver_cuda_without_sync.cu cublas_utils.o cuda_utils.o heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	nvcc $(CUDA_CC_FLAGS) -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o cublas_utils.o cuda_utils.o $(CUDA_LINK_FLAGS) -o $@.out

cuda_once_mem_alloc: src/heat_equation_solver_cuda_once_mem_alloc.cu cublas_utils.o cuda_utils.o heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	nvcc $(CUDA_CC_FLAGS) -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o cublas_utils.o cuda_utils.o $(CUDA_LINK_FLAGS) -o $@.out

cuda_cub_one_block: src/heat_equation_solver_cuda_cub_one_block.cu cublas_utils.o cuda_utils.o heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	nvcc $(CUDA_CC_FLAGS) -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o cublas_utils.o cuda_utils.o $(CUDA_LINK_FLAGS) -o $@.out

cuda_cub_partial_errors: src/heat_equation_solver_cuda_cub_partial_errors.cu cublas_utils.o cuda_utils.o heat_equation_runner.o src/heat_equation_solver.h src/heat_equation_utils.h
	nvcc $(CUDA_CC_FLAGS) -DN_ERR_COMPUTING_IN_DEVICE=$(N_ERR_COMPUTING_IN_DEVICE) $< heat_equation_runner.o cublas_utils.o cuda_utils.o $(CUDA_LINK_FLAGS) -o $@.out

cublas_utils.o: src/cuda_utils/cublas_utils.cu
	nvcc $(CUDA_CC_FLAGS) -c $< -o $@

cuda_utils.o: src/cuda_utils/cuda_utils.cu
	nvcc $(CUDA_CC_FLAGS) -c $< -o $@

heat_equation_runner.o: src/heat_equation_runner.cpp
	nvcc $(COMMON_CC_FLAGS) -c $< -o heat_equation_runner.o

clean:
	rm -rf *.o *.out *.so

.PHONY: run clean