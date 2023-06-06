# This file is a mess. Im sorry for any future users of this file



# Leconte
LEC_CUDA_DIR = /opt/nvidia/hpc_sdk/Linux_ppc64le/22.2/cuda/11.6
LEC_CUDA_MATH_DIR = /opt/nvidia/hpc_sdk/Linux_ppc64le/22.2/math_libs/11.6
LEC_CPU_LIB     = -L/opt/ibmmath/essl/6.3/lib64
LEC_CPU_INC     = -I/opt/ibmmath/essl/6.3/include
#LEC_COMP_LIB    = -L/opt/ibm/xlsmp/5.1.1/lib
#LEC_COMP2_LIB   = -L/opt/ibm/lib
#LEC_RUN_LIB     = -R/opt/ibm/lib
LEC_GPU_LIB     = -L$(LEC_CUDA_DIR)/lib64
LEC_GPU_INC     = -I$(LEC_CUDA_DIR)/include
LEC_CUBLAS_INC  = -I$(LEC_CUDA_MATH_DIR)/include
LEC_CUBLAS_LIB  = -L$(LEC_CUDA_MATH_DIR)/lib64
LEC_DIRS        = $(LEC_CPU_INC) $(LEC_CPU_LIB) $(LEC_COMP_LIB) $(LEC_COMP2_LIB) $(LEC_RUN_LIB) $(LEC_GPU_INC) $(LEC_GPU_LIB) $(LEC_CUBLAS_INC) $(LEC_CUBLAS_LIB)

LEC_CC = xlc_r
LEC_CXXFLAGS= -D_ESV6464 -lesslsmp6464 -lxlsmp $(LEC_DIRS) -O3 -qsmp=omp -qoffload -qtgtarch=sm_70 -lcuda -lcudart -lcublas -pedantic
# LEC_CXXFLAGS= -D_ESV6464 -lessl6464 $(LEC_DIRS) -O3 -qsmp -qoffload -qtgtarch=sm_70 -lcuda -lcudart -lcublas -pedantic

# Summit
SUMMIT_CPU_LIB     = -L$(OLCF_ESSL_ROOT)/lib64
SUMMIT_CPU_INC     = -I$(OLCF_ESSL_ROOT)/include
SUMMIT_COMP_LIB    = -L$(OLCF_XL_ROOT)/xlsmp/5.1.1/lib
SUMMIT_COMP2_LIB   = -L$(OLCF_XL_ROOT)/xlf/16.1.1/lib
SUMMIT_RUN_LIB     = -R$(OLCF_XL_ROOT)/lib

SUMMIT_GPU_LIB     = -L$(OLCF_CUDA_ROOT)/lib64
SUMMIT_GPU_INC     = -I$(OLCF_CUDA_ROOT)/include
SUMMIT_DIRS        =  $(SUMMIT_CPU_INC) $(SUMMIT_CPU_LIB) $(SUMMIT_COMP_LIB) $(SUMMIT_COMP2_LIB) $(SUMMIT_RUN_LIB) $(SUMMIT_GPU_INC) $(SUMMIT_GPU_LIB)
# ESSL libs on summit: libessl.so       libessl6464.so       libesslsmp.so       libesslsmp6464.so       libesslsmpcuda.so
#SUMMIT_CXXFLAGS = -lesslsmp -lxlf90_r -lxlsmp -lxlfmath $(SUMMIT_DIRS) -O3 -qsmp=omp -qoffload -qtgtarch=sm_70 -lcuda -lcudart -lcublas -pedantic 
SUMMIT_CXXFLAGS = -lessl -lxlf90_r -lxlfmath $(SUMMIT_DIRS) -O3 -qsmp=omp -qoffload -qtgtarch=sm_70 -lcuda -lcudart -lcublas -pedantic 
SUMMIT_CC = xlc_r

lb: 
	$(LEC_CC) -o trsm_testing trsm_testing.c mixed_trsm.c mixed_gemm.c $(LEC_CXXFLAGS) -qpath=n:$(LEC_CUDA_DIR)/bin -qstrict
	$(LEC_CC) -o gemm_testing gemm_testing.c mixed_gemm.c -D_ESV6464 -lesslsmp6464 -lxlsmp $(LEC_CPU_INC) $(LEC_CPU_LIB) $(LEC_COMP_LIB) $(LEC_COMP2_LIB) $(LEC_RUN_LIB) -O0 -qsmp=omp

eb: 
	gcc -o gemm_testing gemm_testing.c mixed_gemm.c -I /opt/intel/oneapi/mkl/2022.1.0/include -L /opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -L /opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64 -L /opt/intel/oneapi/intelpython/python3.9/lib -O0 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl -fopenmp -liomp5 -lquadmath

ebc:
	nvcc -o gemm_cuda_testing gemm_testing.c mixed_gemm.c -lquadmath -lcublas -lcuda -lcudart -O3

clean:
	rm -rf *.o *.out *.so *.ptx *.cub *.lst *.mod test test2 test_long trsm_testing output_file gemm_testing
