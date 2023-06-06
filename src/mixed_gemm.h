#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>
#include <quadmath.h>
#include <omp.h>

#ifdef __has_include
    #if !__has_include("cublas_v2.h")
        #warning "Could not load CUDA. GPU GEMMs not compiled."
        typedef typeof(NULL) cublasHandle_t;
    #elif !__has_include("cuda_runtime_api.h")
        #warning "Could not load CUDA. GPU GEMMs not compiled."
        typedef typeof(NULL) cublasHandle_t;
    #else
        #include <cublas_v2.h>
        #include <cuda_runtime_api.h>
        #include <cuda_fp16.h>
        #define CUDA_USABLE 1
    #endif
#else
    #warning "Could not load CUDA. GPU GEMMs not compiled."
    typedef typeof(NULL) cublasHandle_t;
#endif

#ifdef __has_include
    #if __has_include("mkl.h")
        #include "mkl.h"
        #include "mkl_lapacke.h"
    #endif
#endif

typedef __float128 quad;


static unsigned int as_uint(const float x);


static float as_float(const unsigned int x);


static float hacky_truncate(const float x);


static float hacky_truncate_r(const float x, const int r);


static unsigned long as_ulong(const double x);


static double as_double(const unsigned long x);


static double hacky_truncate_double(const double x);


static double hacky_truncate_double_r(const double x, const int r);


float* I_n(int n);


void nvd_trunc_matrix(float* mat, int size);


/************ HALF PRECISION GEMMS *************/

void i_hgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);

// Not implemented
void cublas_hgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


/************ SINGLE PRECISION GEMMS *************/

void i_sgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void cublas_sgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void i_tf32_gemm(float *A_i, float *B_i, float *C, const int A_rows, const int A_cols, const int B_cols);

// Only works on system with A100
void cublas_tf32_gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void cublas_fp16_gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void i_tf32_4gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void i_tf32_3gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);

// Only works on system with A100
void cublas_tf32_4gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);


void cublas_tf32_3gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols);

/************ DOUBLE PRECISION GEMMS *************/

void i_dgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols);


void i_dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols);


void i_4dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols);


void i_3dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols) ;

/************ QUAD PRECISION GEMMS *************/

void i_qgemm(quad *A, quad *B, quad *C, const int A_rows, const int A_cols, const int B_cols);
