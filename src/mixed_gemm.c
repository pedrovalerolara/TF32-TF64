#include "mixed_gemm.h"

// These functions exist for mimicing TF32 on CPU and half precision GEMM
// NOTE: the behavior of these function WILL vary between hardware. 
// These are designed to work on x86 arch or any arch that follows the same floating point storage(these have only been tested on Intel Xeon processor and AMD EPYC 7763)
static unsigned int as_uint(const float x) 
{
    return *(unsigned int*)&x;
}

static float as_float(const unsigned int x) 
{
    return *(float*)&x;
}

// Used to mimic formating to half precision or tf32 value. Truncates to 10 bit mantissa
static float hacky_truncate(const float x)
{
    return hacky_truncate_r(x, 13);
}

// trcuates r bits of float mantissa
static float hacky_truncate_r(const float x, const int r)
{
    return as_float((as_uint(x)>>r)<<r);
}

static unsigned long as_ulong(const double x) 
{
    return *(unsigned long*)&x;
}

static double as_double(const unsigned long x) 
{
    return *(double*)&x;
}

static double hacky_truncate_double(const double x)
{
    return hacky_truncate_double_r(x, 29);
}

static double hacky_truncate_double_r(const double x, const int r)
{
    return as_double((as_ulong(x)>>r)<<r);
}


float* I_n(int n)
{
    // creates identity matrix of size n. User must free returned pointer
    float* In = (float*)calloc(n*n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        In[i*n + i] = 1.0;
    }
    return In;
}


void nvd_trunc_matrix(float* mat, int size)
{
    // Rounds a matrix mat to tf32.
    // Very sloppy function but guarantees the same rounding as nvidia kernels
    // NEEDS A100 GPU
    float* ident = I_n(size);
    cublas_tf32_gemm(mat, ident, mat, size, size, size);

    free(ident);
}


/************ HALF PRECISION GEMMS *************/

void i_hgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols) 
{
    // mimics half precision GEMM kernel.
    // formats inputs into half, multiplies and adds (full precision fma) and then returns C as half.
    // the error in this function is likly dominated by the size of half precision mantissa in output
    int i, k, j;
    float temp;

    #pragma omp parallel for default(shared) private(i, k, j, temp)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            temp = hacky_truncate(A[i * A_cols + k]);
            for (j = 0; j < B_cols; j++) 
            {
                //fp32 fma
                C[i * B_cols + j] = C[i * B_cols + j] + temp * hacky_truncate(B[k * B_cols + j]);
            }
        }
    }
    
    // return output with fp16 data
    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * B_cols; i++)
    {
	    C[i] = hacky_truncate(C[i]);
    }
}


void cublas_hgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    fprintf(stderr, "Not implemented\n");
}


/************ SINGLE PRECISION GEMMS *************/

// Single precision simple GEMM kernel
// mimics cblas_sgemm. Some what naive implementation and may not perfectly align with BLAS libraries.
void i_sgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j;
    float temp;

    #pragma omp parallel for default(shared) private(i, k, j, temp)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            temp = A[i * A_cols + k];
            for (j = 0; j < B_cols; j++) 
            {
                C[i * B_cols + j] += temp * B[k * B_cols + j];
            }
        }
    }
}


// Uses cuBLAS Sgemm(non tenor core) kernel.
// This implementation likly incurs more overhead then needed and should not be used for performance comparisons
// Requires CUDA libraries and device
void cublas_sgemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    #if CUDA_USABLE == 1
    
    float *gpu_A, *gpu_B, *gpu_C;
    cudaMalloc((void**)&gpu_A, A_rows * A_cols * sizeof(float));
    cudaMalloc((void**)&gpu_B, A_cols * B_cols * sizeof(float));
    cudaMalloc((void**)&gpu_C, A_rows * B_cols * sizeof(float));

    cudaMemcpy(gpu_A, A, sizeof( float ) * A_rows * A_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, sizeof( float ) * A_cols * B_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, C, sizeof( float ) * A_rows * B_cols, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    // Do the actual multiplication
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_cols, A_rows, A_cols, &alpha, gpu_B, A_rows, gpu_A,
                          A_cols, &beta, gpu_C, A_rows);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, gpu_C, sizeof( float ) * A_rows * A_cols, cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    
    // Destroy the handle
    cublasDestroy(handle);

    #else
    fprintf(stderr, "CUDA not loaded: cublas_sgemm() has no effect\n");
    #endif
}


// This function is intended to mimic the TF32 compute mode for GEMM
// Takes single prec inputs. Formats to tf32 using nvidia's rounding. multiplies in full precision, accumulates in single and returns
void i_tf32_gemm(float *A_i, float *B_i, float *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j;
    float temp;

    float* A = (float*)malloc(sizeof(float) * A_rows * A_cols);
    float* B = (float*)malloc(sizeof(float) * A_rows * A_cols);

    for (int i = 0; i < A_rows*A_rows; i++)
    {
        A[i] = A_i[i];
        B[i] = B_i[i]; 
    }

    // Round according to TF32
    nvd_trunc_matrix(A, A_rows);
    nvd_trunc_matrix(B, A_rows);

    #pragma omp parallel for default(shared) private(i, k, j, temp)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            // ensures mantissa is truncated to 10 bits
            temp = hacky_truncate(A[i * A_cols + k]);
            for (j = 0; j < B_cols; j++) 
            {
                // tf32 compute mode uses 32bit fma here. I am using hacky_truncate to prevent any possible CPU fma usage here that may allow for higher precsion FMA
                // Probably not needed
                C[i * B_cols + j] = hacky_truncate_r(hacky_truncate_r(temp*B[k * B_cols + j], 0) + C[i * B_cols + j], 0);
            }
        }
    }
}


// Uses Nvidia TF32 compute more GEMM kernel
// NEEDS A100 or ampiere arch. Again should not be used for performance benchmarks
void cublas_tf32_gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    #if CUDA_USABLE == 1

    float *gpu_A, *gpu_B, *gpu_C;
    cudaMalloc((void**)&gpu_A, A_rows * A_cols * sizeof(float));
    cudaMalloc((void**)&gpu_B, A_cols * B_cols * sizeof(float));
    cudaMalloc((void**)&gpu_C, A_rows * B_cols * sizeof(float));

    cudaMemcpy(gpu_A, A, sizeof( float ) * A_rows * A_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, sizeof( float ) * A_cols * B_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, C, sizeof( float ) * A_rows * B_cols, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t status;
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_cols, A_rows, A_cols, &alpha, gpu_B, CUDA_R_32F, A_rows, gpu_A, CUDA_R_32F,
                          A_cols, &beta, gpu_C, CUDA_R_32F, A_rows, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, gpu_C, sizeof( float ) * A_rows * A_cols, cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    
    cublasDestroy(handle);

    #else
    fprintf(stderr, "CUDA not loaded: cublas_tf32_gemm() has no effect\n");
    #endif
}


// Uses fp16 kernel with fp32 accum. This should theretically give identitcal performance outputs to TF32 compute mode
// I was unable to get this to work on V100 hardware. And have not tested on A100.
void cublas_fp16_gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    #if CUDA_USABLE == 1

    float *gpu_A, *gpu_B, *gpu_C;
    cudaMalloc((void**)&gpu_A, A_rows * A_cols * sizeof(float));
    cudaMalloc((void**)&gpu_B, A_cols * B_cols * sizeof(float));
    cudaMalloc((void**)&gpu_C, A_rows * B_cols * sizeof(float));

    cudaMemcpy(gpu_A, A, sizeof( float ) * A_rows * A_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, sizeof( float ) * A_cols * B_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, C, sizeof( float ) * A_rows * B_cols, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t status;
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_cols, A_rows, A_cols, &alpha, gpu_B, CUDA_R_32F, A_rows, gpu_A, CUDA_R_32F,
                          A_cols, &beta, gpu_C, CUDA_R_32F, A_rows, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaDeviceSynchronize();
    
    printf("status: %d\n", status);
    cudaMemcpy(C, gpu_C, sizeof( float ) * A_rows * A_cols, cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    
    cublasDestroy(handle);

    #else
    fprintf(stderr, "CUDA not loaded: tile_gpu_dtrsm() has no effect\n");
    #endif
}


// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the four GEMMS A_1B_1, A_1B_2, A_2B_1, A_2B_2 using i_tf32_gemm
// Uses nvidia harware to trunc. May include a lot of overhead so should likly should not be used for performance metrics
void i_tf32_4gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    float* a1 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* a2 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* b1 = (float *)calloc( A_cols * B_cols, sizeof(float) );
    float* b2 = (float *)calloc( A_cols * B_cols, sizeof(float) );

    float* c1 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c2 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c3 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c4 = (float *)calloc( A_rows * B_cols, sizeof(float) );

    // Use this too specify custom rounding.
    for (int i = 0; i < A_rows*A_rows; i++)
    {
        //a1[i] = hacky_truncate_r(A[i], 0);
        //b1[i] = hacky_truncate_r(B[i], 0);
    }

    cublas_tf32_gemm(A, ident, a1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(B, ident, b1, A_rows, A_cols, B_cols);

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = (float)((double)A[i] - (double)(a1[i]));
        b2[i] = (float)((double)B[i] - (double)(b1[i]));
    }

    i_tf32_gemm(a1, b1, c1, A_rows, A_cols, B_cols);
    i_tf32_gemm(a1, b2, c2, A_rows, A_cols, B_cols);
    i_tf32_gemm(a2, b1, c3, A_rows, A_cols, B_cols);
    i_tf32_gemm(a2, b2, c4, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]) + c4[l]);
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(c4);
    free(ident);
}

// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the three GEMMS A_1B_1, A_1B_2, A_2B_1 using i_tf32_gemm
// Uses nvidia harware to trunc. May include a lot of overhead so should likly should not be used for performance metrics
void i_tf32_3gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    float* a1 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* a2 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* b1 = (float *)calloc( A_cols * B_cols, sizeof(float) );
    float* b2 = (float *)calloc( A_cols * B_cols, sizeof(float) );

    float* c1 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c2 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c3 = (float *)calloc( A_rows * B_cols, sizeof(float) );

    // Use this too specify custom rounding.
    for (int i = 0; i < A_rows*A_rows; i++)
    {
        //a1[i] = hacky_truncate_r(A[i], 0);
        //b1[i] = hacky_truncate_r(B[i], 0);
    }

    cublas_tf32_gemm(A, ident, a1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(B, ident, b1, A_rows, A_cols, B_cols);

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = (float)((double)A[i] - (double)(a1[i]));
        b2[i] = (float)((double)B[i] - (double)(b1[i]));
    }

    i_tf32_gemm(a1, b1, c1, A_rows, A_cols, B_cols);
    i_tf32_gemm(a1, b2, c2, A_rows, A_cols, B_cols);
    i_tf32_gemm(a2, b1, c3, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]));
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(ident);
}


// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the four GEMMS A_1B_1, A_1B_2, A_2B_1, A_2B_2 using cublas_tf32_gemm
// Uses nvidia harware to trunc. May include a lot of overhead so should likly not be used for performance metrics
void cublas_tf32_4gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    float* a1 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* a2 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* b1 = (float *)calloc( A_cols * B_cols, sizeof(float) );
    float* b2 = (float *)calloc( A_cols * B_cols, sizeof(float) );

    float* c1 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c2 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c3 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c4 = (float *)calloc( A_rows * B_cols, sizeof(float) );

    for (int i = 0; i < A_rows*A_rows; i++)
    {
        //a1[i] = hacky_truncate_r(A[i], 0);
        //b1[i] = hacky_truncate_r(B[i], 0);
    }

    cublas_tf32_gemm(A, ident, a1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(B, ident, b1, A_rows, A_cols, B_cols);

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = (float)((double)A[i] - (double)(a1[i]));
        b2[i] = (float)((double)B[i] - (double)(b1[i]));
    }

    cublas_tf32_gemm(a1, b1, c1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(a1, b2, c2, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(a2, b1, c3, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(a2, b2, c4, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]) + c4[l]);
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(c4);
    free(ident);
}

// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the three GEMMS A_1B_1, A_1B_2, A_2B_1 using cublas_tf32_gemm
// Uses nvidia harware to trunc. May include a lot of overhead so should likly not be used for performance metrics
void cublas_tf32_3gemm(float *A, float *B, float *C, const int A_rows, const int A_cols, const int B_cols)
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    float* a1 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* a2 = (float *)calloc( A_rows * A_cols, sizeof(float) );
    float* b1 = (float *)calloc( A_cols * B_cols, sizeof(float) );
    float* b2 = (float *)calloc( A_cols * B_cols, sizeof(float) );

    float* c1 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c2 = (float *)calloc( A_rows * B_cols, sizeof(float) );
    float* c3 = (float *)calloc( A_rows * B_cols, sizeof(float) );

    for (int i = 0; i < A_rows*A_rows; i++)
    {
        //a1[i] = hacky_truncate_r(A[i], 0);
        //b1[i] = hacky_truncate_r(B[i], 0);
    }

    cublas_tf32_gemm(A, ident, a1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(B, ident, b1, A_rows, A_cols, B_cols);

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = (float)((double)A[i] - (double)(a1[i]));
        b2[i] = (float)((double)B[i] - (double)(b1[i]));
    }

    cublas_tf32_gemm(a1, b1, c1, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(a1, b2, c2, A_rows, A_cols, B_cols);
    cublas_tf32_gemm(a2, b1, c3, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]));
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(ident);
}


/************ DOUBLE PRECISION GEMMS *************/

// Computes gemm in double
// mimics cblas_dgemm. Tested to confirm
void i_dgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j;
    double temp;

    #pragma omp parallel for default(shared) private(i, k, j, temp)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            temp = A[i * A_cols + k];
            for (j = 0; j < B_cols; j++) 
            {
                C[i * B_cols + j] += temp * B[k * B_cols + j];
            }
        }
    }
}


// Input matrices are in double. Formated to single. Then multiplied in full precision and accumulated in double.
// Mimics a possible TF64 compute mode. assuming a TF64 would have same mantissa as FP32
void i_dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j;
    float temp, temp2;

    #pragma omp parallel for default(shared) private(i, k, j, temp, temp2)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            temp = (float)A[i * A_cols + k];
            for (j = 0; j < B_cols; j++) 
            {
                temp2 = (float)B[k * B_cols + j]; 
                // I have include the truncate to try and prevent the use of any fma. not sure how much this will effect
                C[i * B_cols + j] = hacky_truncate_double_r((double)temp*(double)temp2 + C[i * B_cols + j], 0);
            }
        }
    }
}


// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the four GEMMS A_1B_1, A_1B_2, A_2B_1 A_2B_2 using i_dsgemm
// Uses manual truncation(may be a source of precision loss). May include a lot of overhead so should likly not be used for performance metrics
void i_4dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    double* a1 = (double *)calloc( A_rows * A_cols, sizeof(double) );
    double* a2 = (double *)calloc( A_rows * A_cols, sizeof(double) );
    double* b1 = (double *)calloc( A_cols * B_cols, sizeof(double) );
    double* b2 = (double *)calloc( A_cols * B_cols, sizeof(double) );
    
    double* c1 = (double *)calloc( A_rows * B_cols, sizeof(double) );
    double* c2 = (double *)calloc( A_rows * B_cols, sizeof(double) );
    double* c3 = (double *)calloc( A_rows * B_cols, sizeof(double) );
    double* c4 = (double *)calloc( A_rows * B_cols, sizeof(double) );

    //manual truncation to float
    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows*A_rows; i++)
    {
        a1[i] = hacky_truncate_double_r(A[i], 29);
        b1[i] = hacky_truncate_double_r(B[i], 29);
    }

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = A[i] - a1[i];
        b2[i] = B[i] - b1[i];
    }

    i_dsgemm(a1, b1, c1, A_rows, A_cols, B_cols);
    i_dsgemm(a1, b2, c2, A_rows, A_cols, B_cols);
    i_dsgemm(a2, b1, c3, A_rows, A_cols, B_cols);
    i_dsgemm(a2, b2, c4, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]) + c4[l]);
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(c4);
    free(ident);
}

// Splits matricies A= A_1+A_2 and B=B_1+B_2 and computes the three GEMMS A_1B_1, A_1B_2, A_2B_1 using i_dsgemm
// Uses manual truncation(may be a source of precision loss). May include a lot of overhead so should likly not be used for performance metrics
void i_3dsgemm(double *A, double *B, double *C, const int A_rows, const int A_cols, const int B_cols) 
{
    int i, k, j, l;
    
    float* ident = I_n(A_rows);

    double* a1 = (double *)calloc( A_rows * A_cols, sizeof(double) );
    double* a2 = (double *)calloc( A_rows * A_cols, sizeof(double) );
    double* b1 = (double *)calloc( A_cols * B_cols, sizeof(double) );
    double* b2 = (double *)calloc( A_cols * B_cols, sizeof(double) );
    
    double* c1 = (double *)calloc( A_rows * B_cols, sizeof(double) );
    double* c2 = (double *)calloc( A_rows * B_cols, sizeof(double) );
    double* c3 = (double *)calloc( A_rows * B_cols, sizeof(double) );

    // manual truncation to float
    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows*A_rows; i++)
    {
        a1[i] = hacky_truncate_double_r(A[i], 29);
        b1[i] = hacky_truncate_double_r(B[i], 29);
    }

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < A_rows * A_cols; i++) 
    {
        a2[i] = A[i] - a1[i];
        b2[i] = B[i] - b1[i];
    }

    i_dsgemm(a1, b1, c1, A_rows, A_cols, B_cols);
    i_dsgemm(a1, b2, c2, A_rows, A_cols, B_cols);
    i_dsgemm(a2, b1, c3, A_rows, A_cols, B_cols);

    for (l = 0; l < A_rows * B_cols; l++)
    {
        C[l] += c1[l] + ((c2[l] + c3[l]));
    }

    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(c1);
    free(c2);
    free(c3);
    free(ident);
}


/************ QUAD PRECISION GEMMS *************/

void i_qgemm(quad *A, quad *B, quad *C, const int A_rows, const int A_cols, const int B_cols)
{
    int i, k, j;
    quad temp;

    #pragma omp parallel for default(shared) private(i, k, j, temp)
    for (i = 0; i < A_rows; i++) 
    {
        for (k = 0; k < A_cols; k++) 
        {
            temp = A[i * A_cols + k];
            for (j = 0; j < B_cols; j++) 
            {
                C[i * B_cols + j] += temp * B[k * B_cols + j];
            }
        }
    }
}
