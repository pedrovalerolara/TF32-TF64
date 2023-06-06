#include "mixed_gemm.h"


void generate_random_matrices(double* mat_a, int A_rows, int A_cols, double* mat_b, int B_rows, int B_cols, double low, double high, int rand_seed)
{
    // populates random matrices mat_a and mat_b, with random doubles between low and high. 
    int i, j;
    double rand_val, rand_val2;
    srand(rand_seed);

    for (i = 0; i < A_rows; i++)
    {
        for (j = 0; j < A_cols; j++)
        {
            mat_a[i*A_cols + j] = (high - low)*((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX)) + low;
        }
    }
    for (i = 0; i < B_rows; i++)
    {
        for (j = 0; j < B_cols; j++)
        {
            mat_b[i*B_cols + j] = (high - low)*((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX)) + low;
        }
    }
}


double fnorm(int rank, float* A, int size)
{
    // Given vector or array of size size , computes 1 2 or Inf norm as float
    double norm_sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        switch (rank)
        {
        case 1:
            norm_sum += fabs((double)A[i]);
            break;
        case 2:
            norm_sum += pow(A[i], 2);
            break;
        default:
            if (norm_sum < fabs((double)A[i]))
                norm_sum = fabs((double)A[i]);
            break;
        }
    }
    switch (rank)
    {
    case 2:
        return sqrt(norm_sum);
    default:
        return norm_sum;
    }
}


double norm(int rank, double* A, int size)
{
    // Given vector or array of size size , computes 1 2 or Inf norm as double
    double norm_sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        switch (rank)
        {
        case 1:
            norm_sum += fabs(A[i]);
            break;
        case 2:
            norm_sum += pow(A[i], 2);
            break;
        default:
            if (norm_sum < fabs(A[i]))
                norm_sum = fabs(A[i]);
            break;
        }
    }
    switch (rank)
    {
    case 2:
        return sqrt(norm_sum);
    default:
        return norm_sum;
    }
}

// do not change. These are only used when mixed_gemm.c is changed
#define NUM_FS 5
#define NUM_FFS 1

// When set to 1 quad precision is used as "truth", when set to 0 cblas_dgemm is used as "truth".
// Quad GEMM is very slow.
#define USE_QUAD_TRUTH 1

void run_precision_tests(int init_size, int max_size, int multiplier, int its_per_size, int rand_seed, int error_metric, char* save_file)
{
    // computes Forward error using the error_metric norm. if error_metric=-1 then it will compute for al error metrics
    // compute DSGEMM functions on square matricies of sizes between init_size and max_size, incrementing in multiples of multiplier
    // For each matrix size, the reported error will be the avage over its_per_size iterations
    // File will be saves to ../data/save_file_<error_metric char>-norm
    int size, r, i, k, it;
    double denom;

    #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
    quad* truth;
    #else
    double* truth;
    #endif

    double* diffs[NUM_FS];
    double* dc_mats[NUM_FS-NUM_FFS];
    float* fc_mats[NUM_FFS];
    double errors[3][NUM_FS];

    FILE *fptr[3];
    char file_name[3][40];
    char error_chars[3] = "I12";
    
    for (i = 0; i < 3; i++)
    {
        if (error_metric == i || error_metric == -1)
        {
            snprintf(file_name[i], 40, "../data/%s_%c-norm", save_file, error_chars[i]);
            fptr[i] = fopen(file_name[i],"w");
            fprintf(fptr[i], "size, sgemm, dgemm, dsgemm, 4dsgemm, 3dsgemm, averaged over %d random multiplciations\n", its_per_size);
        }
    }

    for (size = init_size; size <= max_size; size *= multiplier)
    {
        for (i = 0; i < 3; i++)
        {
            for (k = 0; k < NUM_FS; k++)
            {
                errors[i][k] = 0.0;
            }
        }

        fprintf(stdout, "Computing with size=%d\n", size);

        for (it = 0; it < its_per_size; it++)
        {
            for (k = 0; k < NUM_FS; k++)
            {
                diffs[k] = (double*)calloc(size * size, sizeof(double));
            }

            /************************
             *  generate matricies  *
             ************************/
            #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
            quad* a_mat_quad = (quad*)malloc(sizeof(quad) * size * size);
            quad* b_mat_quad = (quad*)malloc(sizeof(quad) * size * size);
            quad* truth = (quad*)malloc(sizeof(quad) * size * size);
            #else
            double* truth = (double*)calloc(size * size, sizeof(double));
            #endif
            double* a_mat_double = (double*)malloc(sizeof(double) * size * size);
            double* b_mat_double = (double*)malloc(sizeof(double) * size * size);
            float* a_mat_single = (float*)malloc(sizeof(float) * size * size);
            float* b_mat_single = (float*)malloc(sizeof(float) * size * size);

            for (k = 0; k < NUM_FS - NUM_FFS; k++)
            {
                dc_mats[k] = (double*)calloc(size * size, sizeof(double));
            }
            for (k = 0; k < NUM_FFS; k++)
            {
                fc_mats[k] = (float*)calloc(size * size, sizeof(float));
            }

            generate_random_matrices(a_mat_double, size, size, b_mat_double, size, size, -0.5, 0.5, rand_seed);
            
            for (i = 0; i< size*size; i++)
            {
                a_mat_single[i] = (float)a_mat_double[i];
                b_mat_single[i] = (float)b_mat_double[i];
                #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
                a_mat_quad[i] = (quad)a_mat_double[i];
                b_mat_quad[i] = (quad)b_mat_double[i];
                truth[i] = (quad)0.0;
                #endif
            }

            /***************
             *  RUN tests  *
             ***************/

            // Truth data
            #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
            i_qgemm(a_mat_quad, b_mat_quad, truth, size, size, size);
            #else
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1, a_mat_double, size, b_mat_double, size, 0, truth, size);
            #endif
            // Double precision
            i_dgemm(a_mat_double, b_mat_double, dc_mats[0], size, size, size);

            // Single Precision
            i_sgemm(a_mat_single, b_mat_single, fc_mats[0], size, size, size);

            // dsgemms (TF64)
            i_dsgemm(a_mat_double, b_mat_double, dc_mats[1], size, size, size);
            i_4dsgemm(a_mat_double, b_mat_double, dc_mats[2], size, size, size);
            i_3dsgemm(a_mat_double, b_mat_double, dc_mats[3], size, size, size);

            
            /*********************************
             *  Error checking and printing  *
             *********************************/
            for (k = 0; k < NUM_FFS; k++)
            {
                for (i = 0; i < size * size; i++ )
                {
                    #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
                    diffs[k][i] += fabs((double)(truth[i] - (quad)fc_mats[k][i]));
                    #else
                    diffs[k][i] += fabs(truth[i] - fc_mats[k][i]);
                    #endif
                }
            }
            for (k = NUM_FFS; k < NUM_FS; k++)
            {
                for (i = 0; i < size * size; i++ )
                {
                    #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
                    diffs[k][i] += fabs((double)(truth[i] - (quad)dc_mats[k-NUM_FFS][i]));
                    #else
                    diffs[k][i] += fabs(truth[i] - dc_mats[k-NUM_FFS][i]);
                    #endif
                }
            }
            
            for (i = 0; i < 3; i++)
            {
                denom = (norm(i, a_mat_double, size * size) * norm(i, b_mat_double, size * size));
		        for (k = 0; k < NUM_FS; k++)
                {
                    errors[i][k] += norm(i, diffs[k], size * size) / denom;
                }
            }
            
            // Free memory
            for (k = 0; k < NUM_FS; k++)
            {
                free(diffs[k]);
            }

            #if defined USE_QUAD_TRUTH && USE_QUAD_TRUTH == 1
            free(a_mat_quad);
            free(b_mat_quad);
            #endif
            free(a_mat_double);
            free(b_mat_double);
            free(a_mat_single);
            free(b_mat_single);
            free(truth);

            for (k = 0; k < NUM_FS - NUM_FFS; k++)
            {
                free(dc_mats[k]);
            }
            for (k = 0; k < NUM_FFS; k++)
            {
                free(fc_mats[k]);
            }
            
            rand_seed++;
        }
        
        for (i = 0; i < 3; i++)
        {
            if (error_metric == i || error_metric == -1)
            {
                // Print to stdout
                fprintf(stdout, "L-%c Error: sgemm=%.4g, dgemm=%.4g, dsgemm=%.4g, 4dsgemm=%.4g, 3dsgemm=%.4g\n", error_chars[i],
                        errors[i][0]/its_per_size, errors[i][1]/its_per_size, errors[i][2]/its_per_size, errors[i][3]/its_per_size, 
                        errors[i][4]/its_per_size);
                
                // Printing to output files
                fprintf(fptr[i], "%d,%.20g,%.20g,%.20g,%.20g,%.20g\n", size, 
                        errors[i][0]/its_per_size, errors[i][1]/its_per_size, errors[i][2]/its_per_size, errors[i][3]/its_per_size, 
                        errors[i][4]/its_per_size);
            }
        }
    }
    for (i = 0; i < 3; i++)
    {
        if (error_metric == i || error_metric == -1)
        {
            fclose(fptr[i]);
            printf("Saved as csv to %s\n",file_name[i]);
        }
    }
}


#define NUM_CFS 9
#define NUM_CFFS 9
void run_precision_cuda_tests(int init_size, int max_size, int multiplier, int its_per_size, int rand_seed, int error_metric, char* save_file)
{
    // computes Forward error using the error_metric norm. if error_metric=-1 then it will compute for al error metrics
    // compute tf32 functions(both cublas and CPU mimics) on square matricies of sizes between init_size and max_size, incrementing in multiples of multiplier
    // For each matrix size, the reported error will be the avarage over its_per_size iterations
    // File will be saves to ../data/save_file_<error_metric char>-norm
    #if CUDA_USABLE == 1
    int size, r, i, k, it;
    double denom;

    double* truth;

    double* diffs[NUM_CFS];
    double* dc_mats[NUM_CFS-NUM_CFFS];
    float* fc_mats[NUM_CFFS];
    double errors[3][NUM_CFS];

    FILE *fptr[3];
    char file_name[3][40];
    char error_chars[3] = "I12";
    
    for (i = 0; i < 3; i++)
    {
        if (error_metric == i || error_metric == -1)
        {
            snprintf(file_name[i], 40, "../data/%s_%c-norm", save_file, error_chars[i]);
            fptr[i] = fopen(file_name[i],"w");
            fprintf(fptr[i], "size, FP16 GEMM, CPU FP32 GEMM, cuBLAS FP32 GEMM, CPU TF32 GEMM, cuBLAS TF32 GEMM, 4 CPU TF32 GEMMs, 4 cuBLAS TF32 GEMMs, 3 CPU TF32 GEMMs, 3 cuBLAS TF32 GEMMs, averaged over %d random multiplciations\n", its_per_size);
        }
    }

    for (size = init_size; size <= max_size; size *= multiplier)
    {
        for (i = 0; i < 3; i++)
        {
            for (k = 0; k < NUM_CFS; k++)
            {
                errors[i][k] = 0.0;
            }
        }

        fprintf(stdout, "Computing with size=%d\n", size);

        for (it = 0; it < its_per_size; it++)
        {
            for (k = 0; k < NUM_CFS; k++)
            {
                diffs[k] = (double*)calloc(size * size, sizeof(double));
            }

            /************************
             *  generate matricies  *
             ************************/
            double* truth = (double*)calloc(size * size, sizeof(double));
            double* a_mat_double = (double*)malloc(sizeof(double) * size * size);
            double* b_mat_double = (double*)malloc(sizeof(double) * size * size);
            float* a_mat_single = (float*)malloc(sizeof(float) * size * size);
            float* b_mat_single = (float*)malloc(sizeof(float) * size * size);

            for (k = 0; k < NUM_CFS - NUM_CFFS; k++)
            {
                dc_mats[k] = (double*)calloc(size * size, sizeof(double));
            }
            for (k = 0; k < NUM_CFFS; k++)
            {
                fc_mats[k] = (float*)calloc(size * size, sizeof(float));
            }

            generate_random_matrices(a_mat_double, size, size, b_mat_double, size, size, -0.5, 0.5, rand_seed);

            for (i = 0; i< size*size; i++)
            {
                a_mat_single[i] = (float)a_mat_double[i];
                b_mat_single[i] = (float)b_mat_double[i];
            }

            /***************
             *  RUN tests  *
             ***************/
            // Truth data
            i_dgemm(a_mat_double, b_mat_double, truth, size, size, size);
            // "Single" Precision
	        i_hgemm(a_mat_single, b_mat_single, fc_mats[0], size, size, size);
            i_sgemm(a_mat_single, b_mat_single, fc_mats[1], size, size, size);
            cublas_sgemm(a_mat_single, b_mat_single, fc_mats[2], size, size, size);
        
            // TF32
            i_tf32_gemm(a_mat_single, b_mat_single, fc_mats[3], size, size, size);
            cublas_tf32_gemm(a_mat_single, b_mat_single, fc_mats[4], size, size, size);

            i_tf32_4gemm(a_mat_single, b_mat_single, fc_mats[5], size, size, size);
            cublas_tf32_4gemm(a_mat_single, b_mat_single, fc_mats[6], size, size, size);

            i_tf32_3gemm(a_mat_single, b_mat_single, fc_mats[7], size, size, size);
            cublas_tf32_3gemm(a_mat_single, b_mat_single, fc_mats[8], size, size, size);

            
            /*********************************
             *  Error checking and printing  *
             *********************************/
            for (k = 0; k < NUM_CFFS; k++)
            {
                for (i = 0; i < size * size; i++ )
                {
                    diffs[k][i] += fabs(truth[i] - fc_mats[k][i]);
                }
            }
            for (k = NUM_CFFS; k < NUM_CFS; k++)
            {
                for (i = 0; i < size * size; i++ )
                {
                    diffs[k][i] += fabs(truth[i] - dc_mats[k-NUM_CFFS][i]);
                }
            }

            for (i = 0; i < 3; i++)
            {
                denom = (norm(i, a_mat_double, size * size) * norm(i, b_mat_double, size * size));
		        for (k = 0; k < NUM_CFS; k++)
                {
                    errors[i][k] += norm(i, diffs[k], size * size) / denom;
                }
            }
            
            // Free memory
            for (k = 0; k < NUM_CFS; k++)
            {
                free(diffs[k]);
            }

            free(a_mat_double);
            free(b_mat_double);
            free(a_mat_single);
            free(b_mat_single);
            free(truth);

            for (k = 0; k < NUM_CFS - NUM_CFFS; k++)
            {
                free(dc_mats[k]);
            }
            for (k = 0; k < NUM_CFFS; k++)
            {
                free(fc_mats[k]);
            }
            
            rand_seed++;
        }

        for (i = 0; i < 3; i++)
        {
            if (error_metric == i || error_metric == -1)
            {
                // print to stdout
                fprintf(stdout, "L-%c Error: hgemm=%.20g, sgemm=%.20g, cublas_sgemm=%.20g, i_tf32_gemm=%.20g, cublas_tf32_gemm=%.20g, i_tf32_4gemm=%.20g, cublas_tf32_4gemm=%.20g, i_tf32_3gemm=%.20g, cublas_tf32_3gemm=%.20g.\n", 
                        error_chars[i], errors[i][0]/its_per_size, errors[i][1]/its_per_size, errors[i][2]/its_per_size, errors[i][3]/its_per_size, errors[i][4]/its_per_size, 
                        errors[i][5]/its_per_size, errors[i][6]/its_per_size, errors[i][7]/its_per_size, errors[i][8]/its_per_size);

                // Printing to output files
                fprintf(fptr[i], "%d,%.20g,%.20g,%.20g,%.20g,%.20g,%.20g,%.20g,%.20g,%.20g\n", size, 
                        errors[i][0]/its_per_size, errors[i][1]/its_per_size, errors[i][2]/its_per_size, errors[i][3]/its_per_size, 
                        errors[i][4]/its_per_size, errors[i][5]/its_per_size, errors[i][6]/its_per_size, errors[i][7]/its_per_size, errors[i][8]/its_per_size);
            }
        }
    }
    for (i = 0; i < 3; i++)
    {
        if (error_metric == i || error_metric == -1)
        {
            fclose(fptr[i]);
            printf("Saved as csv to %s\n",file_name[i]);
        }
    }
    #else
    printf("No CUDA. Not implemented!\n");
    #endif
}


int main()
{   
    run_precision_tests(128, 4000, 2, 1, 1, -1, "gemm_test");
    //run_precision_cuda_tests(128, 35000, 2, 1, 1, -1, "gemm_cuda_test");
    
    return 0;
}
