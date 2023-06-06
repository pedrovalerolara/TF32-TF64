import numpy as np
from plot_visuals import *

#to_loop = ["c"] # cuda plots
to_loop = ["d"]  # dsgemm plots


for error_metric in ["1", "2", "I"]:
    for prec in to_loop:
        if prec == 'c':
            to_plot = [1,3,5,7,9]
            file_prefix = "gemm_cuda_test"
        else:
            to_plot = [1,2,3,4,5]
            file_prefix = "gemm_test"
        
        file_name = f"../data/{file_prefix}_{error_metric}-norm"

        # FP16 GEMM, CPU FP32 GEMM, cuBLAS FP32 GEMM, CPU TF32 GEMM, cuBLAS TF32 GEMM, 4 CPU TF32 GEMMs, 4 cuBLAS TF32 GEMMs, 3 CPU TF32 GEMMs, 3 cuBLAS TF32 GEMMs
        # sgemm, cublas_sgemm, i_tf32_gemm, cublas_tf32_gemm, cublas_tf32_gemm_test, i_tf32_4gemm, cublas_tf32_4gemm
        
        data = np.genfromtxt(file_name, delimiter=',', skip_header=True).transpose()

        with open(file_name) as f:
            headers = f.readline().rstrip().split(",")
        headers = [headers[i] for i in to_plot]

        fig, ax = plt.subplots(figsize=(10,7))
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.ylabel(f"Forward l-{'inf' if error_metric == 'I' else error_metric} Error")
        plt.xlabel("Matrix Size")
        plt.title(f"Forward l-{'inf' if error_metric == 'I' else error_metric} Error in Mixed Precision GEMM")

        ls = ['-','--','-.',':']
        for i in to_plot:
            # style = 'dashed' if i in [] else "solid"
            ax.plot([round(v) for v in data[0]], data[i])

        #plt.ylim(5e-9, 2e-4)
        #plt.xlim(100, 34000)

        ax.legend(headers)
        # for latex plots: Uncomment eps plots
        # plt.savefig(f"../plots/gemm/{prec}{file_prefix}_l{error_metric}-norm_plot.eps", format='eps', dpi=1000)
        # print(f"Plot saved to: ../plots/gemm/{prec}{file_prefix}_l{error_metric}-norm_plot.eps")

        plt.savefig(f"../plots/gemm/{prec}{file_prefix}_l{error_metric}-norm_plot.png")  #, dpi=1000)
        print(f"Plot saved to: ../plots/gemm/{prec}{file_prefix}_l{error_metric}-norm_plot.png")
