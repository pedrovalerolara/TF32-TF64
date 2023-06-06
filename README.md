## Abstract
With the introduction of high-performance AI cores in new GPUs, such as NVIDIA's Tensor Cores, AMD's Matrix Core, and ARM SME, AI applications have seen large performance increases in accelerating GEMM. We want to extend the successes of AI cores into high-performance computing for scientific applications.  We present an extension to NVIDIA's TF32 mixed precision framework, called DSGEMM for double precision GEMM. The TF32 and DSGEMM mixed precision frameworks work by reducing the precision of the input data. The TF32 mixed precision framework in particular sees an 8x performance increase as compared to SGEMM. Using the DSGEMM framework with the mixed precision method described in [1], we can achieve near double precision accuracy. Currently, there is no hardware support for our proposed framework but we expect to see similar performance results: with a potential 2.6x performance acceleration while maintaining near double precision accuracy.



## References
[1] M. Fasi, N. J. Higham, F. Lopez, T. Mary, and M. Mikaitis, “Matrix
multiplication in multiword arithmetic: Error analysis and application
to gpu tensor cores,” 2022, MIMS Preprint. [Online]. Available:
http://eprints.maths.manchester.ac.uk/id/eprint/286