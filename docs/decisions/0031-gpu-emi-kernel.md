# 0031 The regularized EMI on the GPU is one numba.cuda thread block per point with its matrices in shared memory

## Status

Accepted

## Date

2026-10-06

## Context

The GPU EMI was cupy: batched eigenvalues of |coh| (cuSOLVER Jacobi), inverse, and all eigenpairs of the EMI
matrix; on an A100 42.6 us per point for 92 images (128 CPU cores: 11-12.5 us) and about 445 kB of GPU memory per
point (199 kB for 60 images) beyond its input, so that a batch of 20 000 points needed 8.5 GiB. Studied
2026-10-06 on 2000 Campi Flegrei DS candidates (92 images): cheaper algorithms do not help. Lanczos estimates of
lambda_min of |coh| need about 40 steps, because the EMI phases are sensitive to beta itself (30 steps gave phase
errors of ~pi for 1 % of the points); Cholesky of the EMI matrix with Lanczos or inverse iteration on its inverse
needs 30 to several hundred iterations, as lambda2/lambda1 is below 1.02 for 10 % of the points (p50 1.18), which is
as much work as the Householder reduction and more serial steps. The work per point is a chain of about n small
matrix-vector steps: one A100 holds about as much of it on chip as 128 CPU cores (shared memory 19.5 TB/s, CPU L1
about 18 TB/s), so the GPU kernel is about as fast as such a CPU, not 10 times faster.

## Decision

- The regularized EMI (`regularize=True`, the default) of cupy arrays runs in a numba.cuda kernel
  (`moraine/api/emi_cuda_.py`): one thread block of 128 threads per point; |coh|, its inverse and the EMI matrix as
  packed lower triangles in shared memory; Householder tridiagonalization, Sturm multisection for lambda_min and
  lambda_max of |coh| (beta by the same function as on the CPU, decision 0023) and lambda_1 of the EMI matrix, the
  sweep operator for the inverse, inverse iteration and back transformation of the one eigenvector needed; float32.
- cupy stays for `regularize=False` (the kernel does not pivot; |coh| that is not positive definite is not
  inverted stably) and for more images than the shared memory of a block holds (A100: more than 198 images; more than
  101 images when the kernel cannot be allowed more than 48 kB, e.g. without numba-cuda).
- GPU input stays on the GPU: there is no automatic switch to the CPU, which is faster for many images on a large
  CPU (A100 vs 128 cores: GPU faster below about 100 images).

## Consequences

92 images, A100: 10 instead of 43 us per point, no GPU memory beyond the input and output, phase error p99 3e-4
instead of 2e-3 rad (float64 reference). Fused DS step (92 images, batch 1000): 39 instead of 72 us per point.
With 30 or fewer images it is about as fast as cuSOLVER's batched Jacobi used by cupy (17 images 0.8 vs 1.1 us,
30 images 1.7 vs 1.5 us, 10 images 0.5 vs 0.2 us). The speed falls with the number of images faster than on the CPU
(fewer blocks per multiprocessor: 31 us for 128 images, 75 for 150, 132 for 196, CPU 24, 26, 54); the 03 guide says
so.
The kernel sets the shared memory limit of the function with a numba-cuda internal (`_codelibrary.get_cufunc`).

## Do not

- Do not run the kernel for `regularize=False` without pivoting.
- Do not use static shared arrays whose size depends on the number of images (numba.cuda does not check bounds:
  a fixed size of 128 silently gave wrong phases for 150 images).
