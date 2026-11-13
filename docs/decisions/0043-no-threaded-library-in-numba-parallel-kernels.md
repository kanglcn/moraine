# 0043 Numba parallel kernels call no threaded library

## Status

Accepted

## Date

2026-11-13

## Context

The CPU kernels of the DS stage called BLAS and LAPACK inside numba's parallel loops: `np.dot` in the coherence of
point clouds, `np.linalg.eigvalsh`, `np.linalg.inv` and LAPACK `cheevr` in the EMI. With MKL, whose threads are those of
its OpenMP runtime, a call from numba's threads (TBB) makes them threads of the OpenMP runtime too. That oversubscribed
the cores (a pool of BLAS threads per numba thread, about 2e4 times slower on 128 cores; limited since then by a process
wide limit through `threadpoolctl`) and made every later `fork` of the process hang: the fork handlers run in the
reverse order of their registration, OpenMP takes its lock, then numba's TBB layer joins its worker threads, which wait
for that lock when they exit. `moraine run` runs the steps of a pipeline in one process, so a pipeline with
`cuda = false` hung at its first pyramid after the DS step, and a notebook the same. The problems of these kernels are
small and dense (a matrix of tens to a few hundred images per point), where the overhead of the library calls and the
memory traffic of their matrix vector operations dominate.

## Decision

- A numba parallel kernel (`prange`, `ngpjit`) calls no library with threads of its own: no BLAS or LAPACK, also not
  through numpy's linear algebra in numba. Its linear algebra is written in numba.
- Where LLVM does not vectorize the numba code well enough, the inner kernel is written in LLVM vector instructions
  through a numba intrinsic; otherwise the loops are written so that LLVM vectorizes them (row views with loops from 0,
  rows padded with zeros to whole blocks of 32 columns, FMA contraction allowed in matrix kernels). A kernel that uses
  the input of a point several times copies it once into a buffer of its thread: the input may sit on one memory node
  of several.
- The DS kernels: the coherence `X^H X` by a register blocked micro kernel (4 x 8 block in vector registers, upper
  triangle only); the EMI by the algorithm of the GPU kernel (decision 0031) on full matrices: Householder
  tridiagonalization, Sturm multisection (16 shifts in vector registers), Gauss-Jordan inverse with partial pivoting,
  inverse iteration (`moraine.api.ds.linalg_`).
- No process wide limit of library threads; `threadpoolctl` is no longer a dependency.
- A replacement of a library call is not slower than the call it replaces, measured on one core and on all cores with
  runs of the old and the new code alternating, with inputs on one memory node and spread over all.

## Consequences

- After the DS functions on the CPU the process can fork, and the cores are not oversubscribed.
- On an AMD EPYC 7742 (128 cores, AVX2), 17 - 196 images: the coherence is 1.6 - 2.1 times, the EMI 1.5 - 2.1 times
  faster on one core; on 128 cores both are faster in every case measured (the coherence of 92 images with 7 x 27
  windows only by about 10 %: writing the pages of its new output bounds it). The CPU phase linking step of the Campi
  Flegrei example (1.12 million candidates, 92 images) takes 19 instead of 23.5 s; the compile time is unchanged.
- Results change by float32 rounding: the coherence by at most 9e-7, the EMI phase by about 4e-5 rad (median against
  a float64 EMI). Where the two smallest eigenvalues of the EMI matrix are close, the float32 phase is less accurate,
  as on the GPU, but more accurate than with LAPACK (on the sample data, at the points where the two differ by more
  than 0.01 rad: median error 0.004 instead of 0.015 rad); the DS selected are the same.
- A point with a non-finite coherence gets NaN phase and EMI quality.
- The intrinsics use LLVM vectors of 8 float32: portable through LLVM, tuned for AVX2.
- The example of decision 0029 (LAPACK `cheevr` called by symbol name) is gone; its rule stays for foreign functions
  called outside parallel loops.

## Do not

- Do not call BLAS, LAPACK, numpy's linear algebra or another library with threads of its own inside a numba
  parallel loop, and do not limit the threads of such a library process wide to make it work.
- Do not replace a library call with own code that is slower.
