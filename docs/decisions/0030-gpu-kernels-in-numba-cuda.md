# 0030 GPU kernels are written with numba.cuda (numba-cuda), not as cupy C++ kernels

## Status

Accepted

## Date

2026-10-06

## Context

The GPU code of moraine had cupy ElementwiseKernels: C++ in Python strings, hard to write, test and maintain,
and separate from the numba CPU kernels. numba.cuda kernels are Python, can share device functions with the
CPU kernels (e.g. the KS statistic in `shp.py`), and were as fast or faster where the GPU does real work
(2026-10-06, A100: `temp_coh` 1.8 instead of 7.9 ms, `amp_disp` 1.7 instead of 2.2 ms). Their launch converts
every cupy argument, about 450 us per launch with the CUDA target built into numba (deprecated since numba 0.61)
and 110 us with NVIDIA's `numba-cuda` package, against about 10 us for a cupy kernel: tiny kernels (the per
pixel pre and post processing of `n2f`) take 0.1-1 ms more per call.

## Decision

- GPU kernels are numba.cuda kernels; array operations and linear algebra stay with cupy (e.g. the batched
  matrix products of the coherence). The maintainer prefers one way of writing GPU code to the launch
  overhead of tiny kernels.
- `numba-cuda` is a GPU dependency, installed with conda like cupy, dask-cuda and rmm (README); without it the
  built-in CUDA target of numba is used.

## Consequences

No C++ strings in the code; a tiny kernel costs about 0.1 ms more per call than a cupy one. The CPU and GPU
kernels can share device functions and so give the same results.

## Do not

- Do not add cupy ElementwiseKernel, RawKernel or RawModule code.
