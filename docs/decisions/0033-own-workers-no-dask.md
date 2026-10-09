# 0033 The workers of the commands are moraine's own; dask is not a dependency

## Status

Accepted

## Date

2026-10-09

## Context

With decision 0032 the commands no longer build dask graphs: they hand plain tasks to `Executor`, whose dask
backend only started a `LocalCluster` or `LocalCUDACluster` and submitted the tasks. What dask still cost: 4 to 7 s
to start and stop a cluster in every GPU command (about 50 s of a pipeline of 17 steps), the import of dask,
distributed, dask-cuda, rmm and bokeh in every `moraine` command, conda-only dependencies pinned to the CUDA
version (dask-cuda, rmm), about 10 ms of Python per task in a dask worker instead of 2 to 3 ms in a process
(the fused DS step), the clash of the rmm allocator with the kernel tuning of torch, and worker tracebacks that
agents and users have to read through. moraine never used what dask is good at: array algebra, graph
optimization, spilling, several machines.

## Decision

- `Executor` runs the tasks with its own workers: threads of the command's process for CPU tasks (the kernels
  release the GIL), processes started with ``spawn`` for CPU tasks that hold the GIL and for the GPUs, one
  process per GPU of ``CUDA_VISIBLE_DEVICES`` with ``threads_per_worker`` tasks at a time. A worker process
  binds its GPU by ``CUDA_VISIBLE_DEVICES`` before any CUDA call, takes an rmm memory pool of `rmm_pool_size` of
  the GPU memory when rmm is installed (cupy's own pool otherwise), loads every object shared with `put` once,
  and reports results and errors through a queue; the main process feeds every worker as many tasks as it runs
  at a time, so the memory in flight is bounded by workers x tasks per worker x task memory, as before.
- dask, distributed and dask-cuda are no longer dependencies; rmm is optional (a memory pool). `psutil`, which
  came with distributed, is a dependency of its own (`get_mem_avail`).
- The commands lose `**dask_cluster_arg`; `n_workers`, `threads_per_worker`, `processes`, `rmm_pool_size` keep
  their meaning. `moraine.cli.dask_from_zarr`, `dask_from_zarr_overlap`, `dask_to_zarr` are gone; the zarr
  helpers live in `moraine.cli.zarr_` (`parallel_read_zarr`, `parallel_write_zarr`, `ZarrDir`).
- The progress of a command is logged (every tenth of the tasks), not drawn.

## Consequences

- Same results: every command was compared bit for bit with the dask version on a crop of the sample data (27
  outputs of 23 commands) and the pipelines 02 to 05 on Campi Flegrei agree within the run-to-run rounding of the
  GPU filters.
- A GPU command starts its workers in 2 to 3 s instead of 4 to 7 (the import of moraine in each worker
  process; a worker pool shared by the steps of `moraine run` is a possible next step).
- Pipeline files with `[step.kw]` dask options (`memory_limit`, ...) fail with "unknown worker arguments".
- A task function must be importable (module level) for the process workers; errors that cannot be pickled
  come back as `TaskError` with the traceback of the worker.

## Do not

- Do not add dask, distributed or dask-cuda back, also not as optional dependencies.
- Do not start processes in a command outside the executor (except the subprocesses of GAMMA and the per
  process pools that existed before: `transform`, the pyramids).
- Do not share large arrays with the tasks through their arguments; `put` or zarr.
