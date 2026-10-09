# 0032 The commands run their work as tasks of an executor, not as dask array graphs

## Status

Accepted

## Date

2026-10-09

## Context

Every CLI function built a dask array graph by hand: start a cluster, delayed reads, `from_delayed`, `da.block`,
delayed writes, `persist`, a progress bar. About 40 lines, repeated 36 times, full of dask details (`meta`,
chunk sizes, overlap trimming) that hid the actual work: which chunk goes in, what comes out. dask served only as a
pool of workers (threads, processes, one process per GPU) and as a progress bar; no array algebra, no multi-node.
Starting and stopping a cluster cost 4 to 7 s per GPU command (Campi Flegrei run of 2026-10-08), about 50 s per
pipeline. The parallel layer should be small enough for an AI agent to reason about, and replaceable.

## Decision

`moraine.cli.executor.Executor` is the only way a command runs work in parallel:

- `map(fn, tasks)` runs a plain function with picklable arguments for every task and returns the results in
  order; `chunk_task` with `Chunk` is the usual task (read chunks of zarr arrays, call an API function, write the
  results to chunks of zarr arrays); `put` shares one read-only object with all tasks; `run_on_workers` runs a
  setup function once per worker (e.g. the GPU memory allocator).
- A task reads its inputs and writes its outputs itself; the parts it writes cover whole chunks of the output
  (`check_aligned`), so tasks never write the same chunk.
- The backend of the executor is dask (`LocalCluster`, `dask_cuda.LocalCUDACluster`) for now; the commands do
  not see it. Pools of pure functions in one process (`mcf-pc`, `emcf-pc`, `unwrap-correct-closure-pc`) use
  `concurrent.futures` directly.
- The progress of a command is logged at every tenth of its tasks.

## Consequences

- The commands shrink to the list of their tasks; `moraine.cli.dask_from_zarr`, `dask_from_zarr_overlap` and
  `dask_to_zarr` are no longer used by them (kept until the backend changes).
- Every ported command writes the same values, dtype and chunks as before (checked bit for bit on a crop of the
  sample data for 23 commands, and on Campi Flegrei for the pipelines 02 to 05).
- The backend can be replaced by a small implementation of moraine's own without touching the commands (a
  separate record); the worker arguments of the commands (`n_workers`, `threads_per_worker`, `processes`,
  `rmm_pool_size`, `**dask_cluster_arg`) keep their meaning.
- `math` writes the dtype of its expression (before: always float64). `temp-coh` accepts an integer chunk size for
  point clouds (it failed before).

## Do not

- Do not build dask arrays or task graphs in a command, and do not import dask in `moraine/cli/` outside the executor.
- Do not pass large arrays as task arguments: share them with `put` or read them in the task from zarr.
- Do not let two tasks write parts of the same chunk; a processing chunk is a multiple of the output chunks.
