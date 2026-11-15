# 0046 The pyramids render in processes started by a forkserver

## Status

Accepted

## Date

2026-11-15

## Context

`ras-pyramid` and `pc-pyramid` render the channels of the data in processes of their own (8 by default, outside
the executor, decision 0034), each channel into thousands of 256 x 256 tiles. Processes, not threads: zarr 3 runs
the synchronous calls of a process on one I/O thread and handles about 1 - 2 thousand chunks a second per process,
whatever the threads. With the channels in 8 threads instead of 8 processes the pyramids took 3 - 5 times longer
(2048 x 4096 x 32 complex64: 12.4 instead of 2.4 s; 10 million points x 32: 21 instead of 3.9 s for the
channels), with the same output.

The processes were forks of the calling process. A fork copies only the thread that calls it; the locks held by the
other threads stay locked in the child. The process of a pyramid always has threads (zarr's I/O thread, numba's
threads after a parallel kernel, a threaded BLAS), Python 3.12 warns at every such fork, and a fork after the DS
step hung (decision 0043). `spawn` starts a new interpreter that imports moraine in every process: 1.9 s for a
pool of 8.

## Decision

- The pyramids start their processes from a forkserver (the `multiprocessing` start method `forkserver`), whose
  server imports `moraine.cli.plot` when it starts, once per process. The server has no threads; each pool forks
  its processes from it. Where there is no forkserver (Windows), the processes are spawned.
- The arguments of the processes are pickled, so the indices every process of `pc-pyramid` reads (the cell of each
  point, the cells of the coarser levels, the cells with points: 8 bytes per point and about 4 per cell) are saved
  once in a temporary directory (`TMPDIR`) and memory mapped by the processes: one copy for all of them, as with
  fork. The directory is removed at the end.

## Consequences

- The calling process is not forked: no warning, also where it has threads (`tests/test_cli_plot.py`). The
  pyramids are the same bit for bit.
- The first pyramid of a process takes about 1.4 s longer, the start of the server; the next pools start in
  0.03 s, as with fork. The server (about 170 MB) stays until the process ends, e.g. in a notebook kernel; `moraine
  run` starts it once for all its pyramid steps. On 10 million points x 16 channels (`mean`, a new process each
  time) `pc-pyramid` takes 8.8 - 9.0 instead of 7.4 - 7.9 s, with the same memory (6.6 - 8.2 GB for all the
  processes at the peak). The 14 pyramid steps of the examples on Campi Flegrei in one process take 21.7 instead
  of 23.5 - 25.0 s: the first 2 s longer, the next the same or up to 1.2 s faster.
- The processes run the main script again, as with `spawn` (decision 0034): a script that calls `ras_pyramid` or
  `pc_pyramid` needs the `if __name__ == '__main__'` guard. The command line, pipelines, notebooks and `python -m`
  do not.
- `pc-pyramid` writes 8 bytes per point and about 4 per cell to `TMPDIR` while it renders.

## Do not

- Do not start processes in moraine with `fork` (the start method or `os.fork`): the process has threads.
- Do not render the channels of a pyramid in threads of one process: zarr's handling of the chunks of a process
  is the limit, not moraine's code.
- Do not pass large arrays to processes as arguments, which copies them into each process: save them once and
  memory map them.
