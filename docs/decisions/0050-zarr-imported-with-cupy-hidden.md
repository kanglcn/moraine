# 0050 zarr is imported with cupy hidden

## Status

Accepted

## Date

2026-11-15

## Context

moraine imports cupy only in its GPU code paths, so that `import moraine` works and starts fast where cupy is not
needed. zarr 3.4.1 imports cupy itself wherever cupy is installed: `import zarr` imports `zarr.core.buffer.gpu`
(through `zarr.storage`), which tries `import cupy` for zarr's GPU buffers, and cupy 13.6 imports pytest. Every moraine
process of a GPU environment paid for it, also for work on the CPU: a command, every worker process, the forkserver
of the pyramids. With `CUDA_VISIBLE_DEVICES` set, `import moraine.cli` took 1.82 - 1.87 s (median of 5) and imported
1875 modules. moraine does not use zarr's GPU buffers: a GPU task reads its zarr chunks into numpy arrays, copies them
to the GPU and copies the results back before they are written.

## Decision

- `moraine/__init__.py` imports zarr first, with `sys.modules['cupy'] = None`, so that zarr's `import cupy` raises
  ImportError, which zarr catches. The entry is removed right after, also when the import fails: `sys.modules` is as
  before, and the GPU code paths import cupy as usual.
- Only when neither zarr nor cupy is imported yet: a process that imported zarr or cupy before moraine keeps zarr as
  it is.

## Consequences

- `import moraine.cli` with `CUDA_VISIBLE_DEVICES` set takes 1.46 - 1.49 s instead of 1.82 - 1.87 s (median of 5)
  and imports 1497 modules, neither cupy nor pytest. The GPU code paths and commands (spawned GPU workers, which
  import moraine first) run as before.
- The cost: in a process that imported moraine before zarr, zarr's GPU buffers do not work (`zarr.core.buffer.gpu`
  has no cupy; with `zarr.config.enable_gpu()` writing a cupy array fails). A script that needs them imports zarr or
  cupy before moraine.
- If zarr changes, the tests tell: when zarr imports cupy in another way (another module, or later),
  `test_import_moraine_does_not_import_cupy` of `tests/test_architecture.py` fails; when zarr cannot be imported with
  cupy hidden, `import moraine` and so every test fails. Remove the block of `moraine/__init__.py` when zarr no longer
  imports cupy at its import (a new record supersedes this one); otherwise find what imports cupy
  (`python -X importtime -c "import moraine"`) and write a new record.

## Do not

- Do not leave cupy hidden after zarr's import or hide other packages this way: the GPU code paths must import cupy
  normally.
- Do not use zarr's GPU buffers (`zarr.config.enable_gpu`) in moraine.
