# 0029 Compiled numba functions are cached in a directory named after the hash of the sources

## Status

Accepted

## Date

2026-10-06

## Context

Without a disk cache numba compiles every function again in every process: every step of `moraine run` and every
dask worker process; 11 s for the fused DS step, 5 s for the SHP test on 128 cores (2026-10-06). numba's cache
(`cache=True`) checks only the file of a function: a cached function that calls a numba function of another file
keeps the old machine code when the other file changes, and silently gives the old results (checked 2026-10-06:
f in a.py calls g in b.py; after g changed, f returned the old value). moraine calls numba functions across files
(e.g. `pl.py` -> `co.py`, `emcf.py` -> `closure.py`, `mcf.py`), so `cache=True` was unsafe while the code changes.

## Decision

- All numba CPU functions are decorated with `moraine.api.utils_.mjit(**options)` or its shorthands `ngjit`,
  `ngpjit`; they are cached in `~/.cache/moraine/numba/<hash>` (`$XDG_CACHE_HOME/moraine/numba/<hash>`, or
  `$NUMBA_CACHE_DIR/moraine/<hash>` if that is set). The hash covers all python files of the moraine package and the
  numba version, so any change of the code uses a new, empty directory.
- The numba cache directory is set only while a function is decorated; numba code of the user keeps its cache.
- Directories of other hashes not used for 30 days are removed when moraine is imported. If the directory cannot
  be created, the functions are not cached.
- Foreign functions are called by symbol name (`llvmlite.binding.add_symbol` and `numba.types.ExternalFunction`,
  e.g. LAPACK `cheevr` in `pl.py`), not through ctypes pointers, which numba cannot cache.

## Consequences

The first call in a process loads the compiled code (0.2-0.4 s) instead of compiling it, after the first run of a
version of the code. Every change of any moraine file compiles everything again once, also while developing. The
The CUDA kernels (numba.cuda) are cached the same way through `mcuda_jit` (2026-10-08): the first call of the GPU EMI in
a process loads the kernel in about 0.5 s instead of compiling it for 15 s.

## Do not

- Do not use `numba.jit(cache=True)`, `njit(cache=True)` or `cuda.jit(cache=True)` in moraine; use `mjit`, `ngjit`,
  `ngpjit` and `mcuda_jit`.
- Do not call foreign functions through ctypes function pointers inside numba functions.
