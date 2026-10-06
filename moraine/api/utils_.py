"""internal utilities for developing"""


__all__ = ['mjit', 'ngjit', 'ngpjit', 'is_cuda_available', 'get_n_cpus_avail', 'get_mem_avail', 'get_array_module']

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import numba
import numpy as np
from numba import jit

# Compiled numba functions are cached on disk (decision 0029). numba invalidates a cached function only when its own
# file changes, not when a numba function it calls from another file does; so the cache directory is named after a
# hash of all moraine sources and the numba version: any change of the code uses a new, empty directory. It is set
# only while a function is decorated, so that numba code of the user keeps its own cache location.
_CACHE_KEEP_DAYS = 30

def _source_hash(root=None):
    """hash of the python sources under `root` (the moraine package by default) and the numba version"""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    h = hashlib.sha1(numba.__version__.encode())
    for p in sorted(root.rglob('*.py')):
        if '.ipynb_checkpoints' in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]

def _numba_cache_dir():
    """cache directory of this version of the sources, None if it cannot be created; directories of other versions
    not used for `_CACHE_KEEP_DAYS` days are removed"""
    base = os.environ.get('NUMBA_CACHE_DIR')
    base = Path(base)/'moraine' if base else Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'moraine'/'numba'
    path = base/_source_hash()
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.utime(path)
    except OSError:
        return None
    for old in base.iterdir():
        try:
            if old != path and old.is_dir() and time.time()-old.stat().st_mtime > _CACHE_KEEP_DAYS*86400:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return str(path)

_CACHE_DIR = _numba_cache_dir()

def mjit(**options):
    """`numba.jit` with the compiled code cached in moraine's numba cache directory; use it (or `ngjit`, `ngpjit`)
    instead of `cache=True`"""
    def decorate(func):
        if _CACHE_DIR is None:
            return jit(**options)(func)
        old = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = _CACHE_DIR
        try:
            return jit(cache=True, **options)(func)
        finally:
            numba.config.CACHE_DIR = old
    return decorate

# Adapted from spatialpandas at https://github.com/holoviz/spatialpandas under BSD-2-Clause license.
ngjit = mjit(nopython=True, nogil=True)
ngpjit = mjit(nopython=True, nogil=True, parallel=True)

def _default_cuda_home():
    """In a conda environment that is not activated, point CUDA_HOME to the environment, so that numba-cuda finds
    its CUDA libraries (libnvvm) as the CUDA target built into numba did; variables already set are kept."""
    if any(os.environ.get(v) for v in ('CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX')):
        return
    if (Path(sys.prefix)/'nvvm').is_dir():
        os.environ['CUDA_HOME'] = sys.prefix

_default_cuda_home()

def is_cuda_available():
    # an empty CUDA_VISIBLE_DEVICES or -1 hides all GPUs
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    return devices not in ('', '-1')

def get_n_cpus_avail():
    num_available_cpus = len(os.sched_getaffinity(0))
    return num_available_cpus

def get_mem_avail():
    """Memory available to this process in bytes: the free memory of the machine, or less under a cgroup
    limit (e.g. a SLURM job)."""
    import psutil
    avail = psutil.virtual_memory().available
    for limit_file, usage_file in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),     # cgroup v2
                                   ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                                    '/sys/fs/cgroup/memory/memory.usage_in_bytes')):                  # cgroup v1
        try:
            with open(limit_file) as f:
                limit = f.read().strip()
            with open(usage_file) as f:
                usage = int(f.read().strip())
        except (OSError, ValueError):
            continue
        if limit.isdigit() and int(limit) < avail + usage:
            avail = min(avail, int(limit) - usage)
        break
    return max(int(avail), 0)

def get_array_module(array):
    if is_cuda_available():
        try:
            import cupy
            return cupy.get_array_module(array)
        except:
            return np
    else:
        return np
