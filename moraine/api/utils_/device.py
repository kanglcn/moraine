"""The CPU and GPU of the process: GPU detection, numpy / cupy dispatch, available cores and memory, BLAS threads"""

__all__ = ['is_cuda_available', 'get_n_cpus_avail', 'get_mem_avail', 'get_array_module']

import os
import sys
import threading
from pathlib import Path

import numpy as np
# scipy's BLAS / LAPACK, which numba's linear algebra and the EMI kernel call, is loaded before the BLAS controller
# below is made: the controller knows only the libraries loaded at that time (pip wheels of numpy and scipy bring
# separate BLAS libraries)
import scipy.linalg.cython_lapack  # noqa: F401
from threadpoolctl import ThreadpoolController

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

# The CPU kernels call LAPACK inside numba parallel loops. A multithreaded BLAS (MKL, OpenBLAS) does not see
# that it is called from many threads and starts its own threads in each call: with 128 cores 128 x 128
# threads, about 2e4 times slower. Its threads are limited to 1 while a kernel runs; the limit is global, so
# it is kept until the last of concurrent callers (e.g. the threads of the executor) has finished.
_blas_controller = ThreadpoolController()
_blas_lock = threading.Lock()
_blas_users = 0
_blas_limit = None

class _single_thread_blas:
    def __enter__(self):
        global _blas_users, _blas_limit
        with _blas_lock:
            if _blas_users == 0:
                _blas_limit = _blas_controller.limit(limits=1, user_api='blas')
            _blas_users += 1
    def __exit__(self, *exc):
        global _blas_users, _blas_limit
        with _blas_lock:
            _blas_users -= 1
            if _blas_users == 0:
                _blas_limit.restore_original_limits(); _blas_limit = None
