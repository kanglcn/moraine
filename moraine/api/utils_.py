"""internal utilities for developing"""


__all__ = ['ngjit', 'ngpjit', 'is_cuda_available', 'get_n_cpus_avail', 'get_mem_avail', 'get_array_module']

# Adapted from spatialpandas at https://github.com/holoviz/spatialpandas under BSD-2-Clause license.

import numpy as np
from numba import jit
import os

ngjit = jit(nopython=True, nogil=True)
ngpjit = jit(nopython=True, nogil=True, parallel=True)

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
