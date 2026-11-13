"""The CPU and GPU of the process: GPU detection, numpy / cupy dispatch, available cores and memory"""

__all__ = ['is_cuda_available', 'get_n_cpus_avail', 'get_mem_avail', 'get_array_module']

import os
import sys
from pathlib import Path

import numpy as np

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
