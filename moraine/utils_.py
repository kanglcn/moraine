"""internal utilities for developing"""


__all__ = ['ngjit', 'ngpjit', 'is_cuda_available', 'get_n_cpus_avail', 'get_array_module']

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

def get_array_module(array):
    if is_cuda_available():
        try:
            import cupy
            return cupy.get_array_module(array)
        except:
            return np
    else:
        return np
