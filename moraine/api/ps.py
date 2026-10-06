"""persistent scatterers identification"""


__all__ = ['amp_disp']

import numpy as np
import math
import numba
from .utils_ import is_cuda_available, get_array_module
from .utils_ import mjit
if is_cuda_available():
    import cupy as cp

# already robust enough for nan value
@mjit(nopython=True, parallel=True)
def _amp_disp_numba(rslc):
    nlines, width, nimages = rslc.shape
    npixels = nlines*width
    rslc = rslc.reshape(npixels,nimages)
    amp_disp = np.empty(npixels,dtype=np.float32)
    for i in numba.prange(npixels):
        amp = np.abs(rslc[i,:])
        mean = np.mean(amp)
        std = np.std(amp)
        amp_disp[i] = std/mean
    return amp_disp.reshape(nlines,width)

if is_cuda_available():
    from numba import cuda

    @cuda.jit
    def _amp_disp_cuda(rslc, out):
        # one warp per pixel: the lanes read consecutive images (coalesced); mean, then the deviations from it
        w = cuda.grid(1)//32
        nlines, width, nimages = rslc.shape
        if w >= nlines*width:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        i = w//width; j = w%width
        s = np.float32(0.0)
        for k in range(lane, nimages, 32):
            v = rslc[i,j,k]
            s += math.sqrt(v.real*v.real+v.imag*v.imag)
        offset = 16
        while offset > 0:
            s += cuda.shfl_xor_sync(0xffffffff, s, offset)
            offset //= 2
        mean = s/np.float32(nimages)
        s = np.float32(0.0)
        for k in range(lane, nimages, 32):
            v = rslc[i,j,k]
            d = math.sqrt(v.real*v.real+v.imag*v.imag)-mean
            s += d*d
        offset = 16
        while offset > 0:
            s += cuda.shfl_xor_sync(0xffffffff, s, offset)
            offset //= 2
        if lane == 0:
            out[i,j] = math.sqrt(s/np.float32(nimages))/mean

    def _amp_disp_cp(rslc, block_size=128):
        nlines, width, nimages = rslc.shape
        amp_disp = cp.empty((nlines,width),dtype=cp.float32)
        if nlines*width > 0:
            _amp_disp_cuda[(nlines*width*32+block_size-1)//block_size, block_size](rslc, amp_disp)
        return amp_disp

def amp_disp(rslc:np.ndarray,
            )-> np.ndarray:
    """calculation the amplitude dispersion index from SLC stack.

    Parameters
    ----------
    rslc : np.ndarray
        rslc stack, 3D numpy array or cupy array

    Returns
    -------
    np.ndarray
        dispersion index, 2D numpy array or cupy array
    """
    xp = get_array_module(rslc)
    if xp is np:
        return _amp_disp_numba(rslc)
    else:
        return _amp_disp_cp(rslc)
