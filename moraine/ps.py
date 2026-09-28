"""persistent scatterers identification"""


__all__ = ['amp_disp']

import numpy as np
import math
import numba
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp

# already robust enough for nan value
@numba.jit(nopython=True, cache=True,parallel=True)
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

# already robust enough for nan value
if is_cuda_available():
    _amp_disp_kernel = cp.ElementwiseKernel(
        'raw T rmli_stack, int32 nlines, int32 width, int32 nimages',
        'raw T amp_disp_stack',
        '''
        int k;
        float mean = 0;
        float f_nimages = nimages;
        for (k=0;k<nimages;k++) {
            mean += rmli_stack[i*nimages+k];
        }
        mean /= f_nimages;
        float std = 0;
        for (k=0;k<nimages;k++) {
            std += powf(rmli_stack[i*nimages+k]-mean,2);
        }
        std = sqrt(std/f_nimages);
        amp_disp_stack[i] = std/mean;
        ''',
        name = 'amp_disp_kernel',no_return=True)

if is_cuda_available():
    def _amp_disp_cp(rslc):
        rmli = cp.abs(rslc)
        nlines,width,nimages = rmli.shape
        amp_disp = cp.empty((nlines,width),dtype=cp.float32)
        _amp_disp_kernel(rmli,cp.int32(nlines),cp.int32(width),cp.int32(nimages),
                         amp_disp,size=nlines*width,block_size=128)
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
