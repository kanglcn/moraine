"""SLC amplitude calibration"""

__all__ = ['rslc2amp', 'calamp']

import numpy as np
import math
import numba
from .utils_ import is_cuda_available, get_array_module
from .utils_ import mjit
if is_cuda_available():
    import cupy as cp

@mjit(nopython=True, parallel=True)
def _rslc2amp_numba(rslc):
    # support rslc and rslc stack
    rslc_shape = rslc.shape
    rslc = rslc.reshape(-1)
    amp = np.empty(rslc.shape,dtype=np.float32)
    for i in numba.prange(rslc.shape[0]):
        amp[i] = np.abs(rslc[i])
    return amp.reshape(rslc_shape)

if is_cuda_available():
    def _rslc2amp_cp(rslc):
        return cp.abs(rslc)

def rslc2amp(rslc:np.ndarray,
            )-> np.ndarray:
    """calculation the amplitude from SLC/SLC stack

    Parameters
    ----------
    rslc : np.ndarray
        rslc, 2D/3D numpy array or cupy array

    Returns
    -------
    np.ndarray
        amplitude, 2D/3D numpy array or cupy array
    """
    xp = get_array_module(rslc)
    if xp is np:
        return _rslc2amp_numba(rslc)
    else:
        return _rslc2amp_cp(rslc)

@mjit(nopython=True, parallel=True)
def _calamp_numba(amp):
    nlines, width = amp.shape
    npixels = nlines*width

    amp = amp.reshape(npixels)
    nanamp_mean = np.float32(0.0)
    n_valid_pixel = np.int32(0)
    for i in numba.prange(npixels):
        if not np.isnan(amp[i]):
            nanamp_mean += amp[i]
            n_valid_pixel += 1
    nanamp_mean = nanamp_mean/n_valid_pixel
    amp_caled = np.empty(npixels,dtype=amp.dtype)
    for i in numba.prange(npixels):
            amp_caled[i] = amp[i]/nanamp_mean
    return amp_caled.reshape(nlines,width)

if is_cuda_available():
    def _calamp_cp(amp):
        return amp/cp.nanmean(amp)

def calamp(amp:np.ndarray,
            )-> np.ndarray:
    """calculation the calibrated amplitude from amplitude of one single SLC

    Parameters
    ----------
    amp : np.ndarray
        amplitude, 2D numpy array or cupy array

    Returns
    -------
    np.ndarray
        calibrated amplitude, 2D numpy array or cupy array
    """
    xp = get_array_module(amp)
    if xp is np:
        return _calamp_numba(amp)
    else:
        return _calamp_cp(amp)
