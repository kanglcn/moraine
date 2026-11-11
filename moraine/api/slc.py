"""Basic operations on co-registered SLC stacks: amplitude, amplitude calibration, interferograms, multilooking"""

__all__ = ['rslc2amp', 'calamp', 'intf', 'multi_look']

import numpy as np
import math
import numba
from numba import prange
from .utils_.device import is_cuda_available, get_array_module
from .utils_.jit import mjit, ngpjit
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

@ngpjit
def intf(
    rslc1,
    rslc2,
):
    """1 by 1 look interferogram for both raster and point cloud.

    Parameters
    ----------
    rslc1
        reference rslc, arbitrary dims, e.g., 3D(raster stack), 2D(raster or point cloud stack) or 1D(point cloud)
    rslc2
        secondary rslc
    """
    shape = rslc1.shape

    rslc1 = rslc1.reshape(-1)
    rslc2 = rslc2.reshape(-1)
    intf = np.empty(rslc1.shape,dtype=rslc1.dtype)
    for i in prange(rslc1.shape[0]):
        intf[i] = rslc1[i]*np.conj(rslc2[i])/(np.abs(rslc1[i])*np.abs(rslc2[i]))
    intf = intf.reshape(shape)
    return intf

@ngpjit
def multi_look(
    rslc1,
    rslc2,
    looks=(1,1),
):
    """multi looked interferogram for raster data.

    Parameters
    ----------
    rslc1
        reference rslc
    rslc2
        secondary rslc
    looks : default: (1, 1)
        # range and azimuth looks
    """
    az_look, r_look = looks
    out_ny = math.floor(rslc1.shape[0]/az_look)
    out_nx = math.floor(rslc1.shape[1]/r_look)
    out = np.empty((out_ny, out_nx),dtype=rslc1.dtype)
    for i in prange(out_ny):
        for j in range(out_nx):
            az_start, az_stop = i*az_look, (i+1)*az_look
            r_start, r_stop = j*r_look, (j+1)*r_look
            rslc1_ = rslc1[az_start:az_stop, r_start:r_stop].flatten()
            rslc2_ = rslc2[az_start:az_stop, r_start:r_stop].flatten()
            dnmnt = math.sqrt(np.sum(rslc1_.real**2+rslc1_.imag**2)*np.sum(rslc2_.real**2+rslc2_.imag**2))
            out[i,j] = np.sum(rslc1_*rslc2_.conj())/dnmnt
    return out
