"""Pixel quality metrics"""

__all__ = ['temp_coh']

import math
import numpy as np
import moraine as mr
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
from numba import prange
from .utils_ import ngpjit

# @ngpjit
# def _temp_coh_pc_numba(
#     intf:np.ndarray,# complex interferograms/coherence metrix, dtype np.complex64
#     rslc:np.ndarray, # complex rslc/phase history, dtype np.complex64
#     image_pairs:np.ndarray, # image pairs
# ):
#     nimages = rslc.shape[-1]
#     n_points = rslc.shape[0]
#     n_image_pairs = image_pairs.shape[0]
#     temp_coh = np.empty(n_points,dtype=np.float32)
#     for i in prange(n_points):
#         rslc_ = rslc[i]
#         intf_ = intf[i]
#         for j in range(nimages):
#             rslc_[j] = rslc_[j]/abs(rslc_[j])
#         for j in range(n_image_pairs):
#             intf_[j] = intf_[j]/abs(intf_[j])
#         _t_coh = np.float32(0.0)
#         for j in range(n_image_pairs):
#             n, k = image_pairs[j,0],image_pairs[j,1]
#             rslc_intf_ = np.conjugate(rslc_[n])*rslc_[k]
#             diff_ph = intf_[j]*rslc_intf_
#             _t_coh += diff_ph.real
#         _t_coh = _t_coh/n_image_pairs
#         temp_coh[i] = _t_coh
#     return temp_coh

@ngpjit
def _temp_coh_pc_numba(
    intf:np.ndarray,
    rslc:np.ndarray,
    image_pairs:np.ndarray,
):
    """Parameters
    ----------
    intf : np.ndarray
        complex interferograms/coherence metrix, dtype np.complex64
    rslc : np.ndarray
        complex rslc/phase history, dtype np.complex64
    image_pairs : np.ndarray
        image pairs
    """
    nimages = rslc.shape[-1]
    n_points = rslc.shape[0]
    n_image_pairs = image_pairs.shape[0]
    temp_coh = np.empty(n_points,dtype=np.float32)
    for i in prange(n_points):
        # unit amplitude values in local variables: the inputs are not changed
        rslc_ = np.empty(nimages, dtype=rslc.dtype)
        for j in range(nimages):
            rslc_[j] = rslc[i,j]/abs(rslc[i,j])
        _t_coh = np.complex64(0.0)
        for j in range(n_image_pairs):
            n, k = image_pairs[j,0],image_pairs[j,1]
            rslc_intf_ = np.conjugate(rslc_[n])*rslc_[k]
            diff_ph = intf[i,j]/abs(intf[i,j])*rslc_intf_
            _t_coh += diff_ph
        _t_coh = np.abs(_t_coh)/n_image_pairs
        temp_coh[i] = _t_coh
    return temp_coh

if is_cuda_available():
    from numba import cuda

    @cuda.jit
    def _temp_coh_pc_cuda(intf, rslc, ref, sec, temp_coh):
        # one warp per point: the lanes read consecutive image pairs (coalesced), then reduce by shuffles;
        # |sum over the image pairs of the unit interferogram times the unit conj(ref) sec| / n_pairs
        i = cuda.grid(1)//32
        if i >= intf.shape[0]:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        n_pairs = intf.shape[1]
        sr = np.float32(0.0); si = np.float32(0.0)
        for j in range(lane, n_pairs, 32):
            c = intf[i,j]; a = rslc[i,ref[j]]; b = rslc[i,sec[j]]
            p = c*a.conjugate()*b
            norm = math.sqrt((c.real*c.real+c.imag*c.imag)*(a.real*a.real+a.imag*a.imag)*(b.real*b.real+b.imag*b.imag))
            sr += p.real/norm
            si += p.imag/norm
        offset = 16
        while offset > 0:
            sr += cuda.shfl_down_sync(0xffffffff, sr, offset)
            si += cuda.shfl_down_sync(0xffffffff, si, offset)
            offset //= 2
        if lane == 0:
            temp_coh[i] = math.sqrt(sr*sr+si*si)/n_pairs

def temp_coh(
    intf:np.ndarray,
    rslc:np.ndarray,
    image_pairs:np.ndarray=None,
    block_size:int=128,
):
    """Estimation of temporal coherence.

    Parameters
    ----------
    intf : np.ndarray
        complex interferograms/coherence metrix, dtype cp/np.complex64, shape 2D(pc) or 3D(ras)
    rslc : np.ndarray
        complex rslc/phase history, dtype cp/np.complex64, shape 2D(pc) or 3D(ras)
    image_pairs : np.ndarray, optional
        image pairs
    block_size : int, default: 128
        the CUDA block size, a multiple of 32, only applied for cuda
    """
    xp = get_array_module(intf)
    assert intf.ndim == rslc.ndim
    if intf.ndim == 3:
        # convert to pc
        is_ras = True
        height, width = intf.shape[:2]
        n_points =height*width
        intf = intf.reshape((-1,intf.shape[-1]))
        rslc = rslc.reshape((-1,rslc.shape[-1]))
    elif intf.ndim == 2:
        is_ras = False
        n_points = rslc.shape[0]

    nimages = rslc.shape[-1]
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = image_pairs.astype(np.int32)

    if xp is np:
        out_temp_coh = _temp_coh_pc_numba(intf,rslc,image_pairs)
    else:
        out_temp_coh = cp.empty(n_points, dtype=cp.float32)
        if n_points > 0:
            _temp_coh_pc_cuda[(n_points*32+block_size-1)//block_size, block_size](
                cp.ascontiguousarray(intf), cp.ascontiguousarray(rslc), cp.asarray(image_pairs[:,0]),
                cp.asarray(image_pairs[:,1]), out_temp_coh)

    if is_ras:
        out_temp_coh = out_temp_coh.reshape((height,width))
    return out_temp_coh
