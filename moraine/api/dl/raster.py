"""Noise2Fringe filtering of raster interferograms: n2f, and n2fs3d with the amplitude dispersion index as input"""

__all__ = ['n2f', 'n2fs3d']

import math
import random
import numpy as np
from numba import prange
from ..utils_.jit import ngpjit, mcuda_jit
from ..utils_.device import is_cuda_available, get_array_module
from ..utils_.chunk import chunkwise_slicing_mapping
from .model import _import_torch, _get_model, _cuda_device
if is_cuda_available():
    import cupy as cp

def _infer_unet(
    model,
    x,
):
    """run the unet model, output is the same kind of array as the input, float32 with unit norm over the channels

    Parameters
    ----------
    model
    x
        model input, np.ndarray or cp.ndarray, shape (1, in_channels, nlines, width)
    """
    torch = _import_torch()
    param = next(model.parameters())
    def run(t):
        if param.dtype == torch.float32:
            return model(t)
        # half precision model: the input in its dtype and layout, the output normalized again in float32 (the model
        # normalizes in float16, the amplitude is then 1 within 1e-3)
        t = t.to(param.dtype, memory_format=torch.channels_last)
        return torch.nn.functional.normalize(model(t).float(), dim=1).contiguous()
    with torch.inference_mode():
        if isinstance(x, np.ndarray):
            return run(torch.from_numpy(x).to(param.device)).cpu().numpy()
        return cp.from_dlpack(run(torch.from_dlpack(cp.ascontiguousarray(x))))

@ngpjit
def _pre_infer_n2f_numba(intf):
    nlines, width = intf.shape
    out = np.empty((1,2,nlines,width),dtype=np.float32)
    mask = np.empty((nlines,width),dtype=np.bool_)
    for i in prange(nlines):
        for j in prange(width):
            intf_i_j = intf[i,j]
            if math.isnan(intf_i_j.real):
                mask[i,j] = True
                random_phase = random.uniform(-math.pi,math.pi)
                out[0,0,i,j] = math.cos(random_phase)
                out[0,1,i,j] = math.sin(random_phase)
            else:
                mask[i,j] = False
                amp = abs(intf_i_j)
                out[0,0,i,j] = intf_i_j.real/amp
                out[0,1,i,j] = intf_i_j.imag/amp
    return out, mask

if is_cuda_available():
    from numba import cuda

    @mcuda_jit()
    def _pre_infer_cuda(intf, adi, out, mask):
        # one thread per pixel: unit interferogram in the last two channels of out, adi (if not empty) in the first
        t = cuda.grid(1)
        nlines, width = intf.shape
        if t >= nlines*width:
            return
        i = t//width; j = t%width
        c0 = out.shape[1]-2
        v = intf[i,j]
        masked = math.isnan(v.real)
        if adi.size > 0:
            masked = masked or math.isnan(adi[i,j])
            out[0,0,i,j] = np.float32(1.0) if masked else adi[i,j]
        mask[i,j] = masked
        if not masked:
            amp = abs(v)
            out[0,c0,i,j] = v.real/amp
            out[0,c0+1,i,j] = v.imag/amp

    def _pre_infer_cp(intf, adi=None):
        """model input (1, 2 or 3, nlines, width) float32 and the mask of pixels without data; random phase there"""
        nlines, width = intf.shape
        out = cp.empty((1,2 if adi is None else 3,nlines,width),dtype=cp.float32)
        mask = cp.empty((nlines,width),dtype=cp.bool_)
        adi = cp.empty((0,0),dtype=cp.float32) if adi is None else adi
        if nlines*width > 0:
            _pre_infer_cuda[(nlines*width+127)//128, 128](intf, adi, out, mask)
        c = out.shape[1]-2
        nan_pos = cp.where(mask)
        random_phase = cp.random.uniform(-cp.pi,cp.pi,len(nan_pos[0]))
        out[0,c,nan_pos[0],nan_pos[1]] = cp.cos(random_phase)
        out[0,c+1,nan_pos[0],nan_pos[1]] = cp.sin(random_phase)
        return out, mask

    def _pre_infer_n2f_cp(intf):
        return _pre_infer_cp(intf)

@ngpjit
def _after_infer_n2f_numba(
    infer_out,
    mask,
):
    nlines, width = mask.shape
    out = np.empty((nlines,width),dtype=np.complex64)
    for i in prange(nlines):
        for j in prange(width):
            if mask[i,j]:
                out[i,j] = np.nan+1j*np.nan
            else:
                out[i,j] = infer_out[0,0,i,j]+1j*infer_out[0,1,i,j]
    return out

if is_cuda_available():
    @mcuda_jit()
    def _after_infer_n2f_cuda(infer_out, mask, out):
        t = cuda.grid(1)
        nlines, width = mask.shape
        if t >= nlines*width:
            return
        i = t//width; j = t%width
        if mask[i,j]:
            out[i,j] = complex(math.nan, math.nan)
        else:
            out[i,j] = complex(infer_out[0,0,i,j], infer_out[0,1,i,j])

    def _after_infer_n2f_cp(infer_out,mask):
        nlines, width = mask.shape
        out = cp.empty((nlines,width),dtype=cp.complex64)
        if nlines*width > 0:
            _after_infer_n2f_cuda[(nlines*width+127)//128, 128](infer_out, mask, out)
        return out

def _infer_n2f_cpu(
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2f_numba(intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_numba(infer_out,mask)

def _infer_n2f_gpu(
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2f_cp(intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_cp(infer_out,mask)

def _nan_where_zero(x):
    """copy of `x` (numpy or cupy) with NaN where |x| < 1e-30, the input is not changed"""
    # GAMMA writes 0 where there are no data; should be done when loading the GAMMA results and removed here
    xp = get_array_module(x)
    return xp.where(xp.abs(x)<1e-30, x.dtype.type(np.nan+1j*np.nan), x)

def n2f(
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Noise2Fringe filtering of an interferogram; on the GPU (cupy input) the network runs in half precision.

    Parameters
    ----------
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    xp = get_array_module(intf)
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = xp.empty_like(intf)

    if xp is np:
        model = _get_model('n2f', model, 'cpu')
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2f_cpu(_nan_where_zero(intf[in_slice]),model)[map_slice]
    else:
        model = _get_model('n2f', model, _cuda_device(), half=True)
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2f_gpu(_nan_where_zero(intf[in_slice]),model)[map_slice]

    return out

@ngpjit
def _pre_infer_n2fs3d_numba(adi,intf):
    nlines, width = intf.shape
    out = np.empty((1,3,nlines,width),dtype=np.float32)
    mask = np.empty((nlines,width),dtype=np.bool_)
    for i in prange(nlines):
        for j in prange(width):
            intf_i_j = intf[i,j]
            adi_i_j = adi[i,j]
            if math.isnan(intf_i_j.real) or math.isnan(adi_i_j):
                mask[i,j] = True
                out[0,0,i,j] = 1.0
                random_phase = random.uniform(-math.pi,math.pi)
                out[0,1,i,j] = math.cos(random_phase)
                out[0,2,i,j] = math.sin(random_phase)
            else:
                mask[i,j] = False
                out[0,0,i,j] = adi_i_j
                amp = abs(intf_i_j)
                out[0,1,i,j] = intf_i_j.real/amp
                out[0,2,i,j] = intf_i_j.imag/amp
    return out, mask

if is_cuda_available():
    def _pre_infer_n2fs3d_cp(adi,intf):
        return _pre_infer_cp(intf, adi)

def _infer_n2fs3d_cpu(
    adi,
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2fs3d_numba(adi,intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_numba(infer_out,mask)

def _infer_n2fs3d_gpu(
    adi,
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2fs3d_cp(adi,intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_cp(infer_out,mask)

def n2fs3d(
    adi:np.ndarray,
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Parameters
    ----------
    adi : np.ndarray
        amplitude dispersion index, 2d np.float32 or cp.float32
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    xp = get_array_module(intf)
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = xp.empty_like(intf)

    if xp is np:
        model = _get_model('n2fs3d', model, 'cpu')
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2fs3d_cpu(adi[in_slice],_nan_where_zero(intf[in_slice]),model)[map_slice]
    else:
        model = _get_model('n2fs3d', model, _cuda_device())
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2fs3d_gpu(adi[in_slice],_nan_where_zero(intf[in_slice]),model)[map_slice]

    return out
