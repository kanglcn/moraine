"""Raster <-> point cloud, points in the order of raster chunks"""

__all__ = ['pc2ras', 'ras2pc']

import numpy as np
from numba import prange
import math
from ..utils_.device import get_array_module
from ..utils_.jit import ngpjit
from .index import pc_gix

def pc2ras(idx:np.ndarray,
           pc_data:np.ndarray,
           shape:tuple,
          ):
    """convert point cloud data to original raster, filled with nan

    Parameters
    ----------
    idx : np.ndarray
        gix or hix array
    pc_data : np.ndarray
        data, 1D or more
    shape : tuple
        image shape
    """
    xp = get_array_module(pc_data)
    raster = xp.empty((*shape,*pc_data.shape[1:]),dtype=pc_data.dtype)
    raster[:] = xp.nan
    if idx.ndim == 2:
        gix = idx
    else:
        gix = pc_gix(idx, shape)
    raster[gix[:,0],gix[:,1]] = pc_data
    return raster

def ras2pc(ras,gix):
    return ras[gix[:,0],gix[:,1]]

@ngpjit
def _pc_split_by_chunk(
    gix,
    chunks,
    shape,
):
    '''utility for rearrange pc into raster chunks order'''
    nlines, width = shape
    az_chunk, r_chunk = chunks
    az_bounds = np.arange(0,nlines+az_chunk,az_chunk)
    if az_bounds[-1] > nlines: az_bounds[-1] = nlines
    r_bounds = np.arange(0,width+r_chunk,r_chunk)
    if r_bounds[-1] > width: r_bounds[-1] = width

    n_az_chunk = math.ceil(nlines/az_chunk)
    n_r_chunk = math.ceil(width/r_chunk)
    gix_az_bounds = np.searchsorted(gix[:,0],az_bounds)

    chunk_idx = np.empty(gix.shape[0],dtype=np.int64) # index that sort gix ordered pc into ras chunk order
    chunk_bounds = np.empty(n_az_chunk*n_r_chunk+1,dtype=np.int64) # boundarys that split ras chunk ordered pc into individual chunk
    invert_idx = np.empty(gix.shape[0],dtype=np.int64) # index that sort ras-chunk ordered pc to gix order back
    for i in prange(n_az_chunk):
        pc_start,pc_end = gix_az_bounds[i:i+2]
        gix_r_ = gix[pc_start:pc_end,1]
        r_chunk_idx = np.searchsorted(r_bounds,gix_r_,side='right')-1
        _chunk_idx = np.argsort(r_chunk_idx,kind='mergesort')

        chunk_bounds[i*n_r_chunk:(i+1)*n_r_chunk] = np.searchsorted(r_chunk_idx[_chunk_idx],np.arange(n_r_chunk))+pc_start

        chunk_idx[pc_start:pc_end] = _chunk_idx+pc_start

        _invert_idx = np.empty_like(r_chunk_idx,dtype=np.int64)
        _invert_idx[_chunk_idx] = np.arange(pc_start,pc_end)
        invert_idx[pc_start:pc_end] = _invert_idx
    chunk_bounds[-1] = gix.shape[0]
    return chunk_idx, chunk_bounds, invert_idx

@ngpjit
def _gix_ras_chunk(
    sorted_gix,
    chunk_bounds,
    chunks,
    shape,
    overlap=(0,0),
):
    '''split gix into raster chunks'''
    nlines, width = shape
    az_chunk, r_chunk = chunks
    n_r_chunk = math.ceil(width/r_chunk)

    ras_chunk_order_gix = np.empty_like(sorted_gix)
    for i in prange(chunk_bounds.shape[0]-1):
        start = chunk_bounds[i]
        stop = chunk_bounds[i+1]
        az_chunk_idx = i//n_r_chunk
        r_chunk_idx = i%n_r_chunk
        az_offset = -az_chunk_idx*az_chunk
        if az_chunk_idx != 0: az_offset += overlap[0]
        r_offset = -r_chunk_idx*r_chunk
        if r_chunk_idx != 0: r_offset += overlap[1]
        ras_chunk_order_gix[start:stop,0] = sorted_gix[start:stop,0]+az_offset
        ras_chunk_order_gix[start:stop,1] = sorted_gix[start:stop,1]+r_offset
    return ras_chunk_order_gix
