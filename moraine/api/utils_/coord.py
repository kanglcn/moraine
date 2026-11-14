"""Coordinates of raster images"""

__all__ = ['Coord']

import numpy as np
import math
from .jit import ngpjit
from numba import prange

# test shows this automatical parallel can accelerate 3x, 
# probablely because the data calculting is accelerated but the data writing is not
@ngpjit
def _coords2gixs(coords, x0, dx, nx, out):
    '''input is 1d'''
    for i in prange(coords.shape[0]):
        _out = round((coords[i] - x0)/dx)
        if _out < 0: _out = 0 # numba do not support raise error within prange, we can not raise error now
        if _out > nx-1: _out=nx-1
        out[i] = _out

class Coord(object):
    '''utils for digitize raster and point cloud data, in a pyramid.
    The coord is defined as the continuous coordinates, e.g., longitude & latitude.
    The index is defined as the digitized index, (0,0,nx-1,ny-1).
    '''
    def __init__(self,x0,dx,nx,y0,dy,ny):
        self.x0 = x0
        self.dx = dx
        if nx >= 2**31-1:
            raise ValueError(f'nx = {nx} is too large for 32 bit indices')
        self.nx = nx
        self.xm = x0+(nx-1)*dx
        self.y0 = y0
        self.dy = dy
        if ny >= 2**31-1:
            raise ValueError(f'ny = {ny} is too large for 32 bit indices')
        self.ny = ny
        self.ym = y0+(ny-1)*dy
        self.maxlevel = math.floor(math.log2(min(nx,ny)))

    def coords2gixs(
        self,
        coords,
    ):
        """inputs are 2d arrays. Only zoom level = 0 supported.

        Parameters
        ----------
        coords
            shape [N, 2], first column: y coordinates, secondary column, x coordinates
        """
        out = np.empty_like(coords,dtype=np.int32)
        _coords2gixs(coords[:,1], self.x0, self.dx, self.nx, out[:,1])
        _coords2gixs(coords[:,0], self.y0, self.dy, self.ny, out[:,0])
        return out
    def rasterize_iidx(
        self,
        gix
    ):
        # the output is the inverse index (2D, same shape with ras), i.e., rasterized_ras = pc[iidx]
        # but the nan value in rasterized ras is not filled
        # it should be updated as rasterized_ras[iidx==-1] = np.nan
        iidx = np.full((self.ny,self.nx),fill_value=-1,dtype=np.int64)
        iidx[gix[:,0],gix[:,1]] = np.arange(gix.shape[0])
        return iidx
    def rasterize(self,pc,
                  gix,
                  fill_value=np.nan,
                 )->np.ndarray:
        """Parameters
        ----------
        pc
            point cloud data, can be a stack, shape (n_pc,...)
        gix
            2D grid index of the pc, value within (0,0,im,jm), shape (2,n_pc)
        fill_value : default: np.nan
            fill value

        Returns
        -------
        np.ndarray
            the raster data, the index of the original point cloud list
        """
        # the fill_value can not be np.nan if pc is integer
        ras = np.full((self.ny,self.nx,*pc.shape[1:]),fill_value=fill_value,dtype=pc.dtype)
        ras[gix[:,0],gix[:,1]] = pc
        return ras
