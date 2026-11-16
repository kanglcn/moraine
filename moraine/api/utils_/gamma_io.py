"""Read and write GAMMA binary files: rasters, point data and point lists, big endian"""


__all__ = ['read_gamma_image', 'read_gamma_pdata', 'write_gamma_image', 'read_gamma_plist', 'write_gamma_plist']

import os

import numpy as np

_DTYPES = {'float': '>f4', 'fcomplex': '>c8', 'double': '>f8', 'int': '>i4'}   # GAMMA data formats

def read_gamma_image(imag:str,
                     width:int,
                     dtype:str='float',
                     y0:int=0,
                     ny:int=None,
                    ):
    """Read a GAMMA raster into a numpy array.

    Parameters
    ----------
    imag : str
        GAMMA raster file
    width : int
        number of values per line
    dtype : str, default: 'float'
        GAMMA data format: 'float', 'fcomplex', 'double' or 'int'
    y0 : int, default: 0
        first line to read
    ny : int, optional
        number of lines to read, default: to the last line

    Returns
    -------
    np.ndarray
        lines `y0` to `y0 + ny - 1`, shape (ny, width), float32, complex64, float64 or int32 (`dtype`) in the native
        byte order; values with an absolute value below 1e-30 (GAMMA's no data) are nan, except for 'int'
    """
    if dtype not in _DTYPES:
        raise ValueError('Unsupported data type')
    dt = np.dtype(_DTYPES[dtype])
    nlines = os.path.getsize(imag)//(dt.itemsize*width)
    if y0 > nlines:
        raise ValueError(f'y0 = {y0} is beyond the {nlines} lines of the file')
    if ny is None: ny = nlines-y0
    if y0+ny > nlines:
        raise ValueError(f'y0 + ny = {y0+ny} is beyond the {nlines} lines of the file')
    data = np.fromfile(imag, dtype=dt, offset=y0*width*dt.itemsize, count=ny*width)
    if dtype != 'int':
        data[abs(data)<1e-30] = np.nan+1j*np.nan if dtype == 'fcomplex' else np.nan
    return data.astype(dt.newbyteorder('native')).reshape(-1,width)

def read_gamma_pdata(
    imag:str,
    dtype:str='float',
):
    """Read GAMMA point data (one value per point) into a numpy array.

    Parameters
    ----------
    imag : str
        GAMMA point data file
    dtype : str, default: 'float'
        GAMMA data format: 'float', 'fcomplex', 'double' or 'int'

    Returns
    -------
    np.ndarray
        shape (n_points,), float32, complex64, float64 or int32 (`dtype`) in the native byte order; values with an
        absolute value below 1e-30 (GAMMA's no data) are nan, except for 'int'
    """
    return read_gamma_image(imag, 1, dtype)[:, 0]

def write_gamma_image(imag,path):
    """Write a numpy array to a GAMMA binary file (big endian): a raster, point data or a point list.

    Parameters
    ----------
    imag : np.ndarray
        data: a raster of shape (nlines, width), point data of shape (n_points,) or a point list of shape (n_points,
        2); float32, complex64, float64 or int32 for the GAMMA formats 'float', 'fcomplex', 'double' and 'int'
    path : str
        output file
    """
    imag.astype(imag.dtype.newbyteorder('big')).tofile(path)

def read_gamma_plist(plist:str,dtype='int'):
    """Read a GAMMA point list (two values per point, e.g. range and azimuth pixel) into a numpy array.

    Parameters
    ----------
    plist : str
        GAMMA point list file
    dtype : str, default: 'int'
        GAMMA data format: 'float', 'fcomplex', 'double' or 'int'

    Returns
    -------
    np.ndarray
        shape (n_points, 2), as `read_gamma_image` of width 2
    """
    return read_gamma_image(plist,width=2,dtype=dtype)

write_gamma_plist = write_gamma_image
