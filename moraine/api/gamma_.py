"""internal utilities for working with gamma software"""


__all__ = ['read_gamma_image', 'read_gamma_pdata', 'write_gamma_image', 'read_gamma_plist', 'write_gamma_plist']

import numpy as np
import os

def read_gamma_image(imag:str,
                     width:int,
                     dtype:str='float',
                     y0:int=0,
                     ny:int=None,
                    ):
    # zero value in the image is transformed to 
    # gpu:bool=False, # return a cupy array if true 
    # !! nvidia gpu do not support big endian data, so no way to directly read it
    """read gamma image into numpy array.

    Parameters
    ----------
    imag : str
        gamma raster data
    width : int
        data width
    dtype : str, default: 'float'
        data format, only 'float', 'fcomplex', 'double' and 'int' are supported.
    y0 : int, default: 0
        line number to start reading
    ny : int, optional
        number of lines to read, default: to the last line
    """
    if dtype == 'float':
        dt = '>f4'
    elif dtype == 'fcomplex':
        dt = '>c8'
    elif dtype == 'int':
        dt = '>i4'
    elif dtype =='double':
        dt = '>f8'
    else: raise ValueError('Unsupported data type')

    with open(imag,"rb") as datf:
        datf.seek(0,os.SEEK_END)
        size = datf.tell()
    n_byte = int(dt[-1]) # number of byte per data point
    nlines = int(size / n_byte / width)
    assert y0 <= nlines, 'y0 is larger than height.'
    if ny is None: ny = nlines-y0
    assert y0+ny <= nlines, 'y0+ny is larger than height.'
    offset = y0*width*n_byte
    count = ny*width*n_byte
    # if gpu:
    #     import cupy as cp
    #     import kvikio
    #     import kvikio.zarr
    #     with kvikio.CuFile(imag,'rb') as cufile:
    #         gpu_dt = dt.replace('>','=')
    #         data = cp.empty((ny,width),dtype=dt)
    #         cufile.read(data,size=count,file_offset=offset)
    #     return data
    # else:
    with open(imag,'rb') as datf:
        data = np.fromfile(datf,dtype=dt,offset=offset,count=ny*width)
    if dt != '>i4':
        mask = abs(data)<1e-30 # mask gamma nan value
        if dt == '>c8':
            data[mask] = np.nan+1j*np.nan
        else:
            data[mask] = np.nan
    data = data.astype(data.dtype.newbyteorder('native'))
    return data.reshape(-1,width)

def read_gamma_pdata(
    imag:str,
    dtype:str='float',
):
    # zero value in the data is transformed to nan
    """read gamma image into numpy array.

    Parameters
    ----------
    imag : str
        gamma point cloud data
    dtype : str, default: 'float'
        data format, only 'float', 'fcomplex', 'double' and 'int' are supported.
    """
    if dtype == 'float':
        dt = '>f4'
    elif dtype == 'fcomplex':
        dt = '>c8'
    elif dtype == 'int':
        dt = '>i4'
    elif dtype =='double':
        dt = '>f8'
    else: raise ValueError('Unsupported data type')

    data = np.fromfile(imag,dtype=dt)
    if dt != '>i4':
        mask = data==0
        if dt == '>c8':
            data[mask] = np.nan+1j*np.nan
        else:
            data[mask] = np.nan
    data = data.astype(data.dtype.newbyteorder('native'))
    return data

def write_gamma_image(imag,path):
    # support both raster and point cloud
    imag = imag.astype(imag.dtype.newbyteorder('big'))
    imag.tofile(path)

def read_gamma_plist(plist:str,dtype='int'):
    return read_gamma_image(plist,width=2,dtype=dtype)

def write_gamma_plist(imag,path):
    return write_gamma_image(imag,path)
