"""Grid and hilbert indices of point clouds, sorting"""

__all__ = ['pc_hix', 'pc_gix', 'pc_sort']

import numpy as np
from numba import prange
import math
from ..utils_.device import get_array_module
from ..utils_.jit import ngjit, ngpjit
from ..utils_.errors import ArgumentError

def _ravel_gix(gix,dims):
    xp = get_array_module(gix)
    return xp.ravel_multi_index((gix[:,0],gix[:,1]),dims=dims)

def _unravel_gix(gix_1d,dims):
    xp = get_array_module(gix_1d)
    return xp.stack(xp.unravel_index(gix_1d,dims),axis=-1).astype(xp.int32)

def _check_idx_sorted(idx):
    xp = get_array_module(idx)
    if not (xp.diff(idx)>0).all():
        raise ValueError('idx must be sorted and unique')

# Some functions adapted from spatialpandas at https://github.com/holoviz/spatialpandas under BSD-2-Clause license,
# Which is Initially based on https://github.com/galtay/hilbert_curve, but specialized
# for 2 dimensions with numba acceleration

# Further enhanced with numba prange.
@ngjit
def _int_2_binary(v, width):
    """Returns
    -------
    Return a binary byte array representation of `v` zero padded to `width` bits.
    """
    res = np.zeros(width, dtype=np.uint8)
    for i in range(width):
        res[width - i - 1] = v % 2
        v = v >> 1
    return res

@ngjit
def _binary_2_int(bin_vec):
    # Convert a binary byte array to an integer
    res = 0
    next_val = 1
    width = len(bin_vec)
    for i in range(width):
        res += next_val*bin_vec[width - i - 1]
        next_val <<= 1
    return res

@ngjit
def _hilbert_integer_to_transpose(p:int,
                                  h:int,
                                 )->np.array:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    h : int
        integer distance along hilbert curve

    Returns
    -------
    np.array
        x (two element np.array): transpose of h (n components with values between 0 and 2**p-1)
    """
    #Store a hilbert integer (`h`) as its transpose (`x`).
    h_bits = _int_2_binary(h, 2*p)
    x0, x1 = _binary_2_int(h_bits[0::2]), _binary_2_int(h_bits[1::2])
    x = np.array([x0,x1])
    return x

@ngjit
def _transpose_to_hilbert_integer(p:int,
                                  coord:list,
                                 )->int:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    coord : list
        transpose of h (2 components with values between 0 and 2**p-1)

    Returns
    -------
    int
        h (int): integer distance along hilbert curve
    """
    # Restore a hilbert integer (`h`) from its transpose (`x`).
    bins = [_int_2_binary(v, p) for v in coord]
    concat = np.zeros(2*p, dtype=np.uint8)
    for i in range(p):
        for j in range(2):
            concat[2*i + j] = bins[j][i]

    h = _binary_2_int(concat)
    return h

@ngjit
def _coordinate_from_distance(p:int,
                              h:int,
                             )->list:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    h : int
        integer distance along hilbert curve

    Returns
    -------
    list
        coord (list): Coordinate as length-n list
    """
    # Return the coordinate for a hilbert distance.
    coord = _hilbert_integer_to_transpose(p, h)
    Z = 2 << (p-1)

    # Gray decode by H ^ (H/2)
    t = coord[1] >> 1
    for i in range(1, 0, -1):
        coord[i] ^= coord[i-1]
    coord[0] ^= t

    # Undo excess work
    Q = 2
    while Q != Z:
        P = Q - 1
        for i in range(1, -1, -1):
            if coord[i] & Q:
                # invert
                coord[0] ^= P
            else:
                # exchange
                t = (coord[0] ^ coord[i]) & P
                coord[0] ^= t
                coord[i] ^= t
        Q <<= 1

    return coord

@ngjit
def _distance_from_coordinate(p:int,
                              coord:np.ndarray,
                             )->int:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    coord : np.ndarray
        coordinate as 1d array

    Returns
    -------
    int
        distance
    """
    # Return the hilbert distance for a given coordinate.
    M = 1 << (p - 1)
    # Inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(2):
            if coord[i] & Q:
                coord[0] ^= P
            else:
                t = (coord[0] ^ coord[i]) & P
                coord[0] ^= t
                coord[i] ^= t
        Q >>= 1
    # Gray encode
    for i in range(1, 2):
        coord[i] ^= coord[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if coord[1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(2):
        coord[i] ^= t
    h = _transpose_to_hilbert_integer(p, coord)
    return h

@ngpjit
def _coordinates_from_distances(p:int,
                                h:np.ndarray,
                               )->np.ndarray:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    h : np.ndarray
        1d array of integer distances along hilbert curve

    Returns
    -------
    np.ndarray
        2d array of coordinate, each row a coordinate corresponding to associated distance value in input.
    """
    # Return the coordinates for an array of hilbert distances.
    result = np.zeros((len(h), 2), dtype=np.int32)
    for i in prange(len(h)):
        result[i] = _coordinate_from_distance(p, h[i])
    return result

@ngpjit
def _distances_from_coordinates(p:int,
                                coords:np.ndarray,
                               )->np.ndarray:
    """Parameters
    ----------
    p : int
        iterations to use in the hilbert curve
    coords : np.ndarray
        2d array of coordinates, one coordinate per row

    Returns
    -------
    np.ndarray
        1d array of distances
    """
    # Return the hilbert distances for a given set of coordinates.
    coords = np.atleast_2d(coords).copy()
    result = np.zeros(coords.shape[0], dtype=np.int64)
    for i in prange(coords.shape[0]):
        coord = coords[i]
        result[i] = _distance_from_coordinate(p, coord)
    return result

def pc_hix(
    gix,
    shape:tuple,
)->np.ndarray:
    """Compute the hillbert index for point cloud data.

    Parameters
    ----------
    gix
        grid index
    shape : tuple
        (nlines, width)
    """
    ny, nx = shape
    p = math.ceil(math.log2(max(nx,ny)))
    hix = _distances_from_coordinates(p, gix)
    return hix

def pc_gix(
    hix,
    shape:tuple,
)->np.ndarray:
    """convert hillbert index to grid index.

    Parameters
    ----------
    hix
        hillbert index
    shape : tuple
        (nlines, width)
    """
    ny, nx = shape
    p = math.ceil(math.log2(max(nx,ny)))
    gix = _coordinates_from_distances(p, hix)
    return gix

def pc_sort(idx:np.ndarray,
            shape:tuple=None,
           )->np.ndarray:
    """Get the indices that sort the input.

    Parameters
    ----------
    idx : np.ndarray
        unsorted `gix` (2D) or `hix`(1D)
    shape : tuple, optional
        image shape (nlines, width) for grid index input: needed if `idx` is empty, faster if provided

    Returns
    -------
    np.ndarray
        indices that sort input
    """
    xp = get_array_module(idx)
    if idx.ndim == 2:
        if shape is not None:
            dims_az, dims_r = shape
        elif idx.shape[0] == 0:
            raise ArgumentError('shape (nlines, width) is needed when a grid index is empty: '
                             'the grid index has no points')
        else:
            dims_az = int(xp.max(idx[:,0]))+1
            dims_r = int(xp.max(idx[:,1]))+1
        idx_1d = xp.ravel_multi_index((idx[:,0],idx[:,1]),dims=(dims_az,dims_r))
    else:
        idx_1d = idx
    key = xp.argsort(idx_1d,kind='stable')
    return key
