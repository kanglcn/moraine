"""Set operations of point clouds: union, intersection, difference"""

__all__ = ['pc_union', 'pc_intersect', 'pc_diff']

import numpy as np
from ..utils_.device import get_array_module
from ..utils_.jit import ngjit
from .index import _ravel_gix, _unravel_gix, _check_idx_sorted

def _ras_dims(
    gix1:np.ndarray,
    gix2:np.ndarray,
)->tuple:
    """Get the shape of the original raster image from two index, the shape could be smaller than the truth but it doesn't matter.

    Parameters
    ----------
    gix1 : np.ndarray
        int array, grid index of the first point cloud
    gix2 : np.ndarray
        int array, grid index of the second point cloud

    Returns
    -------
    tuple
        the shape of the original raster image
    """
    xp = get_array_module(gix1)
    dims_az = max(int(gix1[-1,0]),int(gix2[-1,0]))+1
    dims_r = max(int(xp.max(gix1[:,1])),int(xp.max(gix2[:,1])))+1
    return (dims_az,dims_r)

def _flat_idx(idx1, idx2, shape):
    """The indices of the two point clouds of `pc_union`, `pc_intersect` or `pc_diff` as 1D indices, checked to be
    sorted and unique, and the raster shape of grid indices (None for hilbert indices)."""
    if idx1.ndim != idx2.ndim or idx1.ndim not in (1, 2):
        raise ValueError(f'idx1 and idx2 must both be gix (n_points, 2) or both hix (n_points,), got {idx1.ndim} and {idx2.ndim} dimensions')
    dims = None
    if idx1.ndim == 2:
        dims = shape if shape is not None else _ras_dims(idx1,idx2)
        idx1 = _ravel_gix(idx1,dims) # automatically the returned 1d index is in int64
        idx2 = _ravel_gix(idx2,dims)
    _check_idx_sorted(idx1); _check_idx_sorted(idx2)
    return idx1, idx2, dims

@ngjit
def _pc_union_numba(idx1, idx2):
    # hand write merge sort
    size1 = idx1.shape[0]
    size2 = idx2.shape[0]
    idx = np.empty(size1+size2,dtype=idx1.dtype)
    inv_iidx1 = np.empty_like(idx1) # 0:size1 for inv_iidx1, size1: for inv_iidx2
    inv_iidx2 = np.empty_like(idx2)
    ninv_iidx2 = np.empty_like(idx2)
    i1 = 0; i2 = 0; i = 0; inv1_i = 0; inv2_i = 0; ninv2_i = 0;
    while((i1<size1) and (i2<size2)):
        if idx1[i1] < idx2[i2]:
            idx[i] = idx1[i1]
            inv_iidx1[inv1_i] = i
            i1 +=1; inv1_i += 1
        elif idx1[i1] > idx2[i2]:
            idx[i] = idx2[i2]
            inv_iidx2[inv2_i] = i
            ninv_iidx2[ninv2_i] = i2
            i2 +=1; inv2_i += 1; ninv2_i += 1
        else:
            idx[i] = idx1[i1]
            inv_iidx1[inv1_i] = i
            i1 += 1; inv1_i += 1; i2 += 1
        i +=1
    while(i1<size1):
        idx[i] = idx1[i1]
        inv_iidx1[inv1_i] = i
        i1 += 1; inv1_i += 1; i += 1
    while(i2<size2):
            idx[i] = idx2[i2]
            inv_iidx2[inv2_i] = i
            ninv_iidx2[ninv2_i] = i2
            i2 +=1; inv2_i += 1; ninv2_i += 1; i += 1

    return idx[:i], inv_iidx1[:inv1_i], inv_iidx2[:inv2_i], ninv_iidx2[:ninv2_i]

def pc_union(idx1:np.ndarray,
             idx2:np.ndarray,
             shape:tuple=None,
            )->tuple:
    """Get the union of two point cloud dataset. For points at their intersection, prefer idx1 rather than idx2

    Parameters
    ----------
    idx1 : np.ndarray
        int array, grid index or hillbert index of the first point cloud, sorted, without duplicates
    idx2 : np.ndarray
        int array, grid index or hillbert index of the second point cloud, sorted, without duplicates
    shape : tuple, optional
        image shape, faster if provided for grid index input

    Returns
    -------
    tuple
        - the union index `idx`
        - index of the point in output union index that originally in the first point cloud `inv_iidx`
        - index of the point in output union index that only exist in the second point cloud `inv_iidx2`
        - index of the point in the second input index that are not in the first input point cloud
    """
    idx1_1d, idx2_1d, dims = _flat_idx(idx1, idx2, shape)
    xp = get_array_module(idx1_1d)
    if xp is np:
        # on cpu use handwrite merge sort
        idx, inv_iidx1, inv_iidx2, ninv_iidx2 = _pc_union_numba(idx1_1d,idx2_1d)
    else:
        # both inputs are sorted and unique: every point of idx1 is in the union, the points of idx2 not in idx1 are added
        idx = xp.union1d(idx1_1d, idx2_1d)
        ninv_iidx2 = xp.flatnonzero(xp.isin(idx2_1d, idx1_1d, assume_unique=True, invert=True))
        inv_iidx1 = xp.searchsorted(idx, idx1_1d)
        inv_iidx2 = xp.searchsorted(idx, idx2_1d[ninv_iidx2])
    return (idx if dims is None else _unravel_gix(idx, dims)), inv_iidx1, inv_iidx2, ninv_iidx2

def pc_intersect(idx1:np.ndarray,
                 idx2:np.ndarray,
                 shape:tuple=None,
                )->tuple:
    """Get the intersection of two point cloud dataset.

    Parameters
    ----------
    idx1 : np.ndarray
        int array, grid index or hillbert index of the first point cloud, sorted, without duplicates
    idx2 : np.ndarray
        int array, grid index or hillbert index of the second point cloud, sorted, without duplicates
    shape : tuple, optional
        image shape, faster if provided for grid index input

    Returns
    -------
    tuple
        - the intersect index `idx`
        - index of the point in first point cloud index that also exist in the second point cloud
        - index of the point in second point cloud index that also exist in the first point cloud
    """
    # Here I do not write the core function by myself since cupy have a different implementation of intersect1d
    idx1_1d, idx2_1d, dims = _flat_idx(idx1, idx2, shape)
    xp = get_array_module(idx1_1d)
    if idx2_1d.shape[0] == 0:
        idx1_1d = idx1_1d[:0] # cupy (13.6) intersect1d reads out of bounds when the second array is empty
    idx, iidx1, iidx2 = xp.intersect1d(idx1_1d,idx2_1d,assume_unique=True,return_indices=True)
    return (idx if dims is None else _unravel_gix(idx, dims)), iidx1, iidx2

def pc_diff(idx1:np.ndarray,
            idx2:np.ndarray,
            shape:tuple=None,
           )->tuple:
    """Get the point cloud in `idx1` that are not in `idx2`.

    Parameters
    ----------
    idx1 : np.ndarray
        int array, grid index or hillbert index of the first point cloud, sorted, without duplicates
    idx2 : np.ndarray
        int array, grid index or hillbert index of the second point cloud, sorted, without duplicates
    shape : tuple, optional
        image shape, faster if provided for grid index input

    Returns
    -------
    tuple
        - the diff index `idx`
        - index of the point in first point cloud index that do not exist in the second point cloud
    """
    idx1_1d, idx2_1d, dims = _flat_idx(idx1, idx2, shape)
    xp = get_array_module(idx1_1d)
    mask = xp.isin(idx1_1d, idx2_1d, assume_unique=True, invert=True)
    idx = idx1_1d[mask]
    return (idx if dims is None else _unravel_gix(idx, dims)), xp.where(mask)[0]
