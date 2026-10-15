"""Deformation products of point clouds: spatial reference of time series (decision 0040)"""

__all__ = ['reference']

import numpy as np
from .utils_ import get_array_module


def reference(
    data,
    ref,
    method:str='median',
):
    """Values relative to a reference: the reference value of every column is subtracted from the column.

    Parameters
    ----------
    data : np.ndarray
        values of the points, shape (n_points,) or (n_points, n), floating point, numpy or cupy; e.g. the unwrapped
        phase of every image in radians (`ts` of `unwrap_correct_closure_pc`) or displacements
    ref : np.ndarray | int
        the reference: a bool mask of the reference points, shape (n_points,), True for at least one point; or the
        index of one reference point, 0 .. n_points - 1
    method : str, default: 'median'
        reference value of a column from the points of a mask: 'median' or 'mean' of their values in the column,
        nan left out; not used with one reference point (its value is the reference value)

    Returns
    -------
    out : np.ndarray
        `data` minus the reference value of its column, same shape and dtype as `data`; the median (mean, value)
        of the reference points is 0 in every column
    ref_value : np.ndarray
        reference value of every column, shape (n,), or () for data of shape (n_points,), dtype of `data`

    Raises
    ------
    ValueError
        when the reference points have no value (all nan) in a column
    """
    xp = get_array_module(data)
    if method not in ('median', 'mean'):
        raise ValueError(f"method must be 'median' or 'mean', not {method!r}")
    if data.ndim not in (1, 2):
        raise ValueError(f'data must have shape (n_points,) or (n_points, n), got shape {tuple(data.shape)}')
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f'data must be floating point, got {data.dtype}')
    n_points = data.shape[0]
    if isinstance(ref, (bool, np.bool_)):
        raise ValueError('ref must be a bool mask of the points or the index of one point, not a single bool')
    if isinstance(ref, (int, np.integer)):
        if not 0 <= ref < n_points:
            raise ValueError(f'the reference point {ref} is not one of the {n_points} points')
        values = data[int(ref):int(ref) + 1]
    else:
        ref = xp.asarray(ref)
        if ref.dtype != bool or ref.shape != (n_points,):
            raise ValueError(f'the reference mask must be bool of shape ({n_points},), got {ref.dtype} {tuple(ref.shape)}')
        if not bool(ref.any()):
            raise ValueError('the reference mask has no point')
        values = data[ref]
    empty = xp.isnan(values).all(axis=0)
    if bool(empty.any()):
        columns = np.flatnonzero(np.atleast_1d(xp.asnumpy(empty) if xp is not np else empty)).tolist()
        raise ValueError(f'the reference points have no value (all nan) in column(s) {columns[:10]}'
                         f'{" ..." if len(columns) > 10 else ""}')
    if values.shape[0] == 1:
        ref_value = values[0]
    elif method == 'median':
        ref_value = xp.nanmedian(values, axis=0)
    else:
        ref_value = xp.nanmean(values, axis=0)
    ref_value = xp.asarray(ref_value, dtype=data.dtype)
    return data - ref_value, ref_value
