"""Deformation products of point clouds (CLI): spatial reference of time series (decision 0040)"""

__all__ = ['pc_reference']

import logging

import numpy as np
import zarr

import moraine as mr
from .logging import mc_logger
from .executor import Executor, Chunk


def _reference_column(pc_part, out_part, ref, method):
    """subtract the reference value of one column of a point cloud, write it and return the reference value"""
    out, value = mr.reference(pc_part.read(), ref, method)
    out_part.write(out)
    return np.asarray(value).reshape(-1)


@mc_logger
def pc_reference(
    pc:str,
    out:str,
    ref_mask:str=None,
    ref_point:int=None,
    method:str='median',
    ref_value:str=None,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Point cloud data relative to a reference: the reference value of every image is subtracted from it.

    The reference value of an image is the median of the values of the reference points of `ref_mask` (by default),
    their mean, or the value of the point `ref_point`; give either `ref_mask` or `ref_point`. Afterwards the reference
    points have the median (mean, value) 0 in every image, e.g. displacements relative to a stable area. The
    reference points should be stable, coherent and in one region of the closure correction. The mask of the
    reference points stays in memory (1 byte per point).

    Parameters
    ----------
    pc : str
        input: point cloud data, shape (n_points,) or (n_points, n), floating point, chunks (points_block, 1); e.g.
        the unwrapped phase of every image (`ts` of `unwrap-correct-closure-pc`, radians) or displacements
    out : str
        output: `pc` minus the reference value of every image, same shape, dtype and unit as `pc`, chunks
        (out_chunks, 1)
    ref_mask : str, optional
        input: the reference points, bool, shape (n_points,), True for at least one point; e.g. from `polygon-mask`
    ref_point : int, optional
        index of the one reference point, 0 .. n_points - 1, e.g. the point of a double click on a map of
        `moraine.cli.view`
    method : str, default: 'median'
        reference value from the points of `ref_mask`: 'median' or 'mean' of their values in the image, nan left
        out; not used with `ref_point`
    ref_value : str, optional
        output: the reference value of every image, shape (n,) (shape (1,) for `pc` of shape (n_points,)), dtype and
        unit of `pc`
    out_chunks : int, optional
        point chunk size of `out`, that of `pc` by default
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 4
        tasks (images) a worker processes at the same time; a task holds two images of all points
    """
    logger = logging.getLogger(__name__)
    if (ref_mask is None) == (ref_point is None):
        raise ValueError('give the reference with exactly one of `ref_mask` and `ref_point`')
    if method not in ('median', 'mean'):
        raise ValueError(f"method must be 'median' or 'mean', not {method!r}")
    pc_zarr = zarr.open(pc, mode='r'); logger.zarr_info(pc, pc_zarr)
    if pc_zarr.ndim not in (1, 2):
        raise ValueError(f'pc must have shape (n_points,) or (n_points, n), got shape {pc_zarr.shape}')
    if not np.issubdtype(pc_zarr.dtype, np.floating):
        raise ValueError(f'pc must be floating point, got {pc_zarr.dtype}')
    n_points = pc_zarr.shape[0]
    if ref_mask is not None:
        mask_zarr = zarr.open(ref_mask, mode='r'); logger.zarr_info(ref_mask, mask_zarr)
        if mask_zarr.dtype != bool or mask_zarr.shape != (n_points,):
            raise ValueError(f'ref_mask must be bool of shape ({n_points},), got {mask_zarr.dtype} {mask_zarr.shape}')
        ref = mask_zarr[:]
        n_ref = int(ref.sum())
        if n_ref == 0:
            raise ValueError(f'{ref_mask} has no reference point')
        logger.info(f'reference: {method} of {n_ref} points')
    else:
        ref = int(ref_point)
        if not 0 <= ref < n_points:
            raise ValueError(f'ref_point {ref} is not one of the {n_points} points')
        logger.info(f'reference: point {ref}')

    if out_chunks is None:
        out_chunks = pc_zarr.chunks[0]
    columns = [slice(k, k + 1) for k in range(pc_zarr.shape[1])] if pc_zarr.ndim == 2 else [None]
    out_zarr = zarr.open(out, mode='w', shape=pc_zarr.shape, dtype=pc_zarr.dtype,
                         chunks=(out_chunks, 1) if pc_zarr.ndim == 2 else (out_chunks,))
    logger.zarr_info(out, out_zarr)

    def part(path, col):
        return Chunk(path, (slice(0, n_points),) if col is None else (slice(0, n_points), col))
    # one task per image: the whole column in, the whole column out (whole chunks of `out`)
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ref_shared = ex.put(ref)
        values = ex.map(_reference_column, [(part(pc, col), part(out, col), ref_shared, method) for col in columns],
                        desc='images')
    values = np.concatenate(values)
    logger.info(f'reference values: min {np.min(values):.4g}, median {np.median(values):.4g}, max {np.max(values):.4g}')
    if ref_value is not None:
        z = zarr.open(ref_value, mode='w', shape=values.shape, dtype=values.dtype, chunks=values.shape)
        z[:] = values
        logger.zarr_info(ref_value, z)
    logger.info('done.')
