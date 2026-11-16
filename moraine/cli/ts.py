"""Deformation products of point clouds (CLI): spatial reference, displacement, DEM error (decision 0048)"""

__all__ = ['pc_reference', 'pc_disp', 'pc_sc_dem_error']

import logging

import numpy as np
import zarr

import moraine as mr
from ..api.ts import _days
from .logging import mc_logger
from .executor import Executor, Chunk
from .utils_ import ArgumentError


def _reference_column(pc_part, out_part, ref, method):
    """subtract the reference value of one column of a point cloud, write it and return the reference value"""
    out, value = mr.pc_reference(pc_part.read(), ref, method)
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
        raise ArgumentError('give the reference with exactly one of `ref_mask` and `ref_point`')
    if method not in ('median', 'mean'):
        raise ArgumentError(f"method must be 'median' or 'mean', not {method!r}")
    pc_zarr = zarr.open(pc, mode='r'); logger.zarr_info(pc, pc_zarr)
    if pc_zarr.ndim not in (1, 2):
        raise ArgumentError(f'pc must have shape (n_points,) or (n_points, n), got shape {pc_zarr.shape}')
    if not np.issubdtype(pc_zarr.dtype, np.floating):
        raise ArgumentError(f'pc must be floating point, got {pc_zarr.dtype}')
    n_points = pc_zarr.shape[0]
    if ref_mask is not None:
        mask_zarr = zarr.open(ref_mask, mode='r'); logger.zarr_info(ref_mask, mask_zarr)
        if mask_zarr.dtype != bool or mask_zarr.shape != (n_points,):
            raise ArgumentError(f'ref_mask must be bool of shape ({n_points},), got {mask_zarr.dtype} {mask_zarr.shape}')
        ref = mask_zarr[:]
        n_ref = int(ref.sum())
        if n_ref == 0:
            raise ValueError(f'{ref_mask} has no reference point')
        logger.info(f'reference: {method} of {n_ref} points')
    else:
        ref = int(ref_point)
        if not 0 <= ref < n_points:
            raise ArgumentError(f'ref_point {ref} is not one of the {n_points} points')
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


@mc_logger
def pc_disp(
    pc:str,
    out:str,
    wavelength:float,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Line of sight displacement in millimeters from unwrapped phase of point clouds.

    Parameters
    ----------
    pc : str
        input: unwrapped phase in radians, shape (n_points,) or (n_points, n), floating point, chunks
        (points_block, 1); the phase of every image relative to a reference image, growing when the range gets
        shorter, e.g. `ts` of `unwrap-correct-closure-pc` or its output of `pc-reference`
    out : str
        output: line of sight displacement in millimeters, same shape as `pc`, float32, chunks (out_chunks, 1);
        positive towards the satellite (a shorter range); nan where `pc` is nan
    wavelength : float
        radar wavelength in meters, between 0 and 1 (`radar_wavelength` of the metadata of `load-gamma-metadata`,
        e.g. 0.0554658 for Sentinel-1)
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
    wavelength = float(wavelength)
    if not 0 < wavelength < 1:
        raise ArgumentError(f'wavelength must be in meters, between 0 and 1, got {wavelength}')
    pc_zarr = zarr.open(pc, mode='r'); logger.zarr_info(pc, pc_zarr)
    if pc_zarr.ndim not in (1, 2):
        raise ArgumentError(f'pc must have shape (n_points,) or (n_points, n), got shape {pc_zarr.shape}')
    if not np.issubdtype(pc_zarr.dtype, np.floating):
        raise ArgumentError(f'pc must be floating point (unwrapped phase in radians), got {pc_zarr.dtype}')
    if out_chunks is None:
        out_chunks = pc_zarr.chunks[0]
    chunks = (out_chunks, 1) if pc_zarr.ndim == 2 else (out_chunks,)
    out_zarr = zarr.open(out, mode='w', shape=pc_zarr.shape, dtype=np.float32, chunks=chunks)
    logger.zarr_info(out, out_zarr)
    logger.info(f'1 rad = {1000 * wavelength / (4 * np.pi):.4f} mm (wavelength {wavelength} m)')
    # one task per image: the whole column in, the whole column out (its chunks read and written in parallel)
    n_points = pc_zarr.shape[0]
    columns = [(slice(0, n_points), slice(k, k + 1)) for k in range(pc_zarr.shape[1])] if pc_zarr.ndim == 2 else [None]
    tasks = [([Chunk(pc, sl)], [Chunk(out, sl)]) for sl in columns]
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(mr.pc_disp, tasks, desc='images', wavelength=wavelength)
    logger.info('done.')


@mc_logger
def pc_sc_dem_error(
    disp:str,
    bperp:str,
    slant_range:str,
    theta:str,
    dh:str,
    out:str,
    dates:list[str],
    ref:int=0,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Spatially correlated DEM error of every point from its displacement time series, and the displacement without it.

    A scatterer dh meters above the height used in the processing (the DEM) adds (c_k - c_ref) dh to the displacement
    of image k, with c_k = 1000 B_perp,k / (R cos(theta)) millimeters per meter and B_perp,k the perpendicular
    baseline of image k. dh is the least squares fit, with equal weights, of the differences of the displacement
    between neighbouring dates by a constant, a constant velocity and the DEM error term: other deformation enters
    the fit only through its changes from one date to the next. After the DEM error of every point relative to its
    neighbourhood was removed from the phase before the unwrapping (`pc-su-dem-error`), dh is the spatially correlated
    part; otherwise it is the whole DEM error.

    Parameters
    ----------
    disp : str
        input: line of sight displacement in millimeters, shape (n_points, nimages), floating point, chunks
        (points_block, 1); the images in date order, relative to the image `ref`, positive towards the satellite
        (`pc-disp`)
    bperp : str
        input: perpendicular baseline of every image at every point in meters, shape (n_points, nimages), floating
        point; relative to the reference image of the co-registration, positive upwards (`load-gamma-bperp`, then
        `ras2pc`)
    slant_range : str
        input: slant range of the points in meters, shape (n_points,), floating point (`load-gamma-range`, then
        `ras2pc`)
    theta : str
        input: elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle),
        shape (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`)
    dh : str
        output: DEM error in meters, shape (n_points,), float32, chunks (out_chunks,): the height of the scatterer
        minus the height used in the processing, positive above it; nan where `disp` or `bperp` of the point has a
        nan
    out : str
        output: `disp` without the DEM error term (c_k - c_ref) dh, in millimeters, shape (n_points, nimages),
        float32, chunks (out_chunks, 1); the column of `ref` is that of `disp`; nan in the rows where `dh` is nan
    dates : list of str
        date of every image, YYYYMMDD, in increasing order (`dates` of the metadata of `load-gamma-metadata`); at
        least 4 images. Only the days between them are used
    ref : int, default: 0
        the image `disp` is relative to (its column is 0), 0 .. nimages - 1: the `ref` of
        `unwrap-correct-closure-pc`
    out_chunks : int, optional
        point chunk size of `dh` and `out`, that of `disp` by default; the points of a task
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 4
        tasks (blocks of `out_chunks` points) a worker processes at the same time; a task holds about 50 bytes per
        point and image
    """
    logger = logging.getLogger(__name__)
    disp_zarr = zarr.open(disp, mode='r'); logger.zarr_info(disp, disp_zarr)
    if disp_zarr.ndim != 2:
        raise ArgumentError(f'disp must have shape (n_points, nimages), got shape {disp_zarr.shape}')
    n_points, nimages = disp_zarr.shape
    for name, path, shape in (('disp', disp, (n_points, nimages)), ('bperp', bperp, (n_points, nimages)),
                              ('slant_range', slant_range, (n_points,)), ('theta', theta, (n_points,))):
        z = disp_zarr if name == 'disp' else zarr.open(path, mode='r')
        if name != 'disp':
            logger.zarr_info(path, z)
        if not np.issubdtype(z.dtype, np.floating):
            raise ArgumentError(f'{name} must be floating point, got {z.dtype}')
        if z.shape != shape:
            raise ArgumentError(f'{name} must have shape {shape}, got shape {z.shape}')
    if nimages < 4:
        raise ArgumentError(f'the DEM error needs at least 4 images, got {nimages}')
    days = _days(dates, nimages)
    dates = [str(d).strip() for d in (dates.replace(',', ' ').split() if isinstance(dates, str) else dates)]
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ArgumentError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    logger.info(f'{nimages} images from {dates[0]} to {dates[-1]} ({int(days[-1])} days), relative to image {ref} '
                f'({dates[ref]})')
    if out_chunks is None:
        out_chunks = disp_zarr.chunks[0]
    dh_zarr = zarr.open(dh, mode='w', shape=(n_points,), dtype=np.float32, chunks=(out_chunks,))
    logger.zarr_info(dh, dh_zarr)
    out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(out, out_zarr)
    # one task per block of points with all its images (whole chunks of `dh` and `out`)
    images = slice(0, nimages)
    tasks = []
    for start in range(0, n_points, out_chunks):
        points = slice(start, min(start + out_chunks, n_points))
        tasks.append(([Chunk(disp, (points, images)), Chunk(bperp, (points, images)), Chunk(slant_range, (points,)),
                       Chunk(theta, (points,))], [Chunk(dh, (points,)), Chunk(out, (points, images))]))
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(mr.pc_sc_dem_error, tasks, desc='point blocks', dates=dates, ref=int(ref))
    values = dh_zarr[:]                 # 4 bytes per point, for the log
    values = values[np.isfinite(values)]
    if values.size:
        p01, p50, p99 = np.percentile(values, [1, 50, 99])
        logger.info(f'dh of {values.size} points ({n_points - values.size} nan): percentiles 1, 50, 99 '
                    f'{p01:.2f}, {p50:.2f}, {p99:.2f} m')
    else:
        logger.warning('dh is nan at every point')
    logger.info('done.')
