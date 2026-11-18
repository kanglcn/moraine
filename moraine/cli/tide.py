"""Solid earth tides (CLI): the line of sight displacement they cause at point clouds (decision 0056)"""

__all__ = ['pc_solid_tide']

import logging

import numpy as np
import zarr

import moraine as mr
from ..api.utils_.times import _times
from ..api.tide import _check_time
from .logging import mc_logger
from .utils_ import ArgumentError
from .executor import Executor, Chunk


def _tide_task(lon, lat, theta, phi, **kwargs):
    """the tides of a block of points, and per image the number, sum and sum of squares of its finite values"""
    tide = mr.pc_solid_tide(lon, lat, theta, phi, **kwargs)
    stats = np.zeros((3, tide.shape[1]))
    for k in range(tide.shape[1]):             # one column at a time: no copy of the whole block
        col = tide[:, k]
        col = col[np.isfinite(col)].astype(np.float64)
        stats[:, k] = col.size, col.sum(), (col * col).sum()
    return tide, stats


@mc_logger
def pc_solid_tide(
    lon:str,
    lat:str,
    theta:str,
    phi:str,
    out:str,
    times:list[str],
    ref:int=0,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Line of sight displacement caused by the solid earth tides at the points, relative to a reference image.

    The displacement of the ground by the tides of the Sun and the Moon of the IERS Conventions (2010) at the time of
    every image, projected on the look vector of the point.

    Parameters
    ----------
    lon : str
        input: longitude of the points in degrees, shape (n_points,), floating point
    lat : str
        input: latitude of the points in degrees, shape (n_points,), floating point
    theta : str
        input: elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle),
        shape (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`)
    phi : str
        input: orientation angle of the look vector of the points in radians, counterclockwise from east, shape
        (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`); the look vector points from the ground
        to the satellite
    out : str
        output: line of sight displacement in millimeters that the solid earth tides cause in every image relative to
        the image `ref`, shape (n_points, nimages), float32, chunks (out_chunks, 1), positive towards the satellite;
        the column of `ref` is 0; the displacement corrected for the tides is `disp` - `out` (`math`); nan where
        `lon`, `lat`, `theta` or `phi` of the point is nan
    times : list of str
        time of every image in UTC, ISO 8601 (e.g. 2019-01-02T05:06:07.123456Z), 1972 or later: the `times` of the
        metadata of `load-gamma-metadata`
    ref : int, default: 0
        the image the displacement is relative to, 0 .. nimages - 1: the `ref` of `unwrap-correct-closure-pc`
    out_chunks : int, optional
        point chunk size of `out`, that of `lon` by default; the points of a task
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 4
        tasks (blocks of `out_chunks` points) a worker processes at the same time; a task holds about 4 bytes per
        point and image
    """
    logger = logging.getLogger(__name__)
    lon_zarr = zarr.open(lon, mode='r'); logger.zarr_info(lon, lon_zarr)
    n_points = lon_zarr.shape[0]
    for name, path in (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi)):
        z = lon_zarr if name == 'lon' else zarr.open(path, mode='r')
        if name != 'lon':
            logger.zarr_info(path, z)
        if not np.issubdtype(z.dtype, np.floating):
            raise ArgumentError(f'{name} must be floating point, got {z.dtype}')
        if z.shape != (n_points,):
            raise ArgumentError(f'{name} must have shape ({n_points},), got shape {z.shape}')
    t_img = _times(times)
    nimages = len(t_img)
    if nimages == 0:
        raise ArgumentError('times must have the time of at least one image')
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ArgumentError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    for t in t_img:
        _check_time(t)
    logger.info(f'{nimages} images from {t_img[0]:%Y-%m-%d %H:%M} to {t_img[-1]:%Y-%m-%d %H:%M} UTC, relative to '
                f'image {ref} ({t_img[ref]:%Y-%m-%d %H:%M:%S} UTC)')

    if out_chunks is None:
        out_chunks = lon_zarr.chunks[0]
    out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(out, out_zarr)
    images = slice(0, nimages)
    tasks = []
    for start in range(0, n_points, out_chunks):
        points = slice(start, min(start + out_chunks, n_points))
        tasks.append(([Chunk(p, (points,)) for p in (lon, lat, theta, phi)], [Chunk(out, (points, images))]))
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        stats = ex.map_chunks(_tide_task, tasks, desc='point blocks', times=t_img, ref=int(ref))
    count, total, squares = np.sum([s[0] for s in stats], axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        std = np.sqrt(np.maximum(squares / count - (total / count) ** 2, 0))
    others = np.delete(std, ref)
    if others.size and np.isfinite(others).any():
        logger.info(f'standard deviation of the solid earth tides over the points per image: median '
                    f'{np.nanmedian(others):.2f} mm, max {np.nanmax(others):.2f} mm (image {int(np.nanargmax(std))})')
    logger.info('done.')
