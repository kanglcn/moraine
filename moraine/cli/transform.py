"""transform coordinate between projection"""


__all__ = ['transform']

import functools
import logging
from pathlib import Path

import numpy as np
import zarr
from pyproj import Transformer

from ..api.utils_.chunk import all_chunk_slices
from .executor import Chunk, Executor
from .logging import mc_logger
from .utils_ import ArgumentError


@functools.lru_cache(maxsize=None)
def _transformer(epsg_in, epsg_out):
    # one per pair of EPSG codes and process; a pyproj Transformer may be used by several threads at the same time
    return Transformer.from_crs(f'EPSG:{epsg_in}', f'EPSG:{epsg_out}', always_xy=True)


def _transform_chunk(xx, yy, epsg_in, epsg_out):
    # PROJ computes without holding the GIL: the threads of the executor transform their chunks in parallel.
    # pyproj returns two floats for a chunk of one element (taken for a point, numpy warns): back to the chunk shape
    x, y = _transformer(epsg_in, epsg_out).transform(xx, yy)
    return np.reshape(x, xx.shape), np.reshape(y, xx.shape)


@mc_logger
def transform(xx_in,
              yy_in,
              xx_out,
              yy_out,
              epsg_in=4326,
              epsg_out=3857,
              n_workers=1,
              threads_per_worker=4,
             ):
    """Coordinate transformation between two coordinate reference systems (EPSG codes).
    By default, longitude (`xx_in`) and latitude (`yy_in`) in degrees are transformed to the x (east) and y (north)
    coordinates of the web mercator projection in meters, the coordinates of the maps of `moraine view`.
    x is always the east axis (longitude, easting) and y the north axis (latitude, northing), whatever the axis order
    of the EPSG definition. A point with a NaN coordinate gets NaN in both outputs.

    Parameters
    ----------
    xx_in
        input: x coordinate of every pixel or point, any shape, e.g. (nlines, width) for a raster or (n_points,) for
        a point cloud (longitude in degrees for the default epsg_in)
    yy_in
        input: y coordinate, same shape as `xx_in` (latitude in degrees for the default epsg_in)
    xx_out
        output: x coordinate in `epsg_out`, same shape, chunks and dtype as `xx_in` (east in meters for the default
        epsg_out); a new array, not one of the inputs
    yy_out
        output: y coordinate in `epsg_out`, same shape, chunks and dtype as `xx_in` (north in meters for the default
        epsg_out); a new array, not one of the inputs
    epsg_in : int, default: 4326
        EPSG code of the input coordinates (4326: WGS84 longitude / latitude)
    epsg_out : int, default: 3857
        EPSG code of the output coordinates (3857: web mercator)
    n_workers : int, default: 1
        number of workers
    threads_per_worker : int, default: 4
        tasks a worker runs at the same time
    """
    logger = logging.getLogger(__name__)
    xx_in_zarr = zarr.open(xx_in, mode='r'); yy_in_zarr = zarr.open(yy_in, mode='r')
    logger.zarr_info('xx_in', xx_in_zarr); logger.zarr_info('yy_in', yy_in_zarr)
    if yy_in_zarr.shape != xx_in_zarr.shape:
        raise ArgumentError(f'yy_in: shape {yy_in_zarr.shape} differs from xx_in: {xx_in_zarr.shape}')
    # the outputs are created (emptied) before any input is read
    ins, outs = {Path(p).resolve() for p in (xx_in, yy_in)}, {Path(p).resolve() for p in (xx_out, yy_out)}
    if len(outs) < 2 or ins & outs:
        raise ArgumentError('xx_out and yy_out must be two arrays other than the inputs')
    transformer = _transformer(epsg_in, epsg_out)   # fails here for an unknown EPSG code
    logger.info(f'EPSG:{epsg_in} -> EPSG:{epsg_out}: {transformer.description}')

    shape, chunks, dtype = xx_in_zarr.shape, xx_in_zarr.chunks, xx_in_zarr.dtype
    for name, path in (('xx_out', xx_out), ('yy_out', yy_out)):
        logger.zarr_info(name, zarr.open(path, mode='w', shape=shape, dtype=dtype, chunks=chunks))
    tasks = [([Chunk(xx_in, sl), Chunk(yy_in, sl)], [Chunk(xx_out, sl), Chunk(yy_out, sl)])
             for sl in all_chunk_slices(shape, chunks)]
    logger.info(f'transform {len(tasks)} chunks')
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker) as ex:
        ex.map_chunks(_transform_chunk, tasks, desc='transform', epsg_in=epsg_in, epsg_out=epsg_out)
    logger.info('done.')
