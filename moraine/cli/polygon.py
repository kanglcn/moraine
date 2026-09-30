"""Masks from polygons drawn on results (CLI)"""

__all__ = ['polygon_mask']

import itertools
import logging

import numpy as np
import zarr

from ..polygon import read_polygons, polygons_contain
from .logging import mc_logger


def _chunk_slices(shape, chunks):
    ranges = [[slice(s, min(s + c, n)) for s in range(0, n, c)] for n, c in zip(shape, chunks)]
    return itertools.product(*ranges)


@mc_logger
def polygon_mask(
    polygons:str,
    mask:str,
    x:str=None,
    y:str=None,
    gix:str=None,
    shape:tuple[int,int]=None,
    keep:str='inside',
    chunks:tuple[int,int]=(1000,1000),
):
    """Mask of the pixels or points inside (or outside) the polygons of a GeoJSON file.

    The polygons come from the tile viewer (`TileView.save_polygons`) or any GeoJSON file. Give the
    positions of the data in the coordinates of the file in one of three ways: `x` and `y` (longitude /
    latitude for 'lonlat' files, range / azimuth for 'radar_grid' files), `gix` (point clouds on the radar
    grid) or `shape` (a whole raster on the radar grid, pixel (i, j) at range j, azimuth i). Apply the mask
    with e.g. `math` (``where(mask, a, nan)``) or `pc-logic-pc`.

    Parameters
    ----------
    polygons : str
        input: GeoJSON file with the polygons
    mask : str
        output: bool mask, True for the data to keep; same shape as `x` / `y`, (n_points,) for `gix`,
        (nlines, width) for `shape`
    x : str, optional
        input: x coordinate of the data (longitude or range), any shape; with `y`
    y : str, optional
        input: y coordinate of the data (latitude or azimuth), same shape as `x`
    gix : str, optional
        input: grid index (azimuth, range) of a point cloud, shape (n_points, 2); 'radar_grid' polygons only
    shape : tuple[int, int], optional
        raster shape (nlines, width); 'radar_grid' polygons only
    keep : str, default: 'inside'
        'inside' keeps the data inside any polygon (True there), 'outside' keeps the data outside all polygons
    chunks : tuple[int, int], default: (1000, 1000)
        output chunk size for `shape`; the other outputs are chunked like their input
    """
    logger = logging.getLogger(__name__)
    if keep not in ('inside', 'outside'):
        raise ValueError(f"keep must be 'inside' or 'outside', not {keep!r}")
    given = [name for name, v in (('x/y', x if x is not None else y), ('gix', gix), ('shape', shape)) if v is not None]
    if len(given) != 1:
        raise ValueError('give the data positions with exactly one of `x` and `y`, `gix` or `shape`')
    polys, coordinates = read_polygons(polygons)
    logger.info(f'{len(polys)} polygons in {coordinates} coordinates read from {polygons}')
    if coordinates == 'lonlat' and x is None:
        raise ValueError(f'{polygons} has longitude / latitude polygons: give the longitude and latitude of the '
                         f'data as `x` and `y`')

    if x is not None:
        if y is None:
            raise ValueError('give `y` with `x`')
        x_zarr, y_zarr = zarr.open(x, mode='r'), zarr.open(y, mode='r')
        logger.zarr_info(x, x_zarr); logger.zarr_info(y, y_zarr)
        if x_zarr.shape != y_zarr.shape:
            raise ValueError(f'x {x_zarr.shape} and y {y_zarr.shape} have different shapes')
        out_shape, out_chunks = x_zarr.shape, x_zarr.chunks
        def positions(sl):
            return x_zarr[sl], y_zarr[sl]
    elif gix is not None:
        gix_zarr = zarr.open(gix, mode='r')
        logger.zarr_info(gix, gix_zarr)
        if gix_zarr.ndim != 2 or gix_zarr.shape[1] != 2:
            raise ValueError(f'gix must have shape (n_points, 2), not {gix_zarr.shape}')
        out_shape, out_chunks = gix_zarr.shape[:1], gix_zarr.chunks[:1]
        def positions(sl):
            g = gix_zarr[sl[0]]
            return g[:, 1], g[:, 0]
    else:
        out_shape, out_chunks = tuple(int(n) for n in shape), tuple(int(c) for c in chunks)
        def positions(sl):
            yi, xi = np.mgrid[sl[0], sl[1]]
            return xi, yi

    mask_zarr = zarr.open(mask, mode='w', shape=out_shape, dtype=bool, chunks=out_chunks)
    logger.zarr_info(mask, mask_zarr)
    n_keep = 0
    for sl in _chunk_slices(out_shape, out_chunks):
        inside = polygons_contain(polys, *positions(sl))
        block = inside if keep == 'inside' else ~inside
        mask_zarr[sl] = block
        n_keep += int(block.sum())
    logger.info(f'{n_keep} of {int(np.prod(out_shape))} kept ({keep} the polygons)')
