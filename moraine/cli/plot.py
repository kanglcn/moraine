"""Pyramids of rasters and point clouds (multi resolution copies for views and statistics) and the rules to
read and show them"""


__all__ = ['ras_pyramid', 'pc_pyramid']

import logging
import zarr
import numpy as np
import math
from pathlib import Path
import numpy as np
from numba import prange

import os
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor

from ..api.utils_ import ngpjit
from ..api.pc.rtree import HilbertRtree
from .logging import mc_logger
from ..api.utils_.coord import Coord
from . import mk_clean_dir, parallel_write_zarr, parallel_read_zarr
from .utils_ import write_toml

# layout version of the pyramids, see docs/contracts/pyramid.md
PYRAMID_VERSION = 1
_RTREE_PAGE = 512        # points per leaf of the bounding box tree of a point cloud pyramid
_SAMPLE_VALUES = 2**23   # values of the regular sample for the percentiles of the statistics (64 MiB of float64)
_STATS_COLUMNS = ('nan_fraction', 'min', 'max', 'mean', 'std', 'p01', 'p50', 'p99')   # per channel table
_METHODS = {'raster': ('decimate', 'mean'), 'point cloud': ('first', 'mean')}   # how a level is made from the finer one

def _pyramid_workers(n_workers):
    return min(8, os.cpu_count() or 1) if n_workers is None else n_workers

def _pyramid_pool(n_workers, initializer, initargs):
    """process pool rendering the channels of a pyramid (decision 0046). zarr handles the chunks of a process in one
    I/O thread, about 1 - 2 thousand chunks a second, too few for the many small tiles of a pyramid: the channels
    are rendered in processes. They start from a forkserver that has imported this module, as fast as a fork of
    this process but without its threads and locks."""
    if 'forkserver' in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context('forkserver')
        context.set_forkserver_preload([__name__])   # used when the server starts: once per process
    else:
        context = multiprocessing.get_context('spawn')
    return ProcessPoolExecutor(max_workers=_pyramid_workers(n_workers), mp_context=context, initializer=initializer,
                               initargs=initargs)

_PYRAMID_WORKER = {}

def _ras_pyramid_init(ras, out_dir, maxlevel, step, rows, method):
    _PYRAMID_WORKER.update(ras_zarr=zarr.open(ras,mode='r'), step=step, rows=rows, method=method,
                           ras_zarrs=[zarr.open(Path(out_dir)/f'{level}.zarr',mode='r+') for level in range(maxlevel+1)])

def _ras_pyramid_channel(channel_idx):
    """One channel of the raster, by bands of `rows` lines: the levels a band holds whole are written band by
    band, the coarser levels from the coarsest of them once all bands are done. A level is every 2nd pixel of
    the finer one (`method` decimate) or the mean of its 2 x 2 blocks over the finite pixels (mean)."""
    w = _PYRAMID_WORKER
    src, zarrs, rows, mean = w['ras_zarr'], w['ras_zarrs'], w['rows'], w['method'] == 'mean'
    channel = tuple(slice(i, i + 1) for i in channel_idx)
    ny = src.shape[0]
    band_levels = min(len(zarrs) - 1, int(round(math.log2(rows))))     # rows is a power of 2: 2**level divides it
    parts, tops, counts = [], [], []
    for r0 in range(0, ny, rows):
        band = parallel_read_zarr(src, (slice(r0, min(r0 + rows, ny)), slice(None), *channel))
        parts.append(_part_stats(band, w['step']))
        cur, cnt = band, (_finite_count(band) if mean else None)
        for level in range(band_levels + 1):
            if level:
                cur, cnt = _block_mean(cur, cnt) if mean else (cur[::2, ::2], None)
            f = 2 ** level
            parallel_write_zarr(cur.astype(zarrs[level].dtype, copy=False), zarrs[level],
                                (slice(r0 // f, r0 // f + cur.shape[0]), slice(None), *channel))
        tops.append(cur.copy()); counts.append(cnt)     # a copy: a decimated level is a view of the whole band
        del band, cur, cnt      # one band in memory: released before the next one is read (cur may be a view of it)
    if band_levels < len(zarrs) - 1:
        cur = np.concatenate(tops, axis=0)
        cnt = np.concatenate(counts, axis=0) if mean else None
        for level in range(band_levels + 1, len(zarrs)):
            cur, cnt = _block_mean(cur, cnt) if mean else (cur[::2, ::2], None)
            parallel_write_zarr(cur.astype(zarrs[level].dtype, copy=False), zarrs[level],
                                (slice(None), slice(None), *channel))
    return _merge_parts(parts)

def _mean_dtype(dtype):
    """dtype of the levels made by means: float32 for integer and boolean data, the dtype itself otherwise."""
    dtype = np.dtype(dtype)
    return dtype if dtype.kind in 'fc' else np.dtype(np.float32)

def _finite_count(a):
    """(h, w) int64: 1 where the pixel of `a` (h, w, ...) is finite, 0 where not."""
    return np.isfinite(a).reshape(a.shape[0], a.shape[1], -1).all(axis=2).astype(np.int64)

def _block_mean(values, count):
    """Means of the 2 x 2 blocks of `values` (h, w, ...) weighted by `count` (h, w), the finite pixels of the data
    in each cell: (means (ceil(h / 2), ceil(w / 2), ...) as float64 / complex128, nan where a block has none; the
    counts of the blocks)."""
    h, w = values.shape[:2]
    H, W = -(-h // 2), -(-w // 2)
    tail = values.shape[2:]
    acc = np.complex128 if np.iscomplexobj(values) else np.float64
    v = np.zeros((2 * H, 2 * W) + tail, acc)
    c = np.zeros((2 * H, 2 * W), np.int64)
    weight = count[(...,) + (None,) * len(tail)]
    v[:h, :w] = np.where(weight > 0, values, 0) * weight
    c[:h, :w] = count
    v = v.reshape(H, 2, W, 2, *tail).sum(axis=(1, 3))
    c = c.reshape(H, 2, W, 2).sum(axis=(1, 3))
    cw = c[(...,) + (None,) * len(tail)]
    return np.where(cw > 0, v / np.maximum(cw, 1), np.nan), c

def _merge_parts(parts):
    """One `_part_stats` of the parts (bands) of a channel."""
    finite = [p for p in parts if p['n'] - p['nan'] - p['inf']]
    _, mean, m2 = _moments(parts)
    return {'n': sum(p['n'] for p in parts), 'nan': sum(p['nan'] for p in parts), 'inf': sum(p['inf'] for p in parts),
            'mean': mean, 'm2': m2,
            'min': min(p['min'] for p in finite) if finite else np.nan,
            'max': max(p['max'] for p in finite) if finite else np.nan,
            'sample': np.concatenate([p['sample'] for p in parts]) if parts else np.empty(0)}


def _moments(parts):
    """(number, mean, sum of the squared deviations from the mean) of the finite values of all `parts`."""
    # the parts are combined two at a time by their numbers, means and sums of squared deviations (Chan, Golub and
    # LeVeque 1983); sums of the values and of their squares lose the digits of the variance where the mean is
    # large against the standard deviation (1e4 +- 0.01, the amplitude of phases of modulus 1)
    m = [(p['n'] - p['nan'] - p['inf'], p['mean'], p['m2']) for p in parts]
    while len(m) > 1:
        m = [_combine(*m[i:i + 2]) for i in range(0, len(m), 2)]
    return m[0] if m else (0, 0.0, 0.0)


def _combine(a, b=None):
    """(number, mean, sum of the squared deviations) of the values of two parts given as such triples."""
    if b is None or not b[0]:
        return a
    if not a[0]:
        return b
    n = a[0] + b[0]
    d = b[1] - a[1]
    return n, a[1] + d * b[0] / n, a[2] + b[2] + d * d * a[0] * b[0] / n


def _sample_step(n_values):
    """Every `step`-th finite value of the data goes into the sample for the percentiles of the statistics."""
    return max(1, math.ceil(n_values / _SAMPLE_VALUES))


def _part_stats(a, step):
    """Numbers of values, nan and infinite values of a part of the data (a channel or a band), and the mean, the
    sum of the squared deviations from it (``m2``), the minimum, maximum and a regular sample of its finite values,
    pooled by `_pool_stats`: of the amplitude for complex data, of 0 / 1 for booleans."""
    a = np.asarray(a).ravel()
    if a.dtype == bool:
        v, nan = a.astype(np.float64), np.zeros(a.shape, bool)
    elif np.iscomplexobj(a):
        nan = np.isnan(a.real) | np.isnan(a.imag)
        v = np.abs(a).astype(np.float64)
    else:
        v = a.astype(np.float64)
        nan = np.isnan(v)
    inf = np.isinf(v)
    finite = v[~(nan | inf)]
    mean = float(finite.mean()) if finite.size else 0.0
    dev = finite - mean
    # a copy: a view of `finite` would keep all the values of the part, until a channel's bands are merged
    return {'n': int(a.size), 'nan': int(nan.sum()), 'inf': int(inf.sum()), 'mean': mean,
            'm2': float(np.square(dev, out=dev).sum()), 'min': float(finite.min()) if finite.size else np.nan,
            'max': float(finite.max()) if finite.size else np.nan, 'sample': finite[::step].copy()}


def _pool_stats(parts, dtype):
    """Statistics of the data from the `_part_stats` of all its parts: the exact nan fraction, minimum, maximum,
    mean and standard deviation, the percentiles of the pooled sample, and warnings."""
    n = sum(p['n'] for p in parts)
    if n == 0:
        return {'warnings': ['no values']}
    nan, inf = sum(p['nan'] for p in parts), sum(p['inf'] for p in parts)
    n_finite = n - nan - inf
    out, warnings = {}, []
    if n_finite:
        _, mean, m2 = _moments(parts)
        std = math.sqrt(m2 / n_finite)
        lo = min(p['min'] for p in parts if p['n'] - p['nan'] - p['inf'])
        hi = max(p['max'] for p in parts if p['n'] - p['nan'] - p['inf'])
        pct = np.percentile(np.concatenate([p['sample'] for p in parts]), [1, 50, 99])
    if np.dtype(dtype) == bool:
        out['true_fraction'] = _round(mean)
        if lo == hi:
            warnings.append(f'all values are {bool(lo)}')
        return {**out, 'warnings': warnings} if warnings else out
    prefix = 'amplitude_' if np.iscomplexobj(np.empty(0, dtype)) else ''
    out['nan_fraction'] = _round(nan / n)
    if inf:
        warnings.append(f'{inf} infinite values')
    if not n_finite:
        warnings.append('all values are nan')
    else:
        out.update({prefix + 'min': _round(lo), prefix + 'max': _round(hi), prefix + 'mean': _round(mean),
                    prefix + 'std': _round(std), prefix + 'p01': _round(pct[0]), prefix + 'p50': _round(pct[1]),
                    prefix + 'p99': _round(pct[2])})
        if lo == hi:
            warnings.append(f'all values are {_round(lo)}')
    if warnings:
        out['warnings'] = warnings
    return out


def _channel_row(p):
    """Row of the per channel table (`_STATS_COLUMNS`) from the `_part_stats` of one channel."""
    n_finite = p['n'] - p['nan'] - p['inf']
    if not n_finite:
        return [p['nan'] / p['n'] if p['n'] else np.nan] + [np.nan] * (len(_STATS_COLUMNS) - 1)
    pct = np.percentile(p['sample'], [1, 50, 99]) if p['sample'].size else [np.nan] * 3
    return [p['nan'] / p['n'], p['min'], p['max'], p['mean'], math.sqrt(p['m2'] / n_finite), *pct]


def _write_stats(out_dir, z0, kind, method, parts, channel_shape, dtype):
    """Write the per channel table stats.zarr and the marker of the pyramid (`kind`, `method`, the statistics of all
    its data) into `z0` (level 0); `parts` are the `_part_stats` of the channels in the order of `np.ndindex`."""
    table = np.array([_channel_row(p) for p in parts], dtype=np.float64).reshape(*channel_shape, len(_STATS_COLUMNS))
    t = zarr.open(str(Path(out_dir) / 'stats.zarr'), mode='w', shape=table.shape, dtype=np.float64, chunks=table.shape)
    t[...] = table
    t.attrs['columns'] = list(_STATS_COLUMNS)
    # the marker after everything else: a pyramid is complete when it has one
    z0.attrs['moraine_pyramid'] = {'version': PYRAMID_VERSION, 'kind': kind, 'method': method,
                                   'stats': _pool_stats(parts, dtype)}


@mc_logger
def ras_pyramid(
    ras:str,
    out_dir:str,
    chunks:tuple[int,int]=(256,256),
    rows:int=1024,
    method:str='decimate',
    n_workers:int=None,
):
    """render raster data to pyramid of difference zoom levels.

    Parameters
    ----------
    ras : str
        input: 2D raster (nlines, width) or raster stack (nlines, width, n)
    out_dir : str
        output directory to store rendered data
    chunks : tuple[int, int], default: (256, 256)
        output raster tile size
    rows : int, default: 1024
        lines of the raster a process reads at a time, a power of 2; a process needs about 0.5 GB plus
        max(40, 8 x itemsize) x rows x width bytes, more if rows is below the lines of a chunk of the raster
    method : str, default: 'decimate'
        how each level is made from the finer one: 'decimate' keeps every 2nd pixel, 'mean' averages the 2 x 2
        blocks (nan pixels left out, complex data as complex numbers; integer and boolean data give float32
        levels). The mean is a multilook of the data: right for real values and for phases relative to a
        reference image (phase histories of phase linking, filtered interferograms), wrong for the SLC pixels
        of an rslc stack, whose scatterer phases differ from pixel to pixel
    n_workers : int, optional
        processes rendering the channels at the same time, default: 8 or the number of cores if fewer; a process
        renders one channel of the raster
    """
    if rows < 1 or rows & (rows - 1):
        raise ValueError(f'rows must be a power of 2, not {rows}')
    if method not in _METHODS['raster']:
        raise ValueError(f"method must be one of {_METHODS['raster']}, not {method!r}")
    logger = logging.getLogger(__name__)
    logger.info('clean out dir')
    out_dir = Path(out_dir); mk_clean_dir(out_dir)

    ras_zarr = zarr.open(ras,mode='r')
    logger.zarr_info(ras, ras_zarr)

    ny, nx = ras_zarr.shape[0:2]
    n_channel = ras_zarr.ndim-2
    out_chunks = chunks
    channel_chunks = ((1,)*n_channel)
    maxlevel = math.floor(math.log2(min(nx,ny))) # so at least 2 pixels

    logger.info(f'rendered raster pyramid with zoom level ranging from 0 (finest resolution) to {maxlevel} (coarsest resolution).')

    downsampled_ras_zarrs = []
    for level in range(maxlevel+1):
        shape = (math.ceil(ny/(2**level)), math.ceil(nx/(2**level)))
        downsampled_ras_zarr = zarr.open(
            out_dir/f'{level}.zarr',mode='w',
            shape=(*shape,*ras_zarr.shape[2:]),
            dtype=ras_zarr.dtype if level == 0 or method == 'decimate' else _mean_dtype(ras_zarr.dtype),
            chunks=(*out_chunks,*channel_chunks),)
        logger.zarr_info(out_dir/f'{level}.zarr',downsampled_ras_zarr)
        downsampled_ras_zarrs.append(downsampled_ras_zarr)

    channel_idxs = list(np.ndindex(ras_zarr.shape[2:]))
    step = _sample_step(ras_zarr.size)
    logger.info(f'rendering {len(channel_idxs)} channels in {_pyramid_workers(n_workers)} processes.')
    with _pyramid_pool(n_workers, _ras_pyramid_init, (ras, out_dir, maxlevel, step, rows, method)) as pool:
        parts = list(pool.map(_ras_pyramid_channel, channel_idxs))
    # the marker with the statistics of the whole raster: written last, a pyramid is complete when it has it
    _write_stats(out_dir, downsampled_ras_zarrs[0], 'raster', method, parts, ras_zarr.shape[2:], ras_zarr.dtype)
    logger.info('rendering finished.')

@ngpjit
def _next_level_idx_from_raster_of_integer(pc_idx, nan_value):
    '''return the raster indices to the next level of raster'''
    assert pc_idx.ndim == 2
    ny, nx = pc_idx.shape
    next_ny, next_nx = math.ceil(ny/2), math.ceil(nx/2)
    xi = np.empty((next_ny,next_nx), dtype=np.int32)
    yi = np.empty((next_ny,next_nx), dtype=np.int32)

    for i in range(next_ny):
        for j in prange(next_nx):
            # Select a 2x2 box from the original array
            box = pc_idx[i*2:min(i*2+2, ny), j*2:min(j*2+2, nx)]
            idx_ = np.argwhere(box != nan_value)
            if len(idx_) == 0:
                yi[i,j]= i*2
                xi[i,j] = j*2
            else:
                yi[i,j] = idx_[0,0] + i*2
                xi[i,j] = idx_[0,1] + j*2
    return yi, xi

def _pc_pyramid_init(pc, out_dir, maxlevel, coord, shared, step, method):
    # the indices saved by pc_pyramid in the directory `shared`, memory mapped: one copy for all processes
    load = lambda name: np.load(Path(shared)/f'{name}.npy', mmap_mode='r')
    _PYRAMID_WORKER.update(pc_zarr=zarr.open(pc,mode='r'), coord=coord, gix=load('gix'), valid=load('valid'),
                           yis=[load(f'yi_{level}') for level in range(1, maxlevel+1)],
                           xis=[load(f'xi_{level}') for level in range(1, maxlevel+1)], step=step, method=method,
                           pc_out=zarr.open(Path(out_dir)/'pc.zarr',mode='r+'),
                           ras_zarrs=[zarr.open(Path(out_dir)/f'{level}.zarr',mode='r+') for level in range(maxlevel+1)])

def _pc_pyramid_channel(channel_idx):
    w = _PYRAMID_WORKER
    pc = parallel_read_zarr(w['pc_zarr'], (slice(None), *[slice(i,i+1) for i in channel_idx]))
    _pc_downsample_all_and_save(pc, w['coord'], w['gix'], w['yis'], w['xis'], w['pc_out'], w['ras_zarrs'], channel_idx,
                                w['method'], w['valid'])
    return _part_stats(pc, w['step'])

def _pc_downsample_all_and_save(pc,coord,gix,yis,xis,pc_zarr,ras_zarrs,channel_idx,method,valid):
    """The points of one channel to pc.zarr, rasterized to level 0 (nan where a cell has no point: integer and
    boolean data as float32), and the coarser levels: the first point of each 2 x 2 block (`method` first, the
    cells of `yis` / `xis`) or the mean of the cells with points (mean, `valid` marks them at level 0)."""
    pc_slices = [slice(None),]
    ras_slices = [slice(None),slice(None)]
    if len(channel_idx) != 0:
        for idx in channel_idx:
            pc_slices.append(slice(idx,idx+1))
            ras_slices.append(slice(idx,idx+1))
    pc_slices = tuple(pc_slices)
    ras_slices = tuple(ras_slices)
    parallel_write_zarr(pc,pc_zarr,pc_slices)

    ras = coord.rasterize(pc.astype(ras_zarrs[0].dtype, copy=False),gix)
    parallel_write_zarr(ras,ras_zarrs[0],ras_slices)

    if method == 'mean':
        cur, cnt = ras, valid.astype(np.int64)
        for level in range(1, len(ras_zarrs)):
            cur, cnt = _block_mean(cur, cnt)
            parallel_write_zarr(cur.astype(ras_zarrs[level].dtype, copy=False), ras_zarrs[level], ras_slices)
        return
    for level in range(1,len(ras_zarrs)):
        ras = ras[yis[level-1],xis[level-1]]
        parallel_write_zarr(ras,ras_zarrs[level],ras_slices)

@mc_logger
def pc_pyramid(
    pc:str,
    out_dir:str,
    x:str=None,
    y:str=None,
    yx:str=None,
    ras_resolution:float=20,
    ras_chunks:tuple[int,int]=(256,256),
    pc_chunks:int=65536,
    method:str='first',
    n_workers:int=None,
):
    """render point cloud data to pyramid of difference zoom levels.

    Parameters
    ----------
    pc : str
        input: point cloud data, shape (n_points,) or (n_points, n), at least one point
    out_dir : str
        output directory to store rendered data
    x : str, optional
        input: x coordinate of the points, e.g. longitude or web mercator x
    y : str, optional
        input: y coordinate of the points, e.g. latitude or web mercator y
    yx : str, optional
        input: (y, x) coordinates of the points, shape (n_points, 2), e.g. gix; alternative to `x` and
        `y`
    ras_resolution : float, default: 20
        minimum resolution of rendered raster data,
    ras_chunks : tuple[int, int], default: (256, 256)
        output raster tile size
    pc_chunks : int, default: 65536
        output pc tile size
    method : str, default: 'first'
        how each level is made from the finer one: 'first' keeps the first cell with a point of each 2 x 2
        block, 'mean' averages the cells with points (complex data as complex numbers). The mean is a multilook
        of the data: right for real values and for phase histories relative to a reference image (phase
        linking, filtered points)
    n_workers : int, optional
        processes rendering the channels at the same time, default: 8 or the number of cores if fewer; a process
        holds one channel: the data of the points and the finest raster of the channel (3 rasters with 'mean')
    """
    if method not in _METHODS['point cloud']:
        raise ValueError(f"method must be one of {_METHODS['point cloud']}, not {method!r}")
    logger = logging.getLogger(__name__)
    pc_zarr = zarr.open(pc,mode='r')
    logger.zarr_info(pc, pc_zarr)
    if pc_zarr.shape[0] == 0:       # the grid of the levels is placed on the points (docs/contracts/pyramid.md)
        raise ValueError(f'the point cloud is empty: {pc} has no points; a point cloud pyramid needs at least one')
    logger.info('clean out dir')
    out_dir = Path(out_dir); mk_clean_dir(out_dir)

    channel_chunks = (1,)*(pc_zarr.ndim-1)
    logger.info(f'rendering point cloud data coordinates:')
    if x is None and y is None:
        yx_zarr = zarr.open(yx,mode='r')
        if yx_zarr.ndim != 2 or yx_zarr.shape[1] != 2:
            raise ValueError(f'yx must have shape (n_points, 2), got {yx_zarr.shape}')
        yx = parallel_read_zarr(yx_zarr,(slice(None),slice(0,2)))
    else:
        y_zarr = zarr.open(y,mode='r')
        yx = np.empty((y_zarr.shape[0],2),dtype=y_zarr.dtype)
        yx[:,0] = parallel_read_zarr(zarr.open(y,mode='r'),(slice(None),))
        yx[:,1] = parallel_read_zarr(zarr.open(x,mode='r'),(slice(None),))
    x, y = yx[:,1], yx[:,0]

    x0, xm, y0, ym = x.min(), x.max(), y.min(), y.max()
    # cell j is centred at x0 + j*ras_resolution and every point goes to its nearest centre, so the grid
    # must reach the cell of the largest coordinates. The cell indices are computed on a grid with one
    # spare cell and the grid is then cut to the cells used, so both come from the same rounding.
    nx, ny = math.ceil((xm-x0)/ras_resolution) + 2, math.ceil((ym-y0)/ras_resolution) + 2
    gix = Coord(x0, ras_resolution, nx, y0, ras_resolution, ny).coords2gixs(yx)
    ny, nx = int(gix[:,0].max()) + 1, int(gix[:,1].max()) + 1
    coord = Coord(x0, ras_resolution, nx, y0, ras_resolution, ny)
    bounds = {'bounds':[x0, y0, coord.xm, coord.ym]}
    logger.info(f"rasterizing point cloud data to grid with bounds: {bounds['bounds']}.")
    write_toml(out_dir/'bounds.toml', bounds)

    maxlevel = coord.maxlevel

    out_x_zarr = zarr.open(out_dir/f'x.zarr',mode='w',shape=x.shape,dtype=x.dtype,chunks=(pc_chunks,))
    out_y_zarr = zarr.open(out_dir/f'y.zarr',mode='w',shape=y.shape,dtype=y.dtype,chunks=(pc_chunks,))
    logger.zarr_info(out_dir/f'x.zarr',out_x_zarr)
    logger.zarr_info(out_dir/f'y.zarr',out_y_zarr)
    parallel_write_zarr(x, out_x_zarr,(slice(None),))
    parallel_write_zarr(y, out_y_zarr,(slice(None),))
    # the bounding box tree of the points, for the views that draw and probe the points one by one
    HilbertRtree.build(x, y, page_size=_RTREE_PAGE).save(str(out_dir/'rtree.zarr'))
    logger.zarr_info(out_dir/'rtree.zarr', zarr.open(str(out_dir/'rtree.zarr'), mode='r'))
    del x, y, yx
    logger.info('pc data coordinates rendering ends.')

    yis = []; xis = []
    last_idx = None   # set in the first iteration
    for level in range(maxlevel+1):
        if level == 0:
            current_idx = coord.rasterize_iidx(gix)
            valid = current_idx != -1
        else:
            yi, xi = _next_level_idx_from_raster_of_integer(last_idx,-1)
            yis.append(yi); xis.append(xi)
            current_idx = last_idx[yi,xi]
        out_idx_zarr = zarr.open(out_dir/f'idx_{level}.zarr',shape=current_idx.shape,dtype=current_idx.dtype,chunks=ras_chunks)
        logger.zarr_info(out_dir/f'idx_{level}.zarr',out_idx_zarr)
        parallel_write_zarr(current_idx,out_idx_zarr,(slice(None),slice(None)))
        last_idx = current_idx
    logger.info('rasterized idx rendering ends')

    out_pc_zarr = zarr.open(out_dir/f'pc.zarr',mode='w',shape=pc_zarr.shape, dtype=pc_zarr.dtype, chunks=(pc_chunks,*channel_chunks))
    logger.zarr_info(out_dir/f'pc.zarr', out_pc_zarr)
    for level in range(maxlevel+1):
        shape = (math.ceil(ny/(2**level)), math.ceil(nx/(2**level)))
        downsampled_ras_zarr = zarr.open(
            out_dir/f'{level}.zarr',mode='w',
            shape=(*shape,*pc_zarr.shape[1:]),
            dtype=_mean_dtype(pc_zarr.dtype),           # nan where a cell has no point, also for integers
            chunks=(*ras_chunks,*channel_chunks),)
        logger.zarr_info(out_dir/f'{level}.zarr',downsampled_ras_zarr)
        if level == 0:
            level0_zarr = downsampled_ras_zarr

    channel_idxs = list(np.ndindex(pc_zarr.shape[1:]))
    step = _sample_step(pc_zarr.size)
    logger.info(f'rendering {len(channel_idxs)} channels in {_pyramid_workers(n_workers)} processes.')
    with tempfile.TemporaryDirectory(prefix='moraine-pyramid-') as shared:
        # every process needs the indices: saved once, memory mapped by the processes (`_pc_pyramid_init`) instead
        # of a copy for each; this process does not need them any more
        arrays = {'gix': gix, 'valid': valid, **{f'yi_{level}': a for level, a in enumerate(yis, 1)},
                  **{f'xi_{level}': a for level, a in enumerate(xis, 1)}}
        for name, a in arrays.items():
            np.save(Path(shared)/f'{name}.npy', a)
        del arrays, gix, valid, yis, xis
        with _pyramid_pool(n_workers, _pc_pyramid_init, (pc, out_dir, maxlevel, coord, shared, step, method)) as pool:
            parts = list(pool.map(_pc_pyramid_channel, channel_idxs))
    # the marker with the statistics of the points: written last, a pyramid is complete when it has it
    _write_stats(out_dir, level0_zarr, 'point cloud', method, parts, pc_zarr.shape[1:], pc_zarr.dtype)
    logger.info('rendering finished.')

# ---------------------------------------------------------------- pyramid reading and plotting rules
# used by `moraine info` / `quicklook` / `view` (moraine/command/summary.py) and the tile viewer

def pyramid_levels(path)->list:
    """Zoom levels [0, 1, ...] of a pyramid made by `ras_pyramid` / `pc_pyramid`, [] if `path` is not one.

    A pyramid carries ``moraine_pyramid = {version, kind, method, stats}`` in the attributes of ``0.zarr``, written
    when it is complete (docs/contracts/pyramid.md). ValueError for a pyramid of a newer moraine and for one to
    rebuild: levels halving in size without the marker (made before moraine 0.10.0, or its command did not finish),
    a marker without ``method`` or ``stats``, a point cloud pyramid without ``rtree.zarr`` (made before 0.10.0).
    Directories of other arrays named ``0.zarr``, ``1.zarr``, ... (the per-chunk arrays of `ras2pc_ras_chunk`) are
    not pyramids.
    """
    p = Path(path)
    if not p.is_dir() or not (p / '0.zarr').exists():
        return []
    try:
        z0 = zarr.open(str(p / '0.zarr'), mode='r')
    except Exception:
        return []
    meta = z0.attrs.get('moraine_pyramid')
    command = 'pc-pyramid' if (p / 'bounds.toml').exists() else 'ras-pyramid'
    if not meta:
        if not _levels_halve(p, z0):
            return []
        old = (f'a pyramid without its marker (the attribute moraine_pyramid of 0.zarr), made by moraine before '
               f'0.10.0 or by a {command} that did not finish')
    elif meta.get('version', 0) > PYRAMID_VERSION:
        raise ValueError(f'{path}: pyramid layout version {meta["version"]} is newer than this moraine '
                         f'supports ({PYRAMID_VERSION}); update moraine')
    elif 'method' not in meta or 'stats' not in meta:
        old = 'a pyramid of moraine before 0.10.0, without the statistics of its data'
    elif command == 'pc-pyramid' and not (p / 'rtree.zarr').exists():
        old = 'a point cloud pyramid of moraine before 0.10.0, without rtree.zarr'
    else:
        return sorted(int(q.stem) for q in p.glob('*.zarr') if q.stem.isdigit())
    raise ValueError(f'{path}: {old}; rebuild it with `moraine {command}`')


def _levels_halve(p, z0):
    """True if 1.zarr of the directory `p` is half the size of 0.zarr (`z0`), as the levels of a pyramid."""
    try:
        z1 = zarr.open(str(p / '1.zarr'), mode='r')
    except Exception:
        return False
    return isinstance(z0, zarr.Array) and isinstance(z1, zarr.Array) and z0.ndim >= 2 and z1.ndim == z0.ndim and \
        tuple(z1.shape[:2]) == tuple(-(-n // 2) for n in z0.shape[:2])


def _round(x):
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return x
    return round(x, 5 - int(math.floor(math.log10(abs(x)))))


def _stats(a):
    """Statistics and warnings of the values of an array, as `_pool_stats` of the array as one part."""
    return _pool_stats([_part_stats(a, 1)], a.dtype)


def _channel_warnings(p, limit=20):
    """Warnings about single channels (images, pairs) of a pyramid, from its per channel table stats.zarr: all
    values nan, constant values. [] for pyramids without the table and for data without channels."""
    path = Path(p) / 'stats.zarr'
    if not path.exists():
        return []
    t = zarr.open(str(path), mode='r')
    table = np.asarray(t[...])
    if table.ndim < 2:
        return []
    col = {name: k for k, name in enumerate(t.attrs['columns'])}
    rows = table.reshape(-1, table.shape[-1])
    names = [f'image {idx[0]}' if table.ndim == 2 else f'channel {idx}' for idx in np.ndindex(table.shape[:-1])]
    out = []
    for what, bad in [('all values nan', np.isnan(rows[:, col['min']])),
                      ('constant values', rows[:, col['min']] == rows[:, col['max']])]:
        which = [names[k] for k in np.flatnonzero(bad)]
        if which:
            more = f' and {len(which) - limit} more' if len(which) > limit else ''
            out.append(f'{what}: {", ".join(which[:limit])}{more}')
    return out


_FRAME = (900, 700)            # largest plot area (width, height) in screen pixels
_MAX_RATIO = 4                 # scenes more elongated than 1:4 are drawn with this ratio


def _cyclic_cmap():
    import colorcet
    return colorcet.colorwheel


def _frame_size(width, height):
    """Plot area with the aspect of the scene, fitted into _FRAME; very elongated scenes are squeezed."""
    ratio = min(max(width / height, 1 / _MAX_RATIO), _MAX_RATIO)
    if ratio >= _FRAME[0] / _FRAME[1]:
        return _FRAME[0], max(1, round(_FRAME[0] / ratio))
    return max(1, round(_FRAME[1] * ratio)), _FRAME[1]
