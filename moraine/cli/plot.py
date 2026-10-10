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
from concurrent.futures import ProcessPoolExecutor

import toml
from ..api.utils_ import ngpjit
from ..api.rtree import HilbertRtree
from .logging import mc_logger
from ..api.coord_ import Coord
from . import mk_clean_dir, parallel_write_zarr, parallel_read_zarr

# layout version of the pyramids, see docs/contracts/pyramid.md
PYRAMID_VERSION = 1
_RTREE_PAGE = 512        # points per leaf of the bounding box tree of a point cloud pyramid

def _pyramid_workers(n_workers):
    return min(8, os.cpu_count() or 1) if n_workers is None else n_workers

def _pyramid_pool(n_workers, initializer, initargs):
    """process pool rendering the channels of a pyramid: the work per channel is Python code that threads cannot
    share, and forked processes start without importing anything"""
    context = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    return ProcessPoolExecutor(max_workers=_pyramid_workers(n_workers), mp_context=context, initializer=initializer,
                               initargs=initargs)

_PYRAMID_WORKER = {}

def _ras_pyramid_init(ras, out_dir, maxlevel):
    _PYRAMID_WORKER.update(ras_zarr=zarr.open(ras,mode='r'),
                           ras_zarrs=[zarr.open(Path(out_dir)/f'{level}.zarr',mode='r+') for level in range(maxlevel+1)])

def _ras_pyramid_channel(channel_idx):
    w = _PYRAMID_WORKER
    ras = parallel_read_zarr(w['ras_zarr'], (slice(None), slice(None), *[slice(i,i+1) for i in channel_idx]))
    _ras_downsample_all_and_save(ras, w['ras_zarrs'], channel_idx)

def _ras_downsample_all_and_save(ras,zarrs,channel_idx):
    slices = [slice(None),slice(None)]
    if len(channel_idx) != 0:
        for idx in channel_idx:
            slices.append(slice(idx,idx+1))
    slices = tuple(slices)

    for level in range(len(zarrs)):
        ras_ = ras[::2**level,::2**level]
        parallel_write_zarr(ras_,zarrs[level],slices)

@mc_logger
def ras_pyramid(
    ras:str,
    out_dir:str,
    chunks:tuple[int,int]=(256,256),
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
    n_workers : int, optional
        processes rendering the channels at the same time, default: 8 or the number of cores if fewer; a process
        holds one channel of the raster
    """
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
            dtype=ras_zarr.dtype,
            chunks=(*out_chunks,*channel_chunks),)
        logger.zarr_info(out_dir/f'{level}.zarr',downsampled_ras_zarr)
        downsampled_ras_zarrs.append(downsampled_ras_zarr)
    downsampled_ras_zarrs[0].attrs['moraine_pyramid'] = {'version': PYRAMID_VERSION, 'kind': 'raster'}

    channel_idxs = list(np.ndindex(ras_zarr.shape[2:]))
    logger.info(f'rendering {len(channel_idxs)} channels in {_pyramid_workers(n_workers)} processes.')
    with _pyramid_pool(n_workers, _ras_pyramid_init, (ras, out_dir, maxlevel)) as pool:
        list(pool.map(_ras_pyramid_channel, channel_idxs))
    logger.info('rendering finished.')

def _default_ras_post_proc(data_zarr, xslice, yslice, *kdims):
    data_n_kdim = data_zarr.ndim - 2
    assert len(kdims) == data_n_kdim
    if len(kdims) == 0:
        # zarr do not support empty tuple as input
        return data_zarr[yslice,xslice]
    else:
        index_tuple = (yslice, xslice, *kdims)
        return data_zarr[index_tuple]

def _ras_phase_post_proc(data_zarr, xslice, yslice, *kdims):
    data_n_kdim = data_zarr.ndim - 2
    assert len(kdims) == 1
    i = kdims[0]
    assert data_n_kdim == 1
    assert np.iscomplexobj(data_zarr)
    return np.angle(data_zarr[yslice,xslice,i])

def _ras_inf_0_post_proc(data_zarr, xslice, yslice, *kdims):
    data_n_kdim = data_zarr.ndim - 2
    assert len(kdims) == 1
    i = kdims[0]
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,0]*data_zarr[yslice,xslice,i].conj())
        else:
            return data_zarr[yslice,xslice,0]-data_zarr[yslice,xslice,i]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,0,i])
        else:
            return data_zarr[yslice,xslice,0,i]

def _ras_inf_seq_post_proc(data_zarr, xslice, yslice, *kdims):
    data_n_kdim = data_zarr.ndim - 2
    assert len(kdims) == 1
    i = kdims[0]
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,i]*data_zarr[yslice,xslice,i+1].conj())
        else:
            return data_zarr[yslice,xslice,i]-data_zarr[yslice,xslice,i+1]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,i,i+1])
        else:
            return data_zarr[yslice,xslice,i,i+1]
def _ras_inf_all_post_proc(data_zarr, xslice, yslice, *kdims):
    data_n_kdim = data_zarr.ndim - 2
    assert len(kdims) == 2
    i,j = kdims
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,i]*data_zarr[yslice,xslice,j].conj())
        else:
            return data_zarr[yslice,xslice,i]-data_zarr[yslice,xslice,j]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[yslice,xslice,i,j])
        else:
            return data_zarr[yslice,xslice,i,j]

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

def _pc_pyramid_init(pc, out_dir, maxlevel, coord, gix, yis, xis):
    _PYRAMID_WORKER.update(pc_zarr=zarr.open(pc,mode='r'), coord=coord, gix=gix, yis=yis, xis=xis,
                           pc_out=zarr.open(Path(out_dir)/'pc.zarr',mode='r+'),
                           ras_zarrs=[zarr.open(Path(out_dir)/f'{level}.zarr',mode='r+') for level in range(maxlevel+1)])

def _pc_pyramid_channel(channel_idx):
    w = _PYRAMID_WORKER
    pc = parallel_read_zarr(w['pc_zarr'], (slice(None), *[slice(i,i+1) for i in channel_idx]))
    _pc_downsample_all_and_save(pc, w['coord'], w['gix'], w['yis'], w['xis'], w['pc_out'], w['ras_zarrs'], channel_idx)

def _pc_downsample_all_and_save(pc,coord,gix,yis,xis,pc_zarr,ras_zarrs,channel_idx):
    pc_slices = [slice(None),]
    ras_slices = [slice(None),slice(None)]
    if len(channel_idx) != 0:
        for idx in channel_idx:
            pc_slices.append(slice(idx,idx+1))
            ras_slices.append(slice(idx,idx+1))
    pc_slices = tuple(pc_slices)
    ras_slices = tuple(ras_slices)
    parallel_write_zarr(pc,pc_zarr,pc_slices)

    ras = coord.rasterize(pc,gix)
    parallel_write_zarr(ras,ras_zarrs[0],ras_slices)

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
    n_workers:int=None,
):
    """render point cloud data to pyramid of difference zoom levels.

    Parameters
    ----------
    pc : str
        input: point cloud data, shape (n_points,) or (n_points, n)
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
    n_workers : int, optional
        processes rendering the channels at the same time, default: 8 or the number of cores if fewer; a process
        holds one channel: the data of the points and the finest raster of the channel
    """
    logger = logging.getLogger(__name__)
    logger.info('clean out dir')
    out_dir = Path(out_dir); mk_clean_dir(out_dir)

    pc_zarr = zarr.open(pc,mode='r')
    logger.zarr_info(pc, pc_zarr)

    n_pc = pc_zarr.shape[0]
    channel_chunks = (1,)*(pc_zarr.ndim-1)
    logger.info(f'rendering point cloud data coordinates:')
    if x is None and y is None:
        yx_zarr = zarr.open(yx,mode='r')
        assert yx_zarr.shape[1] == 2
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
    with open(out_dir/'bounds.toml',mode='w') as f:
        toml.dump(bounds, f, encoder=toml.TomlNumpyEncoder())

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
            dtype=pc_zarr.dtype,
            chunks=(*ras_chunks,*channel_chunks),)
        logger.zarr_info(out_dir/f'{level}.zarr',downsampled_ras_zarr)
        if level == 0:
            downsampled_ras_zarr.attrs['moraine_pyramid'] = {'version': PYRAMID_VERSION, 'kind': 'point cloud'}

    channel_idxs = list(np.ndindex(pc_zarr.shape[1:]))
    logger.info(f'rendering {len(channel_idxs)} channels in {_pyramid_workers(n_workers)} processes.')
    with _pyramid_pool(n_workers, _pc_pyramid_init, (pc, out_dir, maxlevel, coord, gix, yis, xis)) as pool:
        list(pool.map(_pc_pyramid_channel, channel_idxs))
    logger.info('rendering finished.')

class _LazyRtree:
    '''HilbertRtree of the pyramid points, read from the pyramid (`rtree.zarr`) when the points are first
    queried; built from the coordinates for pyramids made without it.

    The points are only drawn when zoomed in to the finest level, so an overview (or a quicklook) of a large
    point cloud does not touch it.'''
    def __init__(self, pyramid_dir):
        self.pyramid_dir = Path(pyramid_dir)
        self._rtree = None

    def bbox_query(self, bounds, x, y):
        if self._rtree is None:
            if (self.pyramid_dir/'rtree.zarr').exists():
                self._rtree = HilbertRtree.load(str(self.pyramid_dir/'rtree.zarr'))
            else:
                x_ = zarr.open(self.pyramid_dir/'x.zarr',mode='r')[:]
                y_ = zarr.open(self.pyramid_dir/'y.zarr',mode='r')[:]
                self._rtree = HilbertRtree.build(x_,y_,page_size=_RTREE_PAGE)
        return self._rtree.bbox_query(bounds, x, y)

def _default_pc_post_proc(data_zarr, idx_array, *kdims):
    data_n_kdim = data_zarr.ndim - 1
    assert len(kdims) == data_n_kdim
    if len(kdims) == 0:
        return data_zarr[idx_array]
    else:
        index_tuple = (idx_array, *kdims)
        return data_zarr[index_tuple]

def _pc_phase_post_proc(data_zarr, idx_array, *kdims):
    data_n_kdim = data_zarr.ndim - 1
    assert len(kdims) == 1
    assert data_n_kdim == 1
    i = kdims[0]
    assert np.iscomplexobj(data_zarr)
    return np.angle(data_zarr[idx_array,i])

def _pc_inf_0_post_proc(data_zarr, idx_array, *kdims):
    data_n_kdim = data_zarr.ndim - 1
    assert len(kdims) == 1
    i = kdims[0]
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,0]*data_zarr[idx_array,i].conj())
        else:
            return data_zarr[idx_array,0]-data_zarr[idx_array,i]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,0,i])
        else:
            return data_zarr[idx_array,0,i]

def _pc_inf_seq_post_proc(data_zarr, idx_array, *kdims):
    data_n_kdim = data_zarr.ndim - 1
    assert len(kdims) == 1
    i = kdims[0]
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,i]*data_zarr[idx_array,i+1].conj())
        else:
            return data_zarr[idx_array,i]-data_zarr[idx_array,i+1]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,i,i+1])
        else:
            return data_zarr[idx_array,i,i+1]

def _pc_inf_all_post_proc(data_zarr, idx_array, *kdims):
    data_n_kdim = data_zarr.ndim - 1
    assert len(kdims) == 2
    i,j = kdims
    if data_n_kdim == 1:
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,i]*data_zarr[idx_array,j].conj())
        else:
            return data_zarr[idx_array,i]-data_zarr[idx_array,j]
    else:
        assert data_n_kdim == 2
        if np.iscomplexobj(data_zarr):
            return np.angle(data_zarr[idx_array,i,j])
        else:
            return data_zarr[idx_array,i,j]

# ---------------------------------------------------------------- pyramid reading and plotting rules
# used by `moraine info` / `quicklook` / `view` (moraine/command/summary.py) and the tile viewer

def pyramid_levels(path)->list:
    """Zoom levels [0, 1, ...] of a pyramid made by `ras_pyramid` / `pc_pyramid`, [] if `path` is not one.

    Pyramids carry ``moraine_pyramid = {version, kind}`` in the attributes of ``0.zarr``
    (docs/contracts/pyramid.md). Older pyramids without it are recognized by their levels halving in size,
    which also tells them apart from directories of per-chunk arrays (``0.zarr``, ``1.zarr``, ... made by
    `ras2pc_ras_chunk`).
    """
    p = Path(path)
    if not p.is_dir() or not (p / '0.zarr').exists():
        return []
    try:
        z0 = zarr.open(str(p / '0.zarr'), mode='r')
    except Exception:
        return []
    levels = sorted(int(q.stem) for q in p.glob('*.zarr') if q.stem.isdigit())
    meta = z0.attrs.get('moraine_pyramid') if hasattr(z0, 'attrs') else None
    if meta:
        if meta.get('version', 0) > PYRAMID_VERSION:
            raise ValueError(f'{path}: pyramid layout version {meta["version"]} is newer than this moraine '
                             f'supports ({PYRAMID_VERSION}); update moraine')
        return levels
    if not (p / '1.zarr').exists():
        return []
    try:
        z1 = zarr.open(str(p / '1.zarr'), mode='r')
    except Exception:
        return []
    if z0.ndim < 2 or z1.ndim != z0.ndim or \
       tuple(z1.shape[:2]) != tuple(-(-n // 2) for n in z0.shape[:2]):
        return []
    return levels


def _round(x):
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return x
    return round(x, 5 - int(math.floor(math.log10(abs(x)))))


def _stats(a):
    """Statistics and warnings of a sample of values (1D array)."""
    out, warnings = {}, []
    if a.size == 0:
        return {'warnings': ['no values']}
    if a.dtype == bool:
        out['true_fraction'] = _round(a.mean())
        if a.all() or not a.any():
            warnings.append(f'all values are {bool(a[0])}')
        return {**out, 'warnings': warnings} if warnings else out
    prefix = ''
    if np.iscomplexobj(a):
        nan = np.isnan(a.real) | np.isnan(a.imag)
        a, prefix = np.abs(a), 'amplitude_'
    else:
        a = a.astype(np.float64)
        nan = np.isnan(a)
    out['nan_fraction'] = _round(nan.mean())
    inf = np.isinf(a)
    v = a[~nan & ~inf]
    if inf.any():
        warnings.append(f'{int(inf.sum())} infinite values in the sample')
    if v.size == 0:
        warnings.append('all values are nan')
    else:
        p = np.percentile(v, [1, 50, 99])
        out.update({prefix + 'min': _round(v.min()), prefix + 'max': _round(v.max()), prefix + 'mean': _round(v.mean()),
                    prefix + 'std': _round(v.std()), prefix + 'p01': _round(p[0]), prefix + 'p50': _round(p[1]),
                    prefix + 'p99': _round(p[2])})
        if v.min() == v.max():
            warnings.append(f'all values are {_round(v.min())}')
    if warnings:
        out['warnings'] = warnings
    return out


def _pyramid_stats(p, levels, max_bytes):
    """Statistics from the finest pyramid level read within `max_bytes`: a regular decimation of the scene."""
    level = levels[-1]
    for lv in levels:
        if zarr.open(str(p / f'{lv}.zarr'), mode='r').nbytes <= max_bytes:
            level = lv
            break
    a = np.asarray(zarr.open(str(p / f'{level}.zarr'), mode='r')[...])
    idx_path = p / f'idx_{level}.zarr'
    if idx_path.exists():   # point cloud pyramid: skip the cells without points (idx == -1)
        a = a[np.asarray(zarr.open(str(idx_path), mode='r')[...]) != -1]
    return {'stats_level': level, **_stats(a.ravel())}


def _phase_2d(data_zarr, xslice, yslice):
    return np.angle(data_zarr[yslice, xslice])


_PHASE_POST_PROC = ('phase', 'intf_0', 'intf_seq', 'intf_all')
_FRAME = (900, 700)            # largest plot area (width, height) in screen pixels
_MAX_RATIO = 4                 # scenes more elongated than 1:4 are drawn with this ratio


def _cyclic_cmap():
    import colorcet
    return colorcet.colorwheel


def _phase_pc_1d(data_zarr, idx_array):
    return np.angle(data_zarr[idx_array])


def _kdim_ranges(shape, post_proc):
    """Range of the image indices i (and j) of a stack for the sliders."""
    extra = shape[2:]
    if not extra:
        return {}
    n = extra[0]
    if post_proc == 'intf_seq':
        return {'i': (0, n - 2)}
    if post_proc == 'intf_all':
        return {'i': (0, n - 1), 'j': (0, n - 1)}
    if post_proc in ('phase', 'intf_0'):
        return {'i': (0, n - 1)}
    return {name: (0, m - 1) for name, m in zip(('i', 'j'), extra)}


def _resolve_post_proc(base, post_proc):
    """(post_proc, raster post processing, point post processing, phase_like) of a pyramid with level 0 `base`:
    complex data show their phase by default."""
    complex_data = np.iscomplexobj(np.empty(0, base.dtype))
    ras_proc = pc_proc = post_proc
    if post_proc is None and complex_data:
        if base.ndim == 3:
            ras_proc = pc_proc = post_proc = 'phase'
        else:
            ras_proc, pc_proc = _phase_2d, _phase_pc_1d
    return post_proc, ras_proc, pc_proc, complex_data or post_proc in _PHASE_POST_PROC


def _frame_size(width, height):
    """Plot area with the aspect of the scene, fitted into _FRAME; very elongated scenes are squeezed."""
    ratio = min(max(width / height, 1 / _MAX_RATIO), _MAX_RATIO)
    if ratio >= _FRAME[0] / _FRAME[1]:
        return _FRAME[0], max(1, round(_FRAME[0] / ratio))
    return max(1, round(_FRAME[1] * ratio)), _FRAME[1]
