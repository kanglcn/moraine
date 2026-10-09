"""Point cloud data utilities (CLI)"""


__all__ = ['gix2bool', 'bool2gix', 'ras2pc', 'pc_concat', 'ras2pc_ras_chunk', 'pc2ras', 'pc_hix', 'pc_gix', 'pc_sort', 'pc_union',
           'pc_intersect', 'pc_diff', 'pc_logic_ras', 'pc_logic_pc', 'pc_select_data', 'data_reduce']

import itertools
import logging
from pathlib import Path
import zarr
import numpy as np
import numexpr as ne
from typing import Callable

from .logging import mc_logger
import moraine as mr
from ..api.chunk_ import all_chunk_slices
from .dask_ import ZarrDir, _parallel_read_pc_dir
from .executor import Executor, Chunk
from .utils_ import mk_clean_dir


def _channels(extra):
    """index tuples of the channels of an array with the trailing dimensions `extra` ([()] for none)"""
    return list(itertools.product(*[range(n) for n in extra]))


def _column(k):
    """slices of channel `k` (an index tuple), one element per trailing dimension"""
    return tuple(slice(i, i+1) for i in k)


def _pc_zarr(path, n_points, extra, dtype, chunks):
    """a point cloud zarr (n_points, *extra) chunked (chunks, 1, ...)"""
    return zarr.open(str(path), mode='w', shape=(n_points, *extra), dtype=dtype, chunks=(chunks, *(1,)*len(extra)))


def _indexing_pc_data(pc_in, iidx):
    return pc_in[iidx]


def _pc_concat_channel(zarr_dir, k, key, out, n_points):
    """read channel `k` of the per chunk zarrs of `zarr_dir`, sort it by `key` and write it to `out`"""
    data = _parallel_read_pc_dir(zarr_dir, k)
    if key is not None:
        data = data[key]
    Chunk(out, (slice(0, n_points), *_column(k))).write(data.reshape(n_points, *(1,)*len(k)))


def _pc2ras(
    pc_data:np.ndarray,
    gix:np.ndarray,
    shape:tuple,
):
    """Parameters
    ----------
    pc_data : np.ndarray
        data, 1D
    gix : np.ndarray
        gix
    shape : tuple
        image shape
    """
    raster = np.empty((*shape,*pc_data.shape[1:]),dtype=pc_data.dtype)
    raster[:] = np.nan
    raster[gix[:,0],gix[:,1]] = pc_data
    return raster


def _pc_union(pc1,pc2,inv_iidx1,inv_iidx2,iidx2,n_pc):
    pc = np.empty((n_pc,*pc1.shape[1:]),dtype=pc1.dtype)
    pc[inv_iidx1] = pc1
    pc[inv_iidx2] = pc2[iidx2]
    return pc


def _reduce_chunk(chunk, map_func, reduce_func, axis):
    data = chunk.read()
    if map_func is not None:
        data = map_func(data)
    return reduce_func(data, axis=axis, keepdims=True)


def _gather_channels(ex, fn, in_paths, out_paths, refs, n_pc, chunks, desc, logger):
    """Run ``fn(column of every input, *refs)`` for every channel of the point clouds `in_paths` (whole columns of
    n_points) and write the result to the point clouds `out_paths` of `n_pc` points, created here with `chunks`."""
    tasks = []
    for ins, out in zip(in_paths, out_paths):
        zs = [zarr.open(path, mode='r') for path in ins]
        for path, z in zip(ins, zs):
            logger.zarr_info(path, z)
        extra = zs[0].shape[1:]
        logger.zarr_info(out, _pc_zarr(out, n_pc, extra, zs[0].dtype, chunks))
        for k in _channels(extra):
            tasks.append(([Chunk(path, (slice(0, z.shape[0]), *_column(k))) for path, z in zip(ins, zs)] + list(refs),
                          [Chunk(out, (slice(0, n_pc), *_column(k)))]))
    ex.map_chunks(fn, tasks, desc=desc)

@mc_logger
def gix2bool(gix:str,
             is_pc:str,
             shape:tuple[int,int],
             chunks:tuple[int,int]= (1000,1000),
            ):
    """Convert pc grid index to bool 2d array

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int
    is_pc : str
        output: bool raster, True at the points, shape (nlines, width)
    shape : tuple[int, int]
        raster shape (nlines, width)
    chunks : tuple[int, int], default: (1000, 1000)
        output chunk size
    """
    logger = logging.getLogger(__name__)
    is_pc_path = is_pc

    gix_zarr = zarr.open(gix,mode='r')
    logger.zarr_info('gix',gix_zarr)
    assert gix_zarr.ndim == 2, "gix dimentation is not 2."
    assert gix_zarr.shape[1] == 2
    logger.info('loading gix into memory.')
    gix = zarr.open(gix,mode='r')[:]

    logger.info('calculate the bool array')
    is_pc = np.zeros(shape,dtype=bool)
    is_pc[gix[:,0],gix[:,1]] = True

    is_pc_zarr = zarr.open(is_pc_path,mode='w',shape=shape,dtype=bool,chunks=chunks)
    logger.zarr_info('is_pc',is_pc_zarr)
    logger.info('write the bool array.')
    is_pc_zarr[:] = is_pc
    logger.info('write done.')

@mc_logger
def bool2gix(is_pc:str,
             gix:str,
             chunks:int=100000,
            ):
    """Convert bool 2d array to grid index

    Parameters
    ----------
    is_pc : str
        input bool array
    gix : str
        output, point cloud grid index
    chunks : int, default: 100000
        output point chunk size
    """
    gix_path = gix
    logger = logging.getLogger(__name__)

    is_pc_zarr = zarr.open(is_pc,mode='r')
    logger.zarr_info('is_pc', is_pc_zarr)
    logger.info('loading is_pc into memory.')
    is_pc = zarr.open(is_pc,mode='r')[:]

    logger.info('calculate the index')
    gix = np.stack(np.where(is_pc),axis=-1)

    gix_zarr = zarr.open(gix_path,mode='w',shape=gix.shape,dtype=bool,chunks=(chunks,1))
    logger.zarr_info('gix', gix_zarr)
    logger.info('write the gix.')
    gix_zarr[:] = gix
    logger.info('write done.')

@mc_logger
def ras2pc(
    idx:str,
    ras:str|list,
    pc:str|list,
    chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Convert raster data to point cloud data

    Parameters
    ----------
    idx : str
        input: grid index or hillbert index of the point cloud
    ras : str | list
        input: path or list of paths of raster data, shape (nlines, width, ...)
    pc : str | list
        output, path (in string) or list of path for point cloud data
    chunks : int, optional
        point chunk size of the output data, same as `idx` by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    if isinstance(ras,str):
        assert isinstance(pc,str)
        ras_list = [ras]; pc_list = [pc]
    else:
        assert isinstance(ras,list); assert isinstance(pc,list)
        ras_list = ras; pc_list = pc
    shape = zarr.open(ras_list[0],mode='r').shape[:2]
    idx_zarr = zarr.open(idx,mode='r'); logger.zarr_info(idx,idx_zarr)
    if chunks is None: chunks = idx_zarr.chunks[0]
    if idx_zarr.ndim == 2:
        logger.info('loading gix into memory.')
        gix = idx_zarr[:]
    else:
        logger.info('loading hix into memory and convert to gix')
        gix = mr.pc_gix(idx_zarr[:],shape=shape)
    n_pc = gix.shape[0]
    # one task per channel of every raster: the image in, the values at the points out
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        gix_ref = ex.put(gix)
        tasks = []
        for ras_path, pc_path in zip(ras_list,pc_list):
            ras_zarr = zarr.open(ras_path,mode='r'); logger.zarr_info(ras_path, ras_zarr)
            extra = ras_zarr.shape[2:]
            logger.zarr_info(pc_path, _pc_zarr(pc_path, n_pc, extra, ras_zarr.dtype, chunks))
            for k in _channels(extra):
                tasks.append(([Chunk(ras_path, (slice(0, shape[0]), slice(0, shape[1]), *_column(k))), gix_ref],
                              [Chunk(pc_path, (slice(0, n_pc), *_column(k)))]))
        ex.map_chunks(mr.ras2pc, tasks, desc='channels')
    logger.info('done.')

@mc_logger
def pc_concat(
    pcs:list|str,
    pc:list|str,
    key:list|str=None,
    chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """concatenate (and sort) point cloud dataset.

    Parameters
    ----------
    pcs : list | str
        list of path to pc or directory that hold one pc, or a list of that
    pc : list | str
        output, path of output or a list of that
    key : list | str, optional
        input: key(s) that sort the concatenated data (e.g. from `ras2pc_ras_chunk`); no sorting by
        default
    chunks : int, optional
        pc chunk size in output data, optional, same as first pc in pcs by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    pcs_path = pcs
    pc_path = pc
    key_path = key
    logger = logging.getLogger(__name__)
    if isinstance(pc_path,str):
        pc_path = [pc_path,]
        if isinstance(pcs_path,str):
            pcs_path = Path(pcs_path)
            pcs_path = sorted(pcs_path.glob('*.zarr'),key=lambda path: int(path.stem))
        pcs_path = [pcs_path,]
    elif isinstance(pc_path,list):
        assert isinstance(pcs_path,list)
        pcs_path_ = []
        for one_pcs_path in pcs_path:
            if isinstance(one_pcs_path,str):
                one_pcs_path = Path(one_pcs_path)
                one_pcs_path = sorted(one_pcs_path.glob('*.zarr'),key=lambda path: int(path.stem))
            pcs_path_.append(one_pcs_path)
        pcs_path = pcs_path_
    else:
        raise ValueError("wrong input")
    logger.info(f'input pcs: {pcs_path}')
    logger.info(f'output pc: {pc_path}')
    if key_path is not None:
        logger.info('load key')
        if isinstance(key,list):
            for i, _key_path in enumerate(key_path):
                key_zarr = zarr.open(_key_path,mode='r'); logger.zarr_info(_key_path,key_zarr)
                if i == 0:
                    key = key_zarr[:]
                else:
                    key = key[key_zarr[:]]
        else:
            key_zarr = zarr.open(key_path,mode='r'); logger.zarr_info(key_path,key_zarr)
            key = key_zarr[:]
    zarr_dirs = []
    for one_pcs_path in pcs_path:
        zarr_dir = ZarrDir([str(p) for p in one_pcs_path])
        zarr_dirs.append(zarr_dir)
    if chunks is None: chunks = zarr_dirs[0].chunksize[0]
    # one task per channel of every output: the channel of all chunk zarrs read, sorted and written
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        key_ref = ex.put(key) if key is not None else None
        tasks = []
        for zarr_dir, one_pc_path in zip(zarr_dirs, pc_path):
            n_pc, extra = zarr_dir.shape[0], zarr_dir.shape[1:]
            logger.zarr_info(one_pc_path, _pc_zarr(one_pc_path, n_pc, extra, zarr_dir.dtype, chunks))
            tasks += [(zarr_dir, k, key_ref, one_pc_path, n_pc) for k in _channels(extra)]
        ex.map(_pc_concat_channel, tasks, desc='channels')
    logger.info('done.')

@mc_logger
def ras2pc_ras_chunk(
    gix:str,
    ras:str|list,
    pc:str|list,
    key:str,
    chunks:tuple[int,int]=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Convert raster data to point cloud data that sorted by ras chunk

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int
    ras : str | list
        input: path or list of paths of raster data, shape (nlines, width, ...)
    pc : str | list
        output, path (directory) or list of path for point cloud data
    key : str
        output, path for the key to sort generated pc in the directory back to gix order
    chunks : tuple[int, int], optional
        (azimuth, range) raster chunk size used to split the points, same as the first `ras` by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    if isinstance(ras,str):
        assert isinstance(pc,str)
        ras_list = [ras]; pc_list = [pc]
    else:
        assert isinstance(ras,list); assert isinstance(pc,list)
        ras_list = ras; pc_list = pc
    ras0_zarr = zarr.open(ras_list[0],mode='r')
    shape = ras0_zarr.shape[:2]
    if chunks is None: chunks = ras0_zarr.chunks[:2]
    chunks = list(chunks)
    for i in range(len(chunks)):
        if chunks[i] == -1: chunks[i] = ras0_zarr.shape[i]
    chunks = tuple(chunks)
    gix_zarr = zarr.open(gix,mode='r'); logger.zarr_info(gix,gix_zarr)
    logger.info('loading gix into memory.')
    gix = gix_zarr[:]
    logger.info('convert gix to the order of ras chunk')
    chunk_idx, chunk_bounds, invert_idx = mr.api.pc._pc_split_by_chunk(gix,chunks,shape)
    sorted_gix = gix[chunk_idx]
    ras_chunk_order_gix = mr.api.pc._gix_ras_chunk(sorted_gix,chunk_bounds, chunks, shape)
    logger.info('save key')
    key_zarr = zarr.open(key,mode='w',dtype=invert_idx.dtype,shape=invert_idx.shape,chunks=gix_zarr.chunks[:1])
    key_zarr[:] = invert_idx
    chunk_slices = all_chunk_slices(shape, chunks)
    # one task per raster chunk with points and raster: the chunk in, the values at its points out, one zarr per chunk
    tasks = []
    for ras_path, pc_path in zip(ras_list,pc_list):
        pc_path = Path(pc_path); mk_clean_dir(pc_path)
        ras_zarr = zarr.open(ras_path,mode='r'); logger.zarr_info(ras_path, ras_zarr)
        extra = ras_zarr.shape[2:]
        for j, sl in enumerate(chunk_slices):
            b0, b1 = int(chunk_bounds[j]), int(chunk_bounds[j+1])
            if b1 == b0:
                continue
            _pc_zarr(pc_path/f'{j}.zarr', b1-b0, extra, ras_zarr.dtype, b1-b0)
            tasks.append(([Chunk(ras_path, (*sl, *(slice(0, n) for n in extra))), ras_chunk_order_gix[b0:b1]],
                          [Chunk(str(pc_path/f'{j}.zarr'))]))
    logger.info(f'{len(tasks)} raster chunks with points')
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        ex.map_chunks(mr.ras2pc, tasks, desc='chunks')
    logger.info('done.')

@mc_logger
def pc2ras(
    idx:str,
    pc:str|list,
    ras:str|list,
    shape:tuple[int,int],
    chunks:tuple[int,int]=(1000,1000),
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Convert point cloud data to raster data, filled with nan

    Parameters
    ----------
    idx : str
        input: grid index or hillbert index of the point cloud
    pc : str | list
        input: path or list of paths of point cloud data, shape (n_points, ...)
    ras : str | list
        output, path (in string) or list of path for raster data
    shape : tuple[int, int]
        shape of one image (nlines,width)
    chunks : tuple[int, int], default: (1000, 1000)
        output chunk size
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    idx_zarr = zarr.open(idx,mode='r'); logger.zarr_info(idx,idx_zarr)
    if idx_zarr.ndim == 2:
        logger.info('loading gix into memory.')
        gix = idx_zarr[:]
    else:
        logger.info('loading hix into memory and convert to gix')
        assert shape is not None, "shape not provided for hillbert index input"
        gix = mr.pc_gix(idx_zarr[:],shape=shape)
    n_pc = gix.shape[0]
    shape = tuple(shape); chunks = tuple(chunks)
    if isinstance(pc,str):
        assert isinstance(ras,str)
        pc_list = [pc]; ras_list = [ras]
    else:
        assert isinstance(pc,list); assert isinstance(ras,list)
        pc_list = pc; ras_list = ras
    # one task per channel of every point cloud: the values of the points in, the image out
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        gix_ref = ex.put(gix)
        tasks = []
        for ras_path, pc_path in zip(ras_list,pc_list):
            pc_zarr = zarr.open(pc_path,mode='r'); logger.zarr_info(pc_path,pc_zarr)
            extra = pc_zarr.shape[1:]
            ras_zarr = zarr.open(ras_path,mode='w',shape=(*shape,*extra),dtype=pc_zarr.dtype,chunks=(*chunks,*(1,)*len(extra)))
            logger.zarr_info(ras_path, ras_zarr)
            for k in _channels(extra):
                tasks.append(([Chunk(pc_path, (slice(0, n_pc), *_column(k))), gix_ref, shape],
                              [Chunk(ras_path, (slice(0, shape[0]), slice(0, shape[1]), *_column(k)))]))
        ex.map_chunks(_pc2ras, tasks, desc='channels')
    logger.info('done.')

@mc_logger
def pc_hix(
    gix:str,
    hix:str,
    shape:tuple[int,int],
):
    """Compute the hillbert index from grid index for point cloud data.

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int
    hix : str
        output: hillbert index of the point cloud, shape (n_points,), int64
    shape : tuple[int, int]
        raster shape (nlines, width)
    """
    logger = logging.getLogger(__name__)
    gix_zarr = zarr.open(gix,mode='r'); logger.zarr_info(gix, gix_zarr)
    hix_zarr = zarr.open(hix, mode='w', chunks=gix_zarr.chunks[0], dtype=np.int64, shape=gix_zarr.shape[0])
    logger.zarr_info(hix, hix_zarr)
    logger.info('calculating the hillbert index based on grid index')
    hix_data = mr.pc_hix(gix_zarr[:],shape=shape)
    logger.info("writing the hillbert index")
    hix_zarr[:] = hix_data
    logger.info("done.")

@mc_logger
def pc_gix(
    hix:str,
    gix:str,
    shape:tuple[int,int],
):
    """Compute the grid index from hillbert index for point cloud data.

    Parameters
    ----------
    hix : str
        input: hillbert index of the point cloud, shape (n_points,), int64
    gix : str
        output: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int
    shape : tuple[int, int]
        raster shape (nlines, width)
    """
    logger = logging.getLogger(__name__)
    hix_zarr = zarr.open(hix,mode='r'); logger.zarr_info(hix, hix_zarr)
    gix_zarr = zarr.open(gix, mode='w', chunks=(hix_zarr.chunks[0],1), dtype=np.int32, shape=(hix_zarr.shape[0],2))
    logger.zarr_info(gix, gix_zarr)
    logger.info('calculating the grid index from hillbert index')
    gix_data = mr.pc_gix(hix_zarr[:],shape=shape)
    logger.info("writing the grid index")
    gix_zarr[:] = gix_data
    logger.info("done.")

@mc_logger
def pc_sort(
    idx_in:str,
    idx:str,
    pc_in:str|list=None,
    pc:str|list=None,
    shape:tuple[int,int]=None,
    chunks:int=None,
    key:str=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Sort point cloud data according to the indices that sort `idx_in`.

    Parameters
    ----------
    idx_in : str
        input: unsorted grid index or hillbert index of the input data
    idx : str
        output, the sorted grid index or hillbert index
    pc_in : str | list, optional
        input: path or list of paths of the input point cloud data
    pc : str | list, optional
        output, path (in string) or list of path for the output point cloud data
    shape : tuple[int, int], optional
        (nline, width), faster if provided for grid index input
    chunks : int, optional
        chunk size in output data, same as `idx_in` by default
    key : str, optional
        output, path (in string) for the key of sorting
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    idx_in_path = idx_in
    logger = logging.getLogger(__name__)
    idx_in_zarr = zarr.open(idx_in_path,mode='r'); logger.zarr_info(idx_in_path,idx_in_zarr)
    logger.info('loading idx_in and calculate the sorting indices.')
    idx_in = idx_in_zarr[:]; iidx = mr.pc_sort(idx_in, shape=shape)
    n_pc = idx_in_zarr.shape[0]
    if chunks is None: chunks = idx_in_zarr.chunks[0]
    logger.info(f'output pc chunk size is {chunks}')
    idx_chunk_size = (chunks,1) if idx_in.ndim == 2 else (chunks,)
    idx_zarr = zarr.open(idx,mode='w', shape=idx_in_zarr.shape, dtype=idx_in.dtype, chunks=idx_chunk_size)
    logger.info('write idx'); logger.zarr_info('idx', idx_zarr)
    idx_zarr[:] = idx_in[iidx]
    if key is not None:
        logger.info('saving key for this sorting')
        key_zarr = zarr.open(key,mode='w',shape=iidx.shape,dtype=iidx.dtype,chunks=(chunks,))
        key_zarr[:] = iidx
    if pc_in is None:
        logger.info('no point cloud data provided, exit.')
        return None
    if isinstance(pc_in,str):
        assert isinstance(pc,str)
        pc_in_list = [pc_in]; pc_list = [pc]
    else:
        assert isinstance(pc_in,list); assert isinstance(pc,list)
        pc_in_list = pc_in; pc_list = pc
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        _gather_channels(ex, _indexing_pc_data, [[p] for p in pc_in_list], pc_list, [ex.put(iidx)], n_pc, chunks, 'channels', logger)
    logger.info('done.')

@mc_logger
def pc_union(
    idx1:str,
    idx2:str,
    idx:str,
    pc1:str|list=None,
    pc2:str|list=None,
    pc:str|list=None,
    shape:tuple[int,int]=None,
    chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Get the union of two point cloud datasets. Points in both keep the data of the first point
    cloud.

    Parameters
    ----------
    idx1 : str
        input: grid index or hillbert index of the first point cloud, sorted
    idx2 : str
        input: grid index or hillbert index of the second point cloud, sorted
    idx : str
        output: grid index or hillbert index of the union
    pc1 : str | list, optional
        input: path or list of paths of the first point cloud data
    pc2 : str | list, optional
        input: path or list of paths of the second point cloud data
    pc : str | list, optional
        output: path or list of paths of the point cloud data of the union
    shape : tuple[int, int], optional
        image shape, faster if provided for grid index input
    chunks : int, optional
        point chunk size of the output data, same as `idx1` by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    idx1_zarr = zarr.open(idx1,mode='r'); logger.zarr_info(idx1,idx1_zarr)
    idx2_zarr = zarr.open(idx2,mode='r'); logger.zarr_info(idx2,idx2_zarr)
    logger.info('loading idx1 and idx2 into memory.')
    idx1 = idx1_zarr[:]; idx2 = idx2_zarr[:]
    logger.info('calculate the union')
    idx_path = idx
    idx, inv_iidx1, inv_iidx2, iidx2 = mr.pc_union(idx1,idx2,shape=shape)
    n_pc = idx.shape[0]
    logger.info(f'number of points in the union: {n_pc}')
    if chunks is None: chunks = idx1_zarr.chunks[0]
    idx_chunk_size = (chunks,1) if idx.ndim == 2 else (chunks,)
    idx_zarr = zarr.open(idx_path,mode='w',shape=idx.shape,dtype=idx.dtype,chunks=idx_chunk_size)
    logger.info('write union idx')
    idx_zarr[:] = idx
    logger.info('write done')
    logger.zarr_info(idx_path, idx_zarr)
    if pc1 is None:
        logger.info('no point cloud data provided, exit.')
        return None
    if isinstance(pc1,str):
        assert isinstance(pc2,str); assert isinstance(pc,str)
        pc1_list = [pc1]; pc2_list = [pc2]; pc_list = [pc]
    else:
        assert isinstance(pc1,list); assert isinstance(pc2,list); assert isinstance(pc,list)
        pc1_list = pc1; pc2_list = pc2; pc_list = pc
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        refs = [ex.put(inv_iidx1), ex.put(inv_iidx2), ex.put(iidx2), n_pc]
        _gather_channels(ex, _pc_union, [[a, b] for a, b in zip(pc1_list, pc2_list)], pc_list, refs, n_pc, chunks, 'channels', logger)
    logger.info('done.')

@mc_logger
def pc_intersect(
    idx1:str,
    idx2:str,
    idx:str,
    pc1:str|list=None,
    pc2:str|list=None,
    pc:str|list=None,
    shape:tuple[int,int]=None,
    chunks:int=None,
    prefer_1=True,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Get the intersection of two point cloud datasets.

    Parameters
    ----------
    idx1 : str
        input: grid index or hillbert index of the first point cloud, sorted
    idx2 : str
        input: grid index or hillbert index of the second point cloud, sorted
    idx : str
        output: grid index or hillbert index of the intersection
    pc1 : str | list, optional
        input: path or list of paths of the first point cloud data
    pc2 : str | list, optional
        input: path or list of paths of the second point cloud data
    pc : str | list, optional
        output: path or list of paths of the point cloud data of the intersection, taken from `pc1` or
        `pc2` (see `prefer_1`)
    shape : tuple[int, int], optional
        image shape, faster if provided for grid index input
    chunks : int, optional
        point chunk size of the output data, same as `idx1` by default
    prefer_1 : bool, default: True
        take the output data from `pc1` (True) or from `pc2` (False)
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    idx1_zarr = zarr.open(idx1,mode='r'); logger.zarr_info(idx1,idx1_zarr)
    idx2_zarr = zarr.open(idx2,mode='r'); logger.zarr_info(idx2,idx2_zarr)
    logger.info('loading idx1 and idx2 into memory.')
    idx1 = idx1_zarr[:]; idx2 = idx2_zarr[:]
    logger.info('calculate the intersection')
    idx_path = idx
    idx, iidx1, iidx2 = mr.pc_intersect(idx1,idx2,shape=shape)
    n_pc = idx.shape[0]
    logger.info(f'number of points in the intersection: {n_pc}')
    if chunks is None: chunks = idx1_zarr.chunks[0]
    idx_chunk_size = (chunks,1) if idx.ndim == 2 else (chunks,)
    idx_zarr = zarr.open(idx_path,mode='w',shape=idx.shape,dtype=idx.dtype,chunks=idx_chunk_size)
    logger.info('write intersect idx')
    idx_zarr[:] = idx
    logger.info('write done')
    logger.zarr_info(idx_path, idx_zarr)
    if (pc1 is None) and (pc2 is None):
        logger.info('no point cloud data provided, exit.')
        return None
    if prefer_1:
        logger.info('select pc1 as pc_input.')
        iidx = iidx1; pc_input = pc1
    else:
        logger.info('select pc2 as pc_input.')
        iidx = iidx2; pc_input = pc2
    if isinstance(pc_input,str):
        assert isinstance(pc,str)
        pc_input_list = [pc_input]; pc_list = [pc]
    else:
        assert isinstance(pc_input,list); assert isinstance(pc,list)
        pc_input_list = pc_input; pc_list = pc
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        _gather_channels(ex, _indexing_pc_data, [[p] for p in pc_input_list], pc_list, [ex.put(iidx)], n_pc, chunks, 'channels', logger)
    logger.info('done.')

@mc_logger
def pc_diff(
    idx1:str,
    idx2:str,
    idx:str,
    pc1:str|list=None,
    pc:str|list=None,
    shape:tuple[int,int]=None,
    chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
           ):
    """Get the points of the first point cloud dataset that are not in the second one.

    Parameters
    ----------
    idx1 : str
        input: grid index or hillbert index of the first point cloud, sorted
    idx2 : str
        input: grid index or hillbert index of the second point cloud, sorted
    idx : str
        output: grid index or hillbert index of the points in `idx1` but not in `idx2`
    pc1 : str | list, optional
        input: path or list of paths of the first point cloud data
    pc : str | list, optional
        output: path or list of paths of the point cloud data of these points, taken from `pc1`
    shape : tuple[int, int], optional
        image shape, faster if provided for grid index input
    chunks : int, optional
        point chunk size of the output data, same as `idx1` by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    idx1_zarr = zarr.open(idx1,mode='r'); logger.zarr_info(idx1,idx1_zarr)
    idx2_zarr = zarr.open(idx2,mode='r'); logger.zarr_info(idx2,idx2_zarr)
    logger.info('loading idx1 and idx2 into memory.')
    idx1 = idx1_zarr[:]; idx2 = idx2_zarr[:]
    logger.info('calculate the diff.')
    idx_path = idx
    idx, iidx1 = mr.pc_diff(idx1,idx2,shape=shape)
    n_pc = idx.shape[0]
    logger.info(f'number of points in the diff: {n_pc}')
    if chunks is None: chunks = idx1_zarr.chunks[0]
    idx_chunk_size = (chunks,1) if idx.ndim == 2 else (chunks,)
    idx_zarr = zarr.open(idx_path,mode='w',shape=idx.shape,dtype=idx.dtype,chunks=idx_chunk_size)
    logger.info('write intersect idx')
    idx_zarr[:] = idx
    logger.info('write done')
    logger.zarr_info(idx_path, idx_zarr)
    if pc1 is None:
        logger.info('no point cloud data provided, exit.')
        return None
    if isinstance(pc1,str):
        assert isinstance(pc,str)
        pc1_list = [pc1]; pc_list = [pc]
    else:
        assert isinstance(pc1,list); assert isinstance(pc,list)
        pc1_list = pc1; pc_list = pc
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        _gather_channels(ex, _indexing_pc_data, [[p] for p in pc1_list], pc_list, [ex.put(iidx1)], n_pc, chunks, 'channels', logger)
    logger.info('done.')

@mc_logger
def pc_logic_ras(ras,
                 gix,
                 operation:str,
                 chunks:int=100000,
                ):
    """generate point cloud index based on logical operation of one raster image.

    Parameters
    ----------
    ras
        input: raster used in `operation`, shape (nlines, width)
    gix
        output: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int, of the pixels
        where `operation` is True
    operation : str
        numexpr expression on the raster, which is named `ras`, e.g. '(ras>=0)&(ras<=0.3)'
    chunks : int, default: 100000
        chunk size in output data, optional
    """
    gix_path = gix
    logger = logging.getLogger(__name__)
    ras_zarr = zarr.open(ras, mode='r'); logger.zarr_info(ras,ras_zarr)

    ras = ras_zarr[:]; logger.info('loading ras into memory.')
    is_pc = ne.evaluate(operation,{'ras':ras})
    logger.info(f'select pc based on operation: {operation}')
    gix = np.stack(np.where(is_pc),axis=-1).astype(np.int32)
    n_pc = gix.shape[0]
    logger.info(f'number of selected pixels: {n_pc}.')

    gix_zarr = zarr.open(gix_path,mode='w',dtype=gix.dtype,shape=gix.shape,chunks=(chunks,1))
    logger.zarr_info(gix_path, gix_zarr)
    logger.info('writing gix.')
    gix_zarr[:] = gix
    logger.info('write done.')

@mc_logger
def pc_logic_pc(idx_in:str,
                pc_in:str,
                idx:str,
                operation:str,
                chunks:int=None,
               ):
    """generate point cloud index and data based on logical operation one point cloud data.

    Parameters
    ----------
    idx_in : str
        input: grid index or hillbert index of the input point cloud
    pc_in : str
        input: point cloud data used in `operation`, shape (n_points,)
    idx : str
        output: grid index or hillbert index of the points where `operation` is True
    operation : str
        numexpr expression on the point cloud data, which is named `pc_in`, e.g.
        '(pc_in>=0.1)&(pc_in<=0.5)'
    chunks : int, optional
        point chunk size of the output data, same as `idx_in` by default
    """
    idx_path = idx
    logger = logging.getLogger(__name__)
    idx_in_zarr = zarr.open(idx_in,mode='r'); logger.zarr_info(idx_in,idx_in_zarr)
    pc_in_zarr = zarr.open(pc_in, mode='r'); logger.zarr_info(pc_in,pc_in_zarr)

    idx_in = idx_in_zarr[:]; logger.info('loading idx_in into memory.')
    pc_in = pc_in_zarr[:]; logger.info('loading pc_in into memory.')

    is_pc = ne.evaluate(operation,{'pc_in':pc_in})
    logger.info(f'select pc based on operation: {operation}')
    idx = idx_in[is_pc]
    n_pc = idx.shape[0]
    logger.info(f'number of selected pixels: {n_pc}.')
    if chunks is None: chunks = idx_in_zarr.chunks[0] 
    idx_chunk_size = (chunks,1) if idx.ndim == 2 else (chunks,)
    idx_zarr = zarr.open(idx_path,mode='w',shape=idx.shape,dtype=idx.dtype,chunks=idx_chunk_size)
    logger.zarr_info('idx', idx_zarr)
    logger.info('writing idx.')
    idx_zarr[:] = idx
    logger.info('write done.')

@mc_logger
def pc_select_data(
    idx_in:str,
    idx:str,
    pc_in:str|list,
    pc:str|list,
    shape:tuple[int,int]=None,
    chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """generate point cloud data based on its index and one point cloud data.
    The index of generated point cloud data must in the index of the old one.

    Parameters
    ----------
    idx_in : str
        input: grid index or hillbert index of the input point cloud, sorted
    idx : str
        input: grid index or hillbert index of the points to select, a subset of `idx_in`, sorted
    pc_in : str | list
        input: path or list of paths of the input point cloud data
    pc : str | list
        output, path (in string) or list of path for the output point cloud data
    shape : tuple[int, int], optional
        shape of the raster data the point cloud from, must be provided if `idx` is hix
    chunks : int, optional
        point chunk size of the output data, same as `idx` by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    idx_in_path = idx_in; idx_path = idx
    logger = logging.getLogger(__name__)
    idx_in_zarr = zarr.open(idx_in_path,mode='r'); logger.zarr_info(idx_in_path,idx_in_zarr)
    idx_zarr = zarr.open(idx_path,mode='r'); logger.zarr_info(idx_path,idx_zarr)
    logger.info('loading idx_in and idx into memory.')
    idx_in = idx_in_zarr[:]; idx = idx_zarr[:]
    if idx_in.ndim == idx.ndim:
        iidx_in, iidx = mr.pc_intersect(idx_in,idx,shape)[1:]
    elif (idx_in.ndim == 2) and (idx.ndim == 1):
        hix_in_unsorted = mr.pc_hix(idx_in, shape)
        iidx_in, iidx = np.intersect1d(hix_in_unsorted, idx, assume_unique=True, return_indices=True)[1:]
    else:
        raise NotImplementedError('idx_in as hilbert index while idx as grid index have not been supported yet.')
    np.testing.assert_array_equal(iidx,np.arange(iidx.shape[0]),err_msg='idx have points that are not covered by idx_in.')
    n_pc = iidx_in.shape[0]
    if chunks is None: chunks = idx_zarr.chunks[0]
    if isinstance(pc_in,str):
        assert isinstance(pc,str)
        pc_in_list = [pc_in]; pc_list = [pc]
    else:
        assert isinstance(pc_in,list); assert isinstance(pc,list)
        pc_in_list = pc_in; pc_list = pc
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        _gather_channels(ex, _indexing_pc_data, [[p] for p in pc_in_list], pc_list, [ex.put(iidx_in)], n_pc, chunks, 'channels', logger)
    logger.info('done.')

@mc_logger
def data_reduce(
    data_in:str,
    out:str,
    map_func:Callable=None,
    reduce_func:Callable=np.mean,
    axis=0,
    post_map_func:Callable=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """reduction operation for dataset.

    Parameters
    ----------
    data_in : str
        path (in string) for the input data
    out : str
        output, path (in string) for the output data
    map_func : Callable, optional
        elementwise mapping function for input, no mapping by default
    reduce_func : Callable, default: np.mean
        reduction function after mapping, np.mean by default
    axis : default: 0
        axis to be reduced, 0 for point cloud data, (0,1) for raster data
    post_map_func : Callable, optional
        post mapping after reduction, no mapping by default
    processes : default: False
        use process for dask worker or thread
    n_workers : default: 1
        number of dask worker
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    logger = logging.getLogger(__name__)
    data_in_zarr = zarr.open(data_in,mode='r'); logger.zarr_info(data_in, data_in_zarr)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    slices = all_chunk_slices(data_in_zarr.shape, data_in_zarr.chunks)
    n_blocks = [-(-n // c) for n, c in zip(data_in_zarr.shape, data_in_zarr.chunks)]
    logger.info(f'reduction of {len(slices)} chunks')
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes, **dask_cluster_arg) as ex:
        parts = ex.map(_reduce_chunk, [(Chunk(data_in, sl), map_func, reduce_func, axes) for sl in slices], desc='chunks')
    # the reductions of the chunks in the chunk grid, then the reduction over the chunks
    grid = np.empty(n_blocks, dtype=object)
    for pos, part in zip(itertools.product(*[range(n) for n in n_blocks]), parts):
        grid[pos] = part
    reduced_result = np.block(grid.tolist())
    logger.info('continue the reduction on reduced data over every chunk')
    reduced_result = reduce_func(reduced_result,axis=axis,keepdims=False)
    logger.info('post mapping')
    if post_map_func is not None:
        result = post_map_func(reduced_result)
    else:
        result = reduced_result
    logger.info('writing output.')
    shape = result.shape
    if len(shape) == 0:
        shape = (1,)
    out_zarr = zarr.open(out,mode='w',shape=shape,dtype=result.dtype,chunks=shape)
    out_zarr[:] = result
    logger.info('done.')
