"""Point cloud data utilities (CLI)"""

__all__ = ['gix2bool', 'bool2gix', 'ras2pc', 'pc_concat', 'ras2pc_ras_chunk', 'pc2ras', 'pc_hix', 'pc_gix', 'pc_sort', 'pc_union',
           'pc_intersect', 'pc_diff', 'pc_logic_ras', 'pc_logic_pc', 'pc_select_data', 'data_reduce']

import logging
import glob
from pathlib import Path
import zarr
import numpy as np
import numexpr as ne
import time
from typing import Callable

import dask
from dask import array as da
from dask.distributed import Client, LocalCluster, progress

from .logging import mc_logger
import moraine as mr
import moraine.cli as mc
from ..api.utils_ import ngjit
from . import mk_clean_dir, dask_to_zarr, dask_from_zarr

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
        n_data = len(ras_list)

    shape = zarr.open(ras_list[0],mode='r').shape[:2]

    idx_zarr = zarr.open(idx,mode='r'); logger.zarr_info(idx,idx_zarr)
    if idx_zarr.ndim == 2:
        if chunks is None: chunks = idx_zarr.chunks[0]
        logger.info('loading gix into memory.')
        gix = idx_zarr[:]
    else:
        if chunks is None: chunks = idx_zarr.chunks[0]
        logger.info('loading hix into memory and convert to gix')
        hix = idx_zarr[:]
        gix = mr.pc_gix(hix,shape=shape)

    n_pc = gix.shape[0]

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        _pc_list = ()
        gix_darr = da.from_array(gix,chunks=gix.shape)
        for ras_path, pc_path in zip(ras_list,pc_list):
            logger.info(f'start to slice on {ras_path}')
            ras_zarr = zarr.open(ras_path,mode='r'); logger.zarr_info(ras_path, ras_zarr)
            ras = dask_from_zarr(ras_path,parallel_dims=(0,1)); logger.darr_info('ras',ras)
            pc = da.map_blocks(mr.ras2pc, ras, gix_darr, dtype=ras.dtype, chunks=(n_pc,*ras.chunks[2:]),drop_axis=0)
            logger.darr_info('pc', pc)
            logger.info(f'saving to {pc_path}.')
            _pc = dask_to_zarr(pc,pc_path,chunks=(chunks,*pc.chunksize[1:]))
            #_pc.visualize(filename=f'_pc.svg')
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')

    logger.info('dask cluster closed.')

def _indexing_pc_data(pc_in,iidx):
    return pc_in[iidx]

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

    for one_pcs_path in pcs_path:
        n_pc_file = len(one_pcs_path)
        zarr_0 = zarr.open(one_pcs_path[0],mode='r')
        for i in range(1,n_pc_file):
            zarr_ = zarr.open(one_pcs_path[i],mode='r')
            assert zarr_.shape[1:] == zarr_0.shape[1:], 'pcs shape mismatch'
    if chunks is None: chunks = zarr_0.chunks[0]

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes,n_workers=n_workers, threads_per_worker=threads_per_worker,
                     **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        futures = []
        for one_pcs_path, one_pc_path in zip(pcs_path,pc_path):
            logger.info(f'read pc from {one_pcs_path}')
            pc = mc.dask_._dask_from_pc_zarr_dir(one_pcs_path)
            logger.darr_info('concatenated pc', pc)
            if key is not None:
                logger.info('sort pc according to key')
                pc = da.map_blocks(_indexing_pc_data,pc,key, dtype=pc.dtype, meta=np.array((),dtype=pc.dtype))
                logger.darr_info('sorted pc',pc)
            pc_chunk = (1,)*(pc.ndim-1)
            logger.info(f'save pc to {one_pc_path}')
            _pc = dask_to_zarr(pc,one_pc_path,chunks=(chunks,*pc_chunk))
            futures.append(_pc)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(futures)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

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
        n_data = len(ras_list)

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
    n_pc = gix.shape[0]

    logger.info('convert gix to the order of ras chunk')
    chunk_idx, chunk_bounds, invert_idx = mr.api.pc._pc_split_by_chunk(gix,chunks,shape)
    pc_chunksize = tuple(np.diff(chunk_bounds))
    sorted_gix = gix[chunk_idx]
    ras_chunk_order_gix = mr.api.pc._gix_ras_chunk(sorted_gix,chunk_bounds, chunks, shape)
    logger.info('save key')
    key_zarr = zarr.open(key,mode='w',dtype=invert_idx.dtype,shape=invert_idx.shape,chunks=gix_zarr.chunks[:1])
    key_zarr[:] = invert_idx

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        _pc_list = ()
        ras2pc_delayed = dask.delayed(mr.ras2pc,pure=True,nout=1)
        gix_darr = da.from_array(ras_chunk_order_gix,chunks=(pc_chunksize,(2,)))
        gix_delayed = gix_darr.to_delayed().reshape(-1)
        for ras_path, pc_path in zip(ras_list,pc_list):
            pc_path = Path(pc_path); mk_clean_dir(pc_path)
            logger.info(f'start to slice on {ras_path}')
            ras_zarr = zarr.open(ras_path,mode='r'); logger.zarr_info(ras_path, ras_zarr)
            ras = dask_from_zarr(ras_path,chunks=(*chunks, *ras_zarr.shape[2:]))
            logger.darr_info('ras',ras)
            ras_delayed = ras.to_delayed().reshape(-1)
            pc_delayed = np.empty_like(ras_delayed,dtype=object)
            for i in range(ras_delayed.shape[0]):
                pc_delayed[i] = ras2pc_delayed(ras_delayed[i], gix_delayed[i])
                pc_delayed[i] = da.from_delayed(pc_delayed[i],shape=(pc_chunksize[i],*ras_zarr.shape[2:]),meta=np.array((),dtype=ras_zarr.dtype))
            _out_shape = (1,)*(ras_zarr.ndim-2)
            pc = da.block(pc_delayed.reshape((-1,*_out_shape)).tolist()) #the empty chunks are automatically removed from here
            pc = pc.rechunk((pc_chunksize,*pc.shape[1:])) # so add them back here
            logger.darr_info('pc', pc)
            logger.info(f'saving to {pc_path}.')
            for j in range(pc.numblocks[0]):
                if pc_chunksize[j] > 0:
                    _pc = dask_to_zarr(pc.blocks[j],pc_path/f'{j}.zarr',chunks=(pc_chunksize[j],*_out_shape),log_zarr=False)
                    _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')

    logger.info('dask cluster closed.')

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
        hix = idx_zarr[:]
        assert shape is not None, "shape not provided for hillbert index input"
        gix = mr.pc_gix(hix,shape=shape)

    n_pc = gix.shape[0]
    if isinstance(pc,str):
        assert isinstance(ras,str)
        pc_list = [pc]; ras_list = [ras]
    else:
        assert isinstance(pc,list); assert isinstance(ras,list)
        pc_list = pc; ras_list = ras

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        _ras_list = ()
        gix_darr = da.from_array(gix,chunks=gix.shape)

        for ras_path, pc_path in zip(ras_list,pc_list):
            logger.info(f'start to work on {pc_path}')
            pc_zarr = zarr.open(pc_path,mode='r')
            logger.zarr_info(pc_path,pc_zarr)

            pc = dask_from_zarr(pc_path,parallel_dims=0)
            logger.darr_info('pc', pc)
            logger.info('create ras dask array')
            ras = da.map_blocks(_pc2ras, pc, gix_darr, shape, dtype=pc.dtype, chunks=(*shape,*pc_zarr.chunks[1:]))
            logger.darr_info('ras', ras)
            logger.info(f'save ras to {ras_path}')
            _ras = dask_to_zarr(ras,ras_path,chunks=(*chunks,*pc_zarr.chunks[1:]))
            _ras_list += (_ras,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_ras_list)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')

    logger.info('dask cluster closed.')

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

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        _pc_list = ()
        for pc_in_path, pc_path in zip(pc_in_list,pc_list):
            pc_in_zarr = zarr.open(pc_in_path,mode='r'); logger.zarr_info(pc_in_path, pc_in_zarr)
            pc_in = dask_from_zarr(pc_in_path,parallel_dims=0)
            logger.darr_info('pc_in', pc_in)
            logger.info('set up sorted pc data dask array.')
            pc = da.map_blocks(_indexing_pc_data, pc_in, iidx, chunks=pc_in.chunks, dtype=pc_in.dtype)
            logger.darr_info('pc',pc)
            logger.info(f'write pc to {pc_path}')
            _pc = dask_to_zarr(pc, pc_path, chunks=(chunks,*pc.chunksize[1:]))
            # _pc.visualize(filename=f'_pc.svg')
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

def _pc_union(pc1,pc2,inv_iidx1,inv_iidx2,iidx2,n_pc):
    pc = np.empty((n_pc,*pc1.shape[1:]),dtype=pc1.dtype)
    pc[inv_iidx1] = pc1
    pc[inv_iidx2] = pc2[iidx2]
    return pc

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

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        _pc_list = ()
        for pc1_path, pc2_path, pc_path in zip(pc1_list,pc2_list,pc_list):
            pc1_zarr = zarr.open(pc1_path,mode='r'); pc2_zarr = zarr.open(pc2_path,mode='r')
            logger.zarr_info(pc1_path, pc1_zarr); logger.zarr_info(pc2_path, pc2_zarr);
            pc1 = dask_from_zarr(pc1_path,parallel_dims=0)
            pc2 = dask_from_zarr(pc2_path,parallel_dims=0)
            logger.darr_info('pc1', pc1); logger.darr_info('pc2',pc2)
            logger.info('set up union pc data dask array.')
            pc = da.map_blocks(_pc_union, pc1,pc2,inv_iidx1,inv_iidx2,iidx2,n_pc, chunks=(n_pc,*pc1.chunks[1:]), dtype=pc1.dtype)
            logger.darr_info('pc',pc)
            logger.info(f'write pc to {pc_path}')
            _pc = dask_to_zarr(pc, pc_path, chunks=(chunks,*pc.chunksize[1:]))
            # pc.visualize(filename=f'pc.svg')
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')

    logger.info('dask cluster closed.')

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

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        iidx_darr = da.from_array(iidx,chunks=iidx.shape)
        _pc_list = ()
        for pc_input_path, pc_path in zip(pc_input_list,pc_list):
            pc_input_zarr = zarr.open(pc_input_path,mode='r')
            logger.zarr_info(pc_input_path,pc_input_zarr)
            pc_input = dask_from_zarr(pc_input_path,parallel_dims=0)
            logger.darr_info('pc_input', pc_input)

            logger.info('set up intersect pc data dask array.')
            pc = da.map_blocks(_indexing_pc_data, pc_input, iidx_darr, chunks = (n_pc,*pc_input.chunks[1:]), dtype=pc_input.dtype)
            logger.darr_info('pc',pc)
            logger.info(f'write pc to {pc_path}')
            _pc = dask_to_zarr(pc,pc_path,chunks=(chunks,*pc.chunksize[1:]))
            #pc.visualize(filename=f'pc.svg')
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

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

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes,n_workers=n_workers, threads_per_worker=threads_per_worker,
                     **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        iidx1_darr = da.from_array(iidx1,chunks=iidx1.shape)

        _pc_list = ()
        for pc1_path, pc_path in zip(pc1_list,pc_list):
            pc1_zarr = zarr.open(pc1_path,mode='r'); logger.zarr_info(pc1_path, pc1_zarr)
            pc1 = dask_from_zarr(pc1_path,parallel_dims=0); logger.darr_info('pc1', pc1)
            logger.info('set up diff pc data dask array.')
            pc = da.map_blocks(_indexing_pc_data, pc1, iidx1_darr, chunks = (n_pc,*pc1.chunks[1:]), dtype=pc1.dtype)
            logger.darr_info('pc',pc)

            logger.info(f'write pc to {pc_path}')
            _pc = dask_to_zarr(pc,pc_path,chunks=(chunks,*pc.chunksize[1:]))
            # pc.visualize(filename=f'pc.svg',optimize_graph=True)
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

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

    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        iidx_in_darr = da.from_array(iidx_in,chunks=iidx_in.shape)

        _pc_list = ()
        for pc_in_path, pc_path in zip(pc_in_list,pc_list):
            pc_in_zarr = zarr.open(pc_in_path,mode='r'); logger.zarr_info(pc_in_path, pc_in_zarr)
            pc_in = dask_from_zarr(pc_in_path,parallel_dims=0); logger.darr_info('pc_in', pc_in)
            logger.info('set up selected pc data dask array.')
            pc = da.map_blocks(_indexing_pc_data, pc_in, iidx_in_darr, chunks = (n_pc, *pc_in.chunks[1:]), dtype=pc_in.dtype)
            logger.darr_info('pc',pc)
            logger.info(f'write pc to {pc_path}')
            _pc = dask_to_zarr(pc,pc_path,chunks=(chunks,*pc.chunksize[1:]))
            _pc_list += (_pc,)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_pc_list)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

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
    data_in_path = data_in
    logger = logging.getLogger(__name__)
    data_in_zarr = zarr.open(data_in_path,mode='r'); logger.zarr_info(data_in_path, data_in_zarr)
    logger.info('starting dask local cluster.')
    with LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker,
                      **dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        data_in = da.from_zarr(data_in_path,inline_array=True); logger.darr_info('data_in', data_in)
        if map_func is not None:
            map_data_in = da.map_blocks(map_func, data_in)
        else:
            map_data_in = data_in
        logger.darr_info('maped_data_in', map_data_in)
        reduced_chunks = np.array(map_data_in.chunksize); reduced_chunks[np.array(axis)] = 1
        reduced_chunks = tuple(reduced_chunks)

        reduced_data = da.map_blocks(reduce_func, map_data_in, axis=axis, keepdims=True, chunks=reduced_chunks)
        logger.darr_info('reduced data in every chunk', reduced_data)
        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(reduced_data)
        progress(futures,notebook=False); time.sleep(0.1)
        reduced_result = da.compute(futures)[0]
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
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
