"""Spatially Homogenious Pixels Identification"""


__all__ = ['shp_test', 'select_shp']

from itertools import product
import math
import logging
import time

import zarr
import numcodecs
import numpy as np

import dask
from dask import array as da
from dask import delayed
from dask.distributed import Client, LocalCluster, progress
from ..api.utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
    from dask_cuda import LocalCUDACluster
    from rmm.allocators.cupy import rmm_cupy_allocator
import moraine as mr
from .logging import mc_logger
from . import dask_from_zarr,dask_from_zarr_overlap, dask_to_zarr

@mc_logger
def shp_test(
    rslc:str,
    pvalue:str,
    az_half_win:int,
    r_half_win:int,
    method:str=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """SHP identification through hypothetic test.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    pvalue : str
        output: p value of the test between each pixel and the pixels in its window, shape (nlines,
        width, 2*az_half_win+1, 2*r_half_win+1)
    az_half_win : int
        azimuth half window size
    r_half_win : int
        range half window size
    method : str, optional
        test method, only 'ks' (two-sample Kolmogorov-Smirnov) is implemented. Default: 'ks'
    chunks : tuple[int, int], optional
        (azimuth, range) processing chunk size, same as rslc by default
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the dask workers, only for cpu processing. Default:
        False
    n_workers : optional
        number of dask workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        number of threads per dask worker, only for cpu processing. Default: 1
    rmm_pool_size : default: 0.9
        set the rmm pool size, only applied when cuda==True
    **dask_cluster_arg
        other dask local/cudalocal cluster args
    """
    rslc_path = rslc
    pvalue_path = pvalue

    logger = logging.getLogger(__name__)
    if not method: method = 'ks'
    logger.info(f'hypothetic test method: {method}')
    if method != 'ks':
        logger.warning('Currently only KS test is implented. Switching to it.')
        method = 'ks'

    rslc_zarr = zarr.open(rslc_path,mode='r')
    logger.zarr_info(rslc_path,rslc_zarr)

    assert rslc_zarr.ndim == 3, " rslcs dimentation is not 3."

    if chunks is None: chunks = rslc_zarr.chunks[:2]
    chunks=(*chunks,*rslc_zarr.shape[2:])
    if cuda:
        Cluster = LocalCUDACluster; cluster_args= {
            'n_workers':n_workers,
            'rmm_pool_size':rmm_pool_size}
        cluster_args.update(dask_cluster_arg)
        xp = cp
    else:
        if processes is None: processes = False
        if n_workers is None: n_workers = 1
        if threads_per_worker is None: threads_per_worker = 1
        Cluster = LocalCluster; cluster_args = {'processes':processes, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
        cluster_args.update(dask_cluster_arg)
        xp = np

    logger.info('starting dask local cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda: client.run(cp.cuda.set_allocator, rmm_cupy_allocator)

        az_win = 2*az_half_win+1
        logger.info(f'azimuth half window size: {az_half_win}; azimuth window size: {az_win}')
        r_win = 2*r_half_win+1
        logger.info(f'range half window size: {r_half_win}; range window size: {r_win}')
        depth = (az_half_win, r_half_win, 0); boundary = {0:'none',1:'none',2:'none'}

        cpu_rslc_overlap = dask_from_zarr_overlap(rslc_path,chunks=chunks,depth=depth)
        logger.darr_info('rslc with overlap', cpu_rslc_overlap)

        if cuda:
            rslc_overlap = cpu_rslc_overlap.map_blocks(xp.asarray)
        else:
            rslc_overlap = cpu_rslc_overlap
        rmli_overlap = rslc_overlap.map_blocks(mr.rslc2amp)
        p_chunks = (*rmli_overlap.chunks[:2],(az_win,),(r_win,))
        logger.info('applying test on rmli stack.')
        p = rmli_overlap.map_blocks(mr.ks_test,az_half_win=az_half_win,r_half_win=r_half_win,
                                    new_axis=-1,chunks=p_chunks,meta=xp.array((),dtype=rmli_overlap.dtype))
        logger.info('trim shared boundaries between p value chunks')
        p = da.overlap.trim_overlap(p,depth=depth,boundary=boundary)
        if cuda:
            cpu_p = p.map_blocks(xp.asnumpy)
        else:
            cpu_p = p
        logger.darr_info('p value', cpu_p)

        logger.info('saving p value.')
        _p = dask_to_zarr(cpu_p,pvalue_path,chunks=(*cpu_p.chunksize[:2],1,1))
        # _p = da.to_zarr(cpu_p,pvalue_path,compute=False,overwrite=True)
        # p_zarr = kvikio.zarr.open_cupy_array(pvalue_path,'w',shape=p.shape, chunks=p.chunksize, dtype=p.dtype,compressor=None)
        # _p = da.store(p,p_zarr,compute=False,lock=False)

        logger.info('computing graph setted. doing all the computing.')
        #_p.visualize(filename='_p.svg',color='order',cmap="autumn",optimize_graph=True)
        futures = client.persist(_p)

        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

@mc_logger
def select_shp(
    pvalue:str,
    is_shp:str,
    shp_num:str,
    p_max:float=0.05,
    chunks:tuple[int,int]=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
    **dask_cluster_arg,
):
    """Select SHP based on pvalue of SHP test.

    Parameters
    ----------
    pvalue : str
        input: pvalue of hypothetic test
    is_shp : str
        output: bool array, True for SHPs, same shape as `pvalue`
    shp_num : str
        output: number of SHPs of each pixel, shape (nlines, width), int32
    p_max : float, default: 0.05
        pixels with p value below `p_max` are SHPs
    chunks : tuple[int, int], optional
        (azimuth, range) processing chunk size, same as `pvalue` by default
    processes : default: False
        use process for dask worker over thread, the default is False
    n_workers : default: 1
        number of dask worker, the default is 1
    threads_per_worker : default: 1
        number of threads per dask worker
    **dask_cluster_arg
        other dask local cluster args
    """
    is_shp_path = is_shp
    shp_num_path = shp_num
    logger = logging.getLogger(__name__)

    p_zarr = zarr.open(pvalue,mode='r'); logger.zarr_info(pvalue, p_zarr)
    assert p_zarr.ndim == 4, " pvalue dimentation is not 4."

    if chunks is None: chunks = p_zarr.chunks[:2]
    chunks=(*chunks,*p_zarr.shape[2:])

    logger.info('starting dask cluster.')
    with LocalCluster(processes=processes,n_workers=n_workers,threads_per_worker=threads_per_worker,**dask_cluster_arg) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)

        p = dask_from_zarr(pvalue,chunks=chunks)
        logger.darr_info('pvalue', p)
        p_delayed = p.to_delayed()
        is_shp_delayed = np.empty_like(p_delayed,dtype=object)
        shp_num_delayed = np.empty_like(p_delayed, dtype=object)

        with np.nditer(p_delayed,flags=['multi_index','refs_ok'], op_flags=['readwrite']) as p_it:
            for p_block in p_it:
                idx = p_it.multi_index
                is_shp_delayed[idx], shp_num_delayed[idx] = delayed(mr.select_shp,pure=True,nout=2)(p_delayed[idx],p_max)
                chunk_shape = p.blocks[idx].shape[:-2]
                is_shp_delayed[idx] = da.from_delayed(is_shp_delayed[idx], shape = (*chunk_shape, *p.shape[2:]), meta = np.array((),dtype=np.bool_))
                shp_num_delayed[idx] = da.from_delayed(shp_num_delayed[idx], shape=chunk_shape, meta = np.array((),dtype=np.int32))
        is_shp = da.block(is_shp_delayed.tolist())
        shp_num = da.block(shp_num_delayed[:,:,0,0].tolist())
        logger.info('selecting SHPs based on pvalue threshold: '+str(p_max))
        logger.darr_info('is_shp', is_shp)

        logger.info('calculate shp_num.')
        logger.darr_info('shp_num',shp_num)

        logger.info('saving is_shp.')
        _is_shp = dask_to_zarr(is_shp, is_shp_path, chunks=(*is_shp.chunksize[0:2],1,1))

        logger.info('saving shp_num.')
        _shp_num = dask_to_zarr(shp_num, shp_num_path, chunks=is_shp.chunksize[0:2])
        logger.info('computing graph setted. doing all the computing.')

        futures = client.persist([_is_shp,_shp_num])
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
