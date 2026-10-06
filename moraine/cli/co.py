"""Covariance and coherence matrix estimation (CLI)"""

__all__ = ['emperical_co_pc']

import logging
import time
from pathlib import Path

import zarr
import numpy as np
import math

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
import moraine.cli as mc
from .logging import mc_logger
from . import mk_clean_dir, dask_from_zarr, dask_from_zarr_overlap, dask_to_zarr

@mc_logger
def emperical_co_pc(
    rslc:str,
    is_shp_dir:str,
    gix:str,
    coh_dir:str,
    n_looks_dir:str=None,
    image_pairs:np.ndarray=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """estimate emperical coherence matrix on point cloud data.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    is_shp_dir : str
        input: directory with the SHP bool arrays of the points, one zarr per raster chunk, made by
        `ras2pc_ras_chunk`
    gix : str
        input: grid index of the point cloud (azimuth, range), shape (n_points, 2), int
    coh_dir : str
        output: directory with the complex coherence of the image pairs of the points, shape (n_points,
        n_image_pairs), one zarr per raster chunk; merge with `pc_concat` and the key of
        `ras2pc_ras_chunk`
    n_looks_dir : str, optional
        output: directory with the effective number of independent looks of the SHP set of each point,
        float32, shape (n_points,), one zarr per raster chunk (merge as `coh_dir`): the number of
        independent looks with the same variance of the coherence estimate as the correlated SHPs, from
        their positions and the speckle correlation of `rslc`; input `n_looks` of `ds-temp-coh`
    image_pairs : np.ndarray, optional
        input: image pairs (element in the coherence matrix) to be calculated, all image pairs by default
    chunks : tuple[int, int], optional
        parallel processing (azimuth, range) chunk size. Default: rslc.chunks[:2]
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the dask workers, only for cpu processing. Default:
        False
    n_workers : optional
        number of dask workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        number of threads per dask worker, only for cpu processing. Default: 2
    rmm_pool_size : default: 0.9
        set the rmm pool size, only applied when cuda==True
    **dask_cluster_arg
        other dask local/cudalocal cluster args
    """
    rslc_path = rslc
    is_shp_dir_path = Path(is_shp_dir)
    gix_path = gix
    coh_dir = Path(coh_dir); mk_clean_dir(coh_dir)
    if n_looks_dir is not None: n_looks_dir = Path(n_looks_dir); mk_clean_dir(n_looks_dir)
    logger = logging.getLogger(__name__)

    rslc_zarr = zarr.open(rslc_path,mode='r')
    logger.zarr_info(rslc_path, rslc_zarr)
    assert rslc_zarr.ndim == 3, "rslc dimentation is not 3."
    nlines, width, nimage = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]

    is_shp0 = sorted(is_shp_dir_path.glob('*.zarr'))[0]
    is_shp0_zarr = zarr.open(is_shp0,mode='r')
    az_win, r_win = is_shp0_zarr.shape[1:]
    az_half_win = int((az_win-1)/2)
    r_half_win = int((r_win-1)/2)
    logger.info(f'''azimuth window size and half azimuth window size: {az_win}, {az_half_win}''')
    logger.info(f'''range window size and half range window size: {r_win}, {r_half_win}''')

    az_chunk, r_chunk = chunks
    n_az_chunk = math.ceil(nlines/az_chunk)
    n_r_chunk = math.ceil(width/r_chunk)
    logger.info(f'parallel processing azimuth chunk size: {az_chunk}')
    logger.info(f'parallel processing range chunk size: {r_chunk}')

    depth = (az_half_win, r_half_win, 0); boundary = {0:'none',1:'none',2:'none'}
    gix_zarr = zarr.open(gix_path,mode='r')
    logger.zarr_info(gix_path, gix_zarr)
    assert gix_zarr.ndim == 2, "gix dimentation is not 2."
    logger.info('loading gix into memory.')
    gix = mc.parallel_read_zarr(gix_zarr,(slice(None),slice(None)))
    logger.info('convert gix to the order of ras chunk')
    chunk_idx, chunk_bounds = mr.api.pc._pc_split_by_chunk(gix,chunks,(nlines,width))[:2]
    pc_chunksize = tuple(np.diff(chunk_bounds))
    sorted_gix = gix[chunk_idx]
    ras_chunk_order_gix = mr.api.pc._gix_ras_chunk(sorted_gix,chunk_bounds, chunks, (nlines,width),overlap=(az_half_win,r_half_win))

    if cuda:
        Cluster = LocalCUDACluster; cluster_args= {
            'n_workers':n_workers,
            'rmm_pool_size':rmm_pool_size}
        cluster_args.update(dask_cluster_arg)
        xp = cp
    else:
        if processes is None: processes = False
        if n_workers is None: n_workers = 1
        if threads_per_worker is None: threads_per_worker = 2
        Cluster = LocalCluster; cluster_args = {'processes':processes, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
        cluster_args.update(dask_cluster_arg)
        xp = np

    if image_pairs is None:
        tnet = mr.TempNet.from_bandwidth(nimage)
        image_pairs = tnet.image_pairs

    n_image_pairs = image_pairs.shape[0]

    logger.info('starting dask cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda: client.run(cp.cuda.set_allocator, rmm_cupy_allocator)
        return_n_looks = n_looks_dir is not None
        emperical_co_pc_delayed = delayed(mr.emperical_co_pc,pure=True,nout=2 if return_n_looks else 1)

        cpu_rslc_overlap = dask_from_zarr_overlap(rslc_path, (*chunks, rslc_zarr.shape[2]), depth)
        logger.darr_info('rslc_overlap', cpu_rslc_overlap)
        cpu_gix_darr = da.from_array(ras_chunk_order_gix,chunks=(pc_chunksize,(2,)))
        logger.darr_info('gix in ras chunk order', cpu_gix_darr)
        if cuda:
            rslc_overlap = cpu_rslc_overlap.map_blocks(cp.asarray)
            gix_darr = cpu_gix_darr.map_blocks(cp.asarray)
        else:
            rslc_overlap = cpu_rslc_overlap
            gix_darr = cpu_gix_darr
        rslc_overlap_delayed = rslc_overlap.to_delayed().reshape(-1)
        gix_delayed = gix_darr.to_delayed().reshape(-1)

        logger.info(f'estimating coherence matrix chunk by chunk.')
        futures = []
        for j in range(n_az_chunk*n_r_chunk):
            do_log = j%math.ceil(n_az_chunk*n_r_chunk/10) == 0
            # az_chunk_idx = j//n_az_chunk; r_chunk_idx = j%n_az_chunk
            if pc_chunksize[j] > 0:
                cpu_is_shp = mc.dask_from_zarr(is_shp_dir_path/f'{j}.zarr',chunks=(-1,-1,-1))
                if do_log: logger.darr_info(f'is_shp for chunk {j}',cpu_is_shp)
                if cuda:
                    is_shp = cpu_is_shp.map_blocks(cp.asarray)
                else:
                    is_shp = cpu_is_shp
                is_shp_delayed = is_shp.to_delayed()[0,0,0]
                coh_delayed = emperical_co_pc_delayed(rslc_overlap_delayed[j],gix_delayed[j],is_shp_delayed,image_pairs=image_pairs,
                                                      return_n_looks=return_n_looks)
                if return_n_looks:
                    coh_delayed, n_looks_delayed = tuple(coh_delayed)
                    n_looks = da.from_delayed(n_looks_delayed,shape=(pc_chunksize[j],),meta=xp.array((),dtype=xp.float32))
                    if cuda: n_looks = n_looks.map_blocks(cp.asnumpy)
                    futures.append(dask_to_zarr(n_looks,n_looks_dir/f'{j}.zarr',chunks=(n_looks.shape[0],),log_zarr=do_log))
                coh = da.from_delayed(coh_delayed,shape=(pc_chunksize[j],n_image_pairs),meta=xp.array((),dtype=xp.complex64))
                if cuda:
                    cpu_coh = coh.map_blocks(cp.asnumpy)
                else:
                    cpu_coh = coh
                if do_log: logger.darr_info(f'coh for chunk {j}',cpu_coh)
                if do_log: logger.info(f'saving coh for chunk {j}')
                _coh = dask_to_zarr(cpu_coh,coh_dir/f'{j}.zarr',chunks=(coh.shape[0],1),log_zarr=do_log)
                futures.append(_coh)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(futures)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
