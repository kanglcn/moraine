"""Phase linking (CLI)"""

__all__ = ['emi', 'ds_temp_coh', 'emperical_co_emi_temp_coh_pc']

import logging
import time
import zarr
import numpy as np
from pathlib import Path
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
def emi(
    coh:str,
    ph:str,
    emi_quality:str,
    ref:int=0,
    regularize:bool=True,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """Phase linking with EMI estimator.

    Parameters
    ----------
    coh : str
        input: complex coherence of the points (upper triangle of the coherence matrix), shape
        (n_points, n_image_pairs)
    ph : str
        output: phase history of the points, complex, shape (n_points, nimages)
    emi_quality : str
        output: EMI quality (minimum eigenvalue) of the points, shape (n_points,); 1 for a coherence
        matrix whose phases close; with `regularize` (default), regularized points get qualities closer
        to 1 for the same misfit and the qualities are not comparable between points (select DS by the
        temporal coherence instead); without it negative where the coherence magnitude matrix is not
        positive definite (the phase history is then not reliable)
    ref : int, default: 0
        index of the reference image, its phase is set to 0
    regularize : bool, default: True
        regularize the coherence matrix of the points whose coherence magnitude matrix is not positive
        definite or numerically singular (many negative qualities, e.g. when the number of images
        approaches the number of SHPs); points with a well conditioned positive definite matrix are not
        changed
    chunks : int, optional
        point chunk size of the output data, same as `coh` by default
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
    coh_path = coh
    ph_path = ph
    emi_quality_path = emi_quality

    logger = logging.getLogger(__name__)
    coh_zarr = zarr.open(coh_path,mode='r')
    n_image = mr.nimage_from_npair(coh_zarr.shape[-1])
    logger.zarr_info(coh_path,coh_zarr)

    if chunks is None: chunks = coh_zarr.chunks[0] 
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

    logger.info('starting dask cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda: client.run(cp.cuda.set_allocator, rmm_cupy_allocator)

        cpu_coh = dask_from_zarr(coh_path, chunks=(chunks, *coh_zarr.shape[1:]))
        logger.darr_info('coh', cpu_coh)

        logger.info(f'phase linking with EMI.')
        if cuda:
            coh = cpu_coh.map_blocks(cp.asarray)
        else:
            coh = cpu_coh
        coh_delayed = coh.to_delayed()
        coh_delayed = np.squeeze(coh_delayed,axis=-1)

        ph_delayed = np.empty_like(coh_delayed,dtype=object)
        emi_quality_delayed = np.empty_like(coh_delayed,dtype=object)
        emi_delayed = delayed(mr.emi,pure=True,nout=2)

        with np.nditer(coh_delayed,flags=['multi_index','refs_ok'], op_flags=['readwrite']) as it:
            for block in it:
                idx = it.multi_index
                ph_delayed[idx], emi_quality_delayed[idx] = emi_delayed(coh_delayed[idx],ref=ref,regularize=regularize)
                ph_delayed[idx] = da.from_delayed(ph_delayed[idx],shape=(coh.blocks[idx].shape[0],n_image),meta=xp.array((),dtype=coh.dtype))
                emi_quality_delayed[idx] = da.from_delayed(emi_quality_delayed[idx],shape=coh.blocks[idx].shape[0:1],meta=xp.array((),dtype=xp.float32))

        ph = da.block(ph_delayed[...,None].tolist())
        emi_quality = da.block(emi_quality_delayed.tolist())

        if cuda:
            cpu_ph = ph.map_blocks(cp.asnumpy)
            cpu_emi_quality = emi_quality.map_blocks(cp.asnumpy)
        else:
            cpu_ph = ph; cpu_emi_quality = emi_quality
        logger.info(f'got ph and emi_quality.')
        logger.darr_info('ph', cpu_ph)
        logger.darr_info('emi_quality', cpu_emi_quality)

        logger.info('saving ph and emi_quality.')
        _cpu_ph = dask_to_zarr(cpu_ph,ph_path,chunks=(cpu_ph.chunksize[0],1))
        _cpu_emi_quality = dask_to_zarr(cpu_emi_quality,emi_quality_path,chunks=(cpu_emi_quality.chunksize[0]))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist([_cpu_ph,_cpu_emi_quality])
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

@mc_logger
def ds_temp_coh(
    coh:str,
    ph:str,
    t_coh:str=None,
    tnet:str=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """DS temporal coherence.

    Parameters
    ----------
    coh : str
        input: complex coherence of the points (upper triangle of the coherence matrix), shape
        (n_points, n_image_pairs)
    ph : str
        input: phase history of the points, complex, shape (n_points, nimages)
    t_coh : str, optional
        output: temporal coherence of the points, shape (n_points,)
    tnet : str, optional
        input: path of a saved `TempNet` with the image pairs of `coh`; all image pairs by default
    chunks : int, optional
        point cloud chunk size, same as coh by default
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
    coh_path = coh
    ph_path = ph
    t_coh_path = t_coh

    logger = logging.getLogger(__name__)
    coh_zarr = zarr.open(coh_path,mode='r'); logger.zarr_info(coh_path,coh_zarr)
    ph_zarr = zarr.open(ph_path,mode='r'); logger.zarr_info(ph_path,ph_zarr)
    nimage = ph_zarr.shape[-1]

    if chunks is None: chunks = coh_zarr.chunks[0] 
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

    if tnet is not None:
        tnet = mr.TempNet.load(tnet)
    else:
        tnet = mr.TempNet.from_bandwidth(nimage)
    image_pairs = tnet.image_pairs

    logger.info('starting dask local cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda: client.run(cp.cuda.set_allocator, rmm_cupy_allocator)

        cpu_coh = dask_from_zarr(coh_path,chunks=(chunks,*coh_zarr.shape[1:]))
        logger.darr_info('coh', cpu_coh)

        cpu_ph = dask_from_zarr(ph_path,chunks=(chunks,*ph_zarr.shape[1:]))
        logger.darr_info('ph', cpu_ph)

        logger.info(f'Estimate temporal coherence for DS.')
        if cuda:
            coh = cpu_coh.map_blocks(cp.asarray)
            ph = cpu_ph.map_blocks(cp.asarray)
        else:
            coh = cpu_coh
            ph = cpu_ph

        coh_delayed = coh.to_delayed()
        coh_delayed = np.squeeze(coh_delayed,axis=-1)
        ph_delayed = ph.to_delayed()
        ph_delayed = np.squeeze(ph_delayed,axis=-1)
        t_coh_delayed = np.empty_like(coh_delayed,dtype=object)
        ds_temp_coh_delayed = delayed(mr.ds_temp_coh,pure=True,nout=1)

        with np.nditer(coh_delayed,flags=['multi_index','refs_ok'], op_flags=['readwrite']) as it:
            for block in it:
                idx = it.multi_index
                t_coh_delayed[idx] = ds_temp_coh_delayed(coh_delayed[idx],ph_delayed[idx],image_pairs=image_pairs)
                t_coh_delayed[idx] = da.from_delayed(t_coh_delayed[idx],shape=coh.blocks[idx].shape[0:1],meta=xp.array((),dtype=xp.float32))

            t_coh = da.block(t_coh_delayed.tolist())

        if cuda:
            cpu_t_coh = t_coh.map_blocks(cp.asnumpy)
        else:
            cpu_t_coh = t_coh
        logger.info(f'got temporal coherence t_coh.')
        logger.darr_info('t_coh', t_coh)

        logger.info('saving t_coh.')
        _cpu_t_coh = cpu_t_coh.to_zarr(t_coh_path,compute=False,overwrite=True)

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_cpu_t_coh)
        progress(futures,notebook=False); time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

@mc_logger
def emperical_co_emi_temp_coh_pc(
    rslc:str,
    is_shp_dir:str,
    gix:str,
    ph_dir:str,
    emi_quality_dir:str,
    t_coh_dir:str,
    t_coh_w_dir:str=None,
    eff_n_pairs_dir:str=None,
    batch_size:int=1000,
    regularize:bool=True,
    oversampling:float=1.0,
    rho2:str=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """estimating emperical coherence matrix, phase linking and estimating temporal coherence on point cloud data.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    is_shp_dir : str
        input: directory with the SHP bool arrays of the points, one zarr per raster chunk, made by
        `ras2pc_ras_chunk`
    gix : str
        input: grid index of the point cloud (azimuth, range), shape (n_points, 2), int
    ph_dir : str
        output: directory with the phase history of the points, complex, shape (n_points, nimages), one
        zarr per raster chunk; merge with `pc_concat` and the key of `ras2pc_ras_chunk`
    emi_quality_dir : str
        output: directory with the EMI quality of the points, shape (n_points,), one zarr per raster
        chunk; as `emi_quality` of `emi`
    t_coh_dir : str
        output: directory with the temporal coherence of the points, shape (n_points,), one zarr per
        raster chunk
    t_coh_w_dir : str, optional
        output: directory with the weighted temporal coherence of the points (image pairs weighted by
        their squared coherence without the noise bias, see `ds_temp_coh_weighted`), shape (n_points,),
        float32, 0..1, NaN where no image pair is above the noise level, one zarr per raster chunk
    eff_n_pairs_dir : str, optional
        output: directory with the effective number of image pairs of the weighted temporal coherence,
        shape (n_points,), float32, one zarr per raster chunk; use it with `t_coh_w_dir`, a high weighted
        temporal coherence of few effective pairs is not reliable
    batch_size : int, default: 1000
        number of points processed at once, limits the memory use
    regularize : bool, default: True
        regularize the coherence matrix in the phase linking as `regularize` of `emi`; the temporal
        coherence is computed with the coherence matrix as estimated
    oversampling : float, default: 1.0
        number of pixels per independent look of the SLCs (>= 1), e.g. the attribute `oversampling` of the
        output of `slc-correlation` (Sentinel-1 IW: about 2.6-2.8); the effective number of looks of a
        point is its number of SHPs divided by it; only used for `t_coh_w_dir` and `eff_n_pairs_dir`,
        not with `rho2`
    rho2 : str, optional
        input: |rho|^2 of the speckle, the output of `slc-correlation`; with it the effective number of
        looks of each point is computed from the positions of its SHPs (`moraine.shp_n_looks`, more
        accurate for scattered SHPs) instead of number of SHPs / `oversampling`
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
    ph_dir = Path(ph_dir); mk_clean_dir(ph_dir)
    emi_quality_dir = Path(emi_quality_dir); mk_clean_dir(emi_quality_dir)
    t_coh_dir = Path(t_coh_dir); mk_clean_dir(t_coh_dir)
    weighted = (t_coh_w_dir is not None) or (eff_n_pairs_dir is not None)
    rho2_table = None if rho2 is None else np.asarray(zarr.open(rho2,mode='r')[:],dtype=np.float32)
    if t_coh_w_dir is not None: t_coh_w_dir = Path(t_coh_w_dir); mk_clean_dir(t_coh_w_dir)
    if eff_n_pairs_dir is not None: eff_n_pairs_dir = Path(eff_n_pairs_dir); mk_clean_dir(eff_n_pairs_dir)

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

    depth = [az_half_win, r_half_win, 0]; boundary = {0:'none',1:'none',2:'none'}
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

    logger.info('starting dask cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda: client.run(cp.cuda.set_allocator, rmm_cupy_allocator)
        emperical_co_emi_temp_coh_pc_delayed = delayed(mr.emperical_co_emi_temp_coh_pc,pure=True,nout=5 if weighted else 3)

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
            if pc_chunksize[j] > 0:
                cpu_is_shp = mc.dask_from_zarr(is_shp_dir_path/f'{j}.zarr',chunks=(-1,-1,-1))
                if do_log: logger.darr_info(f'is_shp for chunk {j}',cpu_is_shp)
                if cuda:
                    is_shp = cpu_is_shp.map_blocks(cp.asarray)
                else:
                    is_shp = cpu_is_shp
                is_shp_delayed = is_shp.to_delayed()[0,0,0]
                outs = tuple(emperical_co_emi_temp_coh_pc_delayed(rslc_overlap_delayed[j],gix_delayed[j],is_shp_delayed,batch_size=batch_size,
                                                                  regularize=regularize,weighted=weighted,oversampling=oversampling,
                                                                  rho2=rho2_table))
                ph_delayed, emi_quality_delayed, t_coh_delayed = outs[:3]

                ph = da.from_delayed(ph_delayed,shape=(pc_chunksize[j],nimage),meta=xp.array((),dtype=rslc_overlap.dtype))
                emi_quality = da.from_delayed(emi_quality_delayed,shape=(pc_chunksize[j],),meta=xp.array((),dtype=xp.float32))
                t_coh = da.from_delayed(t_coh_delayed,shape=(pc_chunksize[j],),meta=xp.array((),dtype=xp.float32))

                if cuda:
                    cpu_ph = ph.map_blocks(cp.asnumpy)
                    cpu_emi_quality = emi_quality.map_blocks(cp.asnumpy)
                    cpu_t_coh = t_coh.map_blocks(cp.asnumpy)
                else:
                    cpu_ph = ph
                    cpu_emi_quality = emi_quality
                    cpu_t_coh = t_coh

                if do_log:
                    logger.darr_info(f'ph for chunk {j}',cpu_ph)
                    logger.darr_info(f'emi_quality for chunk {j}',cpu_emi_quality)
                    logger.darr_info(f't_coh for chunk {j}',cpu_t_coh)
                    logger.info(f'saving ph, emi_quality, t_coh for chunk {j}')

                _ph = dask_to_zarr(cpu_ph,ph_dir/f'{j}.zarr',chunks=(cpu_ph.shape[0],1),log_zarr=do_log)
                _emi_quality = dask_to_zarr(cpu_emi_quality,emi_quality_dir/f'{j}.zarr',chunks=(cpu_emi_quality.shape[0],),log_zarr=do_log)
                _t_coh = dask_to_zarr(cpu_t_coh,t_coh_dir/f'{j}.zarr',chunks=(cpu_t_coh.shape[0],),log_zarr=do_log)

                futures.extend((_ph,_emi_quality,_t_coh))
                for out_delayed, out_dir in zip(outs[3:], (t_coh_w_dir, eff_n_pairs_dir)):
                    if out_dir is None: continue
                    out = da.from_delayed(out_delayed,shape=(pc_chunksize[j],),meta=xp.array((),dtype=xp.float32))
                    if cuda: out = out.map_blocks(cp.asnumpy)
                    futures.append(dask_to_zarr(out,out_dir/f'{j}.zarr',chunks=(out.shape[0],),log_zarr=do_log))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(futures)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
