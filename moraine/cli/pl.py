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
    ref : int, default: 0
        index of the reference image, its phase is set to 0
    regularize : bool, default: True
        regularize the coherence matrix of the points whose coherence magnitude matrix is not positive
        definite or numerically singular (e.g. when the number of images approaches the number of
        independent looks of the SHPs; without it their phase history is not reliable); points with a
        well conditioned positive definite matrix are not changed
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
        emi_delayed = delayed(mr.emi,pure=True,nout=1)

        with np.nditer(coh_delayed,flags=['multi_index','refs_ok'], op_flags=['readwrite']) as it:
            for block in it:
                idx = it.multi_index
                ph_delayed[idx] = emi_delayed(coh_delayed[idx],ref=ref,regularize=regularize)
                ph_delayed[idx] = da.from_delayed(ph_delayed[idx],shape=(coh.blocks[idx].shape[0],n_image),meta=xp.array((),dtype=coh.dtype))

        ph = da.block(ph_delayed[...,None].tolist())

        if cuda:
            cpu_ph = ph.map_blocks(cp.asnumpy)
        else:
            cpu_ph = ph
        logger.info(f'got ph.')
        logger.darr_info('ph', cpu_ph)

        logger.info('saving ph.')
        _cpu_ph = dask_to_zarr(cpu_ph,ph_path,chunks=(cpu_ph.chunksize[0],1))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist([_cpu_ph])
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
    n_looks:str=None,
    t_coh_w:str=None,
    eff_n_pairs:str=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """DS temporal coherence, optionally also weighted by the coherence of the image pairs.

    Parameters
    ----------
    coh : str
        input: complex coherence of the points (upper triangle of the coherence matrix), shape
        (n_points, n_image_pairs)
    ph : str
        input: phase history of the points, complex, shape (n_points, nimages)
    t_coh : str, optional
        output: temporal coherence of the points, every image pair weighted the same, shape (n_points,)
    tnet : str, optional
        input: path of a saved `TempNet` with the image pairs of `coh`; all image pairs by default
    n_looks : str, optional
        input: effective number of independent looks of the coherence of the points, float32, shape
        (n_points,), in the order of `coh`: the merged `n_looks_dir` of `emperical-co-pc`; needed by
        `t_coh_w` and `eff_n_pairs`
    t_coh_w : str, optional
        output: weighted temporal coherence of the points, float32, shape (n_points,), 0..1: the image
        pairs weighted by their squared coherence without the noise bias, so that incoherent pairs (e.g.
        long time spans in vegetation) do not lower it; NaN where no image pair is above the noise level
    eff_n_pairs : str, optional
        output: effective number of image pairs the weighted temporal coherence rests on, float32, shape
        (n_points,), 0..n_image_pairs; a high weighted temporal coherence of few effective pairs is not
        reliable
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
    n_looks_path = n_looks
    weighted = (t_coh_w is not None) or (eff_n_pairs is not None)
    if weighted and n_looks_path is None:
        raise ValueError('t_coh_w and eff_n_pairs need n_looks')

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

        coh_delayed = np.squeeze(coh.to_delayed(),axis=-1)
        ph_delayed = np.squeeze(ph.to_delayed(),axis=-1)
        if weighted:
            cpu_n_looks = dask_from_zarr(n_looks_path,chunks=(chunks,))
            logger.darr_info('n_looks', cpu_n_looks)
            n_looks_delayed = (cpu_n_looks.map_blocks(cp.asarray) if cuda else cpu_n_looks).to_delayed()
        n_out = 3 if weighted else 1
        ds_temp_coh_delayed = delayed(mr.ds_temp_coh,pure=True,nout=n_out)

        outs_blocks = [[] for _ in range(n_out)]
        for idx in range(coh_delayed.shape[0]):
            kwargs = {'image_pairs':image_pairs}
            if weighted: kwargs['n_looks'] = n_looks_delayed[idx]
            outs = ds_temp_coh_delayed(coh_delayed[idx],ph_delayed[idx],**kwargs)
            outs = tuple(outs) if weighted else (outs,)
            for o, out in enumerate(outs):
                outs_blocks[o].append(da.from_delayed(out,shape=coh.blocks[idx].shape[0:1],meta=xp.array((),dtype=xp.float32)))
        futures = []
        for blocks, path, name in zip(outs_blocks, (t_coh_path, t_coh_w, eff_n_pairs), ('t_coh', 't_coh_w', 'eff_n_pairs')):
            if path is None: continue
            out = da.concatenate(blocks)
            if cuda: out = out.map_blocks(cp.asnumpy)
            logger.darr_info(name, out)
            logger.info(f'saving {name}.')
            futures.append(out.to_zarr(path,compute=False,overwrite=True))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(futures)
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
    t_coh_dir:str,
    t_coh_w_dir:str=None,
    eff_n_pairs_dir:str=None,
    batch_size:int=1000,
    regularize:bool=True,
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
    t_coh_dir : str
        output: directory with the temporal coherence of the points, shape (n_points,), one zarr per
        raster chunk
    t_coh_w_dir : str, optional
        output: directory with the weighted temporal coherence of the points (see `ds-temp-coh`), shape
        (n_points,), float32, 0..1, NaN where no image pair is above the noise level, one zarr per raster
        chunk; the effective number of looks of each point comes from the positions of its SHPs and the
        speckle correlation of `rslc`
    eff_n_pairs_dir : str, optional
        output: directory with the effective number of image pairs of the weighted temporal coherence,
        shape (n_points,), float32, one zarr per raster chunk; a high weighted temporal coherence of few
        effective pairs is not reliable
    batch_size : int, default: 1000
        number of points processed at once, limits the memory use
    regularize : bool, default: True
        regularize the coherence matrix in the phase linking as `regularize` of `emi`; the temporal
        coherence is computed with the coherence matrix as estimated
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
    t_coh_dir = Path(t_coh_dir); mk_clean_dir(t_coh_dir)
    weighted = (t_coh_w_dir is not None) or (eff_n_pairs_dir is not None)
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
        emperical_co_emi_temp_coh_pc_delayed = delayed(mr.emperical_co_emi_temp_coh_pc,pure=True,nout=4 if weighted else 2)

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
                                                                  regularize=regularize,weighted=weighted))
                ph_delayed, t_coh_delayed = outs[:2]

                ph = da.from_delayed(ph_delayed,shape=(pc_chunksize[j],nimage),meta=xp.array((),dtype=rslc_overlap.dtype))
                t_coh = da.from_delayed(t_coh_delayed,shape=(pc_chunksize[j],),meta=xp.array((),dtype=xp.float32))

                if cuda:
                    cpu_ph = ph.map_blocks(cp.asnumpy)
                    cpu_t_coh = t_coh.map_blocks(cp.asnumpy)
                else:
                    cpu_ph = ph
                    cpu_t_coh = t_coh

                if do_log:
                    logger.darr_info(f'ph for chunk {j}',cpu_ph)
                    logger.darr_info(f't_coh for chunk {j}',cpu_t_coh)
                    logger.info(f'saving ph, t_coh for chunk {j}')

                _ph = dask_to_zarr(cpu_ph,ph_dir/f'{j}.zarr',chunks=(cpu_ph.shape[0],1),log_zarr=do_log)
                _t_coh = dask_to_zarr(cpu_t_coh,t_coh_dir/f'{j}.zarr',chunks=(cpu_t_coh.shape[0],),log_zarr=do_log)

                futures.extend((_ph,_t_coh))
                for out_delayed, out_dir in zip(outs[2:], (t_coh_w_dir, eff_n_pairs_dir)):
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
