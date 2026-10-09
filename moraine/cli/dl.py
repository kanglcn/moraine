"""Deep learning operators (CLI)"""

__all__ = ['n2f', 'n2ft']

import logging
import zarr
import numpy as np
from numba import prange
from pathlib import Path
import math
import cmath
import importlib
from ..api.utils_ import ngjit, ngpjit

import dask
from dask import array as da
from dask import delayed
from dask.distributed import Client, LocalCluster, progress
from ..api.utils_ import is_cuda_available
if is_cuda_available():
    import cupy as cp
    from dask_cuda import LocalCUDACluster
    from rmm.allocators.cupy import rmm_cupy_allocator
import moraine as mr
import moraine.cli as mc
from ..api.utils_ import get_array_module
from ..api.chunk_ import chunkwise_slicing_mapping, chunkwise_knn_mapping
from .dask_ import parallel_read_zarr
from ..api.dl import _get_model, _cuda_device, _infer_unet, _n2ft_prepare, _infer_n2ft_prepared, _prefetched, _n2ft_compile_default, _nan_where_zero
from .logging import mc_logger
from . import mk_clean_dir, dask_from_zarr, dask_from_zarr_overlap, dask_to_zarr

def _torch_use_rmm():
    '''let torch allocate gpu memory from the rmm pool, run it in every dask cuda worker before torch uses the gpu'''
    import torch
    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

@ngpjit
def _cli_pre_infer_n2f_numba(
    ref,
    sec,
):
    """Parameters
    ----------
    ref
        reference rslc
    sec
        secoundary rslc
    """
    nlines, width = ref.shape
    out = np.empty((1,2,nlines,width),dtype=np.float32)
    mask = np.empty((nlines,width),dtype=np.bool_)
    for i in prange(nlines):
        for j in prange(width):
            intf_i_j = ref[i,j]*sec[i,j].conjugate()
            if math.isnan(intf_i_j.real):
                mask[i,j] = True
                random_phase = np.random.uniform(-math.pi,math.pi)
                out[0,0,i,j] = math.cos(random_phase)
                out[0,1,i,j] = math.sin(random_phase)
            else:
                mask[i,j] = False
                amp = abs(intf_i_j)
                out[0,0,i,j] = intf_i_j.real/amp
                out[0,1,i,j] = intf_i_j.imag/amp
    return out, mask

if is_cuda_available():
    from numba import cuda
    from ..api.utils_ import mcuda_jit

    @mcuda_jit()
    def _intf_cuda(ref, sec, out):
        t = cuda.grid(1)
        if t >= out.size:
            return
        i = t//out.shape[1]; j = t%out.shape[1]
        out[i,j] = ref[i,j]*sec[i,j].conjugate()

    def _cli_pre_infer_n2f_cp(ref,sec):
        intf = cp.empty_like(ref)
        if intf.size > 0:
            _intf_cuda[(intf.size+127)//128, 128](ref, sec, intf)
        return mr.api.dl._pre_infer_n2f_cp(intf)

def _cli_n2f_cpu(
    ref,
    sec,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Parameters
    ----------
    ref
    sec
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    shape = ref.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = np.empty_like(ref)

    model = _get_model('n2f', model, 'cpu')
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        input_intf_slice, mask_slice = _cli_pre_infer_n2f_numba(_nan_where_zero(ref[in_slice]),_nan_where_zero(sec[in_slice]))
        infer_out_slice = _infer_unet(model, input_intf_slice)
        out[out_slice] = mr.api.dl._after_infer_n2f_numba(infer_out_slice,mask_slice)[map_slice]
    return out

def _cli_n2f_np_in_gpu(
    ref,
    sec,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Parameters
    ----------
    ref
    sec
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    shape = ref.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = np.empty_like(ref)

    model = _get_model('n2f', model, _cuda_device())
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        ref_slice = _nan_where_zero(cp.asarray(ref[in_slice]))
        sec_slice = _nan_where_zero(cp.asarray(sec[in_slice]))
        input_intf_slice, mask_slice = _cli_pre_infer_n2f_cp(ref_slice,sec_slice)
        output_intf_slice = _infer_unet(model, input_intf_slice)
        out[out_slice] = (mr.api.dl._after_infer_n2f_cp(output_intf_slice,mask_slice)[map_slice]).get()
    return out

@mc_logger
def n2f(
    rslc:str,
    intf:str,
    image_pairs:np.ndarray,
    chunks:tuple[int,int]=None,
    out_chunks:tuple[int,int]=None,
    depths:tuple[int,int]=(0,0),
    model:str=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """Noise2Fringe (n2f) filtering of raster interferograms.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    intf : str
        output: filtered interferograms, complex64 with unit amplitude (nan where the input is nan),
        shape (nlines, width, n_image_pairs)
    image_pairs : np.ndarray
        input: image pairs (reference, secondary), shape (n_image_pairs, 2); make a file with
        `moraine image-pairs`
    chunks : tuple[int, int], optional
        (azimuth, range) processing chunk size, same as rslc by default
    out_chunks : tuple[int, int], optional
        (azimuth, range) chunk size of the output, same as rslc by default
    depths : tuple[int, int], default: (0, 0)
        (azimuth, range) overlap in pixels between processing chunks, reduces chunk border effects
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the dask workers, only for cpu processing. Default:
        True
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
    intf_path = intf
    logger = logging.getLogger(__name__)

    rslc_zarr = zarr.open(rslc_path,mode='r')
    logger.zarr_info(rslc_path, rslc_zarr)
    assert rslc_zarr.ndim == 3, "rslc dimentation is not 3."
    nlines, width, nimage = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]
    if out_chunks is None: out_chunks = rslc_zarr.chunks[:2]
    az_chunk, r_chunk = chunks
    logger.info(f'processing azimuth chunk size: {az_chunk}')
    logger.info(f'processing range chunk size: {r_chunk}')

    if cuda:
        Cluster = LocalCUDACluster; cluster_args= {
            'n_workers':n_workers,
            'rmm_pool_size':rmm_pool_size}
        cluster_args.update(dask_cluster_arg)
    else:
        if processes is None: processes = True
        if n_workers is None: n_workers = 1
        if threads_per_worker is None: threads_per_worker = 1
        Cluster = LocalCluster; cluster_args = {'processes':processes, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
        cluster_args.update(dask_cluster_arg)

    n_image_pairs = image_pairs.shape[0]

    logger.info('starting dask cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda:
            client.run(cp.cuda.set_allocator, rmm_cupy_allocator)
            client.run(_torch_use_rmm)
            n2f_delayed = delayed(_cli_n2f_np_in_gpu,pure=True,nout=1)
        else:
            n2f_delayed = delayed(_cli_n2f_cpu,pure=True,nout=1)

        ref_cpu_rslc = dask_from_zarr(rslc_path, chunks=(*rslc_zarr.shape[0:2],1))
        # use two different rslc to make dask do not hold too much data 
        sec_cpu_rslc = dask_from_zarr(rslc_path, chunks=(*rslc_zarr.shape[0:2],1))
        logger.darr_info('rslc', ref_cpu_rslc)
        intf = np.empty((1,1,n_image_pairs),dtype=object)
        for i in range(n_image_pairs):
            ref_i, sec_i = image_pairs[i]
            intf[0,0,i] = n2f_delayed(ref_cpu_rslc[:,:,ref_i].to_delayed()[0,0],sec_cpu_rslc[:,:,sec_i].to_delayed()[0,0],chunks=chunks,depths=depths,model=model)
            intf[0,0,i] = da.from_delayed(intf[0,0,i],shape=rslc_zarr.shape[:2],meta=np.array((),dtype=ref_cpu_rslc.dtype)).reshape(*rslc_zarr.shape[:2],1)
        intf = da.block(intf.tolist())
        logger.info('got filtered interferograms.')
        logger.darr_info('intf', intf)

        logger.info('saving filtered interferograms.')
        _intf = dask_to_zarr(intf,intf_path,chunks=(*out_chunks,1))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist([_intf,])
        progress(futures,notebook=False)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

def _n2ft_block_bounds(n, chunks, out_chunks):
    """start of every processing chunk of `n` points and `n`: chunks of `chunks` points that do not cross the
    output chunks of `out_chunks` points"""
    starts = [s for o in range(0, n, out_chunks) for s in range(o, min(o+out_chunks, n), chunks)]
    return np.array(starts+[n], dtype=np.int64)

def _cli_n2ft_out_chunk(
    x:str,
    y:str,
    rslc:str,
    intf:str,
    rows:tuple,
    idx:np.ndarray,
    blocks:list,
    image_pairs:np.ndarray,
    model:str=None,
    cuda:bool=False,
    compile:bool=False,
):
    """n2ft of all image pairs on the points of one output chunk, written into `intf`

    Parameters
    ----------
    x : str
        zarr path of the x coordinate of all points, shape (n_points,)
    y : str
        zarr path of the y coordinate of all points, shape (n_points,)
    rslc : str
        zarr path of the rslc of all points, shape (n_points, nimages)
    intf : str
        zarr path of the output, shape (n_points, n_image_pairs); `rows` is one of its chunks
    rows : tuple
        (start, stop) of the points of the output chunk
    idx : np.ndarray
        sorted indices of the points of the output chunk and of their halo points
    blocks : list
        every processing chunk of the output chunk: positions in `idx` of its points with halo, positions of
        its own points among them, slice of its own points in the output chunk
    image_pairs : np.ndarray
        image pairs (reference, secondary), shape (n_image_pairs, 2)
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        use gpu for inference
    compile : bool, default: False
        compile the model with torch.compile (once per process)

    Returns
    -------
    int
        number of points of the output chunk
    """
    model = _get_model('n2ft', model, 'cuda' if cuda else 'cpu', compile)
    device = next(model.parameters()).device
    images = np.unique(image_pairs)
    cols = np.searchsorted(images, image_pairs)
    # the rslc of the points with halo for the images of the pairs; every processing chunk is filtered for all
    # image pairs at once: its structure depends on the coordinates only and is prepared in a thread while the model
    # filters the previous chunk, and the model runs several interferograms per call
    rslc_idx = zarr.open(rslc,mode='r').get_orthogonal_selection((idx,images))
    x_idx = zarr.open(x,mode='r').get_orthogonal_selection(idx)
    y_idx = zarr.open(y,mode='r').get_orthogonal_selection(idx)
    start, stop = rows
    out = np.empty((stop-start, image_pairs.shape[0]), dtype=rslc_idx.dtype)
    def prepare(block):
        pos = block[0]
        rslc_pos = rslc_idx[pos]
        ifg = rslc_pos[:,cols[:,0]]*rslc_pos[:,cols[:,1]].conj()
        return _n2ft_prepare(x_idx[pos],y_idx[pos],ifg,device)
    for (pos, own, out_slice), prepared in _prefetched(blocks, prepare):
        out[out_slice] = _infer_n2ft_prepared(prepared,model)[own]
    zarr.open(intf,mode='r+')[start:stop] = out
    return stop-start

@mc_logger
def n2ft(
    x:str,
    y:str,
    rslc:str,
    intf:str,
    image_pairs:np.ndarray,
    chunks:int=None,
    out_chunks:int=None,
    k:int=128,
    model:str=None,
    cuda:bool=False,
    compile:bool=None,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """Noise2Fringe Transformer (n2ft) filtering of point cloud interferograms.

    Every worker filters one output chunk of points with all image pairs at a time: it holds the rslc of the
    points of the chunk and of their halos for the images of the pairs (8 bytes per point and image), the
    filtered interferograms of the chunk (8 bytes per point and image pair) and about 400 bytes per point of
    the chunk more; with `cuda` a model call holds about a fifth of the GPU memory. Before, the main process finds
    the halos of all processing chunks with about 70 bytes per point and 32 threads of 16 x `k` x `chunks` bytes
    (41 MB each by default).

    Parameters
    ----------
    x : str
        input: x coordinate of the points (e.g. longitude or easting), shape (n_points,)
    y : str
        input: y coordinate of the points (e.g. latitude or northing), shape (n_points,)
    rslc : str
        input: rslc of the points, shape (n_points, nimages)
    intf : str
        output: filtered interferograms, complex64 with unit amplitude, shape (n_points, n_image_pairs)
    image_pairs : np.ndarray
        input: image pairs (reference, secondary), shape (n_image_pairs, 2); make a file with
        `moraine image-pairs`
    chunks : int, optional
        number of points per processing chunk, same as rslc by default; the processing chunks do not cross
        the output chunks
    out_chunks : int, optional
        point chunk size of the output, same as rslc by default; it sets the memory of a worker
    k : int, default: 128
        number of nearest neighbours of every point of a processing chunk that are filtered with it (halo)
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        if use cuda for processing, false by default
    compile : bool, optional
        compile the model with torch.compile in every worker: the model then runs about 3 times faster on a GPU,
        but the compilation takes 15-40 s per worker (less when torch has cached it on disk). Default: when
        points times image pairs is at least 1e8
    processes : optional
        use processes (True) or threads (False) for the dask workers, only for cpu processing. Default:
        True
    n_workers : optional
        number of dask workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        number of threads per dask worker, only for cpu processing. Default: 1
    rmm_pool_size : default: 0.9
        set the rmm pool size, only applied when cuda==True
    **dask_cluster_arg
        other dask local/cudalocal cluster args
    """
    logger = logging.getLogger(__name__)
    logger.info('load coordinates')
    pc_x_data = parallel_read_zarr(zarr.open(x,mode='r'),(slice(None),))
    pc_y_data = parallel_read_zarr(zarr.open(y,mode='r'),(slice(None),))
    logger.info('Done')
    rslc_path = rslc
    intf_path = intf

    rslc_zarr = zarr.open(rslc_path,mode='r')
    logger.zarr_info(rslc_path, rslc_zarr)
    assert rslc_zarr.ndim == 2, "rslc dimentation is not 2."
    npoint, nimage = rslc_zarr.shape

    nimage_pairs = image_pairs.shape[0]
    if compile is None:
        compile = _n2ft_compile_default(npoint, nimage_pairs)
        logger.info(f'compile the model: {compile}')
    if chunks is None: chunks = rslc_zarr.chunks[0]
    if out_chunks is None: out_chunks = rslc_zarr.chunks[0]

    logger.info(f'processing point chunk size: {chunks}')
    logger.info(f'output point chunk size: {out_chunks}, processed with all image pairs in one task')
    logger.info('distributing every processing chunk with halo data')
    in_indices, out_slices, map_indices = chunkwise_knn_mapping(pc_x_data, pc_y_data, chunks, k=k,
                                                                bounds=_n2ft_block_bounds(npoint, chunks, out_chunks))
    in_indices_size = [len(in_idx) for in_idx in in_indices]
    logger.info(f'processing chunk size with halo data: {in_indices_size}')

    # one task per output chunk: its points with the halos of its processing chunks, positions as int32; the
    # halo indices are released as they are used
    del pc_x_data, pc_y_data
    tasks_args = []
    j = 0
    for start in range(0, npoint, out_chunks):
        stop = min(start+out_chunks, npoint)
        jb = []
        while j < len(out_slices) and out_slices[j].start < stop:
            jb.append(j); j += 1
        idx = np.unique(np.concatenate([in_indices[i] for i in jb]))
        blocks = []
        for i in jb:
            blocks.append((np.searchsorted(idx, in_indices[i]).astype(np.int32), map_indices[i].astype(np.int32),
                           slice(out_slices[i].start-start, out_slices[i].stop-start)))
            in_indices[i] = None; map_indices[i] = None
        tasks_args.append(((start, stop), idx, blocks))
    del in_indices, out_slices, map_indices

    if cuda:
        Cluster = LocalCUDACluster; cluster_args= {
            'n_workers':n_workers,
            'rmm_pool_size':rmm_pool_size}
        cluster_args.update(dask_cluster_arg)
    else:
        if processes is None: processes = True
        if n_workers is None: n_workers = 1
        if threads_per_worker is None: threads_per_worker = 1
        Cluster = LocalCluster; cluster_args = {'processes':processes, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
        cluster_args.update(dask_cluster_arg)

    n_image_pairs = image_pairs.shape[0]

    logger.info('starting dask cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask cluster started.')
        logger.dask_cluster_info(cluster)
        if cuda:
            client.run(cp.cuda.set_allocator, rmm_cupy_allocator)
            client.run(_torch_use_rmm)

        intf_zarr = zarr.open(intf_path,mode='w',shape=(npoint,n_image_pairs),dtype=rslc_zarr.dtype,chunks=(out_chunks,1))
        logger.zarr_info(intf_path, intf_zarr)
        n2ft_delayed = delayed(_cli_n2ft_out_chunk,pure=True,nout=1)
        tasks = [n2ft_delayed(x, y, rslc_path, intf_path, rows, idx, blocks, image_pairs, model=model, cuda=cuda, compile=compile)
                 for rows, idx, blocks in tasks_args]

        logger.info(f'filtering and saving the interferograms of {len(tasks)} output chunks.')
        futures = client.compute(tasks)
        progress(futures,notebook=False)
        client.gather(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
