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
from ..utils_ import ngjit, ngpjit

import dask
from dask import array as da
from dask import delayed
from dask.distributed import Client, LocalCluster, progress
from ..utils_ import is_cuda_available
if is_cuda_available():
    import cupy as cp
    from dask_cuda import LocalCUDACluster
    from rmm.allocators.cupy import rmm_cupy_allocator
import moraine as mr
import moraine.cli as mc
from ..utils_ import get_array_module
from ..chunk_ import chunkwise_slicing_mapping, chunkwise_knn_mapping
from ..co import intf as intf_func
from .dask_ import parallel_read_zarr
from ..dl import _get_model, _cuda_device, _infer_unet, _infer_n2ft
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
    _cli_pre_infer_n2f_kernel = cp.ElementwiseKernel(
            'raw T ref_, raw T sec_, int32 nlines, int32 width',
            'raw float32 out, raw bool mask',
            '''
            int npixels = nlines*width;
            if (i >= npixels) return;

            T intf_i = ref_[i]*conj(sec_[i]);
            if (isnan(intf_i.real())){
                mask[i] = true;
            }
            else{
                mask[i] = false;
                float amp = abs(intf_i);
                out[i] = intf_i.real()/amp;
                out[npixels+i] = intf_i.imag()/amp;
            }
            ''',
            #preamble = '#include "curand.h"',
            # I do not find an easy way to generate random number with cupy kernel
            name = 'cli_pre_infer_n2f_kernel',reduce_dims=False,no_return=True)

if is_cuda_available():
    def _cli_pre_infer_n2f_cp(ref,sec):
        nlines, width = ref.shape
        out = cp.empty((1,2,nlines,width),dtype=cp.float32)
        mask = cp.empty((nlines,width),dtype=bool)
        _cli_pre_infer_n2f_kernel(ref,sec,cp.int32(nlines),cp.int32(width),out,mask,size=nlines*width,block_size=128)

        nan_pos = cp.where(mask)
        random_phase = cp.random.uniform(-cp.pi,cp.pi,len(nan_pos[0]))
        out[0,0,nan_pos[0],nan_pos[1]] = cp.cos(random_phase)
        out[0,1,nan_pos[0],nan_pos[1]] = cp.sin(random_phase)
        return out, mask

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
    ref[np.abs(ref)<1e-30] = np.nan+1j*np.nan # in case gamma has nan value, should be done in the load gamma function and remove in the future.
    sec[np.abs(sec)<1e-30] = np.nan+1j*np.nan # in case gamma has nan value, should be done in the load gamma function and remove in the future.

    model = _get_model('n2f', model, 'cpu')
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        input_intf_slice, mask_slice = _cli_pre_infer_n2f_numba(ref[in_slice],sec[in_slice])
        infer_out_slice = _infer_unet(model, input_intf_slice)
        out[out_slice] = mr.dl._after_infer_n2f_numba(infer_out_slice,mask_slice)[map_slice]
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
    ref[np.abs(ref)<1e-30] = np.nan+1j*np.nan # in case gamma has nan value, should be done in the load gamma function and remove in the future.
    sec[np.abs(sec)<1e-30] = np.nan+1j*np.nan # in case gamma has nan value, should be done in the load gamma function and remove in the future.

    model = _get_model('n2f', model, _cuda_device())
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        ref_slice = cp.asarray(ref[in_slice])
        sec_slice = cp.asarray(sec[in_slice])
        input_intf_slice, mask_slice = _cli_pre_infer_n2f_cp(ref_slice,sec_slice)
        output_intf_slice = _infer_unet(model, input_intf_slice)
        out[out_slice] = (mr.dl._after_infer_n2f_cp(output_intf_slice,mask_slice)[map_slice]).get()
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
        input: image pairs (reference, secondary), shape (n_image_pairs, 2); make a file with `moraine
        tnet`
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

def _cli_n2ft(
    x:np.ndarray,
    y:np.ndarray,
    ref:np.ndarray,
    sec:np.ndarray,
    in_indices,
    out_slices,
    map_indices,
    chunks:int=None,
    k:int=128,
    model:str=None,
    cuda:bool=False
):
    """Parameters
    ----------
    x : np.ndarray
        x coordinate, e.g., longitude, shape (n,) np.floating
    y : np.ndarray
        y coordinate, e.g., latitude, shape (n,) np.floating
    ref : np.ndarray
        reference slc, shape(n,) np.complex64
    sec : np.ndarray
        secondary slc, shape(n,) np.complex64
    in_indices
    out_slices
    map_indices
    chunks : int, optional
        chunksize, intf.shape[0] by default
    k : int, default: 128
        halo size for chunkwise processing
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        use gpu for inference
    """
    model = _get_model('n2ft', model, 'cuda' if cuda else 'cpu')

    intf = intf_func(ref, sec)[:,None]
    n = intf.shape[0]
    if (chunks is None) or (chunks >= n):
        out = _infer_n2ft(x, y, intf, model)
    else:
        out = np.empty_like(intf)
        for in_idx, out_slice, map_idx in zip(in_indices, out_slices, map_indices):
            out[out_slice] = _infer_n2ft(x[in_idx],y[in_idx],intf[in_idx],model)[map_idx]
    out = out[:,0]

    return out

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
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
    **dask_cluster_arg,
):
    """Noise2Fringe Transformer (n2ft) filtering of point cloud interferograms.

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
        input: image pairs (reference, secondary), shape (n_image_pairs, 2); make a file with `moraine
        tnet`
    chunks : int, optional
        number of points per processing chunk, same as rslc by default
    out_chunks : int, optional
        point chunk size of the output, same as rslc by default
    k : int, default: 128
        number of nearest neighbours of the chunk border points added as halo to each chunk
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
    if chunks is None: chunks = rslc_zarr.chunks[0]
    if out_chunks is None: out_chunks = rslc_zarr.chunks[0]

    logger.info(f'processing point chunk size: {chunks}')
    logger.info('distributing every processing chunk with halo data')
    in_indices, out_slices, map_indices = chunkwise_knn_mapping(pc_x_data, pc_y_data, chunks, k=k)
    in_indices_size = [len(in_idx) for in_idx in in_indices]
    logger.info(f'processing chunk size with halo data: {in_indices_size}')

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
        n2ft_delayed = delayed(_cli_n2ft,pure=True,nout=1)

        ref_cpu_rslc = dask_from_zarr(rslc_path, chunks=(npoint,1))
        # use two different rslc to make dask do not hold too much data 
        sec_cpu_rslc = dask_from_zarr(rslc_path, chunks=(npoint,1))
        logger.darr_info('rslc', ref_cpu_rslc)
        intf = np.empty((1,n_image_pairs),dtype=object)
        for i in range(n_image_pairs):
            ref_i, sec_i = image_pairs[i]
            intf[0,i] = n2ft_delayed(pc_x_data, pc_y_data, ref_cpu_rslc[:,ref_i].to_delayed()[0],sec_cpu_rslc[:,sec_i].to_delayed()[0],
                                     in_indices, out_slices, map_indices,
                                     chunks=chunks, k=k, model=model, cuda=cuda
                                    )
            intf[0,i] = da.from_delayed(intf[0,i],shape=(npoint,),meta=np.array((),dtype=ref_cpu_rslc.dtype)).reshape(npoint,1)
        intf = da.block(intf.tolist())
        logger.info('got filtered interferograms.')
        logger.darr_info('intf', intf)

        logger.info('saving filtered interferograms.')
        _intf = dask_to_zarr(intf,intf_path,chunks=(out_chunks,1))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist([_intf,])
        progress(futures,notebook=False)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')
