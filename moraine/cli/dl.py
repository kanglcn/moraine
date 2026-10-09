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
import itertools
from ..api.utils_ import ngjit, ngpjit

from ..api.utils_ import is_cuda_available
if is_cuda_available():
    import cupy as cp
import moraine as mr
import moraine.cli as mc
from ..api.utils_ import get_array_module
from ..api.chunk_ import chunkwise_knn_mapping
from .dask_ import parallel_read_zarr, parallel_write_zarr
from ..api.dl import _get_model, _cuda_device, _infer_unet, _n2ft_prepare, _infer_n2ft_prepared, _prefetched, _n2ft_compile_default, _nan_where_zero
from .logging import mc_logger
from .executor import Executor

def _torch_use_rmm():
    '''let torch allocate gpu memory from the rmm pool, run it in every dask cuda worker before torch uses the gpu'''
    import torch
    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

def _torch_no_kernel_autotune():
    '''compile the model without tuning its kernels: the tuning needs the memory statistics of the torch allocator,
    which the rmm allocator does not provide'''
    import torch._inductor.config as inductor_config
    inductor_config.triton.autotune_pointwise = False

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

def _n2f_tiles(shape, chunks, out_chunks, depths):
    """processing chunks (tiles) of a raster grouped by output chunk

    Returns {(azimuth, range) slices of the output chunk: [(tile with its overlap in the raster, tile in that
    overlapped tile, tile in the output chunk), ...]}, every item a tuple of (azimuth, range) slices. The tiles
    are the grid of `chunks` cut at the boundaries of the output chunks (the grid of `out_chunks`), with `depths`
    pixels of overlap around every tile.
    """
    dims = []
    for n, chunk, out_chunk, depth in zip(shape, chunks, out_chunks, depths):
        chunk = n if chunk <= 0 else chunk
        out_chunk = n if out_chunk <= 0 else out_chunk
        bounds = sorted({*range(0, n, chunk), *range(0, n, out_chunk), n})
        dim = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            in_start, in_stop = max(start-depth, 0), min(stop+depth, n)
            o = start//out_chunk*out_chunk
            dim.append((slice(o, min(o+out_chunk, n)), slice(in_start, in_stop), slice(start-in_start, stop-in_start),
                        slice(start-o, stop-o)))
        dims.append(dim)
    tiles = {}
    for az, r in itertools.product(*dims):
        tiles.setdefault((az[0], r[0]), []).append(((az[1], r[1]), (az[2], r[2]), (az[3], r[3])))
    return tiles

def _cli_n2f_out_chunk(
    rslc:str,
    intf:str,
    out_slices:tuple,
    tiles:list,
    image_pairs:np.ndarray,
    model:str=None,
    cuda:bool=False,
):
    """filter all image pairs of one output chunk of the raster and write it

    Parameters
    ----------
    rslc : str
        zarr path of the rslc stack, shape (nlines, width, nimages)
    intf : str
        zarr path of the output, shape (nlines, width, n_image_pairs); `out_slices` is one of its chunks
    out_slices : tuple
        (azimuth, range) slices of the output chunk
    tiles : list
        every processing chunk of the output chunk: (azimuth, range) slices of the tile with its overlap in the
        raster, of the tile in that overlapped tile and of the tile in the output chunk
    image_pairs : np.ndarray
        image pairs (reference, secondary), shape (n_image_pairs, 2)
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        filter on the GPU
    """
    rslc_zarr = zarr.open(rslc,mode='r')
    xp = cp if cuda else np
    # the output chunk stays on the GPU until all pairs are filtered: one copy to the host instead of one per pair
    out = xp.empty((out_slices[0].stop-out_slices[0].start, out_slices[1].stop-out_slices[1].start, image_pairs.shape[0]),
                   dtype=rslc_zarr.dtype)
    model = _get_model('n2f', model, _cuda_device() if cuda else 'cpu')
    for in_slices, map_slices, local_slices in tiles:
        # the rslc of the tile with its overlap for all images: every image is read once per tile
        stack = xp.asarray(parallel_read_zarr(rslc_zarr, (*in_slices, slice(None))))
        for k, (ref_i, sec_i) in enumerate(image_pairs):
            ref = _nan_where_zero(stack[:,:,ref_i]); sec = _nan_where_zero(stack[:,:,sec_i])
            if cuda:
                x, mask = _cli_pre_infer_n2f_cp(ref, sec)
                out[(*local_slices, k)] = mr.api.dl._after_infer_n2f_cp(_infer_unet(model, x), mask)[map_slices]
            else:
                x, mask = _cli_pre_infer_n2f_numba(ref, sec)
                out[(*local_slices, k)] = mr.api.dl._after_infer_n2f_numba(_infer_unet(model, x), mask)[map_slices]
    if cuda:
        out = out.get()
    parallel_write_zarr(out, zarr.open(intf,mode='r+'), (*out_slices, slice(None)))

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

    A task filters one output chunk with all image pairs: it holds the rslc of one processing chunk with its
    overlap for all images (8 bytes per pixel and image), the filtered interferograms of the output chunk (8 bytes
    per pixel and image pair) and the model's activations of one interferogram (about 2 kB per pixel of the
    processing chunk); with `cuda` all of it on the GPU. A worker runs `threads_per_worker` tasks at a time.

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
        (azimuth, range) processing chunk size, same as rslc by default; the processing chunks are cut at the
        output chunks
    out_chunks : tuple[int, int], optional
        (azimuth, range) chunk size of the output, same as rslc by default; it sets the memory of a worker
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
        number of threads per dask worker, i.e. output chunks filtered at a time. Default: 1
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
    logger.info(f'processing chunk size: {tuple(chunks)}, overlap: {tuple(depths)}')
    logger.info(f'output chunk size: {tuple(out_chunks)}, processed with all image pairs in one task')
    tiles_by_out = _n2f_tiles((nlines, width), chunks, out_chunks, depths)
    logger.info(f'{len(tiles_by_out)} output chunks, {sum(len(t) for t in tiles_by_out.values())} processing chunks')

    if not cuda and processes is None: processes = True
    n_image_pairs = image_pairs.shape[0]
    intf_zarr = zarr.open(intf_path,mode='w',shape=(nlines,width,n_image_pairs),dtype=rslc_zarr.dtype,chunks=(*out_chunks,1))
    logger.zarr_info(intf_path, intf_zarr)
    tasks = [(rslc_path, intf_path, out_slices, tiles, image_pairs, model, cuda) for out_slices, tiles in tiles_by_out.items()]
    logger.info(f'filtering and saving the interferograms of {len(tasks)} output chunks.')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size, **dask_cluster_arg) as ex:
        if cuda:
            ex.run_on_workers(_torch_use_rmm)
        ex.map(_cli_n2f_out_chunk, tasks, desc='output chunks')
    logger.info('done.')

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
    rmm_pool_size=None,
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
    rmm_pool_size : optional
        rmm memory pool of every worker as a fraction of the GPU memory, shared with torch; none by default: the model
        allocates with torch, which it needs to tune its compiled kernels (with a pool, `compile` runs without the
        tuning and about 2 times slower)
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

    if not cuda and processes is None: processes = True
    n_image_pairs = image_pairs.shape[0]

    intf_zarr = zarr.open(intf_path,mode='w',shape=(npoint,n_image_pairs),dtype=rslc_zarr.dtype,chunks=(out_chunks,1))
    logger.zarr_info(intf_path, intf_zarr)
    tasks = [(x, y, rslc_path, intf_path, rows, idx, blocks, image_pairs, model, cuda, compile) for rows, idx, blocks in tasks_args]
    logger.info(f'filtering and saving the interferograms of {len(tasks)} output chunks.')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size, **dask_cluster_arg) as ex:
        if cuda and rmm_pool_size:
            ex.run_on_workers(_torch_use_rmm)
            if compile:
                logger.info('rmm pool: the compiled model runs without kernel tuning')
                ex.run_on_workers(_torch_no_kernel_autotune)
        ex.map(_cli_n2ft_out_chunk, tasks, desc='output chunks')
    logger.info('done.')
