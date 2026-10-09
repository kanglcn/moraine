"""Pixel quality metrics (CLI)"""


__all__ = ['temp_coh']

import logging
import zarr
import numpy as np

import moraine as mr
from ..api.chunk_ import all_chunk_slices
from .logging import mc_logger
from .executor import Executor, Chunk

@mc_logger
def temp_coh(
    intf:str,
    rslc:str,
    t_coh:str=None,
    image_pairs:np.ndarray=None,
    chunks:int|tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
):
    """temporal coherence.

    Parameters
    ----------
    intf : str
        input: (filtered) interferograms or coherence, complex64, shape (n_points, n_image_pairs) or
        (nlines, width, n_image_pairs)
    rslc : str
        input: rslc stack or phase history, complex64, shape (n_points, nimages) or (nlines, width,
        nimages)
    t_coh : str, optional
        output: temporal coherence, shape (n_points,) or (nlines, width)
    image_pairs : np.ndarray, optional
        image pairs (reference, secondary) of `intf`, shape (n_image_pairs, 2); all image pairs by
        default
    chunks : int | tuple[int, int], optional
        point chunk size or (azimuth, range) chunk size, same as `intf` by default
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the workers, only for cpu processing. Default: False
    n_workers : optional
        number of workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 1
    rmm_pool_size : default: 0.9
        fraction of the GPU memory of each worker taken by an rmm memory pool (rmm must be installed), only with cuda
    """
    logger = logging.getLogger(__name__)
    intf_zarr = zarr.open(intf, mode='r'); logger.zarr_info(intf, intf_zarr)
    rslc_zarr = zarr.open(rslc, mode='r'); logger.zarr_info(rslc, rslc_zarr)
    nimage = rslc_zarr.shape[-1]
    n_pairs = intf_zarr.shape[-1]
    shape = intf_zarr.shape[:-1]           # (n_points,) or (nlines, width)
    if chunks is None: chunks = intf_zarr.chunks[:-1]
    chunks = (chunks,) if isinstance(chunks, int) else tuple(chunks)
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(nimage).image_pairs
    image_pairs = np.asarray(image_pairs).astype(np.int32)
    t_coh_zarr = zarr.open(t_coh, mode='w', shape=shape, dtype=np.float32, chunks=chunks)
    logger.zarr_info(t_coh, t_coh_zarr)
    # one task per chunk of points or pixels: the interferograms and the rslc of the chunk in, the temporal coherence out
    tasks = [([Chunk(intf, (*sl, slice(0, n_pairs))), Chunk(rslc, (*sl, slice(0, nimage)))], [Chunk(t_coh, sl)])
             for sl in all_chunk_slices(shape, chunks)]
    logger.info(f'temporal coherence of {len(tasks)} chunks of {chunks}')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size) as ex:
        ex.map_chunks(mr.temp_coh, tasks, desc='temporal coherence', image_pairs=image_pairs)
    logger.info('done.')
