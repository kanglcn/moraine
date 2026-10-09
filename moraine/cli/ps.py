"""persistent scatterers identification"""


__all__ = ['amp_disp']

import logging
import zarr
import numpy as np

import moraine as mr
from ..api.chunk_ import all_chunk_slices
from .logging import mc_logger
from .executor import Executor, Chunk

@mc_logger
def amp_disp(
    rslc:str,
    adi:str,
    chunks:tuple[int,int]=None,
    out_chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
):
    """calculation the amplitude dispersion index from SLC stack.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    adi : str
        output: amplitude dispersion index, shape (nlines, width), float32
    chunks : tuple[int, int], optional
        data processing chunk size, same as rslc by default
    out_chunks : tuple[int, int], optional
        output data chunk size, same as chunks by default
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
    rslc_zarr = zarr.open(rslc, mode='r')
    logger.zarr_info(rslc, rslc_zarr)
    nlines, width, nimages = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]
    if out_chunks is None: out_chunks = chunks
    adi_zarr = zarr.open(adi, mode='w', shape=(nlines, width), dtype=np.float32, chunks=tuple(out_chunks))
    logger.zarr_info(adi, adi_zarr)
    # one task per (azimuth, range) chunk: the rslc of the chunk in, its amplitude dispersion index out
    tasks = [([Chunk(rslc, (*sl, slice(0, nimages)))], [Chunk(adi, sl)]) for sl in all_chunk_slices((nlines, width), chunks)]
    logger.info(f'amplitude dispersion index of {len(tasks)} chunks of {tuple(chunks)}')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size) as ex:
        ex.map_chunks(mr.amp_disp, tasks, desc='amplitude dispersion')
    logger.info('done.')
