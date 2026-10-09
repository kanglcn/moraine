"""Spatially Homogenious Pixels Identification"""


__all__ = ['shp_test', 'select_shp']

import logging
import zarr
import numpy as np

import moraine as mr
from ..api.chunk_ import all_chunk_slices, chunkwise_slicing_mapping
from .logging import mc_logger
from .executor import Executor, Chunk


def _ks_test_chunk(rslc, az_half_win, r_half_win, map_slice):
    """p values of the pixels of a chunk read with its halo: the test on the amplitudes, cut to the chunk."""
    p = mr.ks_test(mr.rslc2amp(rslc), az_half_win=az_half_win, r_half_win=r_half_win)
    return p[map_slice]

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
        use processes (True) or threads (False) for the workers, only for cpu processing. Default: False
    n_workers : optional
        number of workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 1
    """
    logger = logging.getLogger(__name__)
    if not method: method = 'ks'
    logger.info(f'hypothetic test method: {method}')
    if method != 'ks':
        logger.warning('Currently only KS test is implented. Switching to it.')
        method = 'ks'
    rslc_zarr = zarr.open(rslc, mode='r')
    logger.zarr_info(rslc, rslc_zarr)
    assert rslc_zarr.ndim == 3, " rslcs dimentation is not 3."
    nlines, width, nimages = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]
    chunks = tuple(chunks)
    az_win = 2*az_half_win+1
    logger.info(f'azimuth half window size: {az_half_win}; azimuth window size: {az_win}')
    r_win = 2*r_half_win+1
    logger.info(f'range half window size: {r_half_win}; range window size: {r_win}')
    p_zarr = zarr.open(pvalue, mode='w', shape=(nlines, width, az_win, r_win), dtype=np.float32, chunks=(*chunks, 1, 1))
    logger.zarr_info(pvalue, p_zarr)
    # one task per (azimuth, range) chunk, read with a halo of the half windows; the p values are cut to the chunk
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping((nlines, width), chunks, (az_half_win, r_half_win))
    tasks = [([Chunk(rslc, (*in_sl, slice(0, nimages))), az_half_win, r_half_win, (*map_sl, slice(None), slice(None))],
              [Chunk(pvalue, (*out_sl, slice(0, az_win), slice(0, r_win)))])
             for in_sl, out_sl, map_sl in zip(in_slices, out_slices, map_slices)]
    logger.info(f'KS test of {len(tasks)} chunks of {chunks} with a halo of ({az_half_win}, {r_half_win})')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(_ks_test_chunk, tasks, desc='KS test')
    logger.info('done.')

@mc_logger
def select_shp(
    pvalue:str,
    is_shp:str,
    shp_num:str,
    alpha:float=0.05,
    chunks:tuple[int,int]=None,
    processes=False,
    n_workers=1,
    threads_per_worker=1,
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
    alpha : float, default: 0.05
        significance level of the test, in (0, 1): a pixel is an SHP of the centre pixel of its window when
        its p value is at least `alpha`, i.e. the test does not reject that both have the same distribution;
        a larger `alpha` keeps fewer SHPs
    chunks : tuple[int, int], optional
        (azimuth, range) processing chunk size, same as `pvalue` by default
    processes : default: False
        use processes for the workers instead of threads, the default is False
    n_workers : default: 1
        number of workers, the default is 1
    threads_per_worker : default: 1
        tasks a worker runs at the same time
    """
    logger = logging.getLogger(__name__)
    p_zarr = zarr.open(pvalue, mode='r'); logger.zarr_info(pvalue, p_zarr)
    assert p_zarr.ndim == 4, " pvalue dimentation is not 4."
    nlines, width, az_win, r_win = p_zarr.shape
    if chunks is None: chunks = p_zarr.chunks[:2]
    chunks = tuple(chunks)
    is_shp_zarr = zarr.open(is_shp, mode='w', shape=p_zarr.shape, dtype=bool, chunks=(*chunks, 1, 1))
    logger.zarr_info(is_shp, is_shp_zarr)
    shp_num_zarr = zarr.open(shp_num, mode='w', shape=(nlines, width), dtype=np.int32, chunks=chunks)
    logger.zarr_info(shp_num, shp_num_zarr)
    logger.info('selecting SHPs with p value >= alpha = '+str(alpha))
    win = (slice(0, az_win), slice(0, r_win))
    tasks = [([Chunk(pvalue, (*sl, *win)), alpha], [Chunk(is_shp, (*sl, *win)), Chunk(shp_num, sl)])
             for sl in all_chunk_slices((nlines, width), chunks)]
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(mr.select_shp, tasks, desc='SHP selection')
    logger.info('done.')
