"""Spatially Homogeneous Pixels Identification"""


__all__ = ['shp_test', 'select_shp']

import logging
import zarr
import numpy as np

import moraine as mr
from ..api.chunk_ import all_chunk_slices, chunkwise_slicing_mapping
from .logging import mc_logger
from .zarr_ import check_ndim
from .executor import Executor, Chunk


def _ks_test_chunk(rslc, az_half_win, r_half_win, map_slice, alpha, with_p):
    """p values of the pixels of a chunk read with its halo (the test on the amplitudes, cut to the chunk), the SHP
    flags (p value at least `alpha`; a pixel without a test, nan, is not an SHP) and the number of SHPs of each pixel"""
    p = mr.ks_test(mr.rslc2amp(rslc), az_half_win=az_half_win, r_half_win=r_half_win)[map_slice]
    is_shp = p >= alpha
    shp_num = is_shp.sum(axis=(2, 3), dtype=np.int32)
    return (p, is_shp, shp_num) if with_p else (is_shp, shp_num)

@mc_logger
def shp_test(
    rslc:str,
    is_shp:str,
    shp_num:str,
    az_half_win:int,
    r_half_win:int,
    alpha:float=0.05,
    pvalue:str=None,
    method:str=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
):
    """SHP identification through hypothesis test, and the selection of the SHPs at the level `alpha`.

    Parameters
    ----------
    rslc : str
        input: rslc stack, shape (nlines, width, nimages)
    is_shp : str
        output: True for the SHPs of each pixel, i.e. the pixels of its window whose p value is at least
        `alpha`, shape (nlines, width, 2*az_half_win+1, 2*r_half_win+1), bool
    shp_num : str
        output: number of SHPs of each pixel, shape (nlines, width), int32
    az_half_win : int
        azimuth half window size
    r_half_win : int
        range half window size
    alpha : float, default: 0.05
        significance level of the test, in (0, 1): a pixel is an SHP of the centre pixel of its window when
        its p value is at least `alpha`; a larger `alpha` keeps fewer SHPs
    pvalue : str, optional
        output: p value of the test between each pixel and the pixels in its window, same shape as `is_shp`,
        float32; needed only to select the SHPs again at another level with `select-shp`
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
    logger.info(f'hypothesis test method: {method}')
    if method != 'ks':
        logger.warning('Currently only KS test is implemented. Switching to it.')
        method = 'ks'
    rslc_zarr = zarr.open(rslc, mode='r')
    logger.zarr_info(rslc, rslc_zarr)
    check_ndim('rslc', rslc_zarr, 3, '(nlines, width, nimages)')
    nlines, width, nimages = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]
    chunks = tuple(chunks)
    az_win = 2*az_half_win+1
    logger.info(f'azimuth half window size: {az_half_win}; azimuth window size: {az_win}')
    r_win = 2*r_half_win+1
    logger.info(f'range half window size: {r_half_win}; range window size: {r_win}')
    # window arrays: the whole window of a pixel in one chunk (decision 0032)
    win_shape = (nlines, width, az_win, r_win); win_chunks = (*chunks, az_win, r_win); win = (slice(0, az_win), slice(0, r_win))
    if pvalue is not None:
        logger.zarr_info(pvalue, zarr.open(pvalue, mode='w', shape=win_shape, dtype=np.float32, chunks=win_chunks))
    logger.zarr_info(is_shp, zarr.open(is_shp, mode='w', shape=win_shape, dtype=bool, chunks=win_chunks))
    logger.zarr_info(shp_num, zarr.open(shp_num, mode='w', shape=(nlines, width), dtype=np.int32, chunks=chunks))
    logger.info(f'selecting SHPs with p value >= alpha = {alpha}')
    # one task per (azimuth, range) chunk, read with a halo of the half windows; the p values are cut to the chunk
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping((nlines, width), chunks, (az_half_win, r_half_win))
    tasks = [([Chunk(rslc, (*in_sl, slice(0, nimages))), az_half_win, r_half_win, (*map_sl, slice(None), slice(None)),
               alpha, pvalue is not None],
              ([Chunk(pvalue, (*out_sl, *win))] if pvalue is not None else []) + [Chunk(is_shp, (*out_sl, *win)), Chunk(shp_num, out_sl)])
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
        input: pvalue of hypothesis test
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
    check_ndim('pvalue', p_zarr, 4, '(nlines, width, az_win, r_win)')
    nlines, width, az_win, r_win = p_zarr.shape
    if chunks is None: chunks = p_zarr.chunks[:2]
    chunks = tuple(chunks)
    # the whole window of a pixel in one chunk (decision 0032); the p values may still be one cell per chunk (older runs)
    is_shp_zarr = zarr.open(is_shp, mode='w', shape=p_zarr.shape, dtype=bool, chunks=(*chunks, az_win, r_win))
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
