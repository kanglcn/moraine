"""Covariance and coherence matrix estimation (CLI)"""


__all__ = ['emperical_co_pc']

import logging
from pathlib import Path
import zarr
import numpy as np

import moraine as mr
from ..api.chunk_ import all_chunk_slices_with_overlap
from .logging import mc_logger
from .dask_ import parallel_read_zarr
from .executor import Executor, Chunk, Device
from .utils_ import mk_clean_dir
from .pl import _pc_by_ras_chunk, _make_pc_zarr

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
    logger = logging.getLogger(__name__)
    is_shp_dir = Path(is_shp_dir)
    coh_dir = Path(coh_dir); mk_clean_dir(coh_dir)
    if n_looks_dir is not None: n_looks_dir = Path(n_looks_dir); mk_clean_dir(n_looks_dir)
    rslc_zarr = zarr.open(rslc, mode='r')
    logger.zarr_info(rslc, rslc_zarr)
    assert rslc_zarr.ndim == 3, "rslc dimentation is not 3."
    nlines, width, nimage = rslc_zarr.shape
    if chunks is None: chunks = rslc_zarr.chunks[:2]
    chunks = tuple(chunks)
    if threads_per_worker is None and not cuda: threads_per_worker = 2
    is_shp0_zarr = zarr.open(sorted(is_shp_dir.glob('*.zarr'))[0], mode='r')
    az_win, r_win = is_shp0_zarr.shape[1:]
    az_half_win = int((az_win-1)/2)
    r_half_win = int((r_win-1)/2)
    logger.info(f'azimuth window size and half azimuth window size: {az_win}, {az_half_win}')
    logger.info(f'range window size and half range window size: {r_win}, {r_half_win}')
    logger.info(f'parallel processing azimuth, range chunk size: {chunks}')
    gix_zarr = zarr.open(gix, mode='r')
    logger.zarr_info(gix, gix_zarr)
    assert gix_zarr.ndim == 2, "gix dimentation is not 2."
    logger.info('loading gix into memory.')
    gix = parallel_read_zarr(gix_zarr, (slice(None), slice(None)))
    logger.info('convert gix to the order of ras chunk')
    ras_chunk_order_gix, chunk_bounds = _pc_by_ras_chunk(gix, chunks, (nlines, width), (az_half_win, r_half_win))
    del gix
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(nimage).image_pairs
    image_pairs = np.asarray(image_pairs).astype(np.int32)
    n_image_pairs = image_pairs.shape[0]
    return_n_looks = n_looks_dir is not None
    in_slices = all_chunk_slices_with_overlap((nlines, width, nimage), (*chunks, nimage), (az_half_win, r_half_win, 0))
    # one task per raster chunk with points: the rslc of the chunk with its halo, the grid index of its points
    # (relative to the chunk) and their SHPs in; the coherence (and number of looks) of its points out, one zarr per
    # chunk
    tasks = []
    for j, in_sl in enumerate(in_slices):
        b0, b1 = int(chunk_bounds[j]), int(chunk_bounds[j+1])
        if b1 == b0:
            continue
        _make_pc_zarr(coh_dir/f'{j}.zarr', b1-b0, n_image_pairs, np.complex64)
        outputs = [Chunk(str(coh_dir/f'{j}.zarr'))]
        if return_n_looks:
            _make_pc_zarr(n_looks_dir/f'{j}.zarr', b1-b0, 0, np.float32)
            outputs.append(Chunk(str(n_looks_dir/f'{j}.zarr')))
        inputs = [Chunk(rslc, in_sl), Device(ras_chunk_order_gix[b0:b1]), Chunk(str(is_shp_dir/f'{j}.zarr'))]
        tasks.append((inputs, outputs))
    logger.info(f'coherence of {len(tasks)} raster chunks with points')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size, **dask_cluster_arg) as ex:
        ex.map_chunks(mr.emperical_co_pc, tasks, desc='coherence', image_pairs=image_pairs, return_n_looks=return_n_looks)
    logger.info('done.')
