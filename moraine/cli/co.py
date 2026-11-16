"""Covariance and coherence matrix estimation (CLI)"""


__all__ = ['emperical_co_pc']

import logging
from pathlib import Path
import zarr
import numpy as np

import moraine as mr
from .logging import mc_logger
from .executor import Executor, Chunk
from .utils_ import mk_clean_dir
from .pl import _ras_chunk_inputs, _make_pc_zarr

@mc_logger
def emperical_co_pc(
    rslc:str,
    is_shp_dir:str,
    gix:str,
    coh_dir:str,
    n_looks_dir:str=None,
    coh_mean:str=None,
    image_pairs:np.ndarray=None,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
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
    coh_mean : str, optional
        output: mean coherence magnitude of every image pair over the points, float32, shape (nimages,
        nimages): element (i, j) is the mean of |coherence| of the pair of images i and j over the points
        where it is finite; symmetric, 1 on the diagonal, NaN for the image pairs not in `image_pairs` and for
        those without a finite coherence. An image with low values in its row has a low coherence with the
        other images
    image_pairs : np.ndarray, optional
        input: image pairs (element in the coherence matrix) to be calculated, all image pairs by default
    chunks : tuple[int, int], optional
        parallel processing (azimuth, range) chunk size. Default: rslc.chunks[:2]
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the workers, only for cpu processing. Default: False
    n_workers : optional
        number of workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 2
    """
    logger = logging.getLogger(__name__)
    coh_dir = Path(coh_dir); mk_clean_dir(coh_dir)
    if n_looks_dir is not None: n_looks_dir = Path(n_looks_dir); mk_clean_dir(n_looks_dir)
    if threads_per_worker is None and not cuda: threads_per_worker = 2
    rslc_zarr, chunk_inputs = _ras_chunk_inputs(rslc, is_shp_dir, gix, chunks, logger)
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(rslc_zarr.shape[2]).image_pairs
    image_pairs = np.asarray(image_pairs).astype(np.int32)
    n_image_pairs = image_pairs.shape[0]
    return_n_looks = n_looks_dir is not None
    # one task per raster chunk with points: the rslc of the chunk with its halo, the grid index of its points
    # (relative to the chunk) and their SHPs in; the coherence (and number of looks) of its points out, one zarr per
    # chunk
    tasks = []
    for j, n, inputs in chunk_inputs:
        _make_pc_zarr(coh_dir/f'{j}.zarr', n, n_image_pairs, np.complex64)
        outputs = [Chunk(str(coh_dir/f'{j}.zarr'))]
        if return_n_looks:
            _make_pc_zarr(n_looks_dir/f'{j}.zarr', n, 0, np.float32)
            outputs.append(Chunk(str(n_looks_dir/f'{j}.zarr')))
        tasks.append((inputs, outputs))
    logger.info(f'coherence of {len(tasks)} raster chunks with points')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        sums = ex.map_chunks(_emperical_co_pc_chunk, tasks, desc='coherence', image_pairs=image_pairs,
                             return_n_looks=return_n_looks, coh_sums=coh_mean is not None)
    if coh_mean is not None:
        _write_coh_mean(coh_mean, sums, image_pairs, rslc_zarr.shape[2], logger)
    logger.info('done.')


def _emperical_co_pc_chunk(rslc, gix, is_shp, image_pairs, return_n_looks, coh_sums):
    """the coherence (and the number of looks) of the points of a raster chunk; with `coh_sums` also the sum over the
    points of the finite coherence magnitudes of every image pair, float64, and their number, int64, (n_image_pairs,)"""
    out = mr.emperical_co_pc(rslc, gix, is_shp, image_pairs=image_pairs, return_n_looks=return_n_looks)
    if not coh_sums:
        return out
    out = out if isinstance(out, tuple) else (out,)
    # the sums of the chunk while its coherence is in memory, on its device; the task returns them (`chunk_task`)
    return (*out, *mr.api.ds.co._coh_abs_sums(out[0]))


def _write_coh_mean(path, sums, image_pairs, nimages, logger):
    """write the mean coherence magnitude of the image pairs, (nimages, nimages) float32, from the sums and numbers of
    the finite coherence magnitudes of the tasks"""
    total = np.zeros(len(image_pairs), dtype=np.float64)
    count = np.zeros(len(image_pairs), dtype=np.int64)
    for s, n in sums:
        total += s
        count += n
    mean = np.full((nimages, nimages), np.nan, dtype=np.float32)
    has = count > 0
    ref, sec = image_pairs[has, 0], image_pairs[has, 1]
    mean[ref, sec] = mean[sec, ref] = total[has]/count[has]
    np.fill_diagonal(mean, 1)
    z = zarr.open(path, mode='w', shape=mean.shape, dtype=mean.dtype, chunks=mean.shape)
    z[:] = mean
    logger.zarr_info(path, z)
