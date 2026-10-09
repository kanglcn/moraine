"""Phase linking (CLI)"""


__all__ = ['emi', 'ds_temp_coh', 'emperical_co_emi_temp_coh_pc']

import logging
import zarr
import numpy as np
from pathlib import Path
import math

import moraine as mr
from ..api.chunk_ import all_chunk_slices, all_chunk_slices_with_overlap
from .logging import mc_logger
from .zarr_ import parallel_read_zarr
from .executor import Executor, Chunk, Device, chunk_task
from .utils_ import mk_clean_dir


def _pc_by_ras_chunk(gix, chunks, shape, overlap):
    """The points in the order of the raster chunks: for every chunk its grid index relative to the chunk read with
    `overlap`, and the point bounds of every chunk. The chunks are numbered like `all_chunk_slices` (azimuth major)."""
    chunk_idx, chunk_bounds = mr.api.pc._pc_split_by_chunk(gix, chunks, shape)[:2]
    sorted_gix = gix[chunk_idx]
    ras_chunk_order_gix = mr.api.pc._gix_ras_chunk(sorted_gix, chunk_bounds, chunks, shape, overlap=overlap)
    return ras_chunk_order_gix, chunk_bounds


def _make_pc_zarr(path, n_points, nimages, dtype):
    """A point cloud zarr of one raster chunk: (n_points, nimages) chunked one image per chunk, or (n_points,)."""
    shape = (n_points, nimages) if nimages else (n_points,)
    chunks = (n_points, 1) if nimages else (n_points,)
    zarr.open(str(path), mode='w', shape=shape, dtype=dtype, chunks=chunks)

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
        use processes (True) or threads (False) for the workers, only for cpu processing. Default: False
    n_workers : optional
        number of workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 2
    rmm_pool_size : default: 0.9
        fraction of the GPU memory of each worker taken by an rmm memory pool (rmm must be installed), only with cuda
    """
    logger = logging.getLogger(__name__)
    coh_zarr = zarr.open(coh, mode='r')
    n_points, n_pairs = coh_zarr.shape
    n_image = mr.nimage_from_npair(n_pairs)
    logger.zarr_info(coh, coh_zarr)
    if chunks is None: chunks = coh_zarr.chunks[0]
    chunks = chunks[0] if isinstance(chunks, (tuple, list)) else int(chunks)
    if threads_per_worker is None and not cuda: threads_per_worker = 2
    ph_zarr = zarr.open(ph, mode='w', shape=(n_points, n_image), dtype=coh_zarr.dtype, chunks=(chunks, 1))
    logger.zarr_info(ph, ph_zarr)
    # one task per chunk of points: their coherence in, their phase history out
    tasks = [([Chunk(coh, (sl, slice(0, n_pairs)))], [Chunk(ph, (sl, slice(0, n_image)))])
             for (sl,) in all_chunk_slices((n_points,), (chunks,))]
    logger.info(f'phase linking with EMI of {len(tasks)} chunks of {chunks} points')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size) as ex:
        ex.map_chunks(mr.emi, tasks, desc='phase linking', ref=ref, regularize=regularize)
    logger.info('done.')

@mc_logger
def ds_temp_coh(
    coh:str,
    ph:str,
    t_coh:str=None,
    tnet:str=None,
    n_looks:str=None,
    t_coh_w:str=None,
    eff_n_pairs:str=None,
    n_components:str=None,
    alpha:float=1e-3,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
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
    n_components : str, optional
        output: number of connected components of the graph whose nodes are the images and whose edges are
        the coherent image pairs, int16, shape (n_points,), 1..nimages; 1 means every image is linked to
        every other by coherent pairs, with more the phase between the groups of images is not constrained
        by any coherent pair, whatever the weighted temporal coherence is; needs `n_looks`
    alpha : float, default: 1e-3
        probability that a point without any coherent image pair (pure noise) has all images connected
        (`n_components` 1); it sets the squared coherence above which an image pair counts as coherent,
        depending on the number of looks and of images; 1e-6 <= alpha <= 0.5; only with `n_looks`
    chunks : int, optional
        point cloud chunk size, same as coh by default
    cuda : bool, default: False
        if use cuda for processing, false by default
    processes : optional
        use processes (True) or threads (False) for the workers, only for cpu processing. Default: False
    n_workers : optional
        number of workers. Default: 1 for cpu, one per GPU for cuda
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 2
    rmm_pool_size : default: 0.9
        fraction of the GPU memory of each worker taken by an rmm memory pool (rmm must be installed), only with cuda
    """
    logger = logging.getLogger(__name__)
    weighted = (t_coh_w is not None) or (eff_n_pairs is not None) or (n_components is not None)
    if weighted and n_looks is None:
        raise ValueError('t_coh_w, eff_n_pairs and n_components need n_looks')
    coh_zarr = zarr.open(coh, mode='r'); logger.zarr_info(coh, coh_zarr)
    ph_zarr = zarr.open(ph, mode='r'); logger.zarr_info(ph, ph_zarr)
    n_points, n_pairs = coh_zarr.shape
    nimage = ph_zarr.shape[-1]
    if chunks is None: chunks = coh_zarr.chunks[0]
    chunks = chunks[0] if isinstance(chunks, (tuple, list)) else int(chunks)
    if threads_per_worker is None and not cuda: threads_per_worker = 2
    if tnet is not None:
        tnet = mr.TempNet.load(tnet)
    else:
        tnet = mr.TempNet.from_bandwidth(nimage)
    image_pairs = tnet.image_pairs
    outputs = [(t_coh, np.float32), (t_coh_w, np.float32), (eff_n_pairs, np.float32), (n_components, np.int16)]
    if not weighted:
        outputs = outputs[:1]
    for path, dtype in outputs:
        if path is not None:
            logger.zarr_info(path, zarr.open(path, mode='w', shape=(n_points,), dtype=dtype, chunks=(chunks,)))
    # one task per chunk of points: coherence, phase history (and number of looks) in, the temporal coherences out
    tasks = []
    for (sl,) in all_chunk_slices((n_points,), (chunks,)):
        inputs = [Chunk(coh, (sl, slice(0, n_pairs))), Chunk(ph, (sl, slice(0, nimage)))]
        tasks.append((inputs, [Chunk(path, (sl,)) if path is not None else None for path, _ in outputs]))
    kwargs = {'image_pairs': image_pairs}
    if weighted:
        kwargs['alpha'] = alpha
    logger.info(f'temporal coherence of {len(tasks)} chunks of {chunks} points' + (', weighted' if weighted else ''))
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size) as ex:
        if weighted:    # the number of looks of the chunk is read with the chunk
            ex.map(chunk_task, [(mr.ds_temp_coh, inputs, outs, cuda, {**kwargs, 'n_looks': Chunk(n_looks, (inputs[0].slices[0],))})
                                for inputs, outs in tasks], desc='temporal coherence')
        else:
            ex.map_chunks(mr.ds_temp_coh, tasks, desc='temporal coherence', **kwargs)
    logger.info('done.')

@mc_logger
def emperical_co_emi_temp_coh_pc(
    rslc:str,
    is_shp_dir:str,
    gix:str,
    ph_dir:str,
    t_coh_dir:str,
    t_coh_w_dir:str=None,
    eff_n_pairs_dir:str=None,
    n_components_dir:str=None,
    alpha:float=1e-3,
    batch_size:int=None,
    regularize:bool=True,
    chunks:tuple[int,int]=None,
    cuda:bool=False,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
    rmm_pool_size=0.9,
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
    n_components_dir : str, optional
        output: directory with the number of connected components of the graph whose nodes are the images
        and whose edges are the coherent image pairs (see `ds-temp-coh`), shape (n_points,), int16,
        1..nimages, one zarr per raster chunk; 1 means every image is linked to every other by coherent
        pairs
    alpha : float, default: 1e-3
        probability that a point without any coherent image pair (pure noise) has all images connected;
        1e-6 <= alpha <= 0.5; see `ds-temp-coh`; only with one of the weighted outputs
    batch_size : int, optional
        number of points a worker processes at once; the coherence matrices of a batch take 8 bytes per image
        pair and point, their estimation on the GPU up to 1 GiB more. Default: as many points as 1 GiB of
        coherence matrices hold (small batches cost time: thousands of small GPU operations per batch)
    regularize : bool, default: True
        regularize the coherence matrix in the phase linking as `regularize` of `emi`; the temporal
        coherence is computed with the coherence matrix as estimated
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
    rmm_pool_size : default: 0.9
        fraction of the GPU memory of each worker taken by an rmm memory pool (rmm must be installed), only with cuda
    """
    logger = logging.getLogger(__name__)
    is_shp_dir = Path(is_shp_dir)
    ph_dir = Path(ph_dir); mk_clean_dir(ph_dir)
    t_coh_dir = Path(t_coh_dir); mk_clean_dir(t_coh_dir)
    weighted = (t_coh_w_dir is not None) or (eff_n_pairs_dir is not None) or (n_components_dir is not None)
    extra_dirs = []
    for d in (t_coh_w_dir, eff_n_pairs_dir, n_components_dir):
        if d is not None:
            d = Path(d); mk_clean_dir(d)
        extra_dirs.append(d)
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
    in_slices = all_chunk_slices_with_overlap((nlines, width, nimage), (*chunks, nimage), (az_half_win, r_half_win, 0))
    # one task per raster chunk with points: the rslc of the chunk with its halo, the grid index of its points
    # (relative to the chunk) and their SHPs in; the phase history and the temporal coherences of its points out,
    # one zarr per chunk
    tasks = []
    for j, in_sl in enumerate(in_slices):
        b0, b1 = int(chunk_bounds[j]), int(chunk_bounds[j+1])
        if b1 == b0:
            continue
        _make_pc_zarr(ph_dir/f'{j}.zarr', b1-b0, nimage, rslc_zarr.dtype)
        _make_pc_zarr(t_coh_dir/f'{j}.zarr', b1-b0, 0, np.float32)
        outputs = [Chunk(str(ph_dir/f'{j}.zarr')), Chunk(str(t_coh_dir/f'{j}.zarr'))]
        for d, dtype in zip(extra_dirs, (np.float32, np.float32, np.int16)):
            if d is None:
                outputs.append(None)
            else:
                _make_pc_zarr(d/f'{j}.zarr', b1-b0, 0, dtype)
                outputs.append(Chunk(str(d/f'{j}.zarr')))
        if not weighted:
            outputs = outputs[:2]
        inputs = [Chunk(rslc, in_sl), Device(ras_chunk_order_gix[b0:b1]), Chunk(str(is_shp_dir/f'{j}.zarr'))]
        tasks.append((inputs, outputs))
    logger.info(f'coherence, phase linking and temporal coherence of {len(tasks)} raster chunks with points')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                  rmm_pool_size=rmm_pool_size) as ex:
        ex.map_chunks(mr.emperical_co_emi_temp_coh_pc, tasks, desc='phase linking', batch_size=batch_size,
                      regularize=regularize, weighted=weighted, alpha=alpha)
    logger.info('done.')
