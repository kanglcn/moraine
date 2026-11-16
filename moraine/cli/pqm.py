"""Pixel quality metrics (CLI)"""


__all__ = ['temp_coh', 'pc_su_dem_error']

import logging
import zarr
import numpy as np

import moraine as mr
from ..api.pqm import _check_su_inputs, _remove_su_dem_error
from ..api.utils_.chunk import all_chunk_slices
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
    if intf_zarr.shape[:-1] != rslc_zarr.shape[:-1]:
        raise ValueError(f'temp-coh: intf {intf_zarr.shape} and rslc {rslc_zarr.shape} do not have the same points or pixels')
    if image_pairs.shape[0] != n_pairs or (image_pairs.size and image_pairs.max() >= nimage):
        raise ValueError(f'temp-coh: intf has {n_pairs} interferograms and rslc {nimage} images, but image_pairs has '
                         f'{image_pairs.shape[0]} pairs of images up to {image_pairs.max() if image_pairs.size else "-"}; '
                         'are intf and rslc swapped?')
    t_coh_zarr = zarr.open(t_coh, mode='w', shape=shape, dtype=np.float32, chunks=chunks)
    logger.zarr_info(t_coh, t_coh_zarr)
    # one task per chunk of points or pixels: the interferograms and the rslc of the chunk in, the temporal coherence out
    tasks = [([Chunk(intf, (*sl, slice(0, n_pairs))), Chunk(rslc, (*sl, slice(0, nimage)))], [Chunk(t_coh, sl)])
             for sl in all_chunk_slices(shape, chunks)]
    logger.info(f'temporal coherence of {len(tasks)} chunks of {chunks}')
    with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(mr.temp_coh, tasks, desc='temporal coherence', image_pairs=image_pairs)
    logger.info('done.')


def _su_dem_error_task(intf, rslc, bperp, slant_range, theta, wavelength=None, image_pairs=None, max_dh=None,
                       with_phase=False):
    """DEM error relative to the neighbourhood and temporal coherence of a block of points (and their phase without
    that DEM error)"""
    dh, t_coh = mr.pc_su_dem_error(intf, rslc, bperp, slant_range, theta, wavelength, image_pairs, max_dh)
    if not with_phase:
        return dh, t_coh
    return dh, t_coh, _remove_su_dem_error(rslc, bperp, slant_range, theta, wavelength, dh)


@mc_logger
def pc_su_dem_error(
    intf:str,
    rslc:str,
    bperp:str,
    slant_range:str,
    theta:str,
    dh:str,
    t_coh:str,
    wavelength:float,
    image_pairs:np.ndarray=None,
    out:str=None,
    max_dh:float=20.0,
    chunks:int=None,
    processes=None,
    n_workers=None,
    threads_per_worker=None,
):
    """DEM error of every point relative to its neighbourhood in filtered interferograms, and the temporal coherence
    without it.

    A point dh meters above its neighbourhood (a roof among ground points) differs from the filtered interferogram of
    the images a and b by 4 pi / lambda (B_perp,a - B_perp,b) dh / (R cos(theta)) radians, which the temporal
    coherence takes for noise. dh is the DEM error within -max_dh .. max_dh that maximizes the temporal coherence of
    the point with this term removed, and `t_coh` is that coherence. Fitting dh raises the temporal coherence of noise
    a little: random phases reach about 0.35 (p99.99) with 91 sequential image pairs and +-80 m.

    Parameters
    ----------
    intf : str
        input: filtered interferograms of the points, the phase of their neighbourhood, shape (n_points, n_pairs),
        complex64 (e.g. of `n2ft`)
    rslc : str
        input: phase of the points (rslc or phase history), shape (n_points, nimages), complex64
    bperp : str
        input: perpendicular baseline of every image at every point in meters, shape (n_points, nimages), floating
        point; relative to the reference image of the co-registration, positive upwards (`load-gamma-bperp`, then
        `ras2pc`)
    slant_range : str
        input: slant range of the points in meters, shape (n_points,), floating point (`load-gamma-range`, then
        `ras2pc`)
    theta : str
        input: elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle),
        shape (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`)
    dh : str
        output: DEM error relative to the neighbourhood in meters, shape (n_points,), float32, chunks (chunks,): the
        height of the point minus that of its neighbourhood in `intf`, positive above it; nan where an input of the
        point is not finite or the baselines of all its image pairs are equal
    t_coh : str
        output: temporal coherence of the point without the term of `dh`, shape (n_points,), float32, chunks
        (chunks,), 0 .. 1 (nan where an input is not finite)
    wavelength : float
        radar wavelength in meters, between 0 and 1 (`radar_wavelength` of the metadata of `load-gamma-metadata`)
    image_pairs : np.ndarray, optional
        image pairs (reference, secondary) of `intf`, shape (n_pairs, 2); all image pairs by default
    out : str, optional
        output: `rslc` without the phase of `dh`, shape (n_points, nimages), complex64, chunks (chunks, 1); `rslc`
        where `dh` is nan
    max_dh : float, default: 20
        the DEM error is searched within -max_dh .. max_dh meters; a point further from its neighbourhood gets a low
        temporal coherence
    chunks : int, optional
        point chunk size of the outputs, that of `intf` by default; the points of a task
    processes : optional
        use processes (True) or threads (False) for the workers. Default: False
    n_workers : optional
        number of workers. Default: 1
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 1 (a task uses all cores); a task holds about 50 bytes per
        point and image
    """
    logger = logging.getLogger(__name__)
    z = {}
    for name, path in (('intf', intf), ('rslc', rslc), ('bperp', bperp), ('slant_range', slant_range),
                       ('theta', theta)):
        z[name] = zarr.open(path, mode='r')
        logger.zarr_info(path, z[name])
    image_pairs = _check_su_inputs(*z.values(), wavelength, image_pairs, max_dh)
    n_points, nimages = z['rslc'].shape
    n_pairs = z['intf'].shape[1]
    if chunks is None:
        chunks = z['intf'].chunks[0]
    logger.info(f'{n_pairs} image pairs of {nimages} images; DEM error searched within +-{max_dh} m')
    dh_zarr = zarr.open(dh, mode='w', shape=(n_points,), dtype=np.float32, chunks=(chunks,))
    logger.zarr_info(dh, dh_zarr)
    t_coh_zarr = zarr.open(t_coh, mode='w', shape=(n_points,), dtype=np.float32, chunks=(chunks,))
    logger.zarr_info(t_coh, t_coh_zarr)
    if out is not None:
        out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.complex64, chunks=(chunks, 1))
        logger.zarr_info(out, out_zarr)
    # one task per block of points with all its image pairs and images (whole chunks of the outputs)
    tasks = []
    for start in range(0, n_points, chunks):
        points = slice(start, min(start + chunks, n_points))
        tasks.append(([Chunk(intf, (points, slice(0, n_pairs))), Chunk(rslc, (points, slice(0, nimages))),
                       Chunk(bperp, (points, slice(0, nimages))), Chunk(slant_range, (points,)),
                       Chunk(theta, (points,))],
                      [Chunk(dh, (points,)), Chunk(t_coh, (points,)),
                       Chunk(out, (points, slice(0, nimages))) if out is not None else None]))
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(_su_dem_error_task, tasks, desc='point blocks', wavelength=float(wavelength),
                      image_pairs=image_pairs, max_dh=float(max_dh), with_phase=out is not None)
    values, coh = dh_zarr[:], t_coh_zarr[:]              # 8 bytes per point, for the log
    ok = np.isfinite(values)
    if ok.any():
        p05, p50, p95 = np.percentile(values[ok], [5, 50, 95])
        logger.info(f'dh of {ok.sum()} points ({n_points - ok.sum()} nan): percentiles 5, 50, 95 {p05:.2f}, '
                    f'{p50:.2f}, {p95:.2f} m')
    ok = np.isfinite(coh)
    if ok.any():
        logger.info(f'temporal coherence: median {np.median(coh[ok]):.3f}, > 0.8 for {(coh[ok] > 0.8).sum()} points')
    logger.info('done.')
