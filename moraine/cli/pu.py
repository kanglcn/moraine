"""phase unwrapping"""


__all__ = ['gamma_mcf_pt', 'mcf_pc', 'emcf_pc', 'unwrap_correct_closure_pc']

import logging
import zarr
import time
import numpy as np

import dask
from dask import array as da
from dask import delayed
from dask.distributed import Client, LocalCluster, progress
import moraine as mr
from .logging import mc_logger
from . import dask_from_zarr, dask_to_zarr, parallel_read_zarr

@mc_logger
def gamma_mcf_pt(
    pc_x:str,
    pc_y:str,
    ph:str,
    unw_ph:str,
    image_pairs:np.ndarray,
    ref_point:int=0,
    out_chunks:int=None,
    n_workers=1,
    threads_per_worker=2,
    **dask_cluster_arg,
):
    """A wrapper for mcf_pt in GAMMA software.

    Parameters
    ----------
    pc_x : str
        input: x coordinate of the points, shape (n_points,)
    pc_y : str
        input: y coordinate of the points, shape (n_points,)
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages)
    unw_ph : str
        output: unwrapped phase of the interferograms, shape (n_points, n_image_pairs)
    image_pairs : np.ndarray
        image pairs (reference, secondary) of the interferograms to unwrap, shape (n_image_pairs, 2)
    ref_point : int, default: 0
        index of the reference point (from 0), the first point by default
    out_chunks : int, optional
        point chunk size of `unw_ph`, same as `ph` by default
    n_workers : default: 1
        number of dask worker, number of interferograms to be unwrapped in the same time
    threads_per_worker : default: 2
        number of threads per dask worker
    **dask_cluster_arg
        other dask local/cudalocal cluster args
    """

    logger = logging.getLogger(__name__)
    logger.info('load coordinates')
    pc_x_data = parallel_read_zarr(zarr.open(pc_x,mode='r'),(slice(None),))
    pc_y_data = parallel_read_zarr(zarr.open(pc_y,mode='r'),(slice(None),))
    logger.info('Done')

    ph_path = ph
    unw_ph_path = unw_ph

    ph_zarr = zarr.open(ph_path,mode='r')
    logger.zarr_info(ph_path,ph_zarr)
    npoint, nimage = ph_zarr.shape
    nimage_pairs = image_pairs.shape[0]

    if out_chunks is None: out_chunks = ph_zarr.chunks[0]

    Cluster = LocalCluster; cluster_args = {'processes':True, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
    cluster_args.update(dask_cluster_arg)

    logger.info('starting dask local cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)


        ph = dask_from_zarr(ph_path,chunks=(ph_zarr.shape[0],1))
        logger.darr_info('ph', ph)

        pc_x = da.from_array(pc_x_data,chunks=pc_x_data.shape)
        pc_y = da.from_array(pc_y_data,chunks=pc_y_data.shape)

        logger.info(f'phase wrapping with mcf.')

        pc_x_delayed = pc_x.to_delayed()[0]
        pc_y_delayed = pc_y.to_delayed()[0]
        ph_delayed = ph.to_delayed()[0]

        unw_ph_delayed = np.empty((1,nimage_pairs),dtype=object)
        f_mcf_delayed = delayed(mr.gamma_mcf_pt,pure=True,nout=1)
        f_intf_delayed = delayed(mr.intf,pure=True,nout=1)
        for i, (ref, sec) in enumerate(image_pairs):
            intf_delayed = f_intf_delayed(ph_delayed[ref],ph_delayed[sec])
            unw_ph_delayed[0,i] = f_mcf_delayed(pc_x_delayed, pc_y_delayed, intf_delayed, ref_point=ref_point)
            unw_ph_delayed[0,i] = da.from_delayed(unw_ph_delayed[0,i],shape=(npoint,1),meta=np.array((),dtype=np.float32))
        unw_ph = da.block(unw_ph_delayed.tolist())

        logger.info('got unwrapped phase.')
        logger.darr_info('unw_ph', unw_ph)
        logger.info('save unw_ph')
        _unw_ph = dask_to_zarr(unw_ph, unw_ph_path,chunks=(out_chunks,1))

        logger.info('computing graph setted. doing all the computing.')
        futures = client.persist(_unw_ph)
        progress(futures,notebook=False)
        time.sleep(0.1)
        da.compute(futures)
        logger.info('computing finished.')
    logger.info('dask cluster closed.')

@mc_logger
def mcf_pc(
    gix:str,
    ph:str,
    unw_ph:str,
    image_pairs:np.ndarray,
    range_pixel_spacing:float,
    azimuth_pixel_spacing:float,
    earth_cost:int=1,
    spatial_cost:str='constant',
    n_workers:int=None,
    out_chunks:int=None,
):
    """Minimum cost flow phase unwrapping of point cloud interferograms.

    Every interferogram is unwrapped alone on the Delaunay network of the points. The network stays in
    memory, about 50 bytes per point (100 with `spatial_cost` 'length'), and every interferogram unwrapped
    at the same time needs about 160 bytes per point more.

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the points, shape (n_points, 2), int; the points must be unique
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages), chunked one image per chunk
        (points_block, 1)
    unw_ph : str
        output: unwrapped phase of the interferograms in radians, shape (n_points, n_pairs), float32, chunks
        (out_chunks, 1); the first point keeps its wrapped phase
    image_pairs : np.ndarray
        image pairs (reference, secondary) of the interferograms to unwrap, shape (n_pairs, 2); the
        interferogram is reference * conj(secondary)
    range_pixel_spacing : float
        range pixel spacing in meters
    azimuth_pixel_spacing : float
        azimuth pixel spacing in meters
    earth_cost : int, default: 1
        cost of a phase jump across the convex hull of the points, relative to 1 inside; a larger value
        discourages discharging residues through the border
    spatial_cost : str, default: 'constant'
        where phase jumps are placed first: 'constant' (anywhere alike) or 'length' (long connections
        between points)
    n_workers : int, optional
        number of interferograms unwrapped at the same time; by default as many as the available cores and
        half of the available memory allow
    out_chunks : int, optional
        point chunk size of `unw_ph`, same as `ph` by default
    """
    from ..api.unwrap.emcf import _image_pairs
    from ..api.unwrap.mcf import _mcf_cost_network, _mcf_unwrap, _mcf_worker_bytes
    from ..api.utils_ import get_mem_avail, get_n_cpus_avail
    logger = logging.getLogger(__name__)

    ph_zarr = zarr.open(ph, mode='r')
    logger.zarr_info(ph, ph_zarr)
    n_points, nimages = ph_zarr.shape
    if ph_zarr.chunks[1] != 1:
        raise ValueError(f'{ph}: chunks {ph_zarr.chunks}; stacks are chunked one image per chunk, '
                         f'(points_block, 1) (docs/contracts/data.md)')
    pairs = _image_pairs(image_pairs, nimages)
    n_pairs = pairs.shape[0]
    if out_chunks is None:
        out_chunks = ph_zarr.chunks[0]
    gix_data = parallel_read_zarr(zarr.open(gix, mode='r'), (slice(None), slice(None)))
    if gix_data.shape != (n_points, 2):
        raise ValueError(f'gix must have shape ({n_points}, 2), got {gix_data.shape}')
    x = gix_data[:, 1] * float(range_pixel_spacing)
    y = gix_data[:, 0] * float(azimuth_pixel_spacing)
    del gix_data
    logger.info(f'triangulation of {n_points} points')      # made once, shared by all interferograms
    tri, half, hull, cost = _mcf_cost_network(x, y, spatial_cost)
    del x, y
    earth_cost = int(earth_cost)
    unw_zarr = zarr.open(unw_ph, mode='w', shape=(n_points, n_pairs), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(unw_ph, unw_zarr)

    def unwrap(k):
        intf = ph_zarr[:, pairs[k, 0]] * np.conj(ph_zarr[:, pairs[k, 1]])
        unw_zarr[:, k] = _mcf_unwrap(intf, tri, half, hull, earth_cost, cost)
        return k

    if not n_workers:
        per_worker = _mcf_worker_bytes(n_points, tri.shape[0] // 3)
        n_workers = max(1, min(n_pairs, get_n_cpus_avail(), int(0.5 * get_mem_avail() // per_worker)))
    logger.info(f'{n_pairs} interferograms, {n_workers} at the same time')
    dask.compute(*[delayed(unwrap)(k) for k in range(n_pairs)], scheduler='threads', num_workers=n_workers)
    logger.info('done.')


@mc_logger
def emcf_pc(
    gix:str,
    ph:str,
    image_pairs:np.ndarray,
    unw_ph:str,
    range_pixel_spacing:float,
    azimuth_pixel_spacing:float,
    dates:str|list=None,
    weight:str=None,
    earth_cost:int=1,
    temporal_cost:str='constant',
    spatial_cost:str='constant',
    n_workers:int=None,
    out_chunks:int=None,
):
    """Extended minimum cost flow (EMCF) phase unwrapping of point cloud interferograms.

    The interferograms of a network of image pairs are unwrapped together: the redundancy of the network
    (image pairs that close loops, e.g. (a, b), (b, c) and (a, c)) is used against unwrapping errors. The
    coordinates and the network of the points stay in memory, about 120 bytes per point, and every
    interferogram unwrapped at the same time needs about 200 bytes per point more. Intermediate results are
    written to `unw_ph`.tmp (kept if the command fails, removed at the end).

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the points, shape (n_points, 2), int; the points must be unique
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages), chunked one image per chunk
        (points_block, 1)
    image_pairs : np.ndarray
        the interferograms: (reference, secondary) image indices, shape (n_pairs, 2), e.g. a file made by
        `image-pairs` (`--bandwidth 3`: every image with the next three); the interferogram is
        reference * conj(secondary); without any loop in the network every interferogram is unwrapped alone
    unw_ph : str
        output: unwrapped phase of the interferograms in radians, shape (n_points, n_pairs), float32, chunks
        (out_chunks, 1); at the first point the difference of the wrapped phases of the two images. Loops of
        image pairs may still not add up to zero at some points: see `unwrap-correct-closure-pc`
    range_pixel_spacing : float
        range pixel spacing in meters
    azimuth_pixel_spacing : float
        azimuth pixel spacing in meters
    dates : str or list, optional
        acquisition date of every image (YYYYMMDD), nimages values; needed by `temporal_cost` 'length'
    weight : str, optional
        input: quality of the points from 0 (unreliable) to 1, e.g. the temporal coherence, shape (n_points,);
        needed by `spatial_cost` 'weight'
    earth_cost : int, default: 1
        cost of a phase jump across the border of the network of points, relative to 1 inside; a larger
        value discourages discharging residues through the border
    temporal_cost : str, default: 'constant'
        which interferograms are corrected first where the interferograms of a loop of image pairs disagree:
        'constant' (all alike) or 'length' (the longest in time)
    spatial_cost : str, default: 'constant'
        where phase jumps are placed first in every interferogram: 'constant' (anywhere alike), 'length'
        (long connections between points), 'weight' (points of low `weight`) or 'length+weight'
    n_workers : int, optional
        number of interferograms unwrapped at the same time; by default as many as the available cores and
        half of the available memory allow
    out_chunks : int, optional
        point chunk size of `unw_ph`, same as `ph` by default
    """
    import datetime
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from ..api.unwrap import emcf as _emcf
    from ..api.unwrap.closure import _pair_forest
    from ..api.unwrap.mcf import _edge_length, _mcf_edges
    logger = logging.getLogger(__name__)

    ph_zarr = zarr.open(ph, mode='r')
    logger.zarr_info(ph, ph_zarr)
    n_points, nimages = ph_zarr.shape
    if ph_zarr.chunks[1] != 1:
        raise ValueError(f'{ph}: chunks {ph_zarr.chunks}; stacks are chunked one image per chunk, '
                         f'(points_block, 1) (docs/contracts/data.md)')
    pairs = _emcf._image_pairs(image_pairs, nimages)
    n_pairs = pairs.shape[0]
    t = None
    if dates is not None:
        dates = [dates] if isinstance(dates, (str, int)) else list(dates)
        if len(dates) != nimages:
            raise ValueError(f'dates: {len(dates)} dates for {nimages} images')
        days = [datetime.datetime.strptime(str(d), '%Y%m%d') for d in dates]
        t = np.array([(d - days[0]).days for d in days], dtype=np.float64)
    pair_cost = _emcf._pair_costs(t, pairs, temporal_cost)
    flags = _emcf._spatial_cost_flags(spatial_cost, weight)
    order, parent, n_loops = _pair_forest(pairs, nimages)
    if n_loops == 0:
        logger.warning('the image pairs close no loop: every interferogram is unwrapped alone')
    if out_chunks is None:
        out_chunks = ph_zarr.chunks[0]

    gix_data = parallel_read_zarr(zarr.open(gix, mode='r'), (slice(None), slice(None)))
    if gix_data.shape != (n_points, 2):
        raise ValueError(f'gix must have shape ({n_points}, 2), got {gix_data.shape}')
    x = gix_data[:, 1] * float(range_pixel_spacing)
    y = gix_data[:, 0] * float(azimuth_pixel_spacing)
    del gix_data
    logger.info(f'triangulation of {n_points} points')
    tri, half, hull, edges, edge_of_half, sign_of_half = _mcf_edges(x, y)
    n_edges = edges.shape[0]
    edge_length = _edge_length(x, y, edges) if flags & 1 else np.ones(1, np.float32)
    del x, y
    if weight is not None:
        w = np.clip(parallel_read_zarr(zarr.open(weight, mode='r'), (slice(None),)).astype(np.float32), 0, 1)
    else:
        w = np.ones(1, np.float32)
    logger.info(f'{n_edges} edges, {nimages} images, {n_pairs} interferograms, {n_loops} independent loops')

    tmp = Path(str(unw_ph) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # temporal step per block of edges: the edges are sorted by their smaller point index, so a block reads
    # about one chunk of points of every image (plus the far ends of long edges); the next block is read
    # while the current one is computed (the kernel is parallel itself)
    block = 3 * ph_zarr.chunks[0]
    cycles_zarr = zarr.open(str(tmp / 'cycles.zarr'), mode='w', shape=(n_edges, n_pairs), dtype=np.int8,
                            chunks=(block, 1))
    logger.zarr_info(str(tmp / 'cycles.zarr'), cycles_zarr)

    def read_block(start):
        e = edges[start:start + block]
        rows = np.unique(e)
        return start, e, rows, ph_zarr.get_orthogonal_selection((rows, slice(None)))

    starts = list(range(0, n_edges, block))
    logger.info(f'temporal step: {len(starts)} blocks of {block} edges')
    with ThreadPoolExecutor(max_workers=1) as reader:
        future = reader.submit(read_block, starts[0]) if starts else None
        for j in range(len(starts)):
            start, e, rows, ph_rows = future.result()
            if j + 1 < len(starts):
                future = reader.submit(read_block, starts[j + 1])
            p = np.searchsorted(rows, e[:, 0])
            q = np.searchsorted(rows, e[:, 1])
            cycles_zarr[start:start + e.shape[0]] = _emcf._emcf_temporal(
                np.ascontiguousarray(ph_rows), p, q, pairs, pair_cost, order, parent)
            del ph_rows

    # spatial step per interferogram, in dask threads
    unw_zarr = zarr.open(unw_ph, mode='w', shape=(n_points, n_pairs), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(unw_ph, unw_zarr)
    earth_cost = int(earth_cost)

    def spatial(k):
        unw_zarr[:, k] = _emcf._emcf_spatial(
            ph_zarr[:, pairs[k, 0]], ph_zarr[:, pairs[k, 1]], cycles_zarr[:, k], tri, half, hull, edges,
            edge_of_half, sign_of_half, earth_cost, flags, edge_length, w)
        return k

    if not n_workers:
        n_workers = _emcf._spatial_workers(n_points, n_edges, tri.shape[0] // 3, flags, n_pairs)
    logger.info(f'spatial step: {n_pairs} interferograms, {n_workers} at the same time')
    dask.compute(*[delayed(spatial)(k) for k in range(n_pairs)], scheduler='threads', num_workers=n_workers)
    shutil.rmtree(tmp)
    logger.info('done.')


@mc_logger
def unwrap_correct_closure_pc(
    gix:str,
    ph:str,
    unw_ph:str,
    image_pairs:np.ndarray,
    unw_cor:str,
    range_pixel_spacing:float,
    azimuth_pixel_spacing:float,
    misclosure_fraction:str=None,
    region:str=None,
    max_edge_factor:float=4.0,
    min_region_points:int=30,
    n_workers:int=None,
    out_chunks:int=None,
):
    """Correction of unwrapping errors of point cloud interferograms by phase closure.

    The unwrapped phases of every loop of image pairs should add up to zero, e.g.
    unw(a, b) + unw(b, c) = unw(a, c). Where they do not, the interferograms are corrected by whole cycles,
    changing as few interferograms as possible, by the same amount for all points of a region: the points
    connected by edges of the point network no longer than `max_edge_factor` times the median edge length.
    Points of smaller regions than `min_region_points` are corrected one by one. The correction assumes
    that most interferograms of a region are right: where most are wrong, it makes them worse. The region
    of every point stays in memory (4 bytes per point; about 100 bytes per point while the regions are
    made). Intermediate results are written to `unw_cor`.tmp (kept if the command fails, removed at the end).

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the points, shape (n_points, 2), int; the points must be unique
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages), chunked one image per chunk
        (points_block, 1)
    unw_ph : str
        input: unwrapped phase of the interferograms in radians, shape (n_points, n_pairs), chunked one
        interferogram per chunk (points_block, 1), e.g. from `mcf-pc` or `emcf-pc`; rewrapped, it must be the
        phase of ph[:, reference] * conj(ph[:, secondary])
    image_pairs : np.ndarray
        the interferograms: (reference, secondary) image indices of the columns of `unw_ph`, shape
        (n_pairs, 2); the network needs loops (e.g. every image paired with the next three), otherwise
        nothing is corrected (a warning is given)
    unw_cor : str
        output: corrected unwrapped phase in radians, shape (n_points, n_pairs), float32, chunks
        (out_chunks, 1): `unw_ph` plus whole cycles, the same for all points of a region and interferogram
    range_pixel_spacing : float
        range pixel spacing in meters
    azimuth_pixel_spacing : float
        azimuth pixel spacing in meters
    misclosure_fraction : str, optional
        output: fraction of the interferograms of every point that do not fit the loops before the
        correction, shape (n_points,), float32, 0..1; 0 where every loop closes
    region : str, optional
        output: region of every point, shape (n_points,), int32: 0 .. n_regions - 1, or -1 for the points of
        regions smaller than `min_region_points`, corrected one by one (e.g. to be masked)
    max_edge_factor : float, default: 4.0
        edges of the point network longer than this times the median edge length do not connect a region,
        e.g. edges across water or decorrelated areas; larger than 1
    min_region_points : int, default: 30
        in regions of fewer points every point is corrected by its own loops alone, which is less reliable
    n_workers : int, optional
        number of interferograms corrected at the same time; by default as many as the available cores and
        half of the available memory allow
    out_chunks : int, optional
        point chunk size of the outputs, same as `unw_ph` by default
    """
    import shutil
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from ..api.unwrap import closure as _closure
    from ..api.unwrap.emcf import _image_pairs
    from ..api.unwrap.mcf import _mcf_edges
    from ..api.utils_ import get_mem_avail, get_n_cpus_avail
    logger = logging.getLogger(__name__)

    ph_zarr = zarr.open(ph, mode='r')
    unw_zarr = zarr.open(unw_ph, mode='r')
    logger.zarr_info(ph, ph_zarr)
    logger.zarr_info(unw_ph, unw_zarr)
    n_points, nimages = ph_zarr.shape
    n_pairs = unw_zarr.shape[1]
    for path, z in ((ph, ph_zarr), (unw_ph, unw_zarr)):
        if z.chunks[1] != 1:
            raise ValueError(f'{path}: chunks {z.chunks}; stacks are chunked one image (pair) per chunk, '
                             f'(points_block, 1) (docs/contracts/data.md)')
    if unw_zarr.shape[0] != n_points:
        raise ValueError(f'ph {ph_zarr.shape} and unw_ph {unw_zarr.shape} have different points')
    pairs = _image_pairs(image_pairs, nimages)
    if pairs.shape[0] != n_pairs:
        raise ValueError(f'image_pairs: {pairs.shape[0]} pairs for {n_pairs} columns of unw_ph')
    if not max_edge_factor > 1:
        raise ValueError(f'max_edge_factor must be larger than 1, got {max_edge_factor}')
    if int(min_region_points) < 1:
        raise ValueError(f'min_region_points must be at least 1, got {min_region_points}')
    if out_chunks is None:
        out_chunks = unw_zarr.chunks[0]
    order, parent, n_loops = _closure._pair_forest(pairs, nimages)

    gix_data = parallel_read_zarr(zarr.open(gix, mode='r'), (slice(None), slice(None)))
    if gix_data.shape != (n_points, 2):
        raise ValueError(f'gix must have shape ({n_points}, 2), got {gix_data.shape}')
    x = gix_data[:, 1] * float(range_pixel_spacing)
    y = gix_data[:, 0] * float(azimuth_pixel_spacing)
    del gix_data
    logger.info(f'regions of {n_points} points')
    edges = _mcf_edges(x, y)[3]
    reg, n_regions = _closure._closure_regions(x, y, edges, float(max_edge_factor), int(min_region_points))
    del x, y, edges
    logger.info(f'{n_regions} regions of at least {min_region_points} points, {np.mean(reg < 0):.2%} of the points '
                f'in smaller ones')
    if region is not None:
        z = zarr.open(region, mode='w', shape=(n_points,), dtype=np.int32, chunks=(out_chunks,))
        logger.zarr_info(region, z)
        z[:] = reg
    out_zarr = zarr.open(unw_cor, mode='w', shape=(n_points, n_pairs), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(unw_cor, out_zarr)
    mis_zarr = None
    if misclosure_fraction is not None:
        mis_zarr = zarr.open(misclosure_fraction, mode='w', shape=(n_points,), dtype=np.float32,
                             chunks=(out_chunks,))
        logger.zarr_info(misclosure_fraction, mis_zarr)

    if n_loops == 0:
        warnings.warn('the image pairs close no loop: nothing to correct')
        logger.warning('the image pairs close no loop: nothing to correct')
        for k in range(n_pairs):
            out_zarr[:, k] = unw_zarr[:, k]
        if mis_zarr is not None:
            mis_zarr[:] = 0
        return

    tmp = Path(str(unw_cor) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # per point corrections per block of points (one chunk of every image and interferogram); the next block
    # is read while the current one is computed (the kernel is parallel itself)
    block = unw_zarr.chunks[0]
    d_zarr = zarr.open(str(tmp / 'd.zarr'), mode='w', shape=(n_points, n_pairs), dtype=np.int8, chunks=(block, 1))
    mis = np.empty(n_points, np.float32)

    def read_block(start):
        sl = slice(start, min(start + block, n_points))
        return sl, ph_zarr[sl], unw_zarr[sl]

    starts = list(range(0, n_points, block))
    logger.info(f'per point corrections: {len(starts)} blocks of {block} points')
    with ThreadPoolExecutor(max_workers=1) as reader:
        future = reader.submit(read_block, starts[0])
        for j in range(len(starts)):
            sl, ph_rows, unw_rows = future.result()
            if j + 1 < len(starts):
                future = reader.submit(read_block, starts[j + 1])
            d, mis[sl], dev = _closure._closure_estimate(np.ascontiguousarray(ph_rows),
                                                         np.ascontiguousarray(unw_rows, dtype=np.float32),
                                                         pairs, order, parent)
            if dev > 0.1:
                raise ValueError(f'unw_ph does not rewrap to the interferograms of ph (up to {dev:.2f} cycles off)')
            d_zarr[sl] = d
    if mis_zarr is not None:
        mis_zarr[:] = mis
    logger.info(f'loops do not close at {np.mean(mis > 0):.1%} of the points, mean fraction of interferograms '
                f'that do not fit {mis.mean():.3f}')

    # correction per interferogram, in dask threads
    def correct(k):
        col = np.ascontiguousarray(unw_zarr[:, k], dtype=np.float32)
        _closure._closure_apply(col, d_zarr[:, k], reg, n_regions)
        out_zarr[:, k] = col
        return k

    if not n_workers:
        n_workers = max(1, min(n_pairs, get_n_cpus_avail(), int(0.5 * get_mem_avail() // (16 * n_points + 1))))
    logger.info(f'correction: {n_pairs} interferograms, {n_workers} at the same time')
    dask.compute(*[delayed(correct)(k) for k in range(n_pairs)], scheduler='threads', num_workers=n_workers)
    shutil.rmtree(tmp)
    logger.info('done.')
