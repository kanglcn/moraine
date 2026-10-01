"""phase unwrapping"""


__all__ = ['gamma_mcf_pt', 'mcf_pc', 'emcf_pc']

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
    earth_cost:int=1,
    out_chunks:int=None,
    n_workers=1,
    threads_per_worker=2,
    **dask_cluster_arg,
):
    """Minimum cost flow phase unwrapping of point cloud interferograms.

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int; the points must
        be unique
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages)
    unw_ph : str
        output: unwrapped phase of the interferograms, shape (n_points, n_image_pairs)
    image_pairs : np.ndarray
        image pairs (reference, secondary) of the interferograms to unwrap, shape (n_image_pairs, 2)
    earth_cost : int, default: 1
        cost of a phase jump across the convex hull of the points, relative to 1 inside; a larger value
        discourages discharging residues through the border
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
    gix_data = parallel_read_zarr(zarr.open(gix, mode='r'),(slice(None),slice(None)))
    pc_x_data, pc_y_data = gix_data[:,1], gix_data[:,0]
    # pc_x_data = parallel_read_zarr(zarr.open(pc_x,mode='r'),(slice(None),))
    # pc_y_data = parallel_read_zarr(zarr.open(pc_y,mode='r'),(slice(None),))
    logger.info('Done')

    ph_path = ph
    unw_ph_path = unw_ph

    ph_zarr = zarr.open(ph_path,mode='r')
    logger.zarr_info(ph_path,ph_zarr)
    npoint, nimage = ph_zarr.shape
    nimage_pairs = image_pairs.shape[0]

    if out_chunks is None: out_chunks = ph_zarr.chunks[0]

    logger.info('Delaunay triangulation of the points')   # made once, shared by all interferograms
    required_data = mr.api.unwrap.mcf._mcf_network(pc_x_data, pc_y_data)
    logger.info('Done')

    Cluster = LocalCluster; cluster_args = {'processes':True, 'n_workers':n_workers, 'threads_per_worker':threads_per_worker}
    cluster_args.update(dask_cluster_arg)

    logger.info('starting dask local cluster.')
    with Cluster(**cluster_args) as cluster, Client(cluster) as client:
        logger.info('dask local cluster started.')
        logger.dask_cluster_info(cluster)

        required_data_future = [client.scatter(data, broadcast=True) for data in required_data]
        ph = dask_from_zarr(ph_path,chunks=(ph_zarr.shape[0],1))
        logger.darr_info('ph', ph)
        # ph_delayed = ph.to_delayed()[0]
        logger.info(f'phase wrapping with mcf.')

        unw_ph_delayed = np.empty((1,nimage_pairs),dtype=object)
        f_mcf_delayed = delayed(mr.api.unwrap.mcf._mcf_unwrap,pure=True,nout=1)
        f_intf_delayed = delayed(mr.intf,pure=True,nout=1)
        for i, (ref, sec) in enumerate(image_pairs):
            ref_ph_delayed = ph[:,ref].to_delayed()[0]
            sec_ph_delayed = ph[:,sec].to_delayed()[0]
            intf_delayed = f_intf_delayed(ref_ph_delayed,sec_ph_delayed)
            # intf_delayed = f_intf_delayed(ph_delayed[ref],ph_delayed[sec])
            unw_ph_delayed[0,i] = f_mcf_delayed(intf_delayed, *required_data_future, earth_cost)
            unw_ph_delayed[0,i] = da.from_delayed(unw_ph_delayed[0,i],shape=(npoint,),meta=np.array((),dtype=np.float32)).reshape(npoint,1)
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
    spatial_cost:str='length',
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
        image pairs may still not add up to zero at some points: see `unwrap_correct_closure_pc`
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
    spatial_cost : str, default: 'length'
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
    from ..api.unwrap.mcf import _mcf_edges
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
    edge_length = _emcf._edge_length(x, y, edges) if flags & 1 else np.ones(1, np.float32)
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
