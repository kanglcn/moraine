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
    required_data = mr.pu._mcf_network(pc_x_data, pc_y_data)
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
        f_mcf_delayed = delayed(mr.pu._mcf_unwrap,pure=True,nout=1)
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
    meta:str,
    unw_ph:str,
    pairs:str,
    misclosure:str=None,
    weight:str=None,
    earth_cost:int=1,
    t_scale:float=None,
    bperp_scale:float=None,
    temporal_cost:str='length+gradient',
    spatial_cost:str='gradient+correction+length',
    repair:bool=True,
    n_workers:int=None,
    out_chunks:int=None,
):
    """Extended minimum cost flow (EMCF) phase unwrapping of point cloud interferograms (own implementation, GAMMA
    not needed).

    Parameters
    ----------
    gix : str
        input: grid index (azimuth, range) of the point cloud, shape (n_points, 2), int; the points must
        be unique
    ph : str
        input: wrapped phase history (complex), shape (n_points, nimages)
    meta : str
        input: metadata toml file with the `dates` (YYYYMMDD), `perpendicular_baseline` (m),
        `range_pixel_spacing` and `azimuth_pixel_spacing` (m) of the images, as made by `load-gamma-metadata`
    unw_ph : str
        output: unwrapped phase of the interferograms in radians, shape (n_points, n_image_pairs), float32;
        at every point unw(a, b) + unw(b, c) = unw(a, c) for every triangle of images
    pairs : str
        output: text file with the image pairs of the interferograms (reference, secondary), one line per
        column of `unw_ph`
    misclosure : str, optional
        output: fraction of the triangles of images whose interferograms did not add up to zero before the
        repair, shape (n_points,), float32; 0 where the interferograms agreed, a quality measure
    weight : str, optional
        input: quality of the points from 0 to 1, e.g. the temporal coherence, shape (n_points,); used by
        `spatial_cost` 'weight'
    earth_cost : int, default: 1
        cost of a phase jump across the border of the network (of the points and of the images), relative
        to 1 inside; a larger value discourages discharging residues through the border
    t_scale : float, optional
        time distance in days that counts like `bperp_scale` of perpendicular baseline when the images are
        connected; the time span by default
    bperp_scale : float, optional
        perpendicular baseline distance in meters matching `t_scale`; the baseline span by default
    temporal_cost : str, default: 'length+gradient'
        which interferograms are corrected first where the interferograms of a triangle of images disagree:
        'gradient' (those with the least reliable phase difference between the two points), 'length' (the
        longest in time and baseline), 'length+gradient', or 'constant' (all alike)
    spatial_cost : str, default: 'gradient+correction+length'
        where phase jumps are placed first in every interferogram: 'constant' (anywhere alike) or a
        combination with + of 'gradient', 'correction', 'length' and 'weight', see `moraine.emcf_pc`
    repair : bool, default: True
        make the interferograms of every triangle of images add up to zero at every point
    n_workers : int, optional
        number of interferograms unwrapped at the same time; up to 8 by default
    out_chunks : int, optional
        point chunk size of the outputs, same as `ph` by default
    """
    import datetime
    from pathlib import Path
    import toml
    logger = logging.getLogger(__name__)
    m = toml.load(meta)
    dates = [datetime.datetime.strptime(str(d), '%Y%m%d') for d in m['dates']]
    t = np.array([(d - dates[0]).days for d in dates], dtype=np.float64)
    bperp = np.asarray(m['perpendicular_baseline'], dtype=np.float64)
    rps, azps = float(m.get('range_pixel_spacing', 1.0)), float(m.get('azimuth_pixel_spacing', 1.0))
    logger.info(f'{len(dates)} images, {t[-1]:.0f} days, perpendicular baselines {bperp.min():.1f} .. {bperp.max():.1f} m, '
                f'pixel spacing {rps} m (range) x {azps} m (azimuth)')

    gix_data = parallel_read_zarr(zarr.open(gix, mode='r'), (slice(None), slice(None)))
    ph_zarr = zarr.open(ph, mode='r')
    logger.zarr_info(ph, ph_zarr)
    ph_data = parallel_read_zarr(ph_zarr, (slice(None), slice(None)))
    weight_data = parallel_read_zarr(zarr.open(weight, mode='r'), (slice(None),)) if weight is not None else None
    if out_chunks is None: out_chunks = ph_zarr.chunks[0]

    logger.info('EMCF unwrapping')
    unw, image_pairs, mis = mr.emcf_pc(gix_data[:, 1] * rps, gix_data[:, 0] * azps, ph_data, t, bperp,
                                       weight=weight_data, earth_cost=earth_cost, t_scale=t_scale,
                                       bperp_scale=bperp_scale, temporal_cost=temporal_cost,
                                       spatial_cost=spatial_cost, repair=repair, n_workers=n_workers)
    logger.info(f'{image_pairs.shape[0]} interferograms unwrapped; triangles of images that did not close '
                f'before the repair: {mis.mean():.2%} (points with any: {np.mean(mis > 0):.1%})')
    unw_zarr = zarr.open(unw_ph, mode='w', shape=unw.shape, dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(unw_ph, unw_zarr)
    unw_zarr[:] = unw
    if misclosure is not None:
        mis_zarr = zarr.open(misclosure, mode='w', shape=mis.shape, dtype=np.float32, chunks=(out_chunks,))
        logger.zarr_info(misclosure, mis_zarr)
        mis_zarr[:] = mis
    Path(pairs).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(pairs, image_pairs, fmt='%d')
    logger.info(f'image pairs saved to {pairs}')
