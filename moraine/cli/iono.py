"""Ionosphere (CLI): download of the global ionosphere maps of JPL and the line of sight displacement the ionosphere
causes at point clouds (decision 0057)"""

__all__ = ['download_ionex', 'pc_iono_tec']

import datetime
import logging
import os
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import zarr

import moraine as mr
from ..api.utils_.times import _times
from ..api.iono import _wavelength, _ratio
from .logging import mc_logger
from .utils_ import ArgumentError
from .executor import Executor, Chunk

# The final global ionosphere maps of JPL (2.5 by 5 degrees, every 2 hours, 13 maps from 00:00 to 24:00 UTC per day)
# on the server of JPL, without an account; the same files as JPL's in the IGS archive of NASA CDDIS (Earthdata
# login), from 2008-11-19 on, a few weeks after the day.
_JPL_URL = 'https://sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final/y{day:%Y}/{name}'
_FIRST_DAY = datetime.date(2008, 11, 19)


def _ionex_name(day)->str:
    """the file name of the maps of one day (date, UTC)"""
    return f'JPLG{day.timetuple().tm_yday:03d}0.{day:%y}I.gz'


def _days(t_img)->list:
    """the UTC days of the images, sorted"""
    return sorted({t.date() for t in t_img})


def _covers(path, day)->bool:
    """whether `path` is an IONEX file with maps over the whole `day` (00:00 to 24:00 UTC)"""
    try:
        times = mr.read_ionex(path)['times']
    except (OSError, ValueError, EOFError, UnicodeError, zlib.error):     # missing, cut or damaged
        return False
    start = datetime.datetime.combine(day, datetime.time())
    return times[0] <= start and times[-1] >= start + datetime.timedelta(days=1)


def _download_day(day, out_dir, timeout=60, retries=4):
    """download the maps of one day into `out_dir`, through a temporary file"""
    path = Path(out_dir) / _ionex_name(day)
    url = _JPL_URL.format(day=day, name=path.name)
    part = path.with_name(path.name + '.part')
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(f'{url} is not on the server of JPL: its final maps follow the day by a few '
                                        'weeks') from None
            if e.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    try:
        part.write_bytes(data)
        if not _covers(part, day):
            raise ValueError(f'{url} is not an IONEX file with the maps of {day:%Y-%m-%d} from 00:00 to 24:00 UTC')
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


@mc_logger
def download_ionex(
    out_dir:str,
    times:list[str],
    n_workers:int=4,
):
    """Download the final global ionosphere maps of JPL of the days of the images.

    One IONEX file per day (UTC) of the images, with the maps of the vertical total electron content (2.5 by 5
    degrees, every 2 hours) from the server of JPL, without an account, from 2008-11-19 on; the maps of a day come a
    few weeks after it. Files that exist and hold the maps of their day are kept, so an interrupted download continues
    where it stopped.

    Parameters
    ----------
    out_dir : str
        output: directory of the IONEX files, created if missing; one gzip file per day, JPLGddd0.yyI.gz (day of the
        year ddd, year yy), for `pc-iono-tec`
    times : list of str
        time of every image in UTC, ISO 8601 (e.g. 2019-01-02T05:06:07.123456Z), from 2008-11-19 on: the `times` of
        the metadata of `load-gamma-metadata`
    n_workers : int, default: 4
        number of days downloaded at the same time
    """
    logger = logging.getLogger(__name__)
    t_img = _times(times)
    if not t_img:
        raise ArgumentError('times must have the time of at least one image')
    days = _days(t_img)
    if days[0] < _FIRST_DAY:
        raise ArgumentError(f'times must be from {_FIRST_DAY} on (the maps of JPL on its server), got '
                            f'{min(t_img):%Y-%m-%d %H:%M:%S}')
    if isinstance(n_workers, bool) or not isinstance(n_workers, (int, np.integer)) or n_workers < 1:
        raise ArgumentError(f'n_workers must be a positive integer, got {n_workers!r}')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not _covers(out / _ionex_name(d), d)]
    logger.info(f'{len(t_img)} images on {len(days)} days: {len(days) - len(todo)} days are in {out_dir} already, '
                f'{len(todo)} to download from the server of JPL')
    if todo:
        with Executor(n_workers=int(n_workers)) as ex:
            ex.map(_download_day, [(d, out) for d in todo], desc='days')
    logger.info('done.')


# ---------------------------------------------------------------- displacement of point clouds

def _iono_task(lon, lat, theta, phi, **kwargs):
    """the ionosphere of a block of points, and per image the number, sum and sum of squares of its finite values"""
    iono = mr.pc_iono_tec(lon, lat, theta, phi, **kwargs)
    stats = np.zeros((3, iono.shape[1]))
    for k in range(iono.shape[1]):             # one column at a time: no copy of the whole block
        col = iono[:, k]
        col = col[np.isfinite(col)].astype(np.float64)
        stats[:, k] = col.size, col.sum(), (col * col).sum()
    return iono, stats


def _image_maps(ionex_dir, t_img)->dict:
    """the maps around the time of every image from the file of its day: times, maps (nmaps, nlat, nlon), grid and
    layer of the files"""
    days = _days(t_img)
    missing = [d for d in days if not (Path(ionex_dir) / _ionex_name(d)).is_file()]
    if missing:
        raise ValueError(f'{ionex_dir}: no IONEX file of {len(missing)} of the {len(days)} days of the images, e.g. '
                         f'{_ionex_name(missing[0])}; download them with `download-ionex`')
    maps, first = {}, None
    for day in days:
        path = Path(ionex_dir) / _ionex_name(day)
        ionex = mr.read_ionex(path)
        if first is None:
            first = ionex
        for key in ('lat', 'lon'):
            if ionex[key].shape != first[key].shape or not np.allclose(ionex[key], first[key], rtol=0, atol=1e-6):
                raise ValueError(f'{path} has other {key} than {Path(ionex_dir) / _ionex_name(days[0])}: the maps '
                                 'must be of one grid')
        if ionex['height'] != first['height'] or ionex['radius'] != first['radius']:
            raise ValueError(f'{path} has another layer (height, radius) than {Path(ionex_dir) / _ionex_name(days[0])}')
        epochs = ionex['times']
        for t in t_img:
            if t.date() != day:
                continue
            k = int(np.searchsorted(np.array(epochs, dtype='datetime64[us]'), np.datetime64(t, 'us'), side='right'))
            if k == 0 or (k == len(epochs) and epochs[-1] != t):
                raise ValueError(f'{path}: its maps from {epochs[0]:%Y-%m-%d %H:%M} to {epochs[-1]:%Y-%m-%d %H:%M} '
                                 f'do not hold the image of {t:%Y-%m-%d %H:%M:%S}')
            k = min(k, len(epochs) - 1)
            # the two maps around the image, from the file of its day; where the files of two days give a map of the
            # same time (24:00 of one, 00:00 of the next), the later day's is kept
            for j in (k - 1, k):
                maps[epochs[j]] = ionex['tec'][j]
    times = sorted(maps)
    return {'times': times, 'tec': np.stack([maps[t] for t in times]), 'lat': first['lat'], 'lon': first['lon'],
            'height': first['height'], 'radius': first['radius']}


@mc_logger
def pc_iono_tec(
    ionex_dir:str,
    lon:str,
    lat:str,
    theta:str,
    phi:str,
    out:str,
    times:list[str],
    wavelength:float,
    ratio:float=1.0,
    ref:int=0,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Line of sight displacement caused by the ionosphere at the points, from the global ionosphere maps, relative to a reference image.

    The vertical total electron content of the maps at the pierce point of the look vector of a point through the
    layer of the maps, interpolated in time between the maps around the time of the image, times the mapping factor
    and `ratio`: the electron content along the line of sight below the satellite. The ionosphere shortens the phase
    path by 40.31 STEC / f^2 (STEC in electrons per square meter, f the radar frequency): it corrects the long
    wavelengths, of the scale of the maps (hundreds of kilometers), not the smaller structures of the ionosphere.

    Parameters
    ----------
    ionex_dir : str
        input: directory of the IONEX files of `download-ionex`, one file per day (UTC) of the images, JPLGddd0.yyI.gz
        (plain text or gzip), all of one grid
    lon : str
        input: longitude of the points in degrees, shape (n_points,), floating point
    lat : str
        input: latitude of the points in degrees, shape (n_points,), floating point
    theta : str
        input: elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle),
        shape (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`)
    phi : str
        input: orientation angle of the look vector of the points in radians, counterclockwise from east, shape
        (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`); the look vector points from the ground
        to the satellite
    out : str
        output: line of sight displacement in millimeters that the ionosphere causes in every image relative to the
        image `ref`, shape (n_points, nimages), float32, chunks (out_chunks, 1), positive towards the satellite (more
        electrons); the column of `ref` is 0; the displacement corrected for the ionosphere is `disp` - `out`
        (`math`); nan where `lon`, `lat`, `theta` or `phi` of the point is nan
    times : list of str
        time of every image in UTC, ISO 8601 (e.g. 2019-01-02T05:06:07.123456Z): the `times` of the metadata of
        `load-gamma-metadata`, as given to `download-ionex`
    wavelength : float
        radar wavelength in meters, between 0 and 1 (`radar_wavelength` of the metadata of `load-gamma-metadata`,
        0.0554658 for Sentinel-1)
    ratio : float, default: 1.0
        part of the electron content of the maps below the satellite, 0 < ratio <= 1; the maps hold all electrons up
        to the orbits of the GNSS satellites (20 200 km), a SAR satellite flies at 500 to 800 km
    ref : int, default: 0
        the image the displacement is relative to, 0 .. nimages - 1: the `ref` of `unwrap-correct-closure-pc`
    out_chunks : int, optional
        point chunk size of `out`, that of `lon` by default; the points of a task
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 4
        tasks (blocks of `out_chunks` points) a worker processes at the same time; a task holds about 4 bytes per
        point and image
    """
    logger = logging.getLogger(__name__)
    lon_zarr = zarr.open(lon, mode='r'); logger.zarr_info(lon, lon_zarr)
    n_points = lon_zarr.shape[0]
    for name, path in (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi)):
        z = lon_zarr if name == 'lon' else zarr.open(path, mode='r')
        if name != 'lon':
            logger.zarr_info(path, z)
        if not np.issubdtype(z.dtype, np.floating):
            raise ArgumentError(f'{name} must be floating point, got {z.dtype}')
        if z.shape != (n_points,):
            raise ArgumentError(f'{name} must have shape ({n_points},), got shape {z.shape}')
    t_img = _times(times)
    nimages = len(t_img)
    if nimages == 0:
        raise ArgumentError('times must have the time of at least one image')
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ArgumentError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    wavelength = _wavelength(wavelength)
    ratio = _ratio(ratio)
    maps = _image_maps(ionex_dir, t_img)
    logger.info(f'{nimages} images from {t_img[0]:%Y-%m-%d %H:%M} to {t_img[-1]:%Y-%m-%d %H:%M} UTC on '
                f'{len(_days(t_img))} days, relative to image {ref} ({t_img[ref]:%Y-%m-%d %H:%M:%S} UTC); maps of '
                f'{maps["lat"].size} x {maps["lon"].size} nodes, layer at {maps["height"] / 1000:g} km, ratio {ratio:g}')
    kwargs = dict(times=t_img, wavelength=wavelength, tec_times=maps['times'], tec_lat=maps['lat'],
                  tec_lon=maps['lon'], height=maps['height'], radius=maps['radius'], ratio=ratio)

    # the vertical electron content at the scene: at the pierce point of the median look vector of the first block
    n0 = min(n_points, out_chunks or lon_zarr.chunks[0])
    first = [np.nanmedian(np.asarray(zarr.open(p, mode='r')[:n0], dtype=np.float64)) for p in (lon, lat, theta, phi)]
    if np.isfinite(first).all():
        ipp_lon, ipp_lat, mapping = mr.pierce_point(*(np.array([v]) for v in first), maps['height'], maps['radius'])
        vtec = np.array([mr.ionex_vtec(ipp_lon, ipp_lat, t, maps['tec'], maps['times'], maps['lat'], maps['lon'])[0]
                         for t in t_img])
        if np.isfinite(vtec).any():
            lo, hi = int(np.nanargmin(vtec)), int(np.nanargmax(vtec))
            logger.info(f'vertical electron content at the pierce point ({ipp_lon[0]:.2f}, {ipp_lat[0]:.2f}), mapping '
                        f'factor {mapping[0]:.3f}: median {np.nanmedian(vtec):.1f} TECU, min {vtec[lo]:.1f} '
                        f'({t_img[lo]:%Y-%m-%d}), max {vtec[hi]:.1f} ({t_img[hi]:%Y-%m-%d})')

    if out_chunks is None:
        out_chunks = lon_zarr.chunks[0]
    out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(out, out_zarr)
    images = slice(0, nimages)
    tasks = []
    for start in range(0, n_points, out_chunks):
        points = slice(start, min(start + out_chunks, n_points))
        tasks.append(([Chunk(p, (points,)) for p in (lon, lat, theta, phi)], [Chunk(out, (points, images))]))
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        stats = ex.map_chunks(_iono_task, tasks, desc='point blocks', tec=ex.put(maps['tec']), ref=int(ref), **kwargs)
    count, total, squares = np.sum([s[0] for s in stats], axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        std = np.sqrt(np.maximum(squares / count - (total / count) ** 2, 0))
    others = np.delete(std, ref)
    if others.size and np.isfinite(others).any():
        logger.info(f'standard deviation of the ionosphere over the points per image: median '
                    f'{np.nanmedian(others):.2f} mm, max {np.nanmax(others):.2f} mm (image {int(np.nanargmax(std))})')
    logger.info('done.')
