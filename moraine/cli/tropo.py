"""Tropospheric delay (CLI): from ERA5, download of its pressure levels and delay of point clouds (decision 0051); the
part stratified with height from the displacement (decision 0053)"""

__all__ = ['download_era5', 'pc_tropo_era5', 'pc_tropo_powerlaw']

import datetime
import io
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import zarr

import moraine as mr
from ..api.utils_.times import _times
from ..api.tropo import _era5_hours, _geoid, _powerlaw_x, _estimation_points, _powerlaw_column
from .logging import mc_logger
from .utils_ import ArgumentError
from .executor import Executor, Chunk

_DATASET = 'reanalysis-era5-pressure-levels'
# the 37 pressure levels of ERA5, hPa
_LEVELS = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
           650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000)
_GRID = 0.25                    # degrees, the grid of ERA5
_TOL = 1e-6                     # degrees


def _era5_path(era5_dir, hour)->Path:
    """the file of one hour (naive UTC datetime) in `era5_dir`"""
    return Path(era5_dir) / f'era5_{hour:%Y%m%d_%H}.nc'


def _bbox(bbox)->tuple:
    """(west, south, east, north) in degrees from a list of 4 numbers or text with commas or spaces"""
    if isinstance(bbox, str):
        bbox = bbox.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ArgumentError(f'bbox must be [west, south, east, north] in degrees, got {bbox!r}') from None
    if not (-180 <= w < e <= 360 and e - w <= 360 and -90 <= s < n <= 90):
        raise ArgumentError(f'bbox must be [west, south, east, north] in degrees with west < east and south < '
                            f'north, got {[w, s, e, n]}')
    return w, s, e, n


def _area(bbox)->list:
    """[north, west, south, east] of the ERA5 grid around `bbox`: the grid nodes around it and one more on every
    side"""
    w, s, e, n = _bbox(bbox)
    west = math.floor(w / _GRID + 1e-9) * _GRID - _GRID
    east = math.ceil(e / _GRID - 1e-9) * _GRID + _GRID
    south = max(math.floor(s / _GRID + 1e-9) * _GRID - _GRID, -90.0)
    north = min(math.ceil(n / _GRID - 1e-9) * _GRID + _GRID, 90.0)
    return [north, west, south, east]


# ---------------------------------------------------------------- netCDF files of ERA5

def _text(v)->str:
    if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else v.tolist()
    return v.decode() if isinstance(v, bytes) else str(v)


def _values(d, sel=()):
    """values of (a selection of) a netCDF variable with its CF packing and fill value applied (nan where
    missing)"""
    raw = np.asarray(d[sel])
    attrs = d.attrs
    scale, offset = attrs.get('scale_factor'), attrs.get('add_offset')
    out = raw
    if scale is not None or offset is not None:
        out = raw.astype(np.float64) * (1.0 if scale is None else float(np.ravel(scale)[0])) \
              + (0.0 if offset is None else float(np.ravel(offset)[0]))
    for key in ('_FillValue', 'missing_value'):
        if key in attrs and np.issubdtype(out.dtype, np.floating):
            fill = np.ravel(attrs[key])[0]
            out = np.where(raw == fill, np.nan, out)
    return out


def _cf_times(d)->list:
    """naive UTC datetimes of a CF time variable ('seconds since 1970-01-01', 'hours since 1900-01-01 00:00:00.0',
    ...)"""
    units = _text(d.attrs.get('units', ''))
    m = re.fullmatch(r'\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*', units)
    if not m:
        raise ValueError(f'time units {units!r} are not "<seconds|minutes|hours|days> since <date-time>"')
    origin = m.group(2).replace('UTC', '').strip().replace(' ', 'T')
    origin = datetime.datetime.fromisoformat(origin)
    if origin.tzinfo is not None:
        origin = origin.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    step = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}[m.group(1)]
    return [origin + datetime.timedelta(seconds=round(float(v) * step, 3)) for v in np.ravel(d[()])]


def _read_era5(path)->dict:
    """the fields of an ERA5 file of one hour (netCDF4 of the Climate Data Store): time (naive UTC datetime),
    pressure (Pa, (n_levels,)), lat, lon (degrees), z, t, q ((n_levels, n_lat, n_lon) float32)"""
    import h5py
    try:
        f = h5py.File(path, 'r')
    except OSError as e:
        raise ValueError(f'{path}: not a netCDF4 (HDF5) file ({e}); download it again with `download-era5`') from None
    with f:
        def var(*names):
            for name in names:
                if name in f:
                    return f[name]
            raise ValueError(f'{path}: no variable {" or ".join(names)}')
        times = _cf_times(var('valid_time', 'time'))
        level = var('pressure_level', 'level')
        units = _text(level.attrs.get('units', 'hPa')).strip().lower()
        pressure = np.ravel(_values(level)).astype(np.float64) * (1.0 if units == 'pa' else 100.0)
        lat = np.ravel(_values(var('latitude', 'lat'))).astype(np.float64)
        lon = np.ravel(_values(var('longitude', 'lon'))).astype(np.float64)
        if len(times) != 1:
            raise ValueError(f'{path}: {len(times)} times, an ERA5 file of download-era5 has one hour')
        fields = {}
        for name in ('z', 't', 'q'):
            d = var(name)
            a = _values(d)
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            if a.shape != (pressure.size, lat.size, lon.size):
                raise ValueError(f'{path}: {name} has shape {d.shape}, not (time, level, latitude, longitude) = '
                                 f'(1, {pressure.size}, {lat.size}, {lon.size})')
            fields[name] = a.astype(np.float32, copy=False)
    return dict(time=times[0], pressure=pressure, lat=lat, lon=lon, **fields)


def _lon_span(lon, west)->tuple:
    """(min, max) of longitudes in the convention that starts at `west`"""
    x = west + np.mod(np.asarray(lon, dtype=np.float64) - west + _TOL, 360.0) - _TOL
    return x.min(), x.max()


def _covers(path, hour, area)->bool:
    """whether the file holds the fields of `hour` on the 37 levels around `area` ([north, west, south, east])"""
    try:
        era5 = _read_era5(path)
    except (ValueError, KeyError, OSError):
        return False
    north, west, south, east = area
    lon_min, lon_max = _lon_span(era5['lon'], west)
    return (era5['time'] == hour and set(_LEVELS) <= set(np.round(era5['pressure'] / 100, 3))
            and era5['lat'].min() <= south + _TOL and era5['lat'].max() >= north - _TOL
            and lon_min <= west + _TOL and lon_max >= east - _TOL)


def _download_cds_hour(path, hour, area):
    """download the fields of one hour from the Climate Data Store into `path` (through a temporary file next to it)"""
    import cdsapi
    request = {
        'product_type': ['reanalysis'],
        'variable': ['geopotential', 'specific_humidity', 'temperature'],
        'year': [f'{hour:%Y}'], 'month': [f'{hour:%m}'], 'day': [f'{hour:%d}'], 'time': [f'{hour:%H}:00'],
        'pressure_level': [str(p) for p in _LEVELS],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'area': list(area),
    }
    part = path.with_name(path.name + '.part')
    try:
        cdsapi.Client(quiet=True, progress=False).retrieve(_DATASET, request, str(part))
        if not _covers(part, hour, area):
            raise RuntimeError(f'the file of {hour:%Y-%m-%d %H:00} from the Climate Data Store does not hold the '
                               f'fields of that hour on the 37 levels around {area} (north, west, south, east)')
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def _write_era5(path, hour, pressure, lat, lon, z, t, q):
    """write the fields of one hour as the Climate Data Store does (netCDF4: valid_time, pressure_level in hPa,
    latitude, longitude, z / t / q with dimensions (time, level, latitude, longitude)), through a temporary file"""
    import h5py
    part = path.with_name(path.name + '.part')
    try:
        with h5py.File(part, 'w') as f:
            seconds = round((hour - datetime.datetime(1970, 1, 1)).total_seconds())
            scales = [f.create_dataset('valid_time', data=np.array([seconds], dtype=np.int64)),
                      f.create_dataset('pressure_level', data=np.asarray(pressure, dtype=np.float64) / 100),
                      f.create_dataset('latitude', data=np.asarray(lat, dtype=np.float64)),
                      f.create_dataset('longitude', data=np.asarray(lon, dtype=np.float64))]
            for scale, units in zip(scales, ('seconds since 1970-01-01', 'hPa', 'degrees_north', 'degrees_east')):
                scale.attrs['units'] = units
                scale.make_scale(scale.name.strip('/'))
            for name, a, units in (('z', z, 'm**2 s**-2'), ('t', t, 'K'), ('q', q, 'kg kg**-1')):
                d = f.create_dataset(name, data=np.asarray(a, dtype=np.float32)[None], compression='gzip')
                d.attrs['units'] = units
                for i, scale in enumerate(scales):
                    d.dims[i].attach_scale(scale)
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


# ---------------------------------------------------------------- the NSF NCAR archive of ERA5 on AWS

# one file per day and variable, 24 hours x 37 levels x the earth (0.25 degrees), stored by hour
_NCAR_URL = ('https://nsf-ncar-era5.s3.amazonaws.com/e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{code}.ll025sc.'
             '{day:%Y%m%d}00_{day:%Y%m%d}23.nc')
# (name in the files of one hour, parameter of ECMWF in the file name, variable in the file)
_NCAR_FIELDS = (('z', '128_129_z', 'Z'), ('t', '128_130_t', 'T'), ('q', '128_133_q', 'Q'))


class _HttpFile(io.RawIOBase):
    """A file of a web server read by HTTP range requests, for h5py: small reads (metadata) from cached blocks,
    large reads (a chunk of data) by one request each."""

    def __init__(self, url, block=1 << 20, n_blocks=16, timeout=120, retries=4):
        super().__init__()
        self.url, self.block, self.n_blocks, self.timeout, self.retries = url, block, n_blocks, timeout, retries
        self.pos = 0
        self.size = None
        self.blocks = {}                                   # block number -> bytes, the last n_blocks read
        self.blocks[0] = self._get(0, block - 1)           # the first block also gives the size of the file

    def _get(self, start, stop):
        """bytes start .. stop (inclusive) of the file"""
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{stop}'})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    data = r.read()
                    if self.size is None:
                        content_range = r.headers.get('Content-Range')          # bytes start-stop/size
                        self.size = int(content_range.rsplit('/', 1)[1]) if content_range else len(data)
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):                   # S3 answers 403 for a missing key without listing rights
                    raise FileNotFoundError(self.url) from None
                if e.code < 500 or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            time.sleep(2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), max(self.size - self.pos, 0))
        if n == 0:
            return 0
        start, stop = self.pos, self.pos + n
        if n > 4 * self.block:
            data = self._get(start, stop - 1)
        else:
            first, last = start // self.block, (stop - 1) // self.block
            for i in range(first, last + 1):
                if i not in self.blocks:
                    if len(self.blocks) >= self.n_blocks:
                        self.blocks.pop(next(iter(self.blocks)))
                    self.blocks[i] = self._get(i * self.block, min((i + 1) * self.block, self.size) - 1)
            data = b''.join(self.blocks[i] for i in range(first, last + 1))[start - first * self.block:][:n]
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _ncar_file(url):
    """the file at `url` of the NSF NCAR archive, opened for reading"""
    return _HttpFile(url)


def _read_ncar(url, var, hours, area):
    """the variable `var` of a daily file of the NSF NCAR archive at `hours` over `area` ([north, west, south,
    east]): {hour: (n_levels, n_lat, n_lon) float32}, and the pressure (Pa) of the levels, the latitudes (from the
    north) and the longitudes (in the convention of `area`) of the grid"""
    import h5py
    north, west, south, east = area
    lat_out = np.round(np.arange(north, south - _TOL, -_GRID), 6)
    lon_out = np.round(np.arange(west, east + _TOL, _GRID), 6)
    with _ncar_file(url) as fobj, h5py.File(fobj, 'r') as f:
        lat, lon = f['latitude'][:], f['longitude'][:]
        iy = np.round((lat[0] - lat_out) / (lat[0] - lat[1])).astype(np.int64)
        ix = np.round(np.mod(lon_out - lon[0], 360) / (lon[1] - lon[0])).astype(np.int64)
        if (iy.min() < 0 or iy.max() >= lat.size or ix.max() >= lon.size or np.any(np.diff(iy) != 1)
                or np.abs(lat[iy] - lat_out).max() > _TOL
                or np.abs(np.mod(lon[ix] - lon_out + 180, 360) - 180).max() > _TOL):
            raise ValueError(f'{url}: its grid does not hold the area {area} (north, west, south, east)')
        order = np.argsort(ix)                     # h5py reads increasing indices; the area may cross longitude 0
        back = np.argsort(order)
        stamps = [int(v) for v in f['utc_date'][:]]
        out = {}
        for hour in hours:
            stamp = int(f'{hour:%Y%m%d%H}')
            if stamp not in stamps:
                raise ValueError(f'{url} has no field of {hour:%Y-%m-%d %H:00}')
            a = _values(f[var], (stamps.index(stamp), slice(None), slice(int(iy[0]), int(iy[-1]) + 1), ix[order]))
            out[hour] = a[..., back].astype(np.float32, copy=False)
        level = f['level']
        units = _text(level.attrs.get('units', 'hPa')).strip().lower()
        pressure = np.ravel(_values(level)).astype(np.float64) * (1.0 if units == 'pa' else 100.0)
    return out, pressure, lat_out, lon_out


def _download_ncar_day(day, hours, area, out_dir):
    """download `hours` of one day from the NSF NCAR archive into files of one hour"""
    fields = {}
    for name, code, var in _NCAR_FIELDS:
        url = _NCAR_URL.format(day=day, code=code)
        try:
            fields[name], pressure, lat, lon = _read_ncar(url, var, hours, area)
        except FileNotFoundError:
            raise FileNotFoundError(f'{url} is not in the NSF NCAR archive of ERA5, which follows ERA5 by a few '
                                    'months: download these hours with source cds') from None
    for hour in hours:
        path = _era5_path(out_dir, hour)
        _write_era5(path, hour, pressure, lat, lon, fields['z'][hour], fields['t'][hour], fields['q'][hour])
        if not _covers(path, hour, area):
            path.unlink()
            raise RuntimeError(f'the fields of {hour:%Y-%m-%d %H:00} from the NSF NCAR archive are not on the 37 '
                               f'levels around {area} (north, west, south, east)')


@mc_logger
def download_era5(
    out_dir:str,
    times:list[str],
    bbox:list[float]|str,
    source:str='ncar',
    n_workers:int=4,
):
    """Download the ERA5 pressure levels of the hours around the image times, from the NSF NCAR archive or the Climate Data Store.

    For every image the whole hour at or before its time and, unless it is a whole hour, the hour after it: one file
    per hour with the geopotential, temperature and specific humidity on the 37 pressure levels, over the grid of
    ERA5 (0.25 degrees) around `bbox` and one grid node more on every side. Files that exist and hold their hour over
    this area are kept, so an interrupted download continues where it stopped. Needs the `era5` extra (`pip install
    'moraine[era5]'`). The NSF NCAR archive of ERA5 on AWS needs no account; it sends the fields of the whole earth,
    about 170 MB per hour, and follows ERA5 by a few months. The Copernicus Climate Data Store needs an account: its
    API key in ~/.cdsapirc, and the licence of the dataset "ERA5 hourly data on pressure levels" accepted on its web
    page; it sends the area only, but every hour waits in its queue, about a minute.

    Parameters
    ----------
    out_dir : str
        output: directory of the ERA5 files, created if missing; one netCDF4 file per hour, era5_YYYYMMDD_HH.nc
        (UTC), for `pc-tropo-era5`
    times : list of str
        time of every image in UTC, ISO 8601 (e.g. 2019-01-02T05:06:07.123456Z): the `times` of the metadata of
        `load-gamma-metadata`
    bbox : list of float
        area of the points, [west, south, east, north] in degrees: the `lonlat_bbox` of the metadata of
        `load-gamma-metadata`
    source : str, default: 'ncar'
        where ERA5 comes from: 'ncar', the NSF NCAR archive of ERA5 on AWS (https://registry.opendata.aws/nsf-ncar-era5),
        or 'cds', the Copernicus Climate Data Store
    n_workers : int, default: 4
        number of downloads at the same time: days for 'ncar' (a worker holds about 250 MB), hours for 'cds'
    """
    logger = logging.getLogger(__name__)
    if source not in ('ncar', 'cds'):
        raise ArgumentError(f"source must be 'ncar' or 'cds', not {source!r}")
    area = _area(bbox)
    hours = _era5_hours(times)
    if not hours:
        raise ArgumentError('times must have the time of at least one image')
    try:                    # after the arguments: a wrong one is a usage error also without h5py
        import h5py  # noqa: F401
    except ImportError:
        raise ImportError("download-era5 needs cdsapi and h5py: pip install 'moraine[era5]'") from None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f'{len(_times(times))} images need {len(hours)} hours of ERA5, area (north, west, south, east) {area}')
    todo = [h for h in hours if not _covers(_era5_path(out, h), h, area)]
    logger.info(f'{len(hours) - len(todo)} hours are in {out_dir} already, {len(todo)} to download')
    if todo and source == 'ncar':
        days = {}
        for h in todo:
            days.setdefault(h.date(), []).append(h)
        logger.info(f'from the NSF NCAR archive of ERA5: {len(days)} days')
        # processes: h5py runs one HDF5 call at a time in a process, and the remote file is read within these calls
        with Executor(n_workers=n_workers, processes=True) as ex:
            ex.map(_download_ncar_day, [(day, hs, area, out) for day, hs in sorted(days.items())], desc='days')
    elif todo:
        # the account is needed only for a download: files copied from elsewhere work without it
        try:
            import cdsapi  # noqa: F401
        except ImportError:
            raise ImportError("download-era5 needs cdsapi and h5py: pip install 'moraine[era5]'") from None
        if not (os.environ.get('CDSAPI_KEY')
                or Path(os.environ.get('CDSAPI_RC', '~/.cdsapirc')).expanduser().is_file()):
            raise RuntimeError('no API key of the Climate Data Store: register at https://cds.climate.copernicus.eu, '
                               'accept the licence of "ERA5 hourly data on pressure levels" and write the url and key '
                               'of your profile page into ~/.cdsapirc')
        logger.info('from the Copernicus Climate Data Store')
        with Executor(n_workers=n_workers) as ex:
            ex.map(_download_cds_hour, [(_era5_path(out, h), h, area) for h in todo], desc='hours')
    logger.info('done.')


# ---------------------------------------------------------------- delay of point clouds

def _tropo_task(lon, lat, hgt, theta, **kwargs):
    """the troposphere of a block of points, and per image the number, sum and sum of squares of its finite values"""
    tropo = mr.pc_tropo_era5(lon, lat, hgt, theta, **kwargs)
    stats = np.zeros((3, tropo.shape[1]))
    for k in range(tropo.shape[1]):            # one column at a time: no copy of the whole block
        col = tropo[:, k]
        col = col[np.isfinite(col)].astype(np.float64)
        stats[:, k] = col.size, col.sum(), (col * col).sum()
    return tropo, stats


@mc_logger
def pc_tropo_era5(
    era5_dir:str,
    lon:str,
    lat:str,
    hgt:str,
    theta:str,
    out:str,
    times:list[str],
    geoid:float,
    ref:int=0,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Line of sight displacement caused by the troposphere at the points, from ERA5, relative to a reference image.

    The delay of a point in an image is its zenith delay from ERA5, interpolated linearly in time between the whole
    hours around the time of the image, divided by sin(theta). The ERA5 fields of all hours stay in memory, about
    450 bytes per grid node and hour.

    Parameters
    ----------
    era5_dir : str
        input: directory of the ERA5 files of `download-era5`, one netCDF4 file per hour (era5_YYYYMMDD_HH.nc, UTC)
        with z, t and q on pressure levels, all over the same grid around the points
    lon : str
        input: longitude of the points in degrees, shape (n_points,), floating point
    lat : str
        input: latitude of the points in degrees, shape (n_points,), floating point
    hgt : str
        input: height of the points in meters, shape (n_points,), floating point: above the WGS84 ellipsoid, or above
        the geoid with `geoid` 0 (`load-gamma-lat-lon-hgt`, then `ras2pc`)
    theta : str
        input: elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle),
        shape (n_points,), floating point (`load-gamma-look-vector`, then `ras2pc`)
    out : str
        output: line of sight displacement in millimeters that the troposphere causes in every image relative to the
        image `ref`, shape (n_points, nimages), float32, chunks (out_chunks, 1): minus the change of the delay,
        positive towards the satellite (a smaller delay); the column of `ref` is 0; the displacement corrected for the
        troposphere is `disp` - `out` (`math`); nan where `lon`, `lat`, `hgt` or `theta` of the point is nan
    times : list of str
        time of every image in UTC, ISO 8601 (e.g. 2019-01-02T05:06:07.123456Z): the `times` of the metadata of
        `load-gamma-metadata`, as given to `download-era5`
    geoid : float
        height of the geoid above the WGS84 ellipsoid at the points in meters, between -150 and 150 (e.g. 46 for the
        Campi Flegrei): `hgt` - `geoid` is the height above the geoid (mean sea level); 0 when `hgt` is the height
        above the geoid
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
    for name, path in (('lon', lon), ('lat', lat), ('hgt', hgt), ('theta', theta)):
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
    geoid = _geoid(geoid)
    try:                    # after the arguments: a wrong one is a usage error also without h5py
        import h5py  # noqa: F401
    except ImportError:
        raise ImportError("pc-tropo-era5 needs h5py: pip install 'moraine[era5]'") from None
    hours = _era5_hours(t_img)
    missing = [h for h in hours if not _era5_path(era5_dir, h).is_file()]
    if missing:
        raise ValueError(f'{era5_dir}: no ERA5 file of {len(missing)} of the {len(hours)} hours that the images need, '
                         f'e.g. {_era5_path(era5_dir, missing[0]).name}; download them with `download-era5`')
    logger.info(f'{nimages} images from {t_img[0]:%Y-%m-%d %H:%M} to {t_img[-1]:%Y-%m-%d %H:%M} UTC, '
                f'{len(hours)} hours of ERA5, relative to image {ref} ({t_img[ref]:%Y-%m-%d %H:%M:%S} UTC)')
    era5 = [_read_era5(_era5_path(era5_dir, h)) for h in hours]
    first = era5[0]
    for h, e in zip(hours, era5):
        if e['time'] != h:
            raise ValueError(f'{_era5_path(era5_dir, h)} holds {e["time"]:%Y-%m-%d %H:%M}, not that hour')
        for key in ('pressure', 'lat', 'lon'):
            if e[key].shape != first[key].shape or not np.allclose(e[key], first[key], rtol=0, atol=1e-6):
                raise ValueError(f'{_era5_path(era5_dir, h)} has other {key} than {_era5_path(era5_dir, hours[0])}: '
                                 'the files must be of one area (download them again with `download-era5`)')
    logger.info(f'ERA5 grid: {first["lat"].size} x {first["lon"].size} nodes (latitude {first["lat"].min():.2f} to '
                f'{first["lat"].max():.2f}, longitude {first["lon"].min():.2f} to {first["lon"].max():.2f}), '
                f'{first["pressure"].size} levels')
    fields = {name: np.stack([e[name] for e in era5]) for name in ('z', 't', 'q')}
    del era5

    if out_chunks is None:
        out_chunks = lon_zarr.chunks[0]
    out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(out, out_zarr)
    images = slice(0, nimages)
    tasks = []
    for start in range(0, n_points, out_chunks):
        points = slice(start, min(start + out_chunks, n_points))
        tasks.append(([Chunk(p, (points,)) for p in (lon, lat, hgt, theta)], [Chunk(out, (points, images))]))
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        stats = ex.map_chunks(
            _tropo_task, tasks, desc='point blocks', times=t_img, geoid=geoid, era5_times=hours,
            pressure=first['pressure'], grid_lon=first['lon'], grid_lat=first['lat'], ref=int(ref),
            geopotential=ex.put(fields['z']), temperature=ex.put(fields['t']), specific_humidity=ex.put(fields['q']))
    count, total, squares = np.sum([s[0] for s in stats], axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        std = np.sqrt(np.maximum(squares / count - (total / count) ** 2, 0))
    others = np.delete(std, ref)
    if others.size and np.isfinite(others).any():
        logger.info(f'standard deviation of the troposphere over the points per image: median '
                    f'{np.nanmedian(others):.1f} mm, max {np.nanmax(others):.1f} mm (image {int(np.nanargmax(std))})')
    logger.info('done.')


@mc_logger
def pc_tropo_powerlaw(
    disp:str,
    hgt:str,
    out:str,
    geoid:float,
    mask:str=None,
    k:str=None,
    alpha:float=1.6,
    h0:float=10000.0,
    out_chunks:int=None,
    processes=False,
    n_workers=1,
    threads_per_worker=4,
):
    """Line of sight displacement caused by the troposphere stratified with height, from the displacement itself.

    In every image the displacement of the points of `mask` is fitted by least squares with c + K (h0 - h)^alpha, h the
    height of the point above the geoid: a constant and a power law of the height (with alpha 1 a linear function of
    the height, whatever h0). The troposphere of every point is K ((h0 - h)^alpha - (h0 - h_m)^alpha), h_m the median
    height of the points of `mask`. Deformation that varies with the height among the points of `mask` gets into it:
    give the points of an area without such deformation. The heights and `mask` stay in memory, about 15 bytes per
    point.

    Parameters
    ----------
    disp : str
        input: line of sight displacement in millimeters, shape (n_points, nimages), floating point, chunks
        (points_block, 1): of every image relative to a reference image (`pc-disp`), or corrected with
        `pc-tropo-era5` (`math`)
    hgt : str
        input: height of the points in meters, shape (n_points,), floating point: above the WGS84 ellipsoid, or above
        the geoid with `geoid` 0 (`load-gamma-lat-lon-hgt`, then `ras2pc`)
    out : str
        output: line of sight displacement in millimeters that the troposphere stratified with height causes, shape
        (n_points, nimages), float32, chunks (out_chunks, 1); 0 at the median height of the points of `mask` and in
        the reference image; the corrected displacement is `disp` - `out` (`math`); nan where `hgt` is nan
    geoid : float
        height of the geoid above the WGS84 ellipsoid at the points in meters, between -150 and 150; 0 when `hgt` is
        the height above the geoid
    mask : str, optional
        input: the points of the fit, bool, shape (n_points,), at least 3 points at different heights with a value in
        every image, e.g. an area without deformation that varies with the height (`polygon-mask`); all points by
        default
    k : str, optional
        output: K and its standard deviation of every image in millimeters per meter^alpha, shape (nimages, 2), float64
    alpha : float, default: 1.6
        exponent of the power law, 0 < alpha <= 5; 1 for a linear function of the height
    h0 : float, default: 10000.0
        height above the geoid in meters above which the troposphere of the images does not differ, above the highest
        point; not used with alpha 1
    out_chunks : int, optional
        point chunk size of `out`, that of `disp` by default
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 4
        tasks (images) a worker processes at the same time; a task holds about 50 bytes per point
    """
    logger = logging.getLogger(__name__)
    disp_zarr = zarr.open(disp, mode='r'); logger.zarr_info(disp, disp_zarr)
    if disp_zarr.ndim != 2 or not np.issubdtype(disp_zarr.dtype, np.floating):
        raise ArgumentError(f'disp must be floating point of shape (n_points, nimages), got {disp_zarr.dtype} '
                            f'{disp_zarr.shape}')
    n_points, nimages = disp_zarr.shape
    hgt_zarr = zarr.open(hgt, mode='r'); logger.zarr_info(hgt, hgt_zarr)
    if hgt_zarr.shape != (n_points,) or not np.issubdtype(hgt_zarr.dtype, np.floating):
        raise ArgumentError(f'hgt must be floating point of shape ({n_points},), got {hgt_zarr.dtype} {hgt_zarr.shape}')
    if mask is not None:
        mask_zarr = zarr.open(mask, mode='r'); logger.zarr_info(mask, mask_zarr)
        if mask_zarr.shape != (n_points,) or mask_zarr.dtype != bool:
            raise ArgumentError(f'mask must be bool of shape ({n_points},), got {mask_zarr.dtype} {mask_zarr.shape}')
    x = _powerlaw_x(hgt_zarr[:], geoid, alpha, h0)
    est, x_m = _estimation_points(x, None if mask is None else mask_zarr[:])
    h = hgt_zarr[:].astype(np.float64)[est] - float(geoid)
    logger.info(f'{int(est.sum())} points of the fit, heights above the geoid {h.min():.0f} to {h.max():.0f} m '
                f'(median {np.median(h):.0f} m); alpha {float(alpha)}' + (f', h0 {float(h0):.0f} m' if float(alpha) != 1 else ''))
    if out_chunks is None:
        out_chunks = disp_zarr.chunks[0]
    out_zarr = zarr.open(out, mode='w', shape=(n_points, nimages), dtype=np.float32, chunks=(out_chunks, 1))
    logger.zarr_info(out, out_zarr)
    points = slice(0, n_points)
    tasks = [([Chunk(disp, (points, slice(j, j + 1)))], [Chunk(out, (points, slice(j, j + 1)))]) for j in range(nimages)]
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        res = ex.map_chunks(_powerlaw_column, tasks, desc='images', x=ex.put(x), est=ex.put(est), x_m=x_m)
    kk = np.stack([r[0] for r in res])
    # the slope at the median height: d/dh of K (h0 - h)^alpha, mm per 100 m
    slope = -100 * kk[:, 0] * float(alpha) * (x_m ** ((float(alpha) - 1) / float(alpha)))
    logger.info(f'slope of the troposphere with height at the median height per image: percentiles 5, 50, 95 '
                f'{np.percentile(slope, 5):.2f}, {np.percentile(slope, 50):.2f}, {np.percentile(slope, 95):.2f} mm per 100 m')
    if k is not None:
        z = zarr.open(k, mode='w', shape=kk.shape, dtype=np.float64, chunks=kk.shape)
        z[:] = kk
        logger.zarr_info(k, z)
    logger.info('done.')
