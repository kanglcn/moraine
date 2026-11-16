"""Tropospheric delay of points from the ERA5 reanalysis on pressure levels (decision 0051)"""

__all__ = ['era5_zenith_delay', 'pc_tropo_era5']

import datetime
import math

import numpy as np
from numba import prange

from .utils_.jit import ngjit, ngpjit

# Refractivity 1e6 (n - 1) = k1 Pd / T + k2 e / T + k3 e / T^2 with the constants of PyAPS (Jolivet et al., 2011),
# split into the hydrostatic part k1 Rd rho, whose integral follows from the pressure alone, and the wet part
# k2' e / T + k3 e / T^2 with k2' = k2 - k1 Rd / Rv (Davis et al., 1985).
_K1 = 0.776                     # K / Pa
_K2 = 0.716                     # K / Pa
_K3 = 3.75e3                    # K^2 / Pa
_RD = 287.05                    # J / (kg K), dry air
_RV = 461.495                   # J / (kg K), water vapour
_K2P = _K2 - _K1 * _RD / _RV    # 0.233 K / Pa
_EPS = _RD / _RV
_G0 = 9.80665                   # m / s^2, standard gravity of the geopotential height
_HOUR = datetime.timedelta(hours=1)


# ---------------------------------------------------------------- times

def _utc(t):
    """a time as a naive datetime in UTC, from an aware datetime, a naive one (taken as UTC), ISO 8601 text or a
    numpy.datetime64"""
    raw = t
    if isinstance(t, np.datetime64):
        t = t.astype('datetime64[us]').astype(datetime.datetime)
    elif isinstance(t, str):
        try:                # with the time of the day: a date alone (e.g. YYYYMMDD) is not the time of an image
            t = datetime.datetime.fromisoformat(t.strip().replace('Z', '+00:00').replace('z', '+00:00')) \
                if ':' in t else None
        except ValueError:
            t = None
    if not isinstance(t, datetime.datetime):
        raise ValueError(f'times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), got {raw!r}')
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def _times(times)->list:
    """naive UTC datetimes of a list of times, of a single time, or of text with commas or spaces"""
    if isinstance(times, str):
        times = times.replace(',', ' ').split()
    elif isinstance(times, (datetime.datetime, np.datetime64)):
        times = [times]
    return [_utc(t) for t in times]


def _hour_weights(times)->list:
    """(hour before, weight of the hour after) of every time: the hour before is the whole hour at or before the time,
    the weight (0 <= weight < 1) the fraction of an hour from it to the time"""
    out = []
    for t in times:
        h0 = t.replace(minute=0, second=0, microsecond=0)
        out.append((h0, (t - h0) / _HOUR))
    return out


def _era5_hours(times)->list:
    """the whole hours (naive UTC datetimes, sorted) of ERA5 that the delays at `times` need: the hour before every
    time and the hour after it unless the time is a whole hour"""
    hours = set()
    for h0, w in _hour_weights(_times(times)):
        hours.add(h0)
        if w > 0:
            hours.add(h0 + _HOUR)
    return sorted(hours)


# ---------------------------------------------------------------- profiles of the grid nodes

def _geometric_height(geopotential, lat):
    """height above the geoid in meters of a geopotential (m^2 / s^2) at the latitude `lat` (degrees)"""
    # Mahoney (2008), as RAiDER: the gravity at sea level and the radius of the ellipsoid as functions of latitude
    phi = np.deg2rad(lat)
    g = 9.80616 * (1 - 0.002637 * np.cos(2 * phi) + 0.0000059 * np.cos(2 * phi) ** 2)
    a, b = 6378137.0, 6356752.3142
    re = np.sqrt(1 / (np.cos(phi) ** 2 / a ** 2 + np.sin(phi) ** 2 / b ** 2))
    h = geopotential / _G0
    return h * re / (g / _G0 * re - h)


def _layer_integral(n0, n1, dz):
    """integral over layers of thickness dz of a refractivity exponential in height from n0 at the bottom to n1 at
    the top (linear where n0 or n1 is not positive)"""
    expo = (n0 > 0) & (n1 > 0)
    log = np.log(np.where(expo, n0, 1.0) / np.where(expo, n1, 1.0))
    curved = expo & (np.abs(log) > 1e-6)
    return np.where(curved, (n0 - n1) * dz / np.where(curved, log, 1.0), 0.5 * (n0 + n1) * dz)


def _columns(pressure, grid_lat, geopotential, temperature, specific_humidity):
    """the profile of every grid node from the lowest level to the highest one, arrays (n_lat, n_lon, n_levels)
    float64: height above the geoid (m), wet refractivity, wet delay from the level to the highest level (m); and
    ln(pressure) of the levels (n_levels,)"""
    order = np.argsort(-pressure, kind='stable')               # the highest pressure (lowest level) first
    p = pressure[order].astype(np.float64)
    z = geopotential[order].astype(np.float64)
    t = temperature[order].astype(np.float64)
    q = specific_humidity[order].astype(np.float64)
    hz = _geometric_height(z, np.asarray(grid_lat, dtype=np.float64)[None, :, None])
    if not np.all(np.diff(hz, axis=0) > 0):
        raise ValueError('the heights of the levels (from the geopotential) must increase as the pressure decreases')
    e = q * p[:, None, None] / (_EPS + (1 - _EPS) * q)          # water vapour pressure, Pa
    nw = _K2P * e / t + _K3 * e / t ** 2
    layer = _layer_integral(nw[:-1], nw[1:], hz[1:] - hz[:-1])
    wc = np.zeros_like(hz)
    wc[:-1] = 1e-6 * np.cumsum(layer[::-1], axis=0)[::-1]
    columns = tuple(np.ascontiguousarray(np.moveaxis(a, 0, -1)) for a in (hz, nw, wc))
    return columns, np.log(p)


# ---------------------------------------------------------------- delays at points

@ngjit
def _column_delay(h, ln_p, hz, nw, wc):
    """pressure (Pa) and wet zenith delay (m) at the height h of one grid node: ln(pressure) linear in height and the
    wet refractivity exponential in height between levels (extended from the lowest layer below the lowest level)"""
    n_lev = hz.shape[0]
    if h <= hz[0]:
        lev = 0
    elif h >= hz[n_lev - 1]:
        lev = n_lev - 2
    else:
        lo, hi = 0, n_lev - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if hz[mid] <= h:
                lo = mid
            else:
                hi = mid
        lev = lo
    dz = hz[lev + 1] - hz[lev]
    f = (h - hz[lev]) / dz
    pres = math.exp(ln_p[lev] + f * (ln_p[lev + 1] - ln_p[lev]))
    if h >= hz[n_lev - 1]:
        return pres, 0.0
    n0 = nw[lev]
    n1 = nw[lev + 1]
    rest = hz[lev + 1] - h                 # from h to the level above
    part = 0.5 * (n0 + f * (n1 - n0) + n1) * rest
    if n0 > 0 and n1 > 0:
        lam = math.log(n0 / n1) / dz       # 1 / scale height of the layer
        if abs(lam * rest) > 1e-6:
            part = (n0 * math.exp(-lam * (h - hz[lev])) - n1) / lam
    return pres, 1e-6 * part + wc[lev + 1]


@ngpjit
def _zenith_delay_numba(lon, lat, hgt, lon0, dlon, lat0, dlat, ln_p, hz, nw, wc, out):
    n_lat, n_lon = hz.shape[0], hz.shape[1]
    for i in prange(lon.shape[0]):
        x, y, h = lon[i], lat[i], hgt[i]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(h)):
            out[i] = np.nan
            continue
        fy = (y - lat0) / dlat
        fx = (x - lon0) / dlon
        iy = min(max(int(math.floor(fy)), 0), n_lat - 2)
        ix = min(max(int(math.floor(fx)), 0), n_lon - 2)
        wy = fy - iy
        wx = fx - ix
        pres = 0.0
        wet = 0.0
        for a in range(2):
            for b in range(2):
                w = (wy if a else 1.0 - wy) * (wx if b else 1.0 - wx)
                if w == 0.0:
                    continue
                p_node, wet_node = _column_delay(h, ln_p, hz[iy + a, ix + b], nw[iy + a, ix + b], wc[iy + a, ix + b])
                pres += w * p_node
                wet += w * wet_node
        # hydrostatic delay from the pressure: the mean gravity of the air column (Saastamoinen, 1972)
        gm = 9.784 * (1.0 - 0.00266 * math.cos(2.0 * math.radians(y)) - 0.00028e-3 * h)
        out[i] = 1e-6 * _K1 * _RD * pres / gm + wet


def _grid_axis(values, name):
    """(first, step, flip) of equally spaced grid coordinates, increasing after the flip"""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size < 2:
        raise ValueError(f'{name} must have shape (n,) with n >= 2, got shape {v.shape}')
    flip = v[1] < v[0]
    if flip:
        v = v[::-1]
    step = (v[-1] - v[0]) / (v.size - 1)
    if not step > 0 or np.abs(np.diff(v) - step).max() > 1e-6 * step:
        raise ValueError(f'{name} must be equally spaced and increasing or decreasing')
    return v[0], step, bool(flip)


def era5_zenith_delay(
    lon,
    lat,
    hgt,
    pressure,
    grid_lon,
    grid_lat,
    geopotential,
    temperature,
    specific_humidity,
):
    """Zenith tropospheric delay at points from the fields of one time of ERA5 on pressure levels.

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points in degrees, shape (n_points,), floating point
    lat : np.ndarray
        latitude of the points in degrees, shape (n_points,), floating point
    hgt : np.ndarray
        height of the points above the geoid (mean sea level) in meters, shape (n_points,), floating point
    pressure : np.ndarray
        pressure of the levels in pascals, shape (n_levels,), at least 2 levels, e.g. the 37 levels of ERA5 from 100
        to 100000 Pa
    grid_lon : np.ndarray
        longitudes of the grid in degrees, shape (n_lon,), at least 2, equally spaced, increasing or decreasing
        (`longitude` of ERA5)
    grid_lat : np.ndarray
        latitudes of the grid in degrees, shape (n_lat,), at least 2, equally spaced, increasing or decreasing
        (`latitude` of ERA5)
    geopotential : np.ndarray
        geopotential of the levels in m^2/s^2, shape (n_levels, n_lat, n_lon), floating point (`z` of ERA5)
    temperature : np.ndarray
        temperature in kelvin, shape (n_levels, n_lat, n_lon), floating point (`t` of ERA5)
    specific_humidity : np.ndarray
        specific humidity in kg/kg, shape (n_levels, n_lat, n_lon), floating point (`q` of ERA5)

    Returns
    -------
    ztd : np.ndarray
        zenith delay in meters, shape (n_points,), float64: the hydrostatic plus the wet delay of the atmosphere
        above the point, positive; nan where `lon`, `lat` or `hgt` is nan

    Raises
    ------
    ValueError
        when a point is outside the grid
    """
    pts = [np.asarray(a) for a in (lon, lat, hgt)]
    for name, a in zip(('lon', 'lat', 'hgt'), pts):
        if a.ndim != 1 or a.shape != pts[0].shape:
            raise ValueError(f'lon, lat and hgt must have one shape (n_points,), got {name} of shape {a.shape}')
        if not np.issubdtype(a.dtype, np.floating):
            raise ValueError(f'{name} must be floating point, got {a.dtype}')
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.ndim != 1 or pressure.size < 2 or not np.all(pressure > 0) or np.unique(pressure).size != pressure.size:
        raise ValueError('pressure must be (n_levels,) different positive pressures in pascals, n_levels >= 2')
    lon0, dlon, flip_lon = _grid_axis(grid_lon, 'grid_lon')
    lat0, dlat, flip_lat = _grid_axis(grid_lat, 'grid_lat')
    shape = (pressure.size, len(grid_lat), len(grid_lon))
    fields = []
    for name, a in (('geopotential', geopotential), ('temperature', temperature),
                    ('specific_humidity', specific_humidity)):
        a = np.asarray(a)
        if a.shape != shape:
            raise ValueError(f'{name} must have shape (n_levels, n_lat, n_lon) = {shape}, got {a.shape}')
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{name} has values that are not finite')
        if flip_lat:
            a = a[:, ::-1]
        if flip_lon:
            a = a[:, :, ::-1]
        fields.append(a)
    if not np.all(fields[1] > 0):
        raise ValueError('temperature must be in kelvin, positive')
    grid_lat = lat0 + dlat * np.arange(shape[1])
    (hz, nw, wc), ln_p = _columns(pressure, grid_lat, *fields)

    x = np.asarray(pts[0], dtype=np.float64)
    x = lon0 + np.mod(x - lon0, 360.0)              # longitudes in the convention of the grid
    x = np.where(np.isclose(x, lon0 + 360.0, rtol=0, atol=1e-9), lon0, x)
    y = np.asarray(pts[1], dtype=np.float64)
    h = np.asarray(pts[2], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(h)
    tol = 1e-9
    outside = ok & ((x > lon0 + dlon * (shape[2] - 1) + tol) | (y < lat0 - tol) | (y > lat0 + dlat * (shape[1] - 1) + tol))
    if outside.any():
        raise ValueError(f'{int(outside.sum())} points are outside the grid (longitude {lon0:.4f} to '
                         f'{lon0 + dlon * (shape[2] - 1):.4f}, latitude {lat0:.4f} to {lat0 + dlat * (shape[1] - 1):.4f})')
    out = np.empty(x.shape, dtype=np.float64)
    _zenith_delay_numba(x, y, h, lon0, dlon, lat0, dlat, ln_p, hz, nw, wc, out)
    return out


def pc_tropo_era5(
    lon,
    lat,
    hgt,
    theta,
    times:list,
    geoid:float,
    era5_times:list,
    pressure,
    grid_lon,
    grid_lat,
    geopotential,
    temperature,
    specific_humidity,
    ref:int=0,
):
    """Line of sight displacement caused by the troposphere at points, from ERA5, relative to a reference image.

    The delay of a point in an image is its zenith delay from ERA5 (`era5_zenith_delay`), interpolated linearly in
    time between the whole hours around the time of the image, divided by sin(theta).

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points in degrees, shape (n_points,), floating point
    lat : np.ndarray
        latitude of the points in degrees, shape (n_points,), floating point
    hgt : np.ndarray
        height of the points in meters, shape (n_points,), floating point: above the WGS84 ellipsoid, or above the
        geoid with `geoid` 0
    theta : np.ndarray
        elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle), shape
        (n_points,), floating point, 0 < theta <= pi/2
    times : list of datetime or str
        time of every image in UTC, nimages date-times (datetime, numpy.datetime64 or ISO 8601 text, e.g.
        2019-01-02T05:06:07.123456Z), e.g. the `times` of `load_gamma_metadata`
    geoid : float
        height of the geoid above the WGS84 ellipsoid at the points in meters (`hgt` - `geoid` is the height above
        the geoid), between -150 and 150; 0 when `hgt` is the height above the geoid
    era5_times : list of datetime or str
        time of the ERA5 fields in UTC, n_hours whole hours: for every image the hour at or before its time and,
        unless it is a whole hour, the hour after it
    pressure : np.ndarray
        pressure of the levels in pascals, shape (n_levels,), as in `era5_zenith_delay`
    grid_lon : np.ndarray
        longitudes of the grid in degrees, shape (n_lon,), as in `era5_zenith_delay`
    grid_lat : np.ndarray
        latitudes of the grid in degrees, shape (n_lat,), as in `era5_zenith_delay`
    geopotential : np.ndarray
        geopotential in m^2/s^2, shape (n_hours, n_levels, n_lat, n_lon), floating point, the hours of `era5_times`
    temperature : np.ndarray
        temperature in kelvin, shape (n_hours, n_levels, n_lat, n_lon), floating point
    specific_humidity : np.ndarray
        specific humidity in kg/kg, shape (n_hours, n_levels, n_lat, n_lon), floating point
    ref : int, default: 0
        the image the displacement is relative to, 0 .. nimages - 1

    Returns
    -------
    tropo : np.ndarray
        line of sight displacement in millimeters that the troposphere causes in every image relative to the image
        `ref`, shape (n_points, nimages), float32: minus the change of the delay, positive towards the satellite (a
        smaller delay); the column of `ref` is 0; the displacement corrected for the troposphere is `disp` - `tropo`;
        nan where `lon`, `lat`, `hgt` or `theta` of the point is nan

    Raises
    ------
    ValueError
        when the fields of an hour that an image needs are missing, or a point is outside the grid
    """
    t_img = _times(times)
    nimages = len(t_img)
    if nimages == 0:
        raise ValueError('times must have the time of at least one image')
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ValueError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    geoid = float(geoid)
    if not -150 <= geoid <= 150:
        raise ValueError(f'geoid must be the height of the geoid above the WGS84 ellipsoid in meters, between -150 '
                         f'and 150, got {geoid}')
    theta = np.asarray(theta)
    if theta.ndim != 1 or theta.shape != np.shape(lon):
        raise ValueError(f'theta must have the shape (n_points,) of lon, got shape {theta.shape}')
    if not np.issubdtype(theta.dtype, np.floating):
        raise ValueError(f'theta must be floating point, got {theta.dtype}')
    th = theta.astype(np.float64)
    with np.errstate(invalid='ignore'):
        bad = np.isfinite(th) & ~((th > 0) & (th <= np.pi / 2 + 1e-6))
    if bad.any():
        raise ValueError('theta must be the elevation angle of the look vector in radians, between 0 and pi/2; got '
                         f'{int(bad.sum())} values outside')

    t_era = _times(era5_times)
    if len(set(t_era)) != len(t_era):
        raise ValueError('era5_times must be different hours')
    if any(t != t.replace(minute=0, second=0, microsecond=0) for t in t_era):
        raise ValueError('era5_times must be whole hours')
    index = {t: i for i, t in enumerate(t_era)}
    shape = (len(t_era), np.size(pressure), np.size(grid_lat), np.size(grid_lon))
    for name, a in (('geopotential', geopotential), ('temperature', temperature),
                    ('specific_humidity', specific_humidity)):
        if np.shape(a) != shape:
            raise ValueError(f'{name} must have shape (n_hours, n_levels, n_lat, n_lon) = {shape}, got {np.shape(a)}')
    plan, missing = [], []
    for h0, w in _hour_weights(t_img):
        need = [h0, h0 + _HOUR] if w > 0 else [h0]
        missing += [h for h in need if h not in index]
        plan.append(([index.get(h) for h in need], w))
    if missing:
        hours = sorted(set(missing))
        raise ValueError(f'no ERA5 fields of {len(hours)} hours that the images need: '
                         f'{", ".join(f"{h:%Y-%m-%d %H:00}" for h in hours[:5])}{" ..." if len(hours) > 5 else ""}')

    h = np.asarray(hgt, dtype=np.float64) - geoid

    def delay(k):
        """zenith delay of image k, m"""
        idx, w = plan[k]
        d = [era5_zenith_delay(lon, lat, h, pressure, grid_lon, grid_lat, geopotential[i], temperature[i],
                               specific_humidity[i]) for i in idx]
        return d[0] if len(d) == 1 else (1 - w) * d[0] + w * d[1]

    scale = -1000.0 / np.sin(th)            # m of zenith delay -> mm of line of sight displacement
    d_ref = delay(ref)
    out = np.empty((th.size, nimages), dtype=np.float32)
    for k in range(nimages):
        d = d_ref if k == ref else delay(k)
        out[:, k] = (d - d_ref) * scale + 0.0          # + 0.0: 0 instead of -0 in the column of `ref`
    return out
