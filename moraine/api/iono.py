"""Ionosphere of points: the line of sight displacement it causes, from maps of the total electron content (decision
0057)"""

__all__ = ['read_ionex', 'pierce_point', 'ionex_vtec', 'pc_iono_tec']

import datetime
import gzip
import math

import numpy as np
from numba import prange

from .utils_.errors import ArgumentError
from .utils_.jit import ngjit, ngpjit
from .utils_.orbit import _geodetic_to_ecef
from .utils_.times import _times

# First order ionospheric effect (e.g. Hofmann-Wellenhof et al., GNSS, 2008, section 5.3.2): the phase of a wave of
# frequency f is advanced, the path appears shorter by K STEC / f^2, STEC the electrons per square meter along it;
# the group (amplitude) is delayed by as much. The phase of InSAR measures the phase path: the ionosphere shortens it,
# an apparent displacement towards the satellite.
_K = 40.31                  # m^3/s^2
_C = 299792458.0            # m/s, speed of light
_TECU = 1e16                # electrons per square meter in one TEC unit
_DEG = math.pi / 180
_DAY = 86400.0              # s
_EPOCH = datetime.datetime(2000, 1, 1)      # times are seconds since this epoch inside


# ---------------------------------------------------------------- IONEX files

def _label(line)->str:
    """the label of a header or data record of IONEX (columns 61 - 80)"""
    return line[60:80].strip()


def _floats(text, width)->list:
    """the numbers of a record of fixed width fields"""
    return [float(text[i:i + width]) for i in range(0, len(text.rstrip()), width) if text[i:i + width].strip()]


def _grid(first, last, step, name, path)->np.ndarray:
    """the nodes first, first + step, ..., last of a grid of IONEX"""
    n = (last - first) / step if step else -1
    if not (step and n >= 1 and abs(n - round(n)) < 1e-6):
        raise ValueError(f'{path}: {name} {first} {last} {step} is not a grid of at least 2 nodes')
    return first + step * np.arange(round(n) + 1)


def read_ionex(path)->dict:
    """Read the maps of the vertical total electron content of an IONEX file.

    IONEX (Schaer et al., 1998) is the format of the global ionosphere maps of the International GNSS Service and its
    analysis centres, e.g. the final maps of JPL; the file may be compressed by gzip. Two dimensional maps of one
    single layer only.

    Parameters
    ----------
    path : str or pathlib.Path
        the IONEX file, plain text or gzip

    Returns
    -------
    ionex : dict
        'times': the epochs of the maps in UTC, a list of nmaps naive datetimes;
        'lat': latitude of the rows of the maps in degrees, shape (nlat,), float64;
        'lon': longitude of the columns of the maps in degrees, shape (nlon,), float64;
        'tec': vertical total electron content in TEC units (1e16 electrons per square meter), shape (nmaps, nlat,
        nlon), float32, nan where the map has no value;
        'height': height of the layer above the base radius in meters, float;
        'radius': base radius (the earth as a sphere) in meters, float
    """
    with open(path, 'rb') as f:
        packed = f.read(2) == b'\x1f\x8b'
    opener = gzip.open if packed else open
    with opener(path, 'rt', encoding='ascii', errors='replace') as f:
        lines = f.read().splitlines()
    head = {}
    for n_head, line in enumerate(lines):
        label = _label(line)
        if label == 'END OF HEADER':
            break
        head.setdefault(label, line[:60])
    else:
        raise ValueError(f'{path}: not an IONEX file (no END OF HEADER)')
    if 'IONOSPHERE MAPS' not in head.get('IONEX VERSION / TYPE', ''):
        raise ValueError(f'{path}: not an IONEX file of ionosphere maps')
    try:
        dimension = int(head.get('MAP DIMENSION', '2').split()[0])
        hgt1, hgt2, dhgt = _floats(head['HGT1 / HGT2 / DHGT'][2:], 6)
        lat1, lat2, dlat = _floats(head['LAT1 / LAT2 / DLAT'][2:], 6)
        lon1, lon2, dlon = _floats(head['LON1 / LON2 / DLON'][2:], 6)
        radius = float(head['BASE RADIUS'].split()[0]) * 1000
        exponent = int(head.get('EXPONENT', '-1').split()[0])
    except (KeyError, ValueError, IndexError):
        raise ValueError(f'{path}: an IONEX header without MAP DIMENSION, HGT1 / HGT2 / DHGT, LAT1 / LAT2 / DLAT, '
                         'LON1 / LON2 / DLON or BASE RADIUS') from None
    if dimension != 2 or hgt1 != hgt2 or dhgt != 0:
        raise ValueError(f'{path}: maps of {dimension} dimensions from {hgt1} to {hgt2} km, not of one single layer')
    lat = _grid(lat1, lat2, dlat, 'LAT1 / LAT2 / DLAT', path)
    lon = _grid(lon1, lon2, dlon, 'LON1 / LON2 / DLON', path)
    rows_per_lat = math.ceil(lon.size / 16)
    times, maps = [], []
    i = n_head + 1
    while i < len(lines):
        label = _label(lines[i])
        if label != 'START OF TEC MAP':
            if label in ('START OF RMS MAP', 'START OF HEIGHT MAP'):        # skipped up to their end
                end = label.replace('START', 'END')
                while i < len(lines) and _label(lines[i]) != end:
                    i += 1
            i += 1
            continue
        epoch, scale, tec, i = None, 10.0 ** exponent, np.full((lat.size, lon.size), np.nan, np.float32), i + 1
        while i < len(lines) and _label(lines[i]) != 'END OF TEC MAP':
            line, label = lines[i], _label(lines[i])
            if label == 'EPOCH OF CURRENT MAP':
                epoch = datetime.datetime(*(int(v) for v in line[:36].split()))
            elif label == 'EXPONENT':
                scale = 10.0 ** int(line[:6])
            elif label == 'LAT/LON1/LON2/DLON/H':
                la, lo1, lo2, dlo, _ = _floats(line[2:32], 6)
                row = int(round((la - lat1) / dlat))
                if not (0 <= row < lat.size and abs(lat[row] - la) < 1e-6 and lo1 == lon1 and lo2 == lon2
                        and dlo == dlon):
                    raise ValueError(f'{path}: line {i + 1}: latitude {la}, longitudes {lo1} {lo2} {dlo} are not '
                                     'of the grid of the header')
                text = ''.join(lines[i + 1 + j][:80].ljust(80) for j in range(rows_per_lat))
                values = np.array([int(text[j:j + 5]) for j in range(0, 5 * lon.size, 5)], dtype=np.float64)
                tec[row] = np.where(values == 9999, np.nan, values * scale)
                i += rows_per_lat
            i += 1
        if epoch is None:
            raise ValueError(f'{path}: a TEC map without EPOCH OF CURRENT MAP')
        times.append(epoch)
        maps.append(tec)
        i += 1
    if not maps:
        raise ValueError(f'{path}: no TEC map')
    return {'times': times, 'lat': lat, 'lon': lon, 'tec': np.stack(maps), 'height': hgt1 * 1000, 'radius': radius}


# ---------------------------------------------------------------- pierce points and interpolation

@ngjit
def _pierce(lon, lat, theta, phi, rs):
    """longitude, latitude (degrees) where the line from a point of the WGS84 ellipsoid along its look vector
    (theta, phi) leaves the sphere of radius rs (m) around the centre of the earth, and 1 / cos of the zenith angle
    of the line there"""
    x, y, z, ux, uy, uz = _geodetic_to_ecef(lon, lat, 0.0)
    sl, cl = math.sin(lon * _DEG), math.cos(lon * _DEG)
    sp, cp = math.sin(lat * _DEG), math.cos(lat * _DEG)
    ce, cn, cu = math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), math.sin(theta)
    # look vector in the earth fixed frame: east (-sl, cl, 0), north (-sp cl, -sp sl, cp), up (ux, uy, uz)
    lx = -ce * sl - cn * sp * cl + cu * ux
    ly = ce * cl - cn * sp * sl + cu * uy
    lz = cn * cp + cu * uz
    # |p + s l| = rs with p inside the sphere: s = -p.l + sqrt((p.l)^2 - |p|^2 + rs^2); there (p + s l).l = sqrt(...)
    b = x * lx + y * ly + z * lz
    root = math.sqrt(b * b - (x * x + y * y + z * z) + rs * rs)
    s = root - b
    qx, qy, qz = x + s * lx, y + s * ly, z + s * lz
    return math.atan2(qy, qx) / _DEG, math.asin(min(max(qz / rs, -1.0), 1.0)) / _DEG, rs / root


@ngjit
def _bilinear(tec, m, lon, lat, lon0, dlon, lat0, dlat):
    """map m of tec (nmaps, nlat, nlon) at a longitude and latitude, bilinear; nan outside the grid"""
    nlat, nlon = tec.shape[1], tec.shape[2]
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return math.nan
    y = (lat - lat0) / dlat
    x = ((lon - lon0) % 360.0) / dlon
    if y < -1e-9 or y > nlat - 1 + 1e-9 or x > nlon - 1 + 1e-9:
        return math.nan
    iy = min(max(int(math.floor(y)), 0), nlat - 2)
    ix = min(int(math.floor(x)), nlon - 2)
    wy = min(max(y - iy, 0.0), 1.0)
    wx = min(x - ix, 1.0)
    return ((1 - wy) * ((1 - wx) * tec[m, iy, ix] + wx * tec[m, iy, ix + 1])
            + wy * ((1 - wx) * tec[m, iy + 1, ix] + wx * tec[m, iy + 1, ix + 1]))


@ngjit
def _vtec(tec, m, w, shift0, shift1, lon, lat, lon0, dlon, lat0, dlat):
    """vertical TEC at a time between the maps m and m + 1 (weight w of map m + 1), each map turned by its shift of
    longitude"""
    return ((1 - w) * _bilinear(tec, m, lon + shift0, lat, lon0, dlon, lat0, dlat)
            + w * _bilinear(tec, m + 1, lon + shift1, lat, lon0, dlon, lat0, dlat))


@ngpjit
def _pierce_numba(lon, lat, theta, phi, rs, ipp_lon, ipp_lat, mapping):
    for i in prange(lon.shape[0]):
        ipp_lon[i], ipp_lat[i], mapping[i] = _pierce(lon[i], lat[i], theta[i], phi[i], rs)


@ngpjit
def _vtec_numba(lon, lat, tec, m, w, shift0, shift1, lon0, dlon, lat0, dlat, out):
    for i in prange(lon.shape[0]):
        out[i] = _vtec(tec, m, w, shift0, shift1, lon[i], lat[i], lon0, dlon, lat0, dlat)


@ngpjit
def _los_numba(lon, lat, theta, phi, rs, tec, m, w, shift0, shift1, lon0, dlon, lat0, dlat, scale, ref, out):
    nimages = m.shape[0]
    for i in prange(lon.shape[0]):
        plon, plat, mapping = _pierce(lon[i], lat[i], theta[i], phi[i], rs)
        s_ref = mapping * _vtec(tec, m[ref], w[ref], shift0[ref], shift1[ref], plon, plat, lon0, dlon, lat0, dlat)
        for k in range(nimages):
            s = mapping * _vtec(tec, m[k], w[k], shift0[k], shift1[k], plon, plat, lon0, dlon, lat0, dlat)
            out[i, k] = scale * (s - s_ref)


def _float_points(name, a, shape=None)->np.ndarray:
    """a 1D floating point array of points as float64"""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        raise ArgumentError(f'{name} must be floating point, got {a.dtype}')
    if a.ndim != 1 or (shape is not None and a.shape != shape):
        want = f'the shape {shape} of lon' if shape is not None else 'shape (n_points,)'
        raise ArgumentError(f'{name} must have {want}, got shape {a.shape}')
    return a.astype(np.float64)


def _check_points(lon, lat, theta=None, phi=None):
    """ValueError for coordinates or look vectors out of range"""
    with np.errstate(invalid='ignore'):
        if (np.abs(lat) > 90).any():
            raise ValueError('lat must be latitudes in degrees between -90 and 90')
        if (np.abs(lon) > 360).any():
            raise ValueError('lon must be longitudes in degrees between -360 and 360')
        if theta is not None and ((theta < -1e-6) | (theta > np.pi / 2 + 1e-6)).any():
            raise ValueError('theta must be the elevation angle of the look vector in radians, between 0 and pi/2')
        if phi is not None and (np.abs(phi) > 2 * np.pi + 1e-6).any():
            raise ValueError('phi must be the orientation angle of the look vector in radians, between -2 pi and 2 pi')


def _shell(height, radius)->float:
    """radius of the layer in meters"""
    try:
        height, radius = float(height), float(radius)
    except (TypeError, ValueError):
        raise ArgumentError(f'height and radius must be numbers in meters, got {height!r}, {radius!r}') from None
    if not (5e4 <= height <= 3e7 and 6e6 <= radius <= 7e6):
        raise ArgumentError(f'height and radius must be in meters, height between 5e4 and 3e7 and radius between 6e6 '
                            f'and 7e6 (e.g. 450e3 and 6371e3), got {height}, {radius}')
    if radius + height <= 6378137.0 + 1e3:
        raise ArgumentError(f'the layer (radius + height {radius + height:.0f} m) must be above the points (the '
                            'equatorial radius of WGS84, 6378137 m)')
    return radius + height


def _maps(tec, tec_times, tec_lat, tec_lon)->tuple:
    """checked maps: tec float32 (nmaps, nlat, nlon), their times in seconds, and the grid origin and steps"""
    tec = np.asarray(tec)
    if not np.issubdtype(tec.dtype, np.floating) or tec.ndim != 3:
        raise ArgumentError(f'tec must be floating point of shape (nmaps, nlat, nlon), got {tec.dtype} of shape '
                            f'{tec.shape}')
    t = np.array([(v - _EPOCH).total_seconds() for v in _times(tec_times)])
    if t.size != tec.shape[0] or t.size < 2 or (np.diff(t) <= 0).any():
        raise ArgumentError(f'tec_times must be the increasing times of the {tec.shape[0]} maps of tec (at least 2), '
                            f'got {t.size} times')
    grid = []
    for name, g, n in (('tec_lon', tec_lon, tec.shape[2]), ('tec_lat', tec_lat, tec.shape[1])):
        g = np.asarray(g, dtype=np.float64)
        step = np.diff(g)
        if g.shape != (n,) or n < 2 or step[0] == 0 or np.abs(step - step[0]).max() > 1e-6:
            raise ArgumentError(f'{name} must be the {n} nodes of an evenly spaced grid of the maps, got shape '
                                f'{g.shape}')
        grid += [float(g[0]), float(step[0])]
    if grid[1] < 0:
        raise ArgumentError('tec_lon must increase')
    return np.ascontiguousarray(tec, dtype=np.float32), t, grid       # no copy of float32 maps (0.1 TEC units)


def _at_times(times, t_maps)->tuple:
    """for times (seconds) the first of the two maps around each, the weight of the second and the shifts of longitude
    of both (degrees), as arrays"""
    m = np.clip(np.searchsorted(t_maps, times, side='right') - 1, 0, t_maps.size - 2)
    outside = (times < t_maps[0]) | (times > t_maps[-1])
    if outside.any():
        k = int(np.argmax(outside))
        first, last = (_EPOCH + datetime.timedelta(seconds=float(v)) for v in (t_maps[0], t_maps[-1]))
        when = _EPOCH + datetime.timedelta(seconds=float(times[k]))
        raise ValueError(f'the maps from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} do not hold the time '
                         f'{when:%Y-%m-%d %H:%M:%S} (image {k})')
    t0, t1 = t_maps[m], t_maps[m + 1]
    # the ionosphere follows the Sun, which moves 360 degrees to the west per day: a map is turned by the time to it
    # (Schaer et al., 1998, IONEX version 1, equation 3)
    return m, (times - t0) / (t1 - t0), (times - t0) * 360 / _DAY, (times - t1) * 360 / _DAY


def pierce_point(
    lon,
    lat,
    theta,
    phi,
    height:float=450e3,
    radius:float=6371e3,
):
    """Ionospheric pierce points of the look vectors of points, and their mapping factors.

    The line from a point on the WGS84 ellipsoid along its look vector (to the satellite) crosses the thin layer of
    the ionosphere, a sphere of radius `radius` + `height` around the centre of the earth, at the pierce point; the
    slant total electron content along the line is the vertical one there times the mapping factor, 1 / cos of the
    zenith angle of the line at the pierce point.

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points in degrees, shape (n_points,), floating point
    lat : np.ndarray
        latitude of the points in degrees, shape (n_points,), floating point
    theta : np.ndarray
        elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle), shape
        (n_points,), floating point, 0 <= theta <= pi/2
    phi : np.ndarray
        orientation angle of the look vector of the points in radians, counterclockwise from east, shape (n_points,),
        floating point, -2 pi <= phi <= 2 pi; the look vector points from the ground to the satellite
    height : float, default: 450e3
        height of the layer above `radius` in meters (the 'height' of `read_ionex`)
    radius : float, default: 6371e3
        radius of the earth as a sphere in meters (the 'radius' of `read_ionex`)

    Returns
    -------
    ipp_lon : np.ndarray
        longitude of the pierce points in degrees, -180 to 180, shape (n_points,), float64
    ipp_lat : np.ndarray
        latitude of the pierce points in degrees (on the sphere), shape (n_points,), float64
    mapping : np.ndarray
        mapping factor, 1 or more, shape (n_points,), float64; all three nan where the inputs of the point are nan
    """
    lon = _float_points('lon', lon)
    lat = _float_points('lat', lat, lon.shape)
    theta = _float_points('theta', theta, lon.shape)
    phi = _float_points('phi', phi, lon.shape)
    rs = _shell(height, radius)
    _check_points(lon, lat, theta, phi)
    out = [np.empty(lon.shape) for _ in range(3)]
    _pierce_numba(lon, lat, theta, phi, rs, *out)
    return tuple(out)


def ionex_vtec(
    lon,
    lat,
    time,
    tec,
    tec_times:list,
    tec_lat,
    tec_lon,
):
    """Vertical total electron content at positions and one time, from maps of it.

    Interpolated bilinearly in the maps, and linearly in time between the two maps around the time, each turned in
    longitude with the Sun (the method recommended by the IONEX format).

    Parameters
    ----------
    lon : np.ndarray
        longitude of the positions in degrees (e.g. the pierce points of `pierce_point`), shape (n_points,), floating
        point
    lat : np.ndarray
        latitude of the positions in degrees, shape (n_points,), floating point
    time : datetime or str
        the time in UTC: a datetime, a numpy.datetime64 or ISO 8601 text (e.g. 2019-01-02T05:06:07.123456Z), within
        the times of the maps
    tec : np.ndarray
        maps of the vertical total electron content in TEC units, shape (nmaps, nlat, nlon), floating point (the
        'tec' of `read_ionex`)
    tec_times : list of datetime or str
        times of the maps in UTC, nmaps increasing date-times, at least 2 (the 'times' of `read_ionex`)
    tec_lat : np.ndarray
        latitude of the rows of the maps in degrees, evenly spaced, shape (nlat,) (the 'lat' of `read_ionex`)
    tec_lon : np.ndarray
        longitude of the columns of the maps in degrees, evenly spaced and increasing, shape (nlon,) (the 'lon' of
        `read_ionex`)

    Returns
    -------
    vtec : np.ndarray
        vertical total electron content in TEC units (1e16 electrons per square meter), shape (n_points,), float64;
        nan where the position is nan or outside the maps, or a map has no value next to it
    """
    lon = _float_points('lon', lon)
    lat = _float_points('lat', lat, lon.shape)
    t = _times(time)
    if len(t) != 1:
        raise ArgumentError(f'time must be one time, got {len(t)}')
    tec, t_maps, (lon0, dlon, lat0, dlat) = _maps(tec, tec_times, tec_lat, tec_lon)
    m, w, shift0, shift1 = _at_times(np.array([(t[0] - _EPOCH).total_seconds()]), t_maps)
    out = np.empty(lon.shape)
    _vtec_numba(lon, lat, tec, int(m[0]), float(w[0]), float(shift0[0]), float(shift1[0]), lon0, dlon, lat0, dlat,
                out)
    return out


def pc_iono_tec(
    lon,
    lat,
    theta,
    phi,
    times:list,
    wavelength:float,
    tec,
    tec_times:list,
    tec_lat,
    tec_lon,
    height:float=450e3,
    radius:float=6371e3,
    ratio:float=1.0,
    ref:int=0,
):
    """Line of sight displacement caused by the ionosphere at points, from maps of the total electron content, relative to a reference image.

    The slant total electron content along the look vector of a point is the vertical one of the maps at its pierce
    point (`ionex_vtec` at the points of `pierce_point`) times the mapping factor, `ratio` of it below the satellite;
    the ionosphere shortens the phase path by 40.31 STEC / f^2 (STEC in electrons per square meter, f the radar
    frequency in Hz), an apparent displacement towards the satellite.

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points in degrees, shape (n_points,), floating point
    lat : np.ndarray
        latitude of the points in degrees, shape (n_points,), floating point
    theta : np.ndarray
        elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle), shape
        (n_points,), floating point, 0 <= theta <= pi/2
    phi : np.ndarray
        orientation angle of the look vector of the points in radians, counterclockwise from east, shape (n_points,),
        floating point, -2 pi <= phi <= 2 pi; the look vector points from the ground to the satellite
    times : list of datetime or str
        time of every image in UTC, nimages date-times (datetime, numpy.datetime64 or ISO 8601 text, e.g.
        2019-01-02T05:06:07.123456Z) within the times of the maps, e.g. the `times` of `load_gamma_metadata`
    wavelength : float
        radar wavelength in meters, between 0 and 1 (0.0554658 for Sentinel-1)
    tec : np.ndarray
        maps of the vertical total electron content in TEC units, shape (nmaps, nlat, nlon), floating point (the
        'tec' of `read_ionex`)
    tec_times : list of datetime or str
        times of the maps in UTC, nmaps increasing date-times, at least 2 (the 'times' of `read_ionex`)
    tec_lat : np.ndarray
        latitude of the rows of the maps in degrees, evenly spaced, shape (nlat,) (the 'lat' of `read_ionex`)
    tec_lon : np.ndarray
        longitude of the columns of the maps in degrees, evenly spaced and increasing, shape (nlon,) (the 'lon' of
        `read_ionex`)
    height : float, default: 450e3
        height of the layer of the maps above `radius` in meters (the 'height' of `read_ionex`)
    radius : float, default: 6371e3
        radius of the earth as a sphere in meters (the 'radius' of `read_ionex`)
    ratio : float, default: 1.0
        part of the electron content of the maps below the satellite, 0 < ratio <= 1; 1 takes all of it
    ref : int, default: 0
        the image the displacement is relative to, 0 .. nimages - 1

    Returns
    -------
    iono : np.ndarray
        line of sight displacement in millimeters that the ionosphere causes in every image relative to the image
        `ref`, shape (n_points, nimages), float32, positive towards the satellite (more electrons); the column of
        `ref` is 0; the displacement corrected for the ionosphere is `disp` - `iono`; nan where `lon`, `lat`, `theta`
        or `phi` of the point is nan, or its pierce point is outside the maps or next to a node without value
    """
    lon = _float_points('lon', lon)
    lat = _float_points('lat', lat, lon.shape)
    theta = _float_points('theta', theta, lon.shape)
    phi = _float_points('phi', phi, lon.shape)
    t_img = _times(times)
    nimages = len(t_img)
    if nimages == 0:
        raise ArgumentError('times must have the time of at least one image')
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ArgumentError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    wavelength = _wavelength(wavelength)
    ratio = _ratio(ratio)
    rs = _shell(height, radius)
    tec, t_maps, (lon0, dlon, lat0, dlat) = _maps(tec, tec_times, tec_lat, tec_lon)
    _check_points(lon, lat, theta, phi)
    m, w, shift0, shift1 = _at_times(np.array([(t - _EPOCH).total_seconds() for t in t_img]), t_maps)
    scale = 1000 * ratio * _K * _TECU * wavelength ** 2 / _C ** 2          # mm per TEC unit along the line of sight
    out = np.empty((lon.shape[0], nimages), dtype=np.float32)
    _los_numba(lon, lat, theta, phi, rs, tec, m, w, shift0, shift1, lon0, dlon, lat0, dlat, scale, int(ref), out)
    return out


def _wavelength(wavelength)->float:
    """the radar wavelength in meters"""
    try:
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        raise ArgumentError(f'wavelength must be a number in meters, got {wavelength!r}') from None
    if not 0 < wavelength < 1:
        raise ArgumentError(f'wavelength must be in meters, between 0 and 1, got {wavelength}')
    return wavelength


def _ratio(ratio)->float:
    """the part of the electron content below the satellite"""
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        raise ArgumentError(f'ratio must be a number, got {ratio!r}') from None
    if not 0 < ratio <= 1:
        raise ArgumentError(f'ratio must be the part of the electron content below the satellite, 0 < ratio <= 1, '
                            f'got {ratio}')
    return ratio
