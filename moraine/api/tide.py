"""Solid earth tides of points: the displacement of the ground by the tides of the Sun and the Moon (decision 0056)"""

__all__ = ['solid_tide', 'pc_solid_tide']

import datetime
import math

import numpy as np
from numba import prange

from .utils_.errors import ArgumentError
from .utils_.jit import ngjit, ngpjit
from .utils_.times import _times

# The displacement is that of the IERS Conventions (2010), section 7.1.1 (Mathews et al., 1997), as computed by their
# routine DEHANTTIDEINEL (version of 2017): step 1 (degree 2 and 3 in phase, out of phase parts of the diurnal and
# semidiurnal band, latitude dependence of l) and step 2 (frequency dependence in the diurnal and long period band).
# The permanent tide is not restored (conventional tide free); it cancels in the difference of two images.
_H20, _L20 = 0.6078, 0.0847     # nominal degree 2 Love and Shida numbers
_H3, _L3 = 0.292, 0.015         # degree 3
_MASS_SUN = 332946.0482         # GM of the Sun / GM of the Earth
_MASS_MOON = 0.0123000371       # GM of the Moon / GM of the Earth
_RE = 6378136.6                 # m, equatorial radius of the Earth
# WGS84: the points are on the ellipsoid (their height changes the tide by less than 0.1 %)
_A = 6378137.0
_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
_DEG = math.pi / 180

# Step 2, diurnal band (Table 7.3a, as DATDI of STEP2DIU): multipliers of s, h, p, N', ps, then dR in phase, dR out
# of phase, dT in phase, dT out of phase in mm; the argument is tau plus the multiples, plus the longitude.
_DIURNAL = np.array([
    [-3, 0, 2, 0, 0, -0.01, 0, 0, 0],
    [-3, 2, 0, 0, 0, -0.01, 0, 0, 0],
    [-2, 0, 1, -1, 0, -0.02, 0, 0, 0],
    [-2, 0, 1, 0, 0, -0.08, 0, -0.01, 0.01],
    [-2, 2, -1, 0, 0, -0.02, 0, 0, 0],
    [-1, 0, 0, -1, 0, -0.10, 0, 0, 0],
    [-1, 0, 0, 0, 0, -0.51, 0, -0.02, 0.03],
    [-1, 2, 0, 0, 0, 0.01, 0, 0, 0],
    [0, -2, 1, 0, 0, 0.01, 0, 0, 0],
    [0, 0, -1, 0, 0, 0.02, 0, 0, 0],
    [0, 0, 1, 0, 0, 0.06, 0, 0, 0],
    [0, 0, 1, 1, 0, 0.01, 0, 0, 0],
    [0, 2, -1, 0, 0, 0.01, 0, 0, 0],
    [1, -3, 0, 0, 1, -0.06, 0, 0, 0],
    [1, -2, 0, -1, 0, 0.01, 0, 0, 0],
    [1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01],
    [1, -1, 0, 0, -1, 0.02, 0, 0, 0],
    [1, -1, 0, 0, 1, 0.04, 0, 0, 0],
    [1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0],
    [1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03],
    [1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0],
    [1, 0, 0, 2, 0, -0.04, 0, 0, 0],
    [1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0],
    [1, 1, 0, 0, 1, 0.01, 0, 0, 0],
    [0, 1, 0, 1, -1, -0.01, 0, 0, 0],
    [1, 2, -2, 0, 0, -0.01, 0, 0, 0],
    [1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0],
    [2, -2, 1, 0, 0, -0.01, 0, 0, 0],
    [2, 0, -1, 0, 0, -0.02, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 1, 0, 0, 0, 0, 0],
])
# Step 2, long period band (Table 7.3b, as DATDI of STEP2LON): multipliers of s, h, p, N', ps, then the coefficients
# in mm of the radial part with cos, the tangential part with cos, the radial part with sin, the tangential part with
# sin of the argument, in the order in which STEP2LON uses its columns
_LONG = np.array([
    [0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07],
    [0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05],
    [1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04],
    [2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07],
    [2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03],
])

# TAI - UTC in seconds from the given day (UTC) on; no leap second after 2017-01-01 announced by the IERS up to the
# writing of this table (Bulletin C)
_LEAP = (
    (1972, 1, 1, 10), (1972, 7, 1, 11), (1973, 1, 1, 12), (1974, 1, 1, 13), (1975, 1, 1, 14), (1976, 1, 1, 15),
    (1977, 1, 1, 16), (1978, 1, 1, 17), (1979, 1, 1, 18), (1980, 1, 1, 19), (1981, 7, 1, 20), (1982, 7, 1, 21),
    (1983, 7, 1, 22), (1985, 7, 1, 23), (1988, 1, 1, 24), (1990, 1, 1, 25), (1991, 1, 1, 26), (1992, 7, 1, 27),
    (1993, 7, 1, 28), (1994, 7, 1, 29), (1996, 1, 1, 30), (1997, 7, 1, 31), (1999, 1, 1, 32), (2006, 1, 1, 33),
    (2009, 1, 1, 34), (2012, 7, 1, 35), (2015, 7, 1, 36), (2017, 1, 1, 37),
)
_MJD0 = datetime.datetime(1858, 11, 17)     # MJD 0
_J2000 = 51544.5                            # MJD of J2000.0 (2000-01-01 12:00 TT)


def _check_time(t):
    """an ArgumentError for a time before the leap second table (1972)"""
    if t < datetime.datetime(*_LEAP[0][:3]):
        raise ArgumentError(f'times must be from 1972-01-01 on (UTC), got {t:%Y-%m-%d %H:%M:%S}')


def _tai_utc(t)->float:
    """TAI - UTC in seconds at the naive UTC datetime `t` (1972 or later)"""
    out = _LEAP[0][3]
    for y, m, d, s in _LEAP:
        if t >= datetime.datetime(y, m, d):
            out = s
    return float(out)


def _mjd(t)->float:
    """modified Julian date of a naive datetime"""
    return (t - _MJD0) / datetime.timedelta(days=1)


def _sun_moon(t)->tuple:
    """positions of the Sun and the Moon in meters in the Earth fixed frame (ECEF), shape (3,) each, at the naive UTC
    datetime `t`"""
    # Low precision series of Montenbruck & Gill (2000, Satellite Orbits, section 3.3.2, equations 3.43 - 3.51) in TT,
    # longitudes referred to the mean equinox of date (precession 1.3972 degrees per century added), the J2000 obliquity
    # of the ecliptic; rotated into the Earth fixed frame by the Greenwich mean sidereal time (equation 2.85, UT1 taken
    # as UTC), polar motion and nutation neglected.
    mjd_utc = _mjd(t)
    T = (mjd_utc + (_tai_utc(t) + 32.184) / 86400 - _J2000) / 36525      # centuries of TT since J2000
    eps = 23.43929111 * _DEG
    # Sun
    M = (357.5256 + 35999.049 * T) * _DEG
    lon = (282.9400 + 1.3972 * T) * _DEG + M + (6892 * math.sin(M) + 72 * math.sin(2 * M)) / 3600 * _DEG
    r = (149.619 - 2.499 * math.cos(M) - 0.021 * math.cos(2 * M)) * 1e9
    sun = np.array([r * math.cos(lon), r * math.sin(lon) * math.cos(eps), r * math.sin(lon) * math.sin(eps)])
    # Moon: mean longitude (of date), mean anomalies of the Moon and the Sun, argument of latitude, mean elongation
    L0 = 218.31617 + 481267.88088 * T
    l = (134.96292 + 477198.86753 * T) * _DEG
    lp = (357.52543 + 35999.04944 * T) * _DEG
    F = (93.27283 + 483202.01873 * T) * _DEG
    D = (297.85027 + 445267.11135 * T) * _DEG
    dlon = (22640 * math.sin(l) + 769 * math.sin(2 * l) - 4586 * math.sin(l - 2 * D) + 2370 * math.sin(2 * D)
            - 668 * math.sin(lp) - 412 * math.sin(2 * F) - 212 * math.sin(2 * l - 2 * D)
            - 206 * math.sin(l + lp - 2 * D) + 192 * math.sin(l + 2 * D) - 165 * math.sin(lp - 2 * D)
            + 148 * math.sin(l - lp) - 125 * math.sin(D) - 110 * math.sin(l + lp)
            - 55 * math.sin(2 * F - 2 * D)) / 3600          # degrees
    lon = (L0 + dlon) * _DEG
    lat = (18520 * math.sin(F + (dlon + (412 * math.sin(2 * F) + 541 * math.sin(lp)) / 3600) * _DEG)
           - 526 * math.sin(F - 2 * D) + 44 * math.sin(l + F - 2 * D) - 31 * math.sin(-l + F - 2 * D)
           - 25 * math.sin(-2 * l + F) - 23 * math.sin(lp + F - 2 * D) + 21 * math.sin(-l + F)
           + 11 * math.sin(-lp + F - 2 * D)) / 3600 * _DEG
    r = (385000 - 20905 * math.cos(l) - 3699 * math.cos(2 * D - l) - 2956 * math.cos(2 * D) - 570 * math.cos(2 * l)
         + 246 * math.cos(2 * l - 2 * D) - 205 * math.cos(lp - 2 * D) - 171 * math.cos(l + 2 * D)
         - 152 * math.cos(l + lp - 2 * D)) * 1e3
    x, y, z = r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon), r * math.sin(lat)
    moon = np.array([x, y * math.cos(eps) - z * math.sin(eps), y * math.sin(eps) + z * math.cos(eps)])
    # equatorial (mean equinox of date) -> Earth fixed
    g = ((280.46061837 + 360.98564736629 * (mjd_utc - _J2000)) % 360) * _DEG
    rot = np.array([[math.cos(g), math.sin(g), 0], [-math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return rot @ sun, rot @ moon


def _step2(t)->np.ndarray:
    """the terms of step 2 at the naive UTC datetime `t`, in meters, shape (6,): (Rs, Rc, Ts, Tc) of the diurnal band
    and (Lr, Ln) of the long period band"""
    # The arguments of STEP2DIU / STEP2LON: T in centuries of TT since J2000, the hours of the day in UTC. A sum over
    # the diurnal terms of a sin(theta + lambda) + b cos(theta + lambda) is cos(lambda) (sum of a sin(theta) + b
    # cos(theta)) + sin(lambda) (sum of a cos(theta) - b sin(theta)): the sums over the terms are made once per image.
    mjd_utc = _mjd(t)
    T = (mjd_utc + (_tai_utc(t) + 32.184) / 86400 - _J2000) / 36525
    fhr = (mjd_utc - math.floor(mjd_utc)) * 24
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * T) * T) * T
    tau = fhr * 15 + 280.4606184 + (36000.7700536 + (0.00038793 - 0.0000000258 * T) * T) * T - s
    s += (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * T) * T) * T) * T
    h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654 * T) * T) * T) * T
    p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * T) * T) * T) * T
    zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * T) * T) * T) * T
    ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334 * T) * T) * T) * T
    args = np.array([s, h, p, zns, ps]) % 360
    th = (tau % 360 + _DIURNAL[:, :5] @ args) * _DEG
    sin, cos = np.sin(th), np.cos(th)
    a, b, c, d = (_DIURNAL[:, 5 + i] for i in range(4))
    rs, rc = np.sum(a * sin + b * cos), np.sum(a * cos - b * sin)
    ts, tc = np.sum(c * sin + d * cos), np.sum(c * cos - d * sin)
    th = (_LONG[:, :5] @ args) * _DEG
    sin, cos = np.sin(th), np.cos(th)
    lr = np.sum(_LONG[:, 5] * cos + _LONG[:, 7] * sin)
    ln = np.sum(_LONG[:, 6] * cos + _LONG[:, 8] * sin)
    return np.array([rs, rc, ts, tc, lr, ln]) / 1000


def _image_terms(times)->tuple:
    """Sun (nimages, 3), Moon (nimages, 3) in meters (ECEF) and the terms of step 2 (nimages, 6) of naive UTC
    datetimes"""
    sun = np.empty((len(times), 3))
    moon = np.empty((len(times), 3))
    step2 = np.empty((len(times), 6))
    for k, t in enumerate(times):
        sun[k], moon[k] = _sun_moon(t)
        step2[k] = _step2(t)
    return sun, moon, step2


@ngjit
def _body(x, y, z, r, sinphi, cosphi, sinla, cosla, h2, l2, b):
    """step 1 for one body (Sun or Moon) with position b (ECEF, m) and mass ratio: the in phase displacement (ECEF,
    m) and the out of phase and latitude dependence corrections (radial, north, east, m)"""
    bx, by, bz, mass = b[0], b[1], b[2], b[3]
    rb = math.sqrt(bx * bx + by * by + bz * bz)
    sc = (x * bx + y * by + z * bz) / (r * rb)
    fac2 = mass * _RE * (_RE / rb) ** 3
    fac3 = fac2 * _RE / rb
    # degree 2 and 3 in phase (equations 7.5 and 7.6)
    p2 = 3 * (h2 / 2 - l2) * sc * sc - h2 / 2
    x2 = 3 * l2 * sc
    p3 = 2.5 * (_H3 - 3 * _L3) * sc ** 3 + 1.5 * (_L3 - _H3) * sc
    x3 = 1.5 * _L3 * (5 * sc * sc - 1)
    fb = (fac2 * x2 + fac3 * x3) / rb
    fs = (fac2 * p2 + fac3 * p3) / r
    dx = fb * bx + fs * x
    dy = fb * by + fs * y
    dz = fb * bz + fs * z
    # out of phase parts in the diurnal (ST1IDIU) and semidiurnal (ST1ISEM) band, latitude dependence of l (ST1L1)
    rb2 = rb * rb
    q1 = fac2 * bz * (bx * sinla - by * cosla) / rb2
    q2 = fac2 * bz * (bx * cosla + by * sinla) / rb2
    cos2la = cosla * cosla - sinla * sinla
    sin2la = 2 * cosla * sinla
    q3 = fac2 * ((bx * bx - by * by) * sin2la - 2 * bx * by * cos2la) / rb2
    q4 = fac2 * ((bx * bx - by * by) * cos2la + 2 * bx * by * sin2la) / rb2
    cos2phi = cosphi * cosphi - sinphi * sinphi
    dr = 3 * 0.0025 * sinphi * cosphi * q1 + 0.75 * 0.0022 * cosphi * cosphi * q3
    dn = 3 * 0.0007 * cos2phi * q1 - 1.5 * 0.0007 * sinphi * cosphi * q3
    de = 3 * 0.0007 * sinphi * q2 + 1.5 * 0.0007 * cosphi * q4
    dn += 3 * (-0.0012 * sinphi * sinphi * q2 - 0.0024 / 2 * sinphi * cosphi * q4)
    de += 3 * (0.0012 * sinphi * cos2phi * q1 - 0.0024 / 2 * sinphi * sinphi * cosphi * q3)
    return dx, dy, dz, dr, dn, de


@ngjit
def _displacement(x, y, z, sun, moon, step2):
    """tidal displacement (ECEF, m) of the point (x, y, z) (ECEF, m) for the Sun and Moon (x, y, z, mass ratio) and the
    terms of step 2 of one time"""
    r = math.sqrt(x * x + y * y + z * z)
    rxy = math.sqrt(x * x + y * y)
    sinphi, cosphi = z / r, rxy / r                 # geocentric latitude
    sinla, cosla = y / rxy, x / rxy
    h2 = _H20 - 0.0006 * (1 - 1.5 * cosphi * cosphi)
    l2 = _L20 + 0.0002 * (1 - 1.5 * cosphi * cosphi)
    sx, sy, sz, sr, sn, se = _body(x, y, z, r, sinphi, cosphi, sinla, cosla, h2, l2, sun)
    mx, my, mz, mr, mn, me = _body(x, y, z, r, sinphi, cosphi, sinla, cosla, h2, l2, moon)
    dr = sr + mr
    dn = sn + mn
    de = se + me
    # step 2
    dr += 2 * sinphi * cosphi * (cosla * step2[0] + sinla * step2[1]) + (3 * sinphi * sinphi - 1) / 2 * step2[4]
    dn += (cosphi * cosphi - sinphi * sinphi) * (cosla * step2[2] + sinla * step2[3]) \
        + 2 * cosphi * sinphi * step2[5]
    de += sinphi * (cosla * step2[3] - sinla * step2[2])
    dx = sx + mx + dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    dy = sy + my + dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    dz = sz + mz + dr * sinphi + dn * cosphi
    return dx, dy, dz


@ngjit
def _ecef(lon, lat):
    """position (m) of a point of the WGS84 ellipsoid and its local east, north, up unit vectors (ECEF)"""
    sl, cl = math.sin(lon * _DEG), math.cos(lon * _DEG)
    sp, cp = math.sin(lat * _DEG), math.cos(lat * _DEG)
    n = _A / math.sqrt(1 - _E2 * sp * sp)
    return (n * cp * cl, n * cp * sl, n * (1 - _E2) * sp,
            (-sl, cl, 0.0), (-sp * cl, -sp * sl, cp), (cp * cl, cp * sl, sp))


@ngpjit
def _enu_numba(lon, lat, sun, moon, step2, out):
    for i in prange(lon.shape[0]):
        x, y, z, e, n, u = _ecef(lon[i], lat[i])
        dx, dy, dz = _displacement(x, y, z, sun, moon, step2)
        out[i, 0] = e[0] * dx + e[1] * dy + e[2] * dz
        out[i, 1] = n[0] * dx + n[1] * dy + n[2] * dz
        out[i, 2] = u[0] * dx + u[1] * dy + u[2] * dz


@ngpjit
def _los_numba(lon, lat, theta, phi, sun, moon, step2, ref, out):
    nimages = sun.shape[0]
    for i in prange(lon.shape[0]):
        x, y, z, e, n, u = _ecef(lon[i], lat[i])
        ce, cn, cu = math.cos(theta[i]) * math.cos(phi[i]), math.cos(theta[i]) * math.sin(phi[i]), math.sin(theta[i])
        lx = ce * e[0] + cn * n[0] + cu * u[0]
        ly = ce * e[1] + cn * n[1] + cu * u[1]
        lz = ce * e[2] + cn * n[2] + cu * u[2]
        dx, dy, dz = _displacement(x, y, z, sun[ref], moon[ref], step2[ref])
        d_ref = lx * dx + ly * dy + lz * dz
        for k in range(nimages):
            dx, dy, dz = _displacement(x, y, z, sun[k], moon[k], step2[k])
            out[i, k] = 1000 * (lx * dx + ly * dy + lz * dz - d_ref)


def _bodies(sun, moon)->tuple:
    """Sun and Moon positions (n, 3) with their mass ratios as a fourth column"""
    return (np.column_stack([sun, np.full(len(sun), _MASS_SUN)]),
            np.column_stack([moon, np.full(len(moon), _MASS_MOON)]))


def _float_points(name, a, shape=None)->np.ndarray:
    """a 1D floating point array of points as float64"""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        raise ArgumentError(f'{name} must be floating point, got {a.dtype}')
    if a.ndim != 1 or (shape is not None and a.shape != shape):
        want = f'the shape {shape} of lon' if shape is not None else 'shape (n_points,)'
        raise ArgumentError(f'{name} must have {want}, got shape {a.shape}')
    return a.astype(np.float64)


def _check_lon_lat(lon, lat):
    """ValueError for longitudes or latitudes out of range"""
    with np.errstate(invalid='ignore'):
        if (np.abs(lat) > 90).any():
            raise ValueError('lat must be latitudes in degrees between -90 and 90')
        if (np.abs(lon) > 360).any():
            raise ValueError('lon must be longitudes in degrees between -360 and 360')


def solid_tide(
    lon,
    lat,
    time,
):
    """Displacement of the ground by the solid earth tides at points at one time.

    The tides of the Sun and the Moon of the IERS Conventions (2010), including their part that does not change with
    time, at points on the WGS84 ellipsoid.

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points in degrees, shape (n_points,), floating point
    lat : np.ndarray
        latitude of the points in degrees, shape (n_points,), floating point
    time : datetime or str
        the time in UTC: a datetime, a numpy.datetime64 or ISO 8601 text (e.g. 2019-01-02T05:06:07.123456Z), 1972
        or later

    Returns
    -------
    tide : np.ndarray
        east, north and up displacement of the points in meters, shape (n_points, 3), float64; nan where `lon` or
        `lat` is nan
    """
    lon = _float_points('lon', lon)
    lat = _float_points('lat', lat, lon.shape)
    t = _times(time)
    if len(t) != 1:
        raise ArgumentError(f'time must be one time, got {len(t)}')
    _check_time(t[0])
    _check_lon_lat(lon, lat)
    sun, moon, step2 = _image_terms(t)
    sun, moon = _bodies(sun, moon)
    out = np.empty((lon.shape[0], 3))
    _enu_numba(lon, lat, sun[0], moon[0], step2[0], out)
    return out


def pc_solid_tide(
    lon,
    lat,
    theta,
    phi,
    times:list,
    ref:int=0,
):
    """Line of sight displacement caused by the solid earth tides at points, relative to a reference image.

    The displacement of the ground of `solid_tide` at the time of every image, projected on the look vector of the
    point.

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
        2019-01-02T05:06:07.123456Z), 1972 or later, e.g. the `times` of `load_gamma_metadata`
    ref : int, default: 0
        the image the displacement is relative to, 0 .. nimages - 1

    Returns
    -------
    tide : np.ndarray
        line of sight displacement in millimeters that the solid earth tides cause in every image relative to the
        image `ref`, shape (n_points, nimages), float32, positive towards the satellite; the column of `ref` is 0; the
        displacement corrected for the tides is `disp` - `tide`; nan where `lon`, `lat`, `theta` or `phi` of the point
        is nan
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
    for t in t_img:
        _check_time(t)
    _check_lon_lat(lon, lat)
    with np.errstate(invalid='ignore'):
        if ((theta < -1e-6) | (theta > np.pi / 2 + 1e-6)).any():
            raise ValueError('theta must be the elevation angle of the look vector in radians, between 0 and pi/2')
        if (np.abs(phi) > 2 * np.pi + 1e-6).any():
            raise ValueError('phi must be the orientation angle of the look vector in radians, between -2 pi and 2 pi')
    sun, moon, step2 = _image_terms(t_img)
    sun, moon = _bodies(sun, moon)
    out = np.empty((lon.shape[0], nimages), dtype=np.float32)
    _los_numba(lon, lat, theta, phi, sun, moon, step2, int(ref), out)
    return out
