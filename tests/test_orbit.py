"""Perpendicular baselines from the orbit state vectors (moraine.api.utils_.orbit), on synthetic orbits."""
import math

import numpy as np
import pytest
from pyproj import Transformer

from moraine.api.utils_ import orbit
from moraine.api.utils_.orbit import bperp

A, H, V = 6378137.0, 700e3, 7000.0     # WGS84 semi-major axis, orbit height, speed


def _line(x, y, t0=0.0, dt=10.0, n=15):
    """state vectors (n, dt s apart) of a straight orbit along z through (x, y), at z = V (t + t0) centred on the
    equator"""
    t = np.arange(n)*dt
    pos = np.stack([np.full(n, x), np.full(n, y), V*(t + t0 - (n - 1)*dt/2)], -1)
    return t, pos, np.tile([0.0, 0.0, V], (n, 1))


def _circle(r, t0=0.0, dt=10.0, n=15):
    """state vectors (n, dt s apart) of a circular polar orbit of radius r over longitude 0, at the angle
    w (t + t0) from the equator, centred on the equator"""
    w = math.sqrt(3.986004418e14/r**3)
    t = np.arange(n)*dt
    a = w*(t + t0 - (n - 1)*dt/2)
    pos = np.stack([r*np.cos(a), np.zeros(n), r*np.sin(a)], -1)
    return t, pos, np.stack([-r*w*np.sin(a), np.zeros(n), r*w*np.cos(a)], -1)


def _points(n, seed=0):
    """points right of the orbits, near the equator"""
    rng = np.random.default_rng(seed)
    return rng.uniform(2.5, 5.5, n), rng.uniform(-1, 1, n), rng.uniform(0, 2000, n).astype(np.float32)


def _ecef(lon, lat, hgt):
    """earth centred, earth fixed coordinates (pyproj) and the upward normal of the ellipsoid"""
    x, y, z = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True).transform(lon, lat, np.float64(hgt))
    phi, lam = np.radians(lat), np.radians(lon)
    return np.stack([x, y, z], -1), np.stack([np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)], -1)


def _perp(S, w, P, up):
    """unit vector perpendicular to the line of sight from S to P and to the flight direction w, pointing upwards"""
    u = P - S
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    n = np.cross(u, w)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n*np.sign((n*up).sum(-1, keepdims=True))


def _stack(*orbits):
    return tuple(np.stack(a) for a in zip(*orbits))


def test_geodetic_to_ecef_as_pyproj():
    lon, lat, hgt = _points(50)
    lat = np.linspace(-89, 89, 50)
    P, up = _ecef(lon, lat, hgt)
    for i in range(50):
        x = orbit._geodetic_to_ecef(lon[i], lat[i], np.float64(hgt[i]))
        np.testing.assert_allclose(x[:3], P[i], rtol=0, atol=1e-6)
        np.testing.assert_allclose(x[3:], up[i], rtol=0, atol=1e-12)


def test_bperp_parallel_straight_orbits():
    """Parallel straight orbits: the satellites see a point at the same z, the baseline is their offset in the x-y
    plane; the orbits of the images have different time origins."""
    offsets = [(0.0, 0.0, 0.0), (40.0, 150.0, 3.7), (-25.0, -80.0, -2.1)]
    otime, opos, ovel = _stack(*(_line(A + H + dx, dy, t0) for dx, dy, t0 in offsets))
    lon, lat, hgt = _points(2000)
    B = bperp(lon, lat, hgt, otime, opos, ovel, ref=0)
    assert B.shape == (2000, 3) and B.dtype == np.float32
    assert (B[:, 0] == 0).all()
    P, up = _ecef(lon, lat, hgt)
    S = np.stack([np.full(2000, A + H), np.zeros(2000), P[:, 2]], -1)
    n = _perp(S, np.array([0, 0, 1.0]), P, up)
    for j, (dx, dy, _) in enumerate(offsets[1:], 1):
        np.testing.assert_allclose(B[:, j], (n*[dx, dy, 0.0]).sum(-1), rtol=0, atol=1e-4)
    # another reference: the baselines relative to image 1
    B1 = bperp(lon, lat, hgt, otime, opos, ovel, ref=1)
    assert (B1[:, 1] == 0).all()
    S1 = S + [40.0, 150.0, 0.0]
    n1 = _perp(S1, np.array([0, 0, 1.0]), P, up)
    np.testing.assert_allclose(B1[:, 0], (n1*[-40.0, -150.0, 0.0]).sum(-1), rtol=0, atol=1e-4)


def test_bperp_is_the_height_sensitivity_of_the_range_difference():
    """Moving a pixel up on the range circle of the reference changes the range of image j relative to the reference
    by -bperp / (R sin(incidence)) per meter of height."""
    otime, opos, ovel = _stack(_line(A + H, 0.0), _line(A + H + 40.0, 150.0, 3.7), _line(A + H - 90.0, 30.0, -1.0))
    lon, lat, hgt = _points(20, seed=3)
    B = bperp(lon, lat, hgt, otime, opos, ovel, ref=0)
    P, up = _ecef(lon, lat, hgt)
    inv = Transformer.from_crs('EPSG:4978', 'EPSG:4979', always_xy=True)
    for i in range(20):
        S = np.array([A + H, 0.0, P[i, 2]])
        R = np.linalg.norm(P[i] - S)
        u = (P[i] - S)/R
        n = _perp(S, np.array([0, 0, 1.0]), P[i], up[i])
        sin_inc = math.sqrt(1 - (u @ up[i])**2)
        P2 = S + R*(math.cos(1e-5)*u + math.sin(1e-5)*n)      # same range, 1e-5 rad upwards on the range circle
        dh = inv.transform(*P2)[2] - float(hgt[i])
        for j, b in ((1, [40.0, 150.0, 0.0]), (2, [-90.0, 30.0, 0.0])):
            Sj = S + b
            ddrho = (np.linalg.norm(P2 - Sj) - R) - (np.linalg.norm(P[i] - Sj) - R)
            np.testing.assert_allclose(ddrho/dh, -B[i, j]/(R*sin_inc), rtol=1e-3)


@pytest.mark.parametrize('dt, n', [(10.0, 15), (60.0, 11)])
def test_bperp_circular_orbits(dt, n):
    """A curved orbit, state vectors 10 s or 60 s apart: the same path sampled at other times has no baseline; an
    orbit 100 m higher in the same plane has the baseline of the radial offset."""
    r = A + H
    otime, opos, ovel = _stack(_circle(r, 0.0, dt, n), _circle(r, 4.3, dt, n), _circle(r + 100.0, 0.0, dt, n))
    lon, lat, hgt = _points(500, seed=1)
    B = bperp(lon, lat, hgt, otime, opos, ovel, ref=0)
    np.testing.assert_allclose(B[:, 1], 0, atol=1e-3)
    P, up = _ecef(lon, lat, hgt)
    a = np.arctan2(P[:, 2], P[:, 0])                       # zero Doppler on a circle: the velocity is normal to P
    S = r*np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], -1)
    w = np.stack([-np.sin(a), np.zeros_like(a), np.cos(a)], -1)
    n = _perp(S, w, P, up)
    np.testing.assert_allclose(B[:, 2], 100.0*(n*S/r).sum(-1), rtol=0, atol=1e-3)


def test_bperp_nan_and_points_not_seen():
    otime, opos, ovel = _stack(_line(A + H, 0.0), _line(A + H + 40.0, 150.0))
    lon = np.array([[3.0, np.nan, 3.0], [3.0, 3.0, 3.0]])
    lat = np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])    # 20 degrees north: beyond the end of the orbits
    hgt = np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 10.0]], dtype=np.float32)
    B = bperp(lon, lat, hgt, otime, opos, ovel)
    assert B.shape == (2, 3, 2)
    expect_nan = np.array([[False, True, True], [False, True, False]])
    assert (np.isnan(B).all(-1) == expect_nan).all() and not np.isnan(B[~expect_nan]).any()


def test_bperp_input_errors():
    otime, opos, ovel = _stack(_line(A + H, 0.0), _line(A + H + 40.0, 150.0))
    lon, lat, hgt = _points(5)
    with pytest.raises(ValueError, match='same shape'):
        bperp(lon, lat[:4], hgt, otime, opos, ovel)
    with pytest.raises(ValueError, match='ref'):
        bperp(lon, lat, hgt, otime, opos, ovel, ref=2)
    with pytest.raises(ValueError, match='increasing'):
        bperp(lon, lat, hgt, otime[:, ::-1], opos, ovel)
    with pytest.raises(ValueError, match='shape'):
        bperp(lon, lat, hgt, otime, opos[:, :, :2], ovel)
