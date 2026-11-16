"""Satellite orbits: the perpendicular baseline of every image at points on the ground, from the state vectors of the
orbits, independent of the SAR processor"""

__all__ = ['bperp']

import math

import numpy as np
from numba import prange

from .jit import ngjit, ngpjit
from .errors import ArgumentError

# WGS84 ellipsoid
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F*(2 - _F)

# The orbit of an image is given by state vectors: time, position and velocity in an earth centred, earth fixed frame,
# e.g. 10 s or 60 s apart. Between two samples the position is the cubic Hermite polynomial of the positions and
# velocities at both ends; its error is below (dt/2)^4/24 max|x''''|, with x'''' ~ omega^4 r ~ 1e-5 m/s^4 on a low
# orbit: 0.2 mm for samples 10 s apart, but 0.3 m for 60 s. The state vectors are therefore first resampled every
# _STEP s by Lagrange interpolation of the positions and of the velocities over the _ORDER nearest state vectors (error
# ~ max|x^(8)| prod|t - t_i| / 8! ~ omega^8 r 50 dt^8 / 8!: 5e-5 m at the ends of an orbit with 60 s, much less
# inside), then interpolated by the Hermite polynomials of these samples (2e-8 m). The velocity and the acceleration
# are the derivatives of the Hermite polynomial.
#
# The satellite sees a point P at the zero-Doppler time t, where f(t) = (S(t) - P).V(t) = 0. f'(t) = |V|^2 + (S - P).A
# is positive on any orbit (|V|^2 ~ 5e7, |S - P| |A| ~ 7e6 m^2/s^2), so f increases with time: the interval of the
# state vectors where f changes sign is found by bisection, then Newton iterations converge in a few steps. The
# satellite of every image is taken at its own zero-Doppler time of P, so the time axes of the orbits of different
# images need not be related.
#
# The baseline b = S_j - S_ref is split along the line of sight u of the reference satellite to P, its velocity w (u
# and w are perpendicular at zero Doppler) and n = u x w, the direction perpendicular to both. The perpendicular
# baseline is b.n with n pointing upwards at P (n.up = sin of the incidence angle > 0): moving a pixel up by dh on its
# range circle changes the range of the satellite of image j relative to that of the reference by
# -b.n dh / (R sin(incidence)).

_STEP = 1.0     # s, time step of the resampled orbits
_ORDER = 8      # state vectors in the Lagrange interpolation

def _resample(t, pos, vel, n_out):
    """positions and velocities of one orbit at n_out times evenly spaced over the times of its state vectors, by
    Lagrange interpolation over the _ORDER nearest state vectors"""
    n = t.shape[0]
    m = min(_ORDER, n)
    t_out = np.linspace(t[0], t[-1], n_out)
    k0 = np.clip(np.searchsorted(t, t_out) - m//2, 0, n - m)
    idx = k0[:, None] + np.arange(m)
    tn = t[idx]
    w = np.ones((n_out, m))
    for i in range(m):
        for j in range(m):
            if j != i:
                w[:, i] *= (t_out - tn[:, j])/(tn[:, i] - tn[:, j])
    return t_out, (w[:, :, None]*pos[idx]).sum(1), (w[:, :, None]*vel[idx]).sum(1)

@ngjit
def _geodetic_to_ecef(lon, lat, hgt):
    """earth centred, earth fixed coordinates (m) of a point at longitude, latitude (degrees) and height above the
    WGS84 ellipsoid (m), and the unit normal of the ellipsoid there (upwards)"""
    phi = lat*math.pi/180
    lam = lon*math.pi/180
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    slam = math.sin(lam)
    clam = math.cos(lam)
    N = _A/math.sqrt(1 - _E2*sphi*sphi)
    x = (N + hgt)*cphi*clam
    y = (N + hgt)*cphi*slam
    z = (N*(1 - _E2) + hgt)*sphi
    return x, y, z, cphi*clam, cphi*slam, sphi

@ngjit
def _comb(c0, c1, c2, c3, k, c, pos, vel):
    return c0*pos[k, c] + c1*vel[k, c] + c2*pos[k+1, c] + c3*vel[k+1, c]

@ngjit
def _hermite(t, k, tt, pos, vel):
    """position, velocity and acceleration of one orbit at time t, by the cubic Hermite polynomial of the state
    vectors k and k + 1"""
    h = tt[k+1] - tt[k]
    s = (t - tt[k])/h
    s2 = s*s
    s3 = s2*s
    # basis of the position, of the velocity and of the acceleration, for (position k, velocity k, position k + 1,
    # velocity k + 1)
    b0, b1, b2, b3 = 2*s3 - 3*s2 + 1, (s3 - 2*s2 + s)*h, -2*s3 + 3*s2, (s3 - s2)*h
    d0, d1, d2, d3 = (6*s2 - 6*s)/h, 3*s2 - 4*s + 1, (-6*s2 + 6*s)/h, 3*s2 - 2*s
    e0, e1, e2, e3 = (12*s - 6)/(h*h), (6*s - 4)/h, (-12*s + 6)/(h*h), (6*s - 2)/h
    return (_comb(b0, b1, b2, b3, k, 0, pos, vel), _comb(b0, b1, b2, b3, k, 1, pos, vel),
            _comb(b0, b1, b2, b3, k, 2, pos, vel),
            _comb(d0, d1, d2, d3, k, 0, pos, vel), _comb(d0, d1, d2, d3, k, 1, pos, vel),
            _comb(d0, d1, d2, d3, k, 2, pos, vel),
            _comb(e0, e1, e2, e3, k, 0, pos, vel), _comb(e0, e1, e2, e3, k, 1, pos, vel),
            _comb(e0, e1, e2, e3, k, 2, pos, vel))

@ngjit
def _doppler_at_node(k, pos, vel, px, py, pz):
    return (pos[k, 0] - px)*vel[k, 0] + (pos[k, 1] - py)*vel[k, 1] + (pos[k, 2] - pz)*vel[k, 2]

@ngjit
def _zero_doppler(tt, pos, vel, px, py, pz):
    """zero-Doppler time of the point (px, py, pz) on one orbit and the position and velocity of the satellite then;
    nan if the satellite does not see the point within the time of the state vectors"""
    n = tt.shape[0]
    nan = np.nan
    lo = 0
    hi = n - 1
    flo = _doppler_at_node(lo, pos, vel, px, py, pz)
    fhi = _doppler_at_node(hi, pos, vel, px, py, pz)
    if not (flo <= 0 <= fhi):
        return nan, nan, nan, nan, nan, nan, nan
    while hi - lo > 1:
        mid = (lo + hi)//2
        fmid = _doppler_at_node(mid, pos, vel, px, py, pz)
        if fmid <= 0:
            lo = mid
            flo = fmid
        else:
            hi = mid
            fhi = fmid
    k = lo
    t = tt[lo] - flo*(tt[hi] - tt[lo])/(fhi - flo) if fhi > flo else tt[lo]
    for it in range(30):
        while k > 0 and t < tt[k]:
            k -= 1
        while k < n - 2 and t > tt[k+1]:
            k += 1
        sx, sy, sz, vx, vy, vz, ax, ay, az = _hermite(t, k, tt, pos, vel)
        dx = sx - px
        dy = sy - py
        dz = sz - pz
        f = dx*vx + dy*vy + dz*vz
        fp = vx*vx + vy*vy + vz*vz + dx*ax + dy*ay + dz*az
        dt = f/fp
        t -= dt
        if abs(dt) < 1e-10:
            break
    while k > 0 and t < tt[k]:
        k -= 1
    while k < n - 2 and t > tt[k+1]:
        k += 1
    sx, sy, sz, vx, vy, vz, ax, ay, az = _hermite(t, k, tt, pos, vel)
    return t, sx, sy, sz, vx, vy, vz

@ngpjit
def _bperp_numba(lon, lat, hgt, tt, pos, vel, ref, out):
    n_points = lon.shape[0]
    nimages = tt.shape[0]
    for i in prange(n_points):
        lo = lon[i]
        la = lat[i]
        h = np.float64(hgt[i])
        if not (np.isfinite(lo) and np.isfinite(la) and np.isfinite(h)):
            for j in range(nimages):
                out[i, j] = np.nan
            continue
        px, py, pz, upx, upy, upz = _geodetic_to_ecef(lo, la, h)
        t, rx, ry, rz, rvx, rvy, rvz = _zero_doppler(tt[ref], pos[ref], vel[ref], px, py, pz)
        if not np.isfinite(t):
            for j in range(nimages):
                out[i, j] = np.nan
            continue
        # line of sight u from the reference satellite to P, its velocity w, n = u x w pointing upwards at P
        ux = px - rx
        uy = py - ry
        uz = pz - rz
        r = math.sqrt(ux*ux + uy*uy + uz*uz)
        ux /= r
        uy /= r
        uz /= r
        w = math.sqrt(rvx*rvx + rvy*rvy + rvz*rvz)
        wx = rvx/w
        wy = rvy/w
        wz = rvz/w
        nx = uy*wz - uz*wy
        ny = uz*wx - ux*wz
        nz = ux*wy - uy*wx
        nn = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nx*upx + ny*upy + nz*upz < 0:
            nn = -nn
        nx /= nn
        ny /= nn
        nz /= nn
        for j in range(nimages):
            if j == ref:
                out[i, j] = 0
                continue
            t, sx, sy, sz, vx, vy, vz = _zero_doppler(tt[j], pos[j], vel[j], px, py, pz)
            out[i, j] = (sx - rx)*nx + (sy - ry)*ny + (sz - rz)*nz

def bperp(
    lon:np.ndarray,
    lat:np.ndarray,
    hgt:np.ndarray,
    orbit_time:np.ndarray,
    orbit_pos:np.ndarray,
    orbit_vel:np.ndarray,
    ref:int=0,
)->np.ndarray:
    """Perpendicular baseline of every image relative to a reference image at points on the ground, from the orbits.

    Parameters
    ----------
    lon : np.ndarray
        longitude of the points (WGS84), degrees, float64, any shape
    lat : np.ndarray
        latitude of the points (WGS84), degrees, float64, shape of `lon`
    hgt : np.ndarray
        height of the points above the WGS84 ellipsoid, meters, float32 or float64, shape of `lon`
    orbit_time : np.ndarray
        times of the orbit state vectors of every image, seconds, float64, shape (nimages, n_state_vectors), at
        least 4 per image, increasing along the second axis; each image may have its own time origin
    orbit_pos : np.ndarray
        satellite positions at these times in earth centred, earth fixed coordinates (WGS84), meters, float64, shape
        (nimages, n_state_vectors, 3)
    orbit_vel : np.ndarray
        satellite velocities at these times, meters per second, float64, shape (nimages, n_state_vectors, 3)
    ref : int, default: 0
        index of the reference image, 0 .. nimages - 1

    Returns
    -------
    np.ndarray
        perpendicular baseline, meters, float32, shape (`lon.shape`, nimages): the component of the position of the
        satellite of each image relative to that of the reference image, each when it sees the point (zero Doppler),
        perpendicular to the line of sight of the reference and to its flight direction, positive upwards; 0 for the
        reference image; nan where `lon`, `lat` or `hgt` is nan or the point is not seen within the times of an orbit
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    hgt = np.asarray(hgt)
    if lat.shape != lon.shape or hgt.shape != lon.shape:
        raise ArgumentError(f'lon, lat and hgt must have the same shape, not {lon.shape}, {lat.shape}, {hgt.shape}')
    orbit_time = np.ascontiguousarray(orbit_time, dtype=np.float64)
    orbit_pos = np.ascontiguousarray(orbit_pos, dtype=np.float64)
    orbit_vel = np.ascontiguousarray(orbit_vel, dtype=np.float64)
    nimages, nsv = orbit_time.shape
    if orbit_pos.shape != (nimages, nsv, 3) or orbit_vel.shape != (nimages, nsv, 3):
        raise ArgumentError(f'orbit_pos and orbit_vel must have the shape {(nimages, nsv, 3)}, '
                         f'not {orbit_pos.shape} and {orbit_vel.shape}')
    if nsv < 4 or not (np.diff(orbit_time, axis=1) > 0).all():
        raise ValueError('every orbit needs at least 4 state vectors at increasing times')
    if not 0 <= ref < nimages:
        raise ArgumentError(f'ref must be in 0 .. {nimages - 1}, not {ref}')
    n_out = int(np.ceil((orbit_time[:, -1] - orbit_time[:, 0]).max()/_STEP)) + 1
    orbit_time, orbit_pos, orbit_vel = (np.ascontiguousarray(np.stack(a)) for a in
                                        zip(*(_resample(*o, n_out) for o in zip(orbit_time, orbit_pos, orbit_vel))))
    out = np.empty((lon.size, nimages), dtype=np.float32)
    _bperp_numba(np.ascontiguousarray(lon).reshape(-1), np.ascontiguousarray(lat).reshape(-1),
                 np.ascontiguousarray(hgt).reshape(-1), orbit_time, orbit_pos, orbit_vel, ref, out)
    return out.reshape(*lon.shape, nimages)
