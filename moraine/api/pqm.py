"""Pixel quality metrics"""

__all__ = ['temp_coh', 'pc_su_dem_error']

import math
import numpy as np
from .utils_.device import is_cuda_available, get_array_module
from .utils_.jit import mcuda_jit
from numba import prange
from .utils_.jit import ngpjit, ngjit
from .tnet import TempNet

@ngpjit
def _temp_coh_pc_numba(
    intf:np.ndarray,
    rslc:np.ndarray,
    image_pairs:np.ndarray,
):
    """Parameters
    ----------
    intf : np.ndarray
        complex interferograms/coherence metrix, dtype np.complex64
    rslc : np.ndarray
        complex rslc/phase history, dtype np.complex64
    image_pairs : np.ndarray
        image pairs
    """
    nimages = rslc.shape[-1]
    n_points = rslc.shape[0]
    n_image_pairs = image_pairs.shape[0]
    temp_coh = np.empty(n_points,dtype=np.float32)
    for i in prange(n_points):
        # unit amplitude values in local variables: the inputs are not changed
        rslc_ = np.empty(nimages, dtype=rslc.dtype)
        for j in range(nimages):
            rslc_[j] = rslc[i,j]/abs(rslc[i,j])
        _t_coh = np.complex64(0.0)
        for j in range(n_image_pairs):
            n, k = image_pairs[j,0],image_pairs[j,1]
            rslc_intf_ = np.conjugate(rslc_[n])*rslc_[k]
            diff_ph = intf[i,j]/abs(intf[i,j])*rslc_intf_
            _t_coh += diff_ph
        _t_coh = np.abs(_t_coh)/n_image_pairs
        temp_coh[i] = _t_coh
    return temp_coh

if is_cuda_available():
    from numba import cuda

    @mcuda_jit()
    def _temp_coh_pc_cuda(intf, rslc, ref, sec, temp_coh):
        # one warp per point: the lanes read consecutive image pairs (coalesced), then reduce by shuffles;
        # |sum over the image pairs of the unit interferogram times the unit conj(ref) sec| / n_pairs
        i = cuda.grid(1)//32
        if i >= intf.shape[0]:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        n_pairs = intf.shape[1]
        sr = np.float32(0.0); si = np.float32(0.0)
        for j in range(lane, n_pairs, 32):
            c = intf[i,j]; a = rslc[i,ref[j]]; b = rslc[i,sec[j]]
            p = c*a.conjugate()*b
            norm = math.sqrt((c.real*c.real+c.imag*c.imag)*(a.real*a.real+a.imag*a.imag)*(b.real*b.real+b.imag*b.imag))
            sr += p.real/norm
            si += p.imag/norm
        offset = 16
        while offset > 0:
            sr += cuda.shfl_down_sync(0xffffffff, sr, offset)
            si += cuda.shfl_down_sync(0xffffffff, si, offset)
            offset //= 2
        if lane == 0:
            temp_coh[i] = math.sqrt(sr*sr+si*si)/n_pairs

def temp_coh(
    intf:np.ndarray,
    rslc:np.ndarray,
    image_pairs:np.ndarray=None,
    block_size:int=128,
):
    """Estimation of temporal coherence.

    Parameters
    ----------
    intf : np.ndarray
        complex interferograms/coherence metrix, dtype cp/np.complex64, shape 2D(pc) or 3D(ras)
    rslc : np.ndarray
        complex rslc/phase history, dtype cp/np.complex64, shape 2D(pc) or 3D(ras)
    image_pairs : np.ndarray, optional
        image pairs
    block_size : int, default: 128
        the CUDA block size, a multiple of 32, only applied for cuda
    """
    xp = get_array_module(intf)
    if intf.ndim != rslc.ndim:
        raise ValueError(f'intf and rslc must both be rasters or both point clouds, got {intf.ndim} and {rslc.ndim} dimensions')
    if intf.ndim == 3:
        # convert to pc
        is_ras = True
        height, width = intf.shape[:2]
        n_points =height*width
        intf = intf.reshape((-1,intf.shape[-1]))
        rslc = rslc.reshape((-1,rslc.shape[-1]))
    elif intf.ndim == 2:
        is_ras = False
        n_points = rslc.shape[0]

    nimages = rslc.shape[-1]
    if image_pairs is None:
        image_pairs = TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = image_pairs.astype(np.int32)

    if xp is np:
        out_temp_coh = _temp_coh_pc_numba(intf,rslc,image_pairs)
    else:
        import cupy as cp
        out_temp_coh = cp.empty(n_points, dtype=cp.float32)
        if n_points > 0:
            _temp_coh_pc_cuda[(n_points*32+block_size-1)//block_size, block_size](
                cp.ascontiguousarray(intf), cp.ascontiguousarray(rslc), cp.asarray(image_pairs[:,0]),
                cp.asarray(image_pairs[:,1]), out_temp_coh)

    if is_ras:
        out_temp_coh = out_temp_coh.reshape((height,width))
    return out_temp_coh


@ngpjit
def _su_dem_error_numba(intf, rslc, ref, sec, bperp, factor, max_dh, dh, t_coh):
    """For every point the DEM error (m) that maximizes the temporal coherence of `rslc` relative to `intf`, and that
    coherence; factor: phase in radians per meter of DEM error and meter of perpendicular baseline."""
    n_points = rslc.shape[0]
    n_pairs = ref.shape[0]
    for i in prange(n_points):
        diff = np.empty(n_pairs, dtype=np.complex128)     # filtered minus own phase of every image pair
        a = np.empty(n_pairs, dtype=np.float64)           # its change per meter of DEM error
        finite = np.isfinite(factor[i])
        a_max = 0.0
        for k in range(n_pairs):
            x, u, v = intf[i, k], rslc[i, ref[k]], rslc[i, sec[k]]
            d = x/abs(x)*np.conj(u)/abs(u)*v/abs(v)
            diff[k] = d
            a[k] = factor[i]*(bperp[i, ref[k]] - bperp[i, sec[k]])
            if not (np.isfinite(d.real) and np.isfinite(d.imag) and np.isfinite(a[k])):
                finite = False
            elif abs(a[k]) > a_max:
                a_max = abs(a[k])
        if not finite:
            dh[i] = np.nan
            t_coh[i] = np.nan
            continue
        if a_max == 0.0:                                   # the baselines do not change: no DEM error to fit
            s = 0j
            for k in range(n_pairs):
                s += diff[k]
            dh[i] = np.nan
            t_coh[i] = abs(s)/n_pairs
            continue
        # trials from -max_dh to max_dh, the phase of the pair with the largest baseline changing by at most pi/4
        # from one trial to the next (as StaMPS), 0 among them; the phasors are rotated from trial to trial
        half = int(math.ceil(max_dh/(math.pi/4/a_max)))
        n_trials = 2*half + 1
        step = max_dh/half
        cur = np.empty(n_pairs, dtype=np.complex128)
        rot = np.empty(n_pairs, dtype=np.complex128)
        for k in range(n_pairs):
            cur[k] = diff[k]*np.exp(-1j*a[k]*max_dh)
            rot[k] = np.exp(1j*a[k]*step)
        best, best_dh = -1.0, 0.0
        for t in range(n_trials):
            s = 0j
            for k in range(n_pairs):
                s += cur[k]
                cur[k] *= rot[k]
            if abs(s) > best:
                best, best_dh = abs(s), -max_dh + t*step
        # golden section search of the largest coherence within a step of the best trial (and the search range)
        lo, hi = max(best_dh - step, -max_dh), min(best_dh + step, max_dh)
        x1, x2 = hi - _GOLDEN*(hi - lo), lo + _GOLDEN*(hi - lo)
        g1, g2 = _coherence_at(diff, a, x1), _coherence_at(diff, a, x2)
        for _ in range(30):
            if g1 >= g2:
                hi, x2, g2 = x2, x1, g1
                x1 = hi - _GOLDEN*(hi - lo)
                g1 = _coherence_at(diff, a, x1)
            else:
                lo, x1, g1 = x1, x2, g2
                x2 = lo + _GOLDEN*(hi - lo)
                g2 = _coherence_at(diff, a, x2)
        if max(g1, g2) > best:
            best, best_dh = (g1, x1) if g1 >= g2 else (g2, x2)
        dh[i] = best_dh
        t_coh[i] = best/n_pairs


_GOLDEN = (math.sqrt(5) - 1)/2


@ngjit
def _coherence_at(diff, a, dh):
    """|sum of diff exp(i a dh)|"""
    s = 0j
    for k in range(diff.shape[0]):
        s += diff[k]*np.exp(1j*a[k]*dh)
    return abs(s)


@ngpjit
def _remove_dem_error_numba(rslc, bperp, factor, dh, out):
    """`rslc` without the phase of the DEM error `dh` (unchanged where dh is nan)"""
    for i in prange(rslc.shape[0]):
        for k in range(rslc.shape[1]):
            if np.isfinite(dh[i]):
                out[i, k] = rslc[i, k]*np.exp(-1j*factor[i]*bperp[i, k]*dh[i])
            else:
                out[i, k] = rslc[i, k]


def _phase_per_m(slant_range, theta, wavelength):
    """phase in radians per meter of DEM error and meter of perpendicular baseline: 4 pi / (lambda R cos(theta))"""
    return (4*np.pi/wavelength/(np.asarray(slant_range, dtype=np.float64)*np.cos(np.asarray(theta, dtype=np.float64))))


def _check_su_inputs(intf, rslc, bperp, slant_range, theta, wavelength, image_pairs, max_dh):
    """checked image pairs (int32) of the inputs of `pc_su_dem_error`"""
    if rslc.ndim != 2 or intf.ndim != 2:
        raise ValueError(f'intf and rslc must have shape (n_points, n_pairs) and (n_points, nimages), got shapes '
                         f'{tuple(intf.shape)} and {tuple(rslc.shape)}')
    n_points, nimages = rslc.shape
    for name, x in (('intf', intf), ('rslc', rslc)):
        if not np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError(f'{name} must be complex, got {x.dtype}')
    for name, x, shape in (('intf', intf, (n_points, intf.shape[1])), ('bperp', bperp, (n_points, nimages)),
                           ('slant_range', slant_range, (n_points,)), ('theta', theta, (n_points,))):
        if tuple(x.shape) != shape:
            raise ValueError(f'{name} must have shape {shape}, got shape {tuple(x.shape)}')
    for name, x in (('bperp', bperp), ('slant_range', slant_range), ('theta', theta)):
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError(f'{name} must be floating point, got {x.dtype}')
    if not 0 < float(wavelength) < 1:
        raise ValueError(f'wavelength must be in meters, between 0 and 1, got {wavelength}')
    if not float(max_dh) > 0:
        raise ValueError(f'max_dh must be positive, got {max_dh}')
    if image_pairs is None:
        image_pairs = TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = np.asarray(image_pairs).astype(np.int32)
    if image_pairs.ndim != 2 or image_pairs.shape != (intf.shape[1], 2) or \
       (image_pairs.size and (image_pairs.min() < 0 or image_pairs.max() >= nimages)):
        raise ValueError(f'intf has {intf.shape[1]} interferograms of {nimages} images, but image_pairs has shape '
                         f'{image_pairs.shape} with images {image_pairs.min() if image_pairs.size else "-"} to '
                         f'{image_pairs.max() if image_pairs.size else "-"}')
    return image_pairs


def pc_su_dem_error(
    intf:np.ndarray,
    rslc:np.ndarray,
    bperp:np.ndarray,
    slant_range:np.ndarray,
    theta:np.ndarray,
    wavelength:float,
    image_pairs:np.ndarray=None,
    max_dh:float=20.0,
):
    """DEM error of points relative to their neighbourhood in filtered interferograms, and the temporal coherence
    without it.

    A point dh meters above its neighbourhood (a roof among ground points) differs from the filtered interferogram of
    the images a and b by 4 pi / lambda (B_perp,a - B_perp,b) dh / (R cos(theta)) radians, which the temporal
    coherence takes for noise. dh is the DEM error within -max_dh .. max_dh that maximizes the temporal coherence of
    the point with this term removed, and `t_coh` is that coherence. Fitting dh raises the temporal coherence of noise
    a little: random phases reach about 0.35 (p99.99) with 91 sequential image pairs and +-80 m.

    Parameters
    ----------
    intf : np.ndarray
        filtered interferograms of the points, the phase of their neighbourhood, shape (n_points, n_pairs), complex64
        (e.g. of `n2ft`)
    rslc : np.ndarray
        phase of the points (rslc or phase history), shape (n_points, nimages), complex64
    bperp : np.ndarray
        perpendicular baseline of every image at every point in meters, shape (n_points, nimages), floating point;
        relative to the reference image of the co-registration, positive upwards (`load_gamma_bperp`)
    slant_range : np.ndarray
        slant range of the points in meters, shape (n_points,), floating point
    theta : np.ndarray
        elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle), shape
        (n_points,), floating point
    wavelength : float
        radar wavelength in meters, between 0 and 1
    image_pairs : np.ndarray, optional
        image pairs (reference, secondary) of `intf`, shape (n_pairs, 2); all image pairs by default
    max_dh : float, default: 20
        the DEM error is searched within -max_dh .. max_dh meters; a point further from its neighbourhood gets a low
        temporal coherence

    Returns
    -------
    dh : np.ndarray
        DEM error relative to the neighbourhood in meters, shape (n_points,), float32: the height of the point minus
        that of its neighbourhood in `intf`, positive above it; nan where an input of the point is not finite or the
        baselines of all its image pairs are equal
    t_coh : np.ndarray
        temporal coherence of the point without the term of `dh`, shape (n_points,), float32, 0 .. 1 (nan where an
        input is not finite)
    """
    image_pairs = _check_su_inputs(intf, rslc, bperp, slant_range, theta, wavelength, image_pairs, max_dh)
    n_points = rslc.shape[0]
    dh = np.empty(n_points, dtype=np.float32)
    t_coh = np.empty(n_points, dtype=np.float32)
    if n_points:
        _su_dem_error_numba(np.ascontiguousarray(intf), np.ascontiguousarray(rslc), image_pairs[:, 0].copy(),
                            image_pairs[:, 1].copy(), np.ascontiguousarray(bperp, dtype=np.float64),
                            _phase_per_m(slant_range, theta, float(wavelength)), float(max_dh), dh, t_coh)
    return dh, t_coh


def _remove_su_dem_error(rslc, bperp, slant_range, theta, wavelength, dh):
    """`rslc` without the phase of the DEM error `dh` of every point (unchanged where dh is nan), complex64"""
    out = np.empty(rslc.shape, dtype=np.complex64)
    if rslc.shape[0]:
        _remove_dem_error_numba(np.ascontiguousarray(rslc), np.ascontiguousarray(bperp, dtype=np.float64),
                                _phase_per_m(slant_range, theta, float(wavelength)),
                                np.ascontiguousarray(dh, dtype=np.float64), out)
    return out
