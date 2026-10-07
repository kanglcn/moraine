"""Spatially Homogenious Pixels Identification"""


__all__ = ['ks_test', 'select_shp']

import numpy as np
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
    from numba import cuda
import math
import numba
from numba import prange
from .utils_ import ngpjit, ngjit

@ngjit
def _ks_p_numba(x):
    x2 = -2*x*x
    p = 0.0
    p2 = 0.0
    sign = 1
    for i in range(1,101):
        p += sign*2*math.exp(x2*i*i)
        if (p == p2):
            return p
        else:
            sign = -sign
            p2 = p
    # the series does not converge for x near 0 (at x = 0 it alternates 2, 0, 2, ...), where the p value is 1
    return 1.0

# The KS statistic of two samples of n values is k/n with an integer k = max |j2 - j1| over the merged order, so
# the p value takes n + 1 values only: it is looked up in a table computed with _ks_p_numba (the same values as
# evaluating the series for every pair). The test is symmetric, KS(a, b) = KS(b, a): on the CPU every pixel computes
# the offsets after the centre of the window and writes both entries. The same _ks_k is compiled for the CPU and
# the GPU, so both give the same results.
def _ks_k(ref, sec, n):
    """k = max |j2 - j1| of the merged sorted samples `ref`, `sec` of n values"""
    j1 = 0; j2 = 0; kmax = 0
    while (j1 < n) and (j2 < n):
        f1 = ref[j1]; f2 = sec[j2]
        if f1 <= f2: j1 += 1
        if f1 >= f2: j2 += 1
        k = abs(j2-j1)
        if k > kmax: kmax = k
    return kmax

_ks_k_numba = ngjit(_ks_k)

def _ks_p_table(n):
    """p value of the KS statistic k/n of two samples of n values, k = 0..n, float64"""
    en = math.sqrt(n/2)
    return np.array([_ks_p_numba((en+0.12+0.11/en)*(k/n)) for k in range(n+1)])

@ngpjit
def _ks_test_numba(
    rmli,
    az_half_win,
    r_half_win,
    p_table,
    p,
    dist,
):
    """p (and dist if it is not empty) of the sorted rmli stack, both (n_az, n_r, az_win, r_win)"""
    n_az, n_r, n = rmli.shape
    aw = 2*az_half_win+1; rw = 2*r_half_win+1
    with_dist = dist.size > 0
    for i in prange(n_az):
        for j in range(n_r):
            ref = rmli[i,j]
            ref_nan = math.isnan(ref[n-1])     # nan are sorted to the end
            p[i,j,az_half_win,r_half_win] = np.nan if ref_nan else p_table[0]
            if with_dist: dist[i,j,az_half_win,r_half_win] = np.nan if ref_nan else 0.0
            # the offsets after the centre, and their mirror at the other pixel
            for l in range(az_half_win, aw):
                for m in range(rw):
                    if l == az_half_win and m <= r_half_win:
                        continue
                    si = i+l-az_half_win; sj = j+m-r_half_win
                    inside = (si < n_az) and (sj >= 0) and (sj < n_r)
                    if (not inside) or ref_nan or math.isnan(rmli[si,sj,n-1]):
                        k = -1
                    else:
                        k = _ks_k_numba(ref, rmli[si,sj], n)
                    p_ = np.nan if k < 0 else p_table[k]
                    p[i,j,l,m] = p_
                    if inside: p[si,sj,aw-1-l,rw-1-m] = p_
                    if with_dist:
                        d_ = np.nan if k < 0 else k/n
                        dist[i,j,l,m] = d_
                        if inside: dist[si,sj,aw-1-l,rw-1-m] = d_
            # the offsets before the centre whose other pixel is outside the image
            for l in range(0, az_half_win+1):
                for m in range(rw):
                    if l == az_half_win and m >= r_half_win:
                        continue
                    si = i+l-az_half_win; sj = j+m-r_half_win
                    if (si < 0) or (sj < 0) or (sj >= n_r):
                        p[i,j,l,m] = np.nan
                        if with_dist: dist[i,j,l,m] = np.nan

@ngpjit
def _sort_numba(
    rmli,
):
    """Parameters
    ----------
    rmli
        rmli stack
    """
    n_az, n_r,n_imag = rmli.shape
    sorted_rmli = np.empty_like(rmli)
    for i in prange(n_az):
        for j in prange(n_r):
            sorted_rmli[i,j] = np.sort(rmli[i,j])
    return sorted_rmli

if is_cuda_available():
    _ks_k_cuda = cuda.jit(device=True)(_ks_k)

    @cuda.jit
    def _ks_test_cuda(rmli, az_half_win, r_half_win, p_table, p, dist):
        # one thread per pixel and offset in its window
        t = cuda.grid(1)
        n_az, n_r, n = rmli.shape
        aw = 2*az_half_win+1; rw = 2*r_half_win+1; win = aw*rw
        if t >= n_az*n_r*win:
            return
        pix = t//win; o = t%win
        i = pix//n_r; j = pix%n_r; l = o//rw; m = o%rw
        si = i+l-az_half_win; sj = j+m-r_half_win
        if si < 0 or si >= n_az or sj < 0 or sj >= n_r or math.isnan(rmli[i,j,n-1]) or math.isnan(rmli[si,sj,n-1]):
            p[i,j,l,m] = math.nan
            if dist.size > 0: dist[i,j,l,m] = math.nan
            return
        k = _ks_k_cuda(rmli[i,j], rmli[si,sj], n)
        p[i,j,l,m] = p_table[k]
        if dist.size > 0: dist[i,j,l,m] = k/n

def ks_test(rmli:np.ndarray,
            az_half_win:int,
            r_half_win:int,
            block_size:int=128,
            return_dist:bool=False,
           ) -> tuple[np.ndarray,np.ndarray] :
    """SHP identification based on Two-Sample Kolmogorov-Smirnov Test.

    Parameters
    ----------
    rmli : np.ndarray
        the rmli stack, dtype: cupy.floating
    az_half_win : int
        SHP identification half search window size in azimuth direction
    r_half_win : int
        SHP identification half search window size in range direction
    block_size : int, default: 128
        the CUDA block size, it only affects the calculation speed, only applied if input is cupy.ndarray
    return_dist : bool, default: False
        if return the KS test statistics `dist`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        if return_dist == True, return `dist` and p value `p`. Otherwise, only `p` is returned.
    """
    xp = get_array_module(rmli)
    az_win = 2*az_half_win+1
    r_win = 2*r_half_win+1
    nlines, width, nimages = rmli.shape
    shape = (nlines, width, az_win, r_win)
    p_table = _ks_p_table(nimages).astype(rmli.dtype)
    p = xp.empty(shape, dtype=rmli.dtype)
    dist = xp.empty(shape if return_dist else (0,0,0,0), dtype=rmli.dtype)
    if xp is np:
        _ks_test_numba(_sort_numba(rmli), az_half_win, r_half_win, p_table, p, dist)
    else:
        n = p.size
        if n > 0:
            _ks_test_cuda[(n+block_size-1)//block_size, block_size](
                cp.sort(rmli,axis=-1), np.int32(az_half_win), np.int32(r_half_win), cp.asarray(p_table), p, dist)
    return (dist, p) if return_dist else p

@ngpjit
def select_shp(
    p,
    alpha,
):
    """Select the SHPs: the pixels whose test against the centre pixel is not rejected at the level `alpha`.

    Parameters
    ----------
    p
        p value of the test between each pixel and the pixels in its window, shape (n_az, n_r, az_win, r_win),
        floating; nan for no test
    alpha
        significance level, float in (0, 1): a pixel is an SHP when its p value is at least `alpha`

    Returns
    -------
    is_shp
        True for SHPs, shape (n_az, n_r, az_win, r_win), bool
    shp_num
        number of SHPs of each pixel, shape (n_az, n_r), int32
    """
    p_shape = p.shape
    is_shp = np.empty(p_shape, dtype=np.bool_)
    shp_num = np.zeros(p_shape[:2], dtype=np.int32)
    for i in prange(p_shape[0]):
        for j in prange(p_shape[1]):
            for k in range(p_shape[2]):
                for l in range(p_shape[3]):
                    is_shp[i,j,k,l] = p[i,j,k,l] >= alpha
                    if is_shp[i,j,k,l]:
                        shp_num[i,j] += 1
    return is_shp, shp_num
