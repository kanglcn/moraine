"""Covariance and coherence matrix estimation"""

__all__ = ['multi_look', 'intf', 'emperical_co', 'emperical_co_pc', 'uncompress_single_coh_numba', 'uncompress_coh', 'ad_intf_pc',
           'isPD', 'nearestPD', 'regularize_spectral']

import math
import numpy as np
from .utils_ import is_cuda_available, get_array_module, mcuda_jit
if is_cuda_available():
    import cupy as cp
    from numba import cuda
from typing import Union
import moraine as mr
from .utils_ import ngpjit, ngjit
from numba import prange

@ngpjit
def multi_look(
    rslc1,
    rslc2,
    looks=(1,1),
):
    """multi looked interferogram for raster data.

    Parameters
    ----------
    rslc1
        reference rslc
    rslc2
        secondary rslc
    looks : default: (1, 1)
        # range and azimuth looks
    """
    az_look, r_look = looks
    out_ny = math.floor(rslc1.shape[0]/az_look)
    out_nx = math.floor(rslc1.shape[1]/r_look)
    out = np.empty((out_ny, out_nx),dtype=rslc1.dtype)
    for i in prange(out_ny):
        for j in range(out_nx):
            az_start, az_stop = i*az_look, (i+1)*az_look
            r_start, r_stop = j*r_look, (j+1)*r_look
            rslc1_ = rslc1[az_start:az_stop, r_start:r_stop].flatten()
            rslc2_ = rslc2[az_start:az_stop, r_start:r_stop].flatten()
            dnmnt = math.sqrt(np.sum(rslc1_.real**2+rslc1_.imag**2)*np.sum(rslc2_.real**2+rslc2_.imag**2))
            out[i,j] = np.sum(rslc1_*rslc2_.conj())/dnmnt
    return out

@ngpjit
def intf(
    rslc1,
    rslc2,
):
    """1 by 1 look interferogram for both raster and point cloud.

    Parameters
    ----------
    rslc1
        reference rslc, arbitrary dims, e.g., 3D(raster stack), 2D(raster or point cloud stack) or 1D(point cloud)
    rslc2
        secondary rslc
    """
    shape = rslc1.shape

    rslc1 = rslc1.reshape(-1)
    rslc2 = rslc2.reshape(-1)
    intf = np.empty(rslc1.shape,dtype=rslc1.dtype)
    for i in prange(rslc1.shape[0]):
        intf[i] = rslc1[i]*np.conj(rslc2[i])/(np.abs(rslc1[i])*np.abs(rslc2[i]))
    intf = intf.reshape(shape)
    return intf

def emperical_co(rslc:np.ndarray,
                 is_shp:np.ndarray,
                 block_size:int=128,
                )-> tuple[np.ndarray,np.ndarray]:
    """Maximum likelihood covariance estimator.

    Parameters
    ----------
    rslc : np.ndarray
        rslc stack, dtype: `cupy.complexfloating`
    is_shp : np.ndarray
        shp bool, dtype: `cupy.bool`
    block_size : int, default: 128
        no effect, kept for compatibility

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        the covariance and coherence matrix `cov` and `coh`
    """
    xp = get_array_module(rslc)
    if xp is np:
        raise NotImplementedError("Currently only cuda version available.")
    nlines, width, nimages = rslc.shape
    az_win, r_win = is_shp.shape[-2:]
    az_half_win = (az_win-1)//2
    r_half_win = (r_win-1)//2

    cov = cp.empty((nlines*width,nimages,nimages),dtype=rslc.dtype)
    coh = cp.empty((nlines*width,nimages,nimages),dtype=rslc.dtype)
    az_idx, r_idx = (cp.asarray(i.ravel(), dtype=cp.int32) for i in np.meshgrid(np.arange(nlines), np.arange(width), indexing='ij'))
    for start, stop, c, n in _shp_products_cp(rslc, az_idx, r_idx, is_shp.reshape(-1, az_win, r_win)):
        c = c.conj()     # sum x_m conj(x_j)
        cov[start:stop] = c/n[:,None,None].astype(cp.float32)
        d = 1/cp.sqrt(cp.diagonal(c, axis1=1, axis2=2).real)
        coh[start:stop] = c*(d[:,:,None]*d[:,None,:])
    return cov.reshape(nlines,width,nimages,nimages), coh.reshape(nlines,width,nimages,nimages)

# Coherence of the SHP samples X (n_shp, nimages) of a point: X^H X normalized by its diagonal, with BLAS on the
# CPU (one BLAS thread per numba thread, see moraine.api.pl._single_thread_blas) and batched matrix products on the
# GPU; every image pair of the point reads the same samples, so all pairs come from one matrix product.
@ngpjit
def _emperical_co_pc_numba(
    rslc,
    az_idx,
    r_idx,
    pc_is_shp,
    image_pairs,
):
    nlines, width, nimages = rslc.shape
    n_pc = az_idx.shape[0]
    az_win, r_win = pc_is_shp.shape[1:]
    az_half_win, r_half_win = az_win//2, r_win//2
    npairs = image_pairs.shape[0]

    coh = np.empty((n_pc, npairs),dtype=rslc.dtype)

    for i in prange(n_pc):
        n_shp = 0
        for k in range(az_win):
            for l in range(r_win):
                az_idx_ = az_idx[i] - az_half_win + k
                r_idx_ = r_idx[i] - r_half_win + l
                if (az_idx_ >= 0) and (az_idx_ < nlines) and (r_idx_ >= 0) and (r_idx_ < width) and pc_is_shp[i,k,l]:
                    n_shp += 1
        if n_shp == 0:
            for pair_i in range(npairs):
                coh[i,pair_i] = np.nan
            continue
        x = np.empty((n_shp, nimages), dtype=rslc.dtype)
        n_shp = 0
        for k in range(az_win):
            for l in range(r_win):
                az_idx_ = az_idx[i] - az_half_win + k
                r_idx_ = r_idx[i] - r_half_win + l
                if (az_idx_ >= 0) and (az_idx_ < nlines) and (r_idx_ >= 0) and (r_idx_ < width) and pc_is_shp[i,k,l]:
                    x[n_shp] = rslc[az_idx_, r_idx_]
                    n_shp += 1
        c = np.dot(np.conj(x).T, x)    # c[m, j] = sum conj(x_m) x_j
        d = np.empty(nimages, dtype=np.float32)
        for m in range(nimages):
            d[m] = 1/math.sqrt(c[m,m].real)
        for pair_i in range(npairs):
            m, j = image_pairs[pair_i]
            coh[i,pair_i] = np.conj(c[m,j])*(d[m]*d[j])
    return coh

if is_cuda_available():
    def _shp_products_cp(rslc, az_idx, r_idx, pc_is_shp, max_bytes=2**30):
        """for batches of points: (start, stop, c, n) with c = X^H X of the SHP samples X (n_shp, nimages) of every
        point, (b, nimages, nimages), and the number of SHPs n, (b,)"""
        nlines, width, nimages = rslc.shape
        n_pc = az_idx.shape[0]
        az_win, r_win = pc_is_shp.shape[1:]
        win = az_win*r_win
        da, dr = np.meshgrid(np.arange(az_win)-az_win//2, np.arange(r_win)-r_win//2, indexing='ij')
        da = cp.asarray(da.ravel(), dtype=cp.int32); dr = cp.asarray(dr.ravel(), dtype=cp.int32)
        # samples, their conjugate transpose and the product of every point of a batch, complex64
        batch = max(1, int(max_bytes//(8*(2*win*nimages+2*nimages*nimages))))
        for start in range(0, n_pc, batch):
            stop = min(start+batch, n_pc)
            a = az_idx[start:stop,None]+da; r = r_idx[start:stop,None]+dr                          # (b, win)
            valid = pc_is_shp[start:stop].reshape(-1, win) & (a >= 0) & (a < nlines) & (r >= 0) & (r < width)
            x = rslc[cp.clip(a, 0, nlines-1), cp.clip(r, 0, width-1)]                             # (b, win, nimages)
            x = cp.where(valid[...,None], x, 0)       # not x*valid: nan outside the SHPs must not count
            c = cp.matmul(x.conj().swapaxes(1,2), x)                                             # (b, nimages, nimages)
            del x
            yield start, stop, c, cp.count_nonzero(valid, axis=1)

    def _emperical_co_pc_cp(rslc, az_idx, r_idx, pc_is_shp, image_pairs):
        ref = cp.asarray(image_pairs[:,0]); sec = cp.asarray(image_pairs[:,1])
        coh = cp.empty((az_idx.shape[0], image_pairs.shape[0]), dtype=rslc.dtype)
        for start, stop, c, _ in _shp_products_cp(rslc, az_idx, r_idx, pc_is_shp):
            d = 1/cp.sqrt(cp.diagonal(c, axis1=1, axis2=2).real)
            coh[start:stop] = c[:,ref,sec].conj()*(d[:,ref]*d[:,sec])
        return coh

def emperical_co_pc(rslc:np.ndarray,
                    idx:np.ndarray,
                    pc_is_shp:np.ndarray,
                    block_size:int=128,
                    image_pairs:np.ndarray=None,
                    return_n_looks:bool=False,
                   ):
    """Maximum likelihood covariance estimator for point cloud data.

    Parameters
    ----------
    rslc : np.ndarray
        rslc stack, dtype: `np.complex64`
    idx : np.ndarray
        index of point target (azimuth_index, range_index), dtype: `np.int32`, shape: (n_pc, 2)
    pc_is_shp : np.ndarray
        shp bool, dtype: `bool`
    block_size : int, default: 128
        no effect, kept for compatibility
    image_pairs : np.ndarray, optional
        only coherence of those image pairs will estimated, dtype: `np.int32`, shape: (n_image_pair, 2)
    return_n_looks : bool, default: False
        also return the effective number of independent looks of the SHP set of each point

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        `coh`, dtype:`np.complex64`, shape(n_pc, n_image_pair); with `return_n_looks` also `n_looks`, dtype
        float32, shape (n_pc,): the number of independent looks with the same variance of the coherence
        estimate as the correlated SHPs of the point (between 1 and the number of SHPs; smaller for compact
        SHP sets than for scattered ones of the same size), from the positions of the SHPs and the speckle
        correlation of `rslc`; the number of SHPs where `rslc` is too small to estimate the correlation
    """
    xp = get_array_module(rslc)
    nlines, width, nimages = rslc.shape
    az_win, r_win = pc_is_shp.shape[-2:]
    az_half_win = (az_win-1)//2
    r_half_win = (r_win-1)//2
    idx = idx.astype(np.int32)
    az_idx = idx[:,0]; r_idx = idx[:,1]
    n_pc = az_idx.shape[0]

    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = image_pairs.astype(np.int32)

    if xp is np:
        from .pl import _single_thread_blas
        with _single_thread_blas():
            coh = _emperical_co_pc_numba(rslc,az_idx,r_idx,pc_is_shp,image_pairs)
    else:
        coh = _emperical_co_pc_cp(rslc,az_idx,r_idx,pc_is_shp,image_pairs)
    if not return_n_looks:
        return coh
    return coh, _n_looks(pc_is_shp, _rslc_rho2(rslc))

@ngjit
def uncompress_single_coh_numba(coh, nimages, image_pairs):
    uncompressed_coh = np.zeros((nimages,nimages),dtype=coh.dtype)
    ref_images, sec_images = image_pairs[:,0], image_pairs[:,1]
    for i in range(ref_images.shape[0]):
        uncompressed_coh[ref_images[i], sec_images[i]] = coh[i]
        uncompressed_coh[sec_images[i], ref_images[i]] = np.conj(coh[i])
    for i in range(nimages):
        uncompressed_coh[i,i] = 1
    return uncompressed_coh

def uncompress_coh(
    coh:np.ndarray,
    image_pairs:np.ndarray=None,
)-> np.ndarray:
    """uncompress coh matrix to a hermitian matrix

    Parameters
    ----------
    coh : np.ndarray
        compressed coherence stack, dtype: `np.complex64`, shape(...,n_image_pair)
    image_pairs : np.ndarray, optional
        image pairs, dtype: `np.int32`, shape: (n_image_pair, 2), all pairs by default

    Returns
    -------
    np.ndarray
        uncompressed, dtype:`np.complex64`, shape(..., n_image_pair)
    """
    xp = get_array_module(coh)
    if image_pairs is None:
        nimages = mr.nimage_from_npair(coh.shape[-1])
        image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    else:
        nimages = image_pairs[-1,-1]+1
    uncompressed_coh = xp.zeros((*coh.shape[:-1],nimages,nimages),dtype=coh.dtype)
    ref_images, sec_images = image_pairs[:,0], image_pairs[:,1]
    uncompressed_coh[...,ref_images,sec_images] = coh
    uncompressed_coh[...,sec_images,ref_images] = coh.conj()
    uncompressed_coh[...,np.arange(nimages),np.arange(nimages)] = 1
    return uncompressed_coh

@ngpjit
def _ad_intf_pc_numba(
    ref_rslc,
    sec_rslc,
    az_idx,
    r_idx,
    pc_is_shp,
):
    nlines, width = ref_rslc.shape
    n_pc = az_idx.shape[0]
    az_win, r_win = pc_is_shp.shape[1:]
    az_half_win, r_half_win = az_win//2, r_win//2

    inf = np.empty(n_pc,dtype=ref_rslc.dtype)

    for i in prange(n_pc):
        _co_nume = 0.0 + 0.0j
        _ref_amp2 = 0.0
        _sec_amp2 = 0.0
        for k in range(az_win):
            for l in range(r_win):
                az_idx_ = az_idx[i] - az_half_win + k
                r_idx_ = r_idx[i] - r_half_win + l
                if (az_idx_ >= 0) and (az_idx_ < nlines) and (r_idx_ >= 0) and (r_idx_ < width) and pc_is_shp[i,k,l]:
                    _ref_rslc = ref_rslc[az_idx_, r_idx_]
                    _sec_rslc = sec_rslc[az_idx_, r_idx_]
                    _ref_amp2 += _ref_rslc.real**2 + _ref_rslc.imag**2
                    _sec_amp2 += _sec_rslc.real**2 + _sec_rslc.imag**2
                    _co_nume += _ref_rslc*np.conj(_sec_rslc)
        _inf = _co_nume/math.sqrt(_ref_amp2*_sec_amp2)
        inf[i] = _inf
    return inf

if is_cuda_available():
    @mcuda_jit()
    def _ad_intf_pc_cuda(ref_rslc, sec_rslc, az_idx, r_idx, pc_is_shp, intf):
        # one thread per point: sum over its SHPs of ref conj(sec), normalized by the powers
        i = cuda.grid(1)
        if i >= intf.shape[0]:
            return
        nlines, width = ref_rslc.shape
        az_win, r_win = pc_is_shp.shape[1], pc_is_shp.shape[2]
        nr = np.float32(0.0); ni = np.float32(0.0); pr = np.float32(0.0); ps = np.float32(0.0)
        for k in range(az_win):
            for l in range(r_win):
                a = az_idx[i]-az_win//2+k; r = r_idx[i]-r_win//2+l
                if a >= 0 and a < nlines and r >= 0 and r < width and pc_is_shp[i,k,l]:
                    x = ref_rslc[a,r]; y = sec_rslc[a,r]
                    nr += x.real*y.real+x.imag*y.imag
                    ni += x.imag*y.real-x.real*y.imag
                    pr += x.real*x.real+x.imag*x.imag
                    ps += y.real*y.real+y.imag*y.imag
        d = math.sqrt(pr*ps)
        intf[i] = complex(nr/d, ni/d)

def ad_intf_pc(
    ref_rslc:np.ndarray,
    sec_rslc:np.ndarray,
    idx:np.ndarray,
    pc_is_shp:np.ndarray,
    block_size:int=128,
)-> np.ndarray:
    """Adaptive multilooking interferogram generation for point cloud data.

    Parameters
    ----------
    ref_rslc : np.ndarray
        reference rslc, dtype: `np.complex64`
    sec_rslc : np.ndarray
        secondary rslc, dtype: `np.complex64`
    idx : np.ndarray
        index of point target (azimuth_index, range_index), dtype: `np.int32`, shape: (n_pc, 2)
    pc_is_shp : np.ndarray
        shp bool, dtype: `bool`
    block_size : int, default: 128
        the CUDA block size, it only affects the calculation speed

    Returns
    -------
    np.ndarray
        `coh`, dtype:`np.complex64`, shape(n_pc, n_image_pair)
    """
    xp = get_array_module(ref_rslc)
    if xp is np:
        return _ad_intf_pc_numba(ref_rslc, sec_rslc, idx[:,0], idx[:,1], pc_is_shp)
    else:
        n_pc = idx.shape[0]
        intf = cp.empty(n_pc,dtype=ref_rslc.dtype)
        if n_pc > 0:
            idx = cp.asarray(idx, dtype=cp.int32)
            _ad_intf_pc_cuda[(n_pc+block_size-1)//block_size, block_size](
                ref_rslc, sec_rslc, idx[:,0], idx[:,1], pc_is_shp, intf)
        return intf

def isPD(co:np.ndarray,
         )-> np.ndarray:
    """Parameters
    ----------
    co : np.ndarray
        absolute value of complex coherence/covariance stack

    Returns
    -------
    np.ndarray
        bool array indicating wheather coherence/covariance is positive define
    """
    xp = get_array_module(co)
    if xp is np:
        # numpy's cholesky raises on a non positive definite matrix instead of returning nan like cupy
        return (np.linalg.eigvalsh(co) > 0).all(axis=-1)
    L = xp.linalg.cholesky(co)
    is_PD = xp.isfinite(L).all(axis=(-2,-1))
    return is_PD

'''
    The method is presented in [1]. John D'Errico implented it in MATLAB [2] under BSD
    Licence and [3] implented it with Python/Numpy based on [2] also under BSD Licence.
    This is a cupy implentation with stack of matrix supported.

    [1] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [3] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
'''
def nearestPD(co:np.ndarray,
             )-> np.ndarray:
    """Find the nearest positive-definite matrix to input matrix.

    Parameters
    ----------
    co : np.ndarray
        stack of matrix with shape [...,N,N]

    Returns
    -------
    np.ndarray
        nearest positive definite matrix of input, shape [...,N,N]
    """
    xp = get_array_module(co)
    B = (co + xp.swapaxes(co,-1,-2))/2
    s, V = xp.linalg.svd(co)[1:]
    I = xp.eye(co.shape[-1],dtype=co.dtype)
    S = s[...,None]*I
    del s

    H = xp.matmul(xp.swapaxes(V,-1,-2), xp.matmul(S, V))
    del S, V
    A2 = (B + H) / 2
    del B, H
    A3 = (A2 + xp.swapaxes(A2,-1,-2))/2
    del A2

    if isPD(A3).all():
        return A3

    co_norm = xp.linalg.norm(co,axis=(-2,-1))
    spacing = xp.nextafter(co_norm,co_norm+1.0)-co_norm

    k = 0
    while True:
        is_pd = isPD(A3)
        is_pd_all = is_pd.all()
        if is_pd_all or k>=100:
            break
        k+=1
        mineig = xp.amin(xp.linalg.eigvalsh(A3),axis=-1)
        assert xp.isfinite(mineig).all()
        A3 += (~is_pd[...,None,None] * I) * (-mineig * k**2 + spacing)[...,None,None]
    #print(k)
    return A3

# Speckle correlation (decision 0025). The SLC is first divided by the square root of its local mean power
# (_SLC_CORRELATION_TEXTURE_WIN pixels square) so that bright extended structures (buildings), whose
# neighbouring pixels are much more correlated than speckle, do not dominate. |rho|^2 at lag (da, dr) is
# estimated per azimuth line pair (y, y+da) as |sum_x a conj(b)|^2 / (sum |a|^2 sum |b|^2) - 1/m over the
# m valid pixel pairs of the lines, then averaged over the line pairs: an azimuth phase ramp (TOPS) is
# constant along a line pair, so it does not reduce |rho| as an average over many lines would. 1/m is the
# expectation of the estimate for rho = 0 with independent pixels; the pixels of a line are correlated in
# range, which leaves a bias of about (f_range - 1)/m per lag. It is removed with the mean of the outermost
# lags (no correlation there), which sums up to 4 % of the oversampling otherwise (117 lags).
_SLC_CORRELATION_MIN_PIXELS = 100
_SLC_CORRELATION_TEXTURE_WIN = 15

@ngjit
def _reflect(i, n):
    # index of a half-sample symmetric extension (d c b a | a b c d | d c b a), as scipy.ndimage 'reflect'
    while i < 0 or i >= n:
        i = -i-1 if i < 0 else 2*n-i-1
    return i

@ngpjit
def _normalize_local_power(slc, half):
    # slc / sqrt(local mean power over the valid pixels of a (2*half+1)^2 window, symmetric extension at the
    # borders); 0 for invalid pixels. Window sums from integral images of the extended power and valid mask.
    nl, nw = slc.shape
    pl = nl+2*half; pw = nw+2*half
    ip = np.zeros((pl+1, pw+1), np.float64); iv = np.zeros((pl+1, pw+1), np.float64)
    for y in prange(pl):
        sy = _reflect(y-half, nl)
        sp = 0.0; sv = 0.0
        for c in range(pw):
            v = slc[sy, _reflect(c-half, nw)]
            p = np.float64(v.real)**2+np.float64(v.imag)**2
            if p > 0 and np.isfinite(p):
                sp += p; sv += 1.0
            ip[y+1, c+1] = sp; iv[y+1, c+1] = sv
    for c in prange(1, pw+1):
        for y in range(1, pl+1):
            ip[y, c] += ip[y-1, c]; iv[y, c] += iv[y-1, c]
    k = 2*half+1
    out = np.zeros((nl, nw), np.complex64)
    for y in prange(nl):
        for c in range(nw):
            v = slc[y, c]
            p = np.float64(v.real)**2+np.float64(v.imag)**2
            if p > 0 and np.isfinite(p):
                sp = ip[y+k, c+k]-ip[y, c+k]-ip[y+k, c]+ip[y, c]
                sv = iv[y+k, c+k]-iv[y, c+k]-iv[y+k, c]+iv[y, c]
                lp = sp/max(sv, 1e-12)
                if lp > 0:
                    out[y, c] = v/np.float32(math.sqrt(lp))
    return out

@ngpjit
def _slc_correlation_numba(slc, max_az, max_r):
    nlines, width = slc.shape
    n_az = 2*max_az+1; n_r = 2*max_r+1
    rho2 = np.empty((n_az, n_r), dtype=np.float64)
    for idx in prange(n_az*n_r):
        da = idx//n_r-max_az; dr = idx%n_r-max_r
        acc = 0.0; n_pairs = 0
        for y in range(max(0, -da), min(nlines, nlines-da)):
            num_r = 0.0; num_i = 0.0; pa = 0.0; pb = 0.0; m = 0
            for x in range(max(0, -dr), min(width, width-dr)):
                a = slc[y,x]; b = slc[y+da,x+dr]
                aa = np.float64(a.real)**2+np.float64(a.imag)**2
                bb = np.float64(b.real)**2+np.float64(b.imag)**2
                if aa > 0 and bb > 0 and np.isfinite(aa) and np.isfinite(bb):
                    num_r += np.float64(a.real)*b.real+np.float64(a.imag)*b.imag
                    num_i += np.float64(a.imag)*b.real-np.float64(a.real)*b.imag
                    pa += aa; pb += bb; m += 1
            if m >= _SLC_CORRELATION_MIN_PIXELS:
                acc += (num_r*num_r+num_i*num_i)/(pa*pb)-1.0/m
                n_pairs += 1
        rho2[da+max_az, dr+max_r] = acc/n_pairs if n_pairs > 0 else np.nan
    rho2[max_az, max_r] = 1.0
    return rho2

def _slc_correlation(slc:np.ndarray,
                     max_lag:tuple[int,int]=(4,6),
                    )-> np.ndarray:
    """Squared magnitude of the spatial correlation coefficient of the speckle of an SLC.

    Its sum over all lags is the number of pixels per independent look (the oversampling): an estimate of
    second order statistics (e.g. coherence) from n pixels of a compact area counts as about n / sum
    independent looks.

    Parameters
    ----------
    slc : np.ndarray
        one SLC, dtype complex64, shape (nlines, width), numpy; pixels that are 0 or not finite are
        ignored; not affected by azimuth phase ramps (e.g. TOPS)
    max_lag : tuple[int, int], default: (4, 6)
        largest (azimuth, range) lag in pixels; the correlation must be about 0 at the largest lags
        (they calibrate the noise of the estimate)

    Returns
    -------
    np.ndarray
        `rho2`, dtype float32, shape (2*max_lag[0]+1, 2*max_lag[1]+1): |rho|^2 of the speckle at the lag
        (azimuth, range) = index - max_lag, 1 at the center, the noise bias of the estimate removed
        (values near 0 can be slightly negative); NaN where no azimuth line pair has 100 valid pixel pairs
    """
    max_az, max_r = int(max_lag[0]), int(max_lag[1])
    slc = np.ascontiguousarray(slc, dtype=np.complex64)
    slc = _normalize_local_power(slc, _SLC_CORRELATION_TEXTURE_WIN//2)
    rho2 = _slc_correlation_numba(slc, max_az, max_r)
    outer = np.ones(rho2.shape, dtype=bool); outer[1:-1,1:-1] = False
    noise = rho2[outer][np.isfinite(rho2[outer])]
    if noise.size > 0: rho2 -= noise.mean()
    rho2[max_az, max_r] = 1.0
    return rho2.astype(np.float32)

def _rslc_rho2(rslc, n_images:int=3):
    """|rho|^2 of the speckle of an rslc stack (nlines, width, nimages), numpy or cupy: median of
    `_slc_correlation` of `n_images` evenly spaced images; None if the block is too small to estimate it."""
    nimages = rslc.shape[2]
    tables = []
    for k in np.unique(np.linspace(0, nimages-1, min(n_images, nimages)).round().astype(int)):
        slc = rslc[:,:,k]
        if get_array_module(slc) is not np: slc = slc.get()
        tables.append(_slc_correlation(slc))
    rho2 = np.median(np.stack(tables), axis=0)
    return rho2.astype(np.float32) if np.isfinite(rho2).all() else None

# Effective number of looks of an SHP set S (decision 0025): n^2 / sum_{p,q in S} |rho(p-q)|^2, the number
# of independent looks with the same variance of second order estimates; |rho|^2 from _slc_correlation,
# negative values (noise of the estimate) taken as 0, lags outside the table as 0. On the CPU the sum is
# taken over the lags, sum_d |rho(d)|^2 C(d), with C(d) the number of SHP pairs at lag d counted by
# popcount on the rows of the mask as bit fields (C(-d) = C(d), weighted by |rho(d)|^2 + |rho(-d)|^2);
# about 100 times faster than the sum over the pairs.
@ngjit
def _popcount64(x):
    x = x-((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333))+((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x+(x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x*np.uint64(0x0101010101010101)) >> np.uint64(56)

@ngpjit
def _shp_n_looks_numba(pc_is_shp, rho2, max_az, max_r):
    n_points, az_win, r_win = pc_is_shp.shape
    n_looks = np.empty(n_points, dtype=np.float32)
    for i in prange(n_points):
        rows = np.empty(az_win, dtype=np.uint64)
        n = 0
        for a in range(az_win):
            v = np.uint64(0)
            for r in range(r_win):
                if pc_is_shp[i,a,r]:
                    v |= np.uint64(1) << np.uint64(r)
            rows[a] = v
            n += _popcount64(v)
        if n == 0:
            n_looks[i] = 0.0
            continue
        s = np.float64(n)*max(np.float64(rho2[max_az,max_r]), 0.0)
        for da in range(0, min(max_az, az_win-1)+1):
            for dr in range(-min(max_r, r_win-1), min(max_r, r_win-1)+1):
                if da == 0 and dr <= 0:
                    continue
                w = max(np.float64(rho2[da+max_az,dr+max_r]), 0.0)+max(np.float64(rho2[max_az-da,max_r-dr]), 0.0)
                if w <= 0:
                    continue
                c = np.uint64(0)
                for a in range(az_win-da):
                    y = rows[a+da] >> np.uint64(dr) if dr >= 0 else rows[a+da] << np.uint64(-dr)
                    c += _popcount64(rows[a] & y)
                s += w*np.float64(c)
        n_looks[i] = np.float64(n)*n/s
    return n_looks

if is_cuda_available():
    from numba import cuda

    @mcuda_jit()
    def _shp_n_looks_cuda(pc_is_shp, rho2, max_az, max_r, n_looks):
        # one warp per point: the lanes take the SHPs p of the window in turn, then reduce by shuffles
        i = cuda.grid(1)//32
        if i >= pc_is_shp.shape[0]:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        az_win = pc_is_shp.shape[1]; r_win = pc_is_shp.shape[2]
        win = az_win*r_win
        n = np.float32(0.0); s = np.float32(0.0)
        for p in range(lane, win, 32):
            pa = p//r_win; pr = p%r_win
            if pc_is_shp[i,pa,pr]:
                n += np.float32(1.0)
                for q in range(win):
                    qa = q//r_win; qr = q%r_win
                    if pc_is_shp[i,qa,qr]:
                        da = qa-pa; dr = qr-pr
                        if abs(da) <= max_az and abs(dr) <= max_r:
                            v = rho2[da+max_az,dr+max_r]
                            if v > 0: s += v
        offset = 16
        while offset > 0:
            n += cuda.shfl_down_sync(0xffffffff, n, offset)
            s += cuda.shfl_down_sync(0xffffffff, s, offset)
            offset //= 2
        if lane == 0:
            n_looks[i] = n*n/s if n > 0 else np.float32(0.0)

def _n_looks(pc_is_shp, rho2):
    """Effective number of looks of the SHP sets with the speckle correlation `rho2`, or the number of
    SHPs if `rho2` is None."""
    if rho2 is None:
        return get_array_module(pc_is_shp).count_nonzero(pc_is_shp, axis=(1,2)).astype(np.float32)
    return _shp_n_looks(pc_is_shp, rho2)

def _shp_n_looks(pc_is_shp:np.ndarray,
                 rho2:np.ndarray,
                 block_size:int=128,
                )-> np.ndarray:
    """Effective number of independent looks of the SHP set of each point.

    The SHPs are correlated pixels; their effective number of looks is the number of independent looks
    that gives the same variance of second order estimates such as coherence (it is smaller for compact
    SHP sets than for scattered ones of the same size).

    Parameters
    ----------
    pc_is_shp : np.ndarray
        SHP masks of the points, dtype bool, shape (n_points, az_win, r_win), numpy or cupy
    rho2 : np.ndarray
        |rho|^2 of the speckle at the lag (azimuth, range) = index - (shape - 1) / 2, dtype float32, shape
        (2*max_az+1, 2*max_r+1), e.g. from `_slc_correlation`; lags outside it count as uncorrelated
    block_size : int, default: 128
        the CUDA block size, a multiple of 32, only affects the speed

    Returns
    -------
    np.ndarray
        effective number of looks of each point, dtype float32, shape (n_points,), between 1 and the
        number of SHPs (number of SHPs for uncorrelated pixels), 0 for points without SHP
    """
    xp = get_array_module(pc_is_shp)
    if pc_is_shp.shape[2] > 64:
        raise ValueError('SHP windows wider than 64 pixels in range are not supported')
    max_az, max_r = (rho2.shape[0]-1)//2, (rho2.shape[1]-1)//2
    rho2 = xp.ascontiguousarray(xp.asarray(rho2, dtype=np.float32))
    pc_is_shp = xp.ascontiguousarray(pc_is_shp)
    if xp is np:
        return _shp_n_looks_numba(pc_is_shp, rho2, max_az, max_r)
    n_points = pc_is_shp.shape[0]
    n_looks = cp.empty(n_points, dtype=cp.float32)
    if n_points > 0:
        n_blocks = (n_points*32+block_size-1)//block_size
        _shp_n_looks_cuda[n_blocks, block_size](pc_is_shp, rho2, np.int32(max_az), np.int32(max_r), n_looks)
    return n_looks

def regularize_spectral(coh:np.ndarray,
                        beta:Union[float, np.ndarray],
                        )-> np.ndarray:
    """Spectral regularizer for coherence matrix.

    Parameters
    ----------
    coh : np.ndarray
        stack of matrix with shape [...,N,N]
    beta : Union[float, np.ndarray]
        the regularization parameter, a float number or cupy ndarray with shape [...]

    Returns
    -------
    np.ndarray
        regularized matrix, shape [...,N,N]
    """
    xp = get_array_module(coh)
    I = xp.eye(coh.shape[-1],dtype=coh.dtype)
    beta = xp.asarray(beta)[...,None,None]

    regularized_coh = (1-beta)*coh + beta* I
    return regularized_coh
