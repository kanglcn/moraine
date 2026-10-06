"""Phase linking"""

__all__ = ['emi', 'ds_temp_coh', 'emperical_co_emi_temp_coh_pc']

import math
import numpy as np
import moraine as mr
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
    from numba import cuda
from numba import prange
from .utils_ import ngpjit, ngjit
from .co import _rslc_rho2, _n_looks

# Adaptive regularization of EMI (decision 0023): the coherence matrix coh of a point is replaced by
# (1-beta)*coh + beta*I, i.e. the off-diagonal elements are scaled by 1-beta, with the smallest beta
# (0 <= beta < 1) such that the eigenvalues of |coh| after the regularization satisfy
#   lambda_min >= (1-beta)*delta, delta = max(0, -lambda_min(|coh|)): the true |coh| is positive definite,
#     so delta is a lower bound of the norm of the estimation error of |coh| and (1-beta)*delta the same
#     bound for the regularized matrix; the solution is beta = 2*delta/(1+2*delta);
#   lambda_min >= lambda_max/_EMI_MAX_COND: the float32 inverse of |coh| keeps about two digits.
# Points with a positive definite |coh| whose condition number is at most _EMI_MAX_COND get beta = 0 and
# are not changed. The data and the weights are regularized together, so a coherence matrix whose phases
# close is still solved exactly with quality 1.
_EMI_MAX_COND = 1e5

@ngjit
def _emi_reg_beta_numba(lam_min, lam_max):
    lam_min = np.float64(lam_min); lam_max = np.float64(lam_max)
    delta = max(0.0, -lam_min)
    beta = 2*delta/(1+2*delta)
    beta_cond = (lam_max/_EMI_MAX_COND-lam_min)/((1-lam_min)+(lam_max-1)/_EMI_MAX_COND)
    return max(max(beta, beta_cond), 0.0)

def _emi_reg_beta(lam_min, lam_max):
    """Same as `_emi_reg_beta_numba` for arrays of eigenvalues (numpy or cupy)."""
    xp = get_array_module(lam_min)
    lam_min = lam_min.astype(np.float64); lam_max = lam_max.astype(np.float64)
    delta = xp.maximum(0.0, -lam_min)
    beta = 2*delta/(1+2*delta)
    beta_cond = (lam_max/_EMI_MAX_COND-lam_min)/((1-lam_min)+(lam_max-1)/_EMI_MAX_COND)
    return xp.maximum(xp.maximum(beta, beta_cond), 0.0)

if is_cuda_available():
    def _emi_cp(
        coh,
        n_images,
        image_pairs,
        ref:int=0,
        regularize:bool=False,
    ):
        """Parameters
        ----------
        coh
            complex coherence metrix, dtype, cp.complex64
        n_images
        image_pairs
        ref : int, default: 0
        regularize : bool, default: False
        """
        n_points = coh.shape[0]
        max_batch_size = 2**20
        num_batchs = np.ceil(n_points/max_batch_size).astype(int)
        ph = cp.empty((n_points,n_images),dtype=coh.dtype)
        emi_quality = cp.empty(n_points, dtype=cp.float32)
        diag = cp.arange(n_images)
        for i in range(num_batchs):
            start = i*max_batch_size
            end = (i+1)*max_batch_size
            if end >= n_points: end = n_points
            _coh = mr.uncompress_coh(coh[start:end],image_pairs)
            coh_mag = cp.abs(_coh)
            if regularize:
                lam = cp.linalg.eigvalsh(coh_mag)
                scale = (1-_emi_reg_beta(lam[:,0],lam[:,-1])).astype(cp.float32)
                del lam
                # in place; a factor of exactly 1 (beta = 0) leaves the point unchanged
                _coh *= scale[:,None,None]
                _coh[:,diag,diag] = 1
                cp.abs(_coh,out=coh_mag)
            coh_mag_inv = cp.linalg.inv(coh_mag)
            min_eigval, min_eig = cp.linalg.eigh(coh_mag_inv*_coh)
            min_eigval = min_eigval[...,0]
            min_eig = min_eig[...,0]*min_eig[...,[ref],0].conj()
            ph[start:end] = min_eig/abs(min_eig)
            emi_quality[start:end] = min_eigval
        return ph, emi_quality

@ngpjit
def _emi_numba(
    coh,
    n_images,
    image_pairs,
    ref:int=0,
    regularize:bool=False,
):
    n_points = coh.shape[0]
    ph = np.empty((n_points,n_images),dtype=coh.dtype)
    emi_quality = np.empty(n_points,dtype=np.float32)
    for i in prange(n_points):
        _coh = mr.uncompress_single_coh_numba(coh[i],n_images,image_pairs)
        coh_mag = np.abs(_coh)
        if regularize:
            lam = np.linalg.eigvalsh(coh_mag)
            beta = _emi_reg_beta_numba(lam[0],lam[-1])
            if beta > 0:
                scale = np.float32(1-beta)
                for j in range(n_images):
                    for k in range(n_images):
                        if j != k:
                            _coh[j,k] = _coh[j,k]*scale
                coh_mag = np.abs(_coh)
        coh_mag_inv = np.linalg.inv(coh_mag)
        # the inverse of the symmetric coh_mag is symmetric; the float32 LU inverse is not exactly, and the
        # EMI quality amplifies the difference by the condition number (0.1 at 2e4)
        coh_mag_inv = (coh_mag_inv+coh_mag_inv.T)*np.float32(0.5)
        min_eigval, min_eig = np.linalg.eigh(coh_mag_inv*_coh)
        min_eigval = min_eigval[0]
        min_eig = min_eig[:,0]*np.conj(min_eig[ref,0])
        for j in range(ph.shape[-1]):
            ph[i,j] = min_eig[j]/abs(min_eig[j])
        emi_quality[i] = min_eigval
    return ph, emi_quality

def _emi(coh, ref=0, regularize=True):
    """`emi` that also returns the EMI quality (minimum eigenvalue, float32, (n_points,)), for tests."""
    xp = get_array_module(coh)
    nimages = mr.nimage_from_npair(coh.shape[-1])
    image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    if xp is np:
        return _emi_numba(coh,nimages,image_pairs,ref,regularize)
    else:
        return _emi_cp(coh,nimages,image_pairs,ref,regularize)

def emi(coh:np.ndarray,
        ref:int=0,
        regularize:bool=True,
       )-> np.ndarray:
    """Phase linking with the EMI estimator.

    Parameters
    ----------
    coh : np.ndarray
        complex coherence of the points (upper triangle of the coherence matrix, all image pairs),
        dtype complex64, shape (n_points, n_image_pairs), numpy or cupy
    ref : int, default: 0
        index of the reference image, its phase is set to 0
    regularize : bool, default: True
        regularize the coherence matrix of the points whose coherence magnitude matrix is not positive
        definite or numerically singular (e.g. when the number of images approaches the number of
        independent looks of the SHPs; without it their phase history is not reliable); points with a
        well conditioned positive definite matrix are not changed

    Returns
    -------
    np.ndarray
        phase history `ph`, dtype complex64, shape (n_points, nimages), unit amplitude, the phase of
        image `ref` is 0
    """
    return _emi(coh,ref,regularize)[0]

# DS temporal coherence t = |sum_k exp(j psi_k)| / n_pairs over the image pairs k, with the residual phase
# psi_k of the phase history. Weighted (decision 0024): t_w = |sum_k w_k exp(j psi_k)| / sum_k w_k and
# eff_n_pairs = (sum w)^2 / sum w^2, with w_k = max(0, (|coh_k|^2 - 1/n_looks) / (1 - 1/n_looks)), the squared
# coherence without the noise bias E|coh|^2 = 1/n_looks of incoherent pairs (pairs at the noise level get 0);
# NaN and 0 where n_looks <= 1 or no pair has a positive weight. One kernel per device computes both
# (1/n_looks = 1 skips the weights), so t is the same bit for bit with and without n_looks; two kernels do not
# guarantee that on the GPU (the compiler places fused multiply-adds differently).
@ngpjit
def _ds_temp_coh_numba(coh, ph, ref, sec, inv_n):
    n_points, n_pairs = coh.shape
    t_coh = np.empty(n_points, dtype=np.float32)
    t_coh_w = np.empty(n_points, dtype=np.float32)
    eff_n_pairs = np.empty(n_points, dtype=np.float32)
    zero = np.float32(0.0); one = np.float32(1.0)
    for i in prange(n_points):
        inv = inv_n[i]
        weighted = inv < one
        denom = one-inv
        acc = np.complex64(0.0)
        sr = zero; si = zero; sw = zero; sw2 = zero
        for k in range(n_pairs):
            c = coh[i,k]
            e = c*(np.conjugate(ph[i,ref[k]])*ph[i,sec[k]])/np.abs(c)
            acc += e
            if weighted:
                w = (c.real*c.real+c.imag*c.imag-inv)/denom
                if w > zero:
                    sr += w*e.real
                    si += w*e.imag
                    sw += w
                    sw2 += w*w
        t_coh[i] = np.abs(acc)/n_pairs
        if sw > zero:
            t_coh_w[i] = np.sqrt(sr*sr+si*si)/sw
            eff_n_pairs[i] = sw*sw/sw2
        else:
            t_coh_w[i] = np.nan
            eff_n_pairs[i] = zero
    return t_coh, t_coh_w, eff_n_pairs

if is_cuda_available():
    @cuda.jit
    def _ds_temp_coh_cuda(coh, ph, ref, sec, inv_n, t_coh, t_coh_w, eff_n_pairs):
        # one warp per point: the 32 lanes read consecutive image pairs (coalesced), then reduce by shuffles
        i = cuda.grid(1)//32
        if i >= coh.shape[0]:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        zero = np.float32(0.0); one = np.float32(1.0)
        inv = inv_n[i]
        weighted = inv < one
        denom = one-inv
        ur = zero; ui = zero; sr = zero; si = zero; sw = zero; sw2 = zero
        for k in range(lane, coh.shape[1], 32):
            c = coh[i,k]
            mag2 = c.real*c.real+c.imag*c.imag
            p = c*ph[i,ref[k]].conjugate()*ph[i,sec[k]]
            s = one/math.sqrt(mag2)
            ur += s*p.real
            ui += s*p.imag
            if weighted:
                w = (mag2-inv)/denom
                if w > zero:
                    sr += w*(s*p.real)
                    si += w*(s*p.imag)
                    sw += w
                    sw2 += w*w
        offset = 16
        while offset > 0:
            ur += cuda.shfl_down_sync(0xffffffff, ur, offset)
            ui += cuda.shfl_down_sync(0xffffffff, ui, offset)
            sr += cuda.shfl_down_sync(0xffffffff, sr, offset)
            si += cuda.shfl_down_sync(0xffffffff, si, offset)
            sw += cuda.shfl_down_sync(0xffffffff, sw, offset)
            sw2 += cuda.shfl_down_sync(0xffffffff, sw2, offset)
            offset //= 2
        if lane == 0:
            t_coh[i] = math.sqrt(ur*ur+ui*ui)/coh.shape[1]
            if sw > zero:
                t_coh_w[i] = math.sqrt(sr*sr+si*si)/sw
                eff_n_pairs[i] = sw*sw/sw2
            else:
                t_coh_w[i] = np.nan
                eff_n_pairs[i] = zero

def ds_temp_coh(coh:np.ndarray,
                ph:np.ndarray,
                image_pairs:np.ndarray=None,
                block_size:int=128,
                n_looks=None,
            ):
    """DS temporal coherence, optionally also weighted by the coherence of the image pairs.

    Parameters
    ----------
    coh : np.ndarray
        complex coherence of the points as estimated (upper triangle of the coherence matrix), dtype
        complex64, shape (n_points, n_image_pairs), numpy or cupy
    ph : np.ndarray
        phase history of the points, dtype complex64, shape (n_points, nimages), unit amplitude
    image_pairs : np.ndarray, optional
        image pairs of `coh`, dtype int32, shape (n_image_pairs, 2), all image pairs by default
    block_size : int, default: 128
        the CUDA block size, a multiple of 32, only applied for cuda
    n_looks : float or np.ndarray, optional
        effective number of independent looks of the coherence of each point, float or shape (n_points,),
        e.g. from `emperical_co_pc(..., return_n_looks=True)`; with it also the weighted temporal coherence
        and the effective number of image pairs are returned; points with n_looks <= 1 get NaN and 0

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        temporal coherence `t_coh`, dtype float32, shape (n_points,), 0..1, every image pair weighted the
        same; with `n_looks` also the weighted temporal coherence `t_coh_w`, dtype float32, shape
        (n_points,), 0..1: the image pairs weighted by their squared coherence without the noise bias of
        `n_looks` looks, so that incoherent pairs (e.g. long time spans in vegetation) do not lower it, NaN
        where no image pair is above the noise level; and the effective number of image pairs it rests on
        `eff_n_pairs` ((sum of the weights)^2 / sum of the squared weights), dtype float32, shape (n_points,),
        0..n_image_pairs, 0 where `t_coh_w` is NaN; a high `t_coh_w` of few effective pairs is not reliable
    """
    xp = get_array_module(coh)
    n_points = ph.shape[0]
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(ph.shape[-1]).image_pairs
    image_pairs = xp.asarray(image_pairs, dtype=np.int32)
    ref = xp.ascontiguousarray(image_pairs[:,0]); sec = xp.ascontiguousarray(image_pairs[:,1])
    inv_n = xp.ones((), dtype=np.float32) if n_looks is None else 1/xp.asarray(n_looks, dtype=np.float32)
    inv_n = xp.ascontiguousarray(xp.broadcast_to(inv_n, (n_points,)))
    coh = xp.ascontiguousarray(coh); ph = xp.ascontiguousarray(ph)
    if xp is np:
        out = _ds_temp_coh_numba(coh, ph, ref, sec, inv_n)
    else:
        out = tuple(cp.empty(n_points, dtype=cp.float32) for _ in range(3))
        if n_points > 0:
            n_blocks = (n_points*32+block_size-1)//block_size
            _ds_temp_coh_cuda[n_blocks, block_size](coh, ph, ref, sec, inv_n, *out)
    return out[0] if n_looks is None else out

def emperical_co_emi_temp_coh_pc(
    rslc:np.ndarray,
    idx:np.ndarray,
    pc_is_shp:np.ndarray,
    batch_size:int=1000,
    regularize:bool=True,
    weighted:bool=False,
):
    """Coherence matrix estimation, EMI phase linking and DS temporal coherence of points in one pass,
    without keeping the coherence matrices.

    Parameters
    ----------
    rslc : np.ndarray
        rslc stack, dtype complex64, shape (nlines, width, nimages), numpy or cupy
    idx : np.ndarray
        grid index of the points (azimuth, range) in `rslc`, dtype int32, shape (n_points, 2)
    pc_is_shp : np.ndarray
        SHP masks of the points, dtype bool, shape (n_points, az_win, r_win)
    batch_size : int, default: 1000
        number of points processed at once, limits the memory use
    regularize : bool, default: True
        regularize the coherence matrix in the phase linking as in `emi`; the temporal coherence is
        computed with the coherence matrix as estimated
    weighted : bool, default: False
        also return the weighted temporal coherence and the effective number of image pairs of
        `ds_temp_coh`, with the effective number of looks of the SHPs of each point (as
        `emperical_co_pc(..., return_n_looks=True)`)

    Returns
    -------
    tuple
        phase history `ph`, dtype complex64, shape (n_points, nimages), and temporal coherence `t_coh`,
        dtype float32, shape (n_points,); with `weighted` also `t_coh_w` and `eff_n_pairs` of
        `ds_temp_coh`, dtype float32, shape (n_points,) each
    """
    xp = get_array_module(rslc)
    n_pc = idx.shape[0]
    nimages = rslc.shape[2]
    ph = xp.empty((n_pc, nimages),dtype=xp.complex64)
    t_coh = xp.empty(n_pc,dtype=np.float32)
    if weighted:
        t_coh_w = xp.empty(n_pc,dtype=np.float32)
        eff_n_pairs = xp.empty(n_pc,dtype=np.float32)
        rho2 = _rslc_rho2(rslc)
    batch_bounds = np.arange(0,n_pc+batch_size,batch_size)
    # I forgot why I have to split data into batches, probably due to memory issue.
    if batch_bounds[-1]>n_pc: batch_bounds[-1]=n_pc
    for i in range(batch_bounds.shape[0]-1):
        start = batch_bounds[i]; stop = batch_bounds[i+1]
        _coh = mr.emperical_co_pc(rslc,idx[start:stop],pc_is_shp[start:stop])
        ph[start:stop] = emi(_coh,regularize=regularize)
        if weighted:
            n_looks = _n_looks(pc_is_shp[start:stop],rho2)
            t_coh[start:stop],t_coh_w[start:stop],eff_n_pairs[start:stop] = ds_temp_coh(_coh,ph[start:stop],n_looks=n_looks)
        else:
            t_coh[start:stop] = ds_temp_coh(_coh,ph[start:stop])
    if weighted:
        return ph, t_coh, t_coh_w, eff_n_pairs
    return ph, t_coh
