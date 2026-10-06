"""Phase linking"""

__all__ = ['emi', 'ds_temp_coh', 'emperical_co_emi_temp_coh_pc']

import contextlib
import llvmlite.binding as _ll
import functools
import math
import threading
import numpy as np
from numba import types as _nbtypes
from numba.extending import get_cython_function_address
from threadpoolctl import ThreadpoolController
import moraine as mr
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
    from numba import cuda
import numba
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

# The CPU kernels call LAPACK inside numba parallel loops. A multithreaded BLAS (MKL, OpenBLAS) does not see
# that it is called from many threads and starts its own threads in each call: with 128 cores 128 x 128
# threads, about 2e4 times slower. Its threads are limited to 1 while a kernel runs; the limit is global, so
# it is kept until the last of concurrent callers (e.g. dask threads) has finished.
_blas_lock = threading.Lock()
_blas_users = 0
_blas_limit = None

class _single_thread_blas:
    def __enter__(self):
        global _blas_users, _blas_limit
        with _blas_lock:
            if _blas_users == 0:
                _blas_limit = _blas_controller.limit(limits=1, user_api='blas')
            _blas_users += 1
    def __exit__(self, *exc):
        global _blas_users, _blas_limit
        with _blas_lock:
            _blas_users -= 1
            if _blas_users == 0:
                _blas_limit.restore_original_limits(); _blas_limit = None

# Only the smallest eigenpair of the EMI matrix is needed: LAPACK cheevr with RANGE='I', IL=IU=1
# (tridiagonalization, then one eigenvector by MRRR and its back transformation) instead of all of them
# (2.2 times faster for 60-92 images). It is the LAPACK numba's np.linalg calls, through scipy. A numba
# array in C order is the Fortran order of its transpose, the conjugate of a Hermitian matrix, so the
# eigenvector comes out conjugated. It is called by symbol name, not through a ctypes pointer, so that the kernels
# can be cached (decision 0029).
_ll.add_symbol('moraine_cheevr', get_cython_function_address('scipy.linalg.cython_lapack', 'cheevr'))
_cheevr = _nbtypes.ExternalFunction('moraine_cheevr', _nbtypes.void(*([_nbtypes.voidptr]*23)))
# after loading scipy's LAPACK (pip wheels of numpy and scipy bring separate BLAS libraries; numba calls scipy's)
_blas_controller = ThreadpoolController()

@ngjit
def _cheevr_call(a, n_images, w, z, work, lwork, rwork, lrwork, iwork, liwork, isuppz, m, info):
    jobz = np.array([ord('V')], np.int8); rng = np.array([ord('I')], np.int8); uplo = np.array([ord('L')], np.int8)
    n = np.array([n_images], np.int32); one = np.array([1], np.int32)
    vl = np.zeros(1, np.float32); vu = np.zeros(1, np.float32)
    abstol = np.array([np.finfo(np.float32).tiny], np.float32)    # highest accuracy with MRRR
    lw = np.array([lwork], np.int32); lrw = np.array([lrwork], np.int32); liw = np.array([liwork], np.int32)
    _cheevr(jobz.ctypes, rng.ctypes, uplo.ctypes, n.ctypes, a.ctypes, n.ctypes, vl.ctypes, vu.ctypes, one.ctypes,
            one.ctypes, abstol.ctypes, m.ctypes, w.ctypes, z.ctypes, n.ctypes, isuppz.ctypes, work.ctypes, lw.ctypes,
            rwork.ctypes, lrw.ctypes, iwork.ctypes, liw.ctypes, info.ctypes)

@ngjit
def _cheevr_workspace(n_images):
    """Optimal (lwork, lrwork, liwork) of cheevr for n_images x n_images."""
    a = np.zeros((n_images, n_images), np.complex64)
    w = np.empty(n_images, np.float32); z = np.empty((1, n_images), np.complex64)
    work = np.empty(1, np.complex64); rwork = np.empty(1, np.float32); iwork = np.empty(1, np.int32)
    isuppz = np.empty(2, np.int32); m = np.zeros(1, np.int32); info = np.zeros(1, np.int32)
    _cheevr_call(a, n_images, w, z, work, -1, rwork, -1, iwork, -1, isuppz, m, info)
    return (max(int(work[0].real), 2*n_images), max(int(rwork[0]), 24*n_images), max(int(iwork[0]), 10*n_images))

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
    lwork, lrwork, liwork = _cheevr_workspace(n_images)
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
        a = coh_mag_inv*_coh
        w = np.empty(n_images, np.float32); z = np.empty((1, n_images), np.complex64)
        work = np.empty(lwork, np.complex64); rwork = np.empty(lrwork, np.float32)
        iwork = np.empty(liwork, np.int32); isuppz = np.empty(2, np.int32)
        m = np.zeros(1, np.int32); info = np.zeros(1, np.int32)
        _cheevr_call(a, n_images, w, z, work, lwork, rwork, lrwork, iwork, liwork, isuppz, m, info)
        if info[0] != 0 or m[0] != 1:
            ph[i] = np.nan; emi_quality[i] = np.nan
            continue
        min_eig = np.conj(z[0])
        min_eig = min_eig*np.conj(min_eig[ref])
        for j in range(ph.shape[-1]):
            ph[i,j] = min_eig[j]/abs(min_eig[j])
        emi_quality[i] = w[0]
    return ph, emi_quality

def _emi(coh, ref=0, regularize=True):
    """`emi` that also returns the EMI quality (minimum eigenvalue, float32, (n_points,)), for tests."""
    xp = get_array_module(coh)
    nimages = mr.nimage_from_npair(coh.shape[-1])
    image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    if xp is np:
        with _single_thread_blas():
            return _emi_numba(coh,nimages,image_pairs,ref,regularize)
    else:
        # decision 0031: own kernel for the regularized EMI; cupy without regularization (no pivoting in the kernel)
        # and for more images than its shared memory holds
        if regularize:
            from .emi_cuda_ import emi_cuda, emi_cuda_supported
            if emi_cuda_supported(nimages):
                return emi_cuda(coh,nimages,ref)
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

# Connectivity of the image pairs (decision 0028). The image pairs of a point whose squared coherence is above
# the noise level are the edges of a graph on the images; the phase of images in different connected components
# is not constrained by any coherent pair, whatever the weighted temporal coherence is. For a point without any
# coherent pair (|coh|^2 ~ Beta(1, n_looks-1), P(|coh|^2 > x) = (1-x)^(n_looks-1)) a pair is above the threshold
# x = 1 - p^(1/(n_looks-1)) with probability p, and the graph is a random graph G(n_images, p) (the edges of the
# simulated speckle noise are not independent, but the probability that such a graph is connected agrees with
# G(n, p) to within the sampling error of 4000 trials at n_images = 17 and 92). p is chosen so that the noise
# graph is connected with probability alpha. With p = P(|coh|^2 > 1/n_looks) = 0.37 (every pair with a positive
# weight of decision 0024) all points, noise included, are connected.
_MAX_SHARED_BYTES = 48*1024    # shared memory per block available to every CUDA device (4 bytes per image and warp)

@functools.lru_cache(maxsize=None)
def _edge_pfa(n_images, alpha):
    """Probability p of an edge so that the random graph G(n_images, p) is connected with probability alpha.

    Gilbert's recursion C_m = 1 - sum_{k<m} binom(m-1, k-1) C_k (1-p)^(k (m-k)) in log space, solved for p
    by bisection on log p (C_n increases with p). Accurate to about 1e-6 relative for alpha >= 1e-6; below
    that 1 - sum loses all digits.
    """
    n = int(n_images)
    if n < 2: return 1.0
    lf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n+1)))))    # log k!
    def p_connected(p):
        logq = math.log1p(-p)
        lc = np.zeros(n+1)                                                # log C_k, C_1 = 1
        for m in range(2, n+1):
            k = np.arange(1, m)
            t = lf[m-1]-lf[k-1]-lf[m-k]+lc[k]+k*(m-k)*logq
            mx = t.max()
            ls = mx+math.log(np.exp(t-mx).sum())
            lc[m] = math.log(max(-math.expm1(min(ls, 0.0)), 1e-300))
        return lc[n]
    lo, hi = math.log(1e-9), math.log(1-1e-9)
    la = math.log(alpha)
    for _ in range(45):
        mid = 0.5*(lo+hi)
        if p_connected(math.exp(mid)) < la: lo = mid
        else: hi = mid
    return math.exp(0.5*(lo+hi))

# DS temporal coherence t = |sum_k exp(j psi_k)| / n_pairs over the image pairs k, with the residual phase
# psi_k of the phase history. Weighted (decision 0024): t_w = |sum_k w_k exp(j psi_k)| / sum_k w_k and
# eff_n_pairs = (sum w)^2 / sum w^2, with w_k = max(0, (|coh_k|^2 - 1/n_looks) / (1 - 1/n_looks)), the squared
# coherence without the noise bias E|coh|^2 = 1/n_looks of incoherent pairs (pairs at the noise level get 0);
# NaN and 0 where n_looks <= 1 or no pair has a positive weight. n_components is the number of connected
# components of the graph of the pairs with |coh_k|^2 > 1 - p^(1/(n_looks-1)) (see _edge_pfa); n_images where
# n_looks <= 1. One kernel per device computes everything (1/n_looks = 1 skips the weights and the graph), so t
# is the same bit for bit with and without n_looks; two kernels do not guarantee that on the GPU (the compiler
# places fused multiply-adds differently).
@ngpjit
def _ds_temp_coh_numba(coh, ph, ref, sec, inv_n, log_p):
    # real float32 arithmetic: complex division and np.abs (hypot) were 2-3 times slower
    n_points, n_pairs = coh.shape
    n_images = ph.shape[1]
    t_coh = np.empty(n_points, dtype=np.float32)
    t_coh_w = np.empty(n_points, dtype=np.float32)
    eff_n_pairs = np.empty(n_points, dtype=np.float32)
    n_components = np.empty(n_points, dtype=np.int16)
    zero = np.float32(0.0); one = np.float32(1.0)
    for i in prange(n_points):
        inv = inv_n[i]
        weighted = inv < one
        denom = one-inv
        thr = zero
        parent = np.empty(n_images if weighted else 0, dtype=np.int32)    # union-find forest of the images
        if weighted:
            thr = -np.expm1(log_p*inv/denom)     # 1/(n_looks-1) = inv/(1-inv)
            for x in range(n_images):
                parent[x] = x
        ur = zero; ui = zero; sr = zero; si = zero; sw = zero; sw2 = zero
        for k in range(n_pairs):
            c = coh[i,k]; a = ph[i,ref[k]]; b = ph[i,sec[k]]
            cr = c.real; ci = c.imag
            pr = a.real*b.real+a.imag*b.imag; pi = a.real*b.imag-a.imag*b.real    # conj(a)*b
            mag2 = cr*cr+ci*ci
            s = one/np.sqrt(mag2)
            er = (cr*pr-ci*pi)*s; ei = (cr*pi+ci*pr)*s
            ur += er; ui += ei
            if weighted:
                w = (mag2-inv)/denom
                if w > zero:
                    sr += w*er
                    si += w*ei
                    sw += w
                    sw2 += w*w
                if mag2 > thr:
                    ra = ref[k]; rb = sec[k]
                    while parent[ra] != ra:
                        parent[ra] = parent[parent[ra]]
                        ra = parent[ra]
                    while parent[rb] != rb:
                        parent[rb] = parent[parent[rb]]
                        rb = parent[rb]
                    if ra < rb:
                        parent[rb] = ra
                    elif rb < ra:
                        parent[ra] = rb
        t_coh[i] = np.sqrt(ur*ur+ui*ui)/n_pairs
        if sw > zero:
            t_coh_w[i] = np.sqrt(sr*sr+si*si)/sw
            eff_n_pairs[i] = sw*sw/sw2
        else:
            t_coh_w[i] = np.nan
            eff_n_pairs[i] = zero
        if weighted:
            n_root = 0
            for x in range(n_images):
                if parent[x] == x:
                    n_root += 1
            n_components[i] = n_root
        else:
            n_components[i] = n_images
    return t_coh, t_coh_w, eff_n_pairs, n_components

if is_cuda_available():
    @cuda.jit(device=True)
    def _uf_find(parent, base, x):
        # path halving; parent[x] <= x always, so a stale read still points to an ancestor
        p = parent[base+x]
        while p != x:
            pp = parent[base+p]
            parent[base+x] = pp
            x = pp
            p = parent[base+x]
        return x

    @cuda.jit
    def _ds_temp_coh_cuda(coh, ph, ref, sec, inv_n, log_p, t_coh, t_coh_w, eff_n_pairs, n_components):
        # one warp per point: the 32 lanes read consecutive image pairs (coalesced), then reduce by shuffles.
        # The graph of the images is a union-find forest in shared memory, one per warp; the lanes join the
        # roots of the two images of a pair with atomic compare-and-swap (the larger root below the smaller).
        parent = cuda.shared.array(0, dtype=numba.int32)     # dynamic: (block_size // 32) * n_images
        i = cuda.grid(1)//32
        if i >= coh.shape[0]:   # the same for all lanes of a warp
            return
        lane = cuda.laneid
        n_images = ph.shape[1]
        base = (cuda.threadIdx.x//32)*n_images
        zero = np.float32(0.0); one = np.float32(1.0)
        inv = inv_n[i]
        weighted = inv < one
        denom = one-inv
        thr = zero
        if weighted:
            thr = -math.expm1(log_p*inv/denom)
            for x in range(lane, n_images, 32):
                parent[base+x] = x
            cuda.syncwarp()
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
                if mag2 > thr:
                    ra = _uf_find(parent, base, ref[k])
                    rb = _uf_find(parent, base, sec[k])
                    while ra != rb:
                        if ra < rb:
                            ra, rb = rb, ra
                        old = cuda.atomic.cas(parent, base+ra, ra, rb)   # ra > rb: join ra below rb
                        if old == ra:
                            break
                        ra = _uf_find(parent, base, ra)
                        rb = _uf_find(parent, base, rb)
        n_root = np.int32(0)
        if weighted:
            cuda.syncwarp()
            for x in range(lane, n_images, 32):
                if parent[base+x] == x:
                    n_root += 1
        offset = 16
        while offset > 0:
            ur += cuda.shfl_down_sync(0xffffffff, ur, offset)
            ui += cuda.shfl_down_sync(0xffffffff, ui, offset)
            sr += cuda.shfl_down_sync(0xffffffff, sr, offset)
            si += cuda.shfl_down_sync(0xffffffff, si, offset)
            sw += cuda.shfl_down_sync(0xffffffff, sw, offset)
            sw2 += cuda.shfl_down_sync(0xffffffff, sw2, offset)
            n_root += cuda.shfl_down_sync(0xffffffff, n_root, offset)
            offset //= 2
        if lane == 0:
            t_coh[i] = math.sqrt(ur*ur+ui*ui)/coh.shape[1]
            if sw > zero:
                t_coh_w[i] = math.sqrt(sr*sr+si*si)/sw
                eff_n_pairs[i] = sw*sw/sw2
            else:
                t_coh_w[i] = np.nan
                eff_n_pairs[i] = zero
            n_components[i] = n_root if weighted else n_images

def ds_temp_coh(coh:np.ndarray,
                ph:np.ndarray,
                image_pairs:np.ndarray=None,
                block_size:int=128,
                n_looks=None,
                alpha:float=1e-3,
            ):
    """DS temporal coherence, optionally also weighted by the coherence of the image pairs.

    Parameters
    ----------
    coh : np.ndarray
        complex coherence of the points as estimated (upper triangle of the coherence matrix), dtype
        complex64, shape (n_points, n_image_pairs), numpy or cupy
    ph : np.ndarray
        phase history of the points, dtype complex64, shape (n_points, nimages), unit amplitude; with `n_looks`
        on a GPU `block_size` x nimages is at most 393216 (3072 images at the default block size)
    image_pairs : np.ndarray, optional
        image pairs of `coh`, dtype int32, shape (n_image_pairs, 2), all image pairs by default
    block_size : int, default: 128
        the CUDA block size, a multiple of 32, only applied for cuda
    n_looks : float or np.ndarray, optional
        effective number of independent looks of the coherence of each point, float or shape (n_points,),
        e.g. from `emperical_co_pc(..., return_n_looks=True)`; with it also the weighted temporal coherence,
        the effective number of image pairs and the number of connected components are returned; points with
        n_looks <= 1 get NaN, 0 and nimages
    alpha : float, default: 1e-3
        only with `n_looks`; probability that a point without any coherent image pair (pure noise) has all
        images connected (`n_components` 1); it sets the squared coherence above which an image pair counts
        as coherent, depending on `n_looks` and the number of images; 1e-6 <= alpha <= 0.5

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        temporal coherence `t_coh`, dtype float32, shape (n_points,), 0..1, every image pair weighted the
        same; with `n_looks` also the weighted temporal coherence `t_coh_w`, dtype float32, shape
        (n_points,), 0..1: the image pairs weighted by their squared coherence without the noise bias of
        `n_looks` looks, so that incoherent pairs (e.g. long time spans in vegetation) do not lower it, NaN
        where no image pair is above the noise level; the effective number of image pairs it rests on
        `eff_n_pairs` ((sum of the weights)^2 / sum of the squared weights), dtype float32, shape (n_points,),
        0..n_image_pairs, 0 where `t_coh_w` is NaN; and `n_components`, dtype int16, shape (n_points,),
        1..nimages: the number of connected components of the graph whose nodes are the images and whose
        edges are the image pairs that are coherent (see `alpha`); 1 means every image is linked to every
        other by coherent pairs, with more the phase between the groups of images is not constrained by
        any coherent pair, whatever `t_coh_w` is
    """
    xp = get_array_module(coh)
    n_points, n_images = ph.shape
    if n_looks is not None:
        if not 1e-6 <= alpha <= 0.5:
            raise ValueError(f'alpha must be between 1e-6 and 0.5, not {alpha}')
        log_p = np.float32(math.log(_edge_pfa(n_images, float(alpha))))
    else:
        log_p = np.float32(0.0)
    shared_bytes = 0
    if xp is not np and n_looks is not None:
        shared_bytes = block_size//32*n_images*4
        if shared_bytes > _MAX_SHARED_BYTES:
            raise ValueError(f'block_size x nimages must be at most {_MAX_SHARED_BYTES*8} on a GPU, not {block_size*n_images}')
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(ph.shape[-1]).image_pairs
    image_pairs = xp.asarray(image_pairs, dtype=np.int32)
    ref = xp.ascontiguousarray(image_pairs[:,0]); sec = xp.ascontiguousarray(image_pairs[:,1])
    inv_n = xp.ones((), dtype=np.float32) if n_looks is None else 1/xp.asarray(n_looks, dtype=np.float32)
    inv_n = xp.ascontiguousarray(xp.broadcast_to(inv_n, (n_points,)))
    coh = xp.ascontiguousarray(coh); ph = xp.ascontiguousarray(ph)
    if xp is np:
        out = _ds_temp_coh_numba(coh, ph, ref, sec, inv_n, log_p)
    else:
        out = tuple(cp.empty(n_points, dtype=cp.float32) for _ in range(3)) + (cp.empty(n_points, dtype=cp.int16),)
        if n_points > 0:
            n_blocks = (n_points*32+block_size-1)//block_size
            _ds_temp_coh_cuda[n_blocks, block_size, 0, shared_bytes](coh, ph, ref, sec, inv_n, log_p, *out)
    return out[0] if n_looks is None else out

def emperical_co_emi_temp_coh_pc(
    rslc:np.ndarray,
    idx:np.ndarray,
    pc_is_shp:np.ndarray,
    batch_size:int=1000,
    regularize:bool=True,
    weighted:bool=False,
    alpha:float=1e-3,
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
        also return the weighted temporal coherence, the effective number of image pairs and the number of
        connected components of `ds_temp_coh`, with the effective number of looks of the SHPs of each point
        (as `emperical_co_pc(..., return_n_looks=True)`)
    alpha : float, default: 1e-3
        `alpha` of `ds_temp_coh`, only with `weighted`

    Returns
    -------
    tuple
        phase history `ph`, dtype complex64, shape (n_points, nimages), and temporal coherence `t_coh`,
        dtype float32, shape (n_points,); with `weighted` also `t_coh_w` and `eff_n_pairs` of
        `ds_temp_coh`, dtype float32, shape (n_points,) each, and `n_components`, dtype int16, shape
        (n_points,)
    """
    xp = get_array_module(rslc)
    n_pc = idx.shape[0]
    nimages = rslc.shape[2]
    ph = xp.empty((n_pc, nimages),dtype=xp.complex64)
    t_coh = xp.empty(n_pc,dtype=np.float32)
    if weighted:
        t_coh_w = xp.empty(n_pc,dtype=np.float32)
        eff_n_pairs = xp.empty(n_pc,dtype=np.float32)
        n_components = xp.empty(n_pc,dtype=np.int16)
        rho2 = _rslc_rho2(rslc)
    batch_bounds = np.arange(0,n_pc+batch_size,batch_size)
    # I forgot why I have to split data into batches, probably due to memory issue.
    if batch_bounds[-1]>n_pc: batch_bounds[-1]=n_pc
    # one BLAS thread limit for all batches: setting it costs a few ms, about one batch of emi on the CPU
    with (_single_thread_blas() if xp is np else contextlib.nullcontext()):
        for i in range(batch_bounds.shape[0]-1):
            start = batch_bounds[i]; stop = batch_bounds[i+1]
            _coh = mr.emperical_co_pc(rslc,idx[start:stop],pc_is_shp[start:stop])
            ph[start:stop] = emi(_coh,regularize=regularize)
            if weighted:
                n_looks = _n_looks(pc_is_shp[start:stop],rho2)
                (t_coh[start:stop],t_coh_w[start:stop],eff_n_pairs[start:stop],
                 n_components[start:stop]) = ds_temp_coh(_coh,ph[start:stop],n_looks=n_looks,alpha=alpha)
            else:
                t_coh[start:stop] = ds_temp_coh(_coh,ph[start:stop])
    if weighted:
        return ph, t_coh, t_coh_w, eff_n_pairs, n_components
    return ph, t_coh
