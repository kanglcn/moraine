"""Phase linking"""

__all__ = ['emi', 'ds_temp_coh', 'emperical_co_emi_temp_coh_pc']

import numpy as np
import moraine as mr
from .utils_ import is_cuda_available, get_array_module
if is_cuda_available():
    import cupy as cp
from numba import prange
from .utils_ import ngpjit, ngjit

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

def emi(coh:np.ndarray,
        ref:int=0,
        regularize:bool=False,
       )-> tuple[np.ndarray,np.ndarray]:
    """Phase linking with the EMI estimator.

    Parameters
    ----------
    coh : np.ndarray
        complex coherence of the points (upper triangle of the coherence matrix, all image pairs),
        dtype complex64, shape (n_points, n_image_pairs), numpy or cupy
    ref : int, default: 0
        index of the reference image, its phase is set to 0
    regularize : bool, default: False
        regularize the coherence matrix of the points whose coherence magnitude matrix is not positive
        definite or numerically singular (e.g. when the number of images approaches the number of
        SHPs); points with a well conditioned positive definite matrix are not changed

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        phase history `ph`, dtype complex64, shape (n_points, nimages), unit amplitude, the phase of
        image `ref` is 0; EMI quality (minimum eigenvalue), dtype float32, shape (n_points,): 1 for a
        coherence matrix whose phases close, negative where the coherence magnitude matrix is not
        positive definite (the phase history of such points is not reliable without `regularize`);
        regularized points get qualities closer to 1 for the same misfit, so their qualities are not
        comparable with those of other points (select by the temporal coherence instead)
    """
    xp = get_array_module(coh)
    nimages = mr.nimage_from_npair(coh.shape[-1])
    image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    if xp is np:
        return _emi_numba(coh,nimages,image_pairs,ref,regularize)
    else:
        return _emi_cp(coh,nimages,image_pairs,ref,regularize)

@ngpjit
def _ds_temp_coh_numba(
    coh:np.ndarray,
    ph:np.ndarray,
    image_pairs:np.ndarray,
):
    """Parameters
    ----------
    coh : np.ndarray
        complex coherence metrix, dtype np.complex64
    ph : np.ndarray
        complex phase history, dtype np.complex64
    image_pairs : np.ndarray
        image pairs
    """
    nimages = ph.shape[-1]
    n_points = ph.shape[0]
    n_image_pairs = image_pairs.shape[0]
    temp_coh = np.empty(n_points,dtype=np.float32)
    for i in prange(n_points):
        _ph = ph[i]
        _coh = coh[i]
        _t_coh = np.complex64(0.0)
        for j in range(n_image_pairs):
            n, k = image_pairs[j,0],image_pairs[j,1]
            int_conj_ph = np.conjugate(_ph[n])*_ph[k]
            diff_ph = _coh[j]*int_conj_ph/np.abs(_coh[j])
            _t_coh += diff_ph
        temp_coh[i] = np.abs(_t_coh)/n_image_pairs
    return temp_coh

if is_cuda_available():
    _ds_temp_coh_kernel = cp.ElementwiseKernel(
        'raw T coh, raw T ph, raw I image_pairs, int32 n_points, int32 nimages, int32 n_image_pairs',
        'raw float32 temp_coh',
        '''
        if (i >= n_points) return;
        T _t_coh = T(0.0,0.0);
        int j; int n; int k;
        int coh_idx; int ph_n_idx; int ph_k_idx;
        for (j = 0; j< n_image_pairs; j++){
            n = image_pairs[j*2];
            k = image_pairs[j*2+1];
            coh_idx = i*n_image_pairs+j;
            ph_n_idx = i*nimages+n;
            ph_k_idx = i*nimages+k;
            _t_coh += coh[coh_idx]/sqrt(norm(coh[coh_idx]))*conj(ph[ph_n_idx])*ph[ph_k_idx];
        }
        temp_coh[i] = abs(_t_coh)/n_image_pairs;
        ''',
        name = 'ds_temp_coh_kernel',no_return=True,
    )

def ds_temp_coh(coh:np.ndarray,
                ph:np.ndarray,
                image_pairs:np.ndarray=None,
                block_size:int=128,
            ):
    """Parameters
    ----------
    coh : np.ndarray
        complex coherence metrix, np.complex64 or cp.complex64
    ph : np.ndarray
        complex phase history, np.complex64 or cp.complex64
    image_pairs : np.ndarray, optional
        image pairs, all image pairs by default
    block_size : int, default: 128
        the CUDA block size, only applied for cuda
    """
    xp = get_array_module(coh)
    n_points = ph.shape[0]
    nimages = ph.shape[-1]
    if image_pairs is None:
        image_pairs = mr.TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = image_pairs.astype(np.int32)

    if xp is np:
        return _ds_temp_coh_numba(coh,ph,image_pairs)
    else:
        image_pairs = cp.asarray(image_pairs)
        n_image_pairs = image_pairs.shape[0]
        temp_coh = cp.empty(n_points, dtype=cp.float32)
        _ds_temp_coh_kernel(coh, ph, image_pairs, cp.int32(n_points),cp.int32(nimages),cp.int32(n_image_pairs), temp_coh, size=n_points, block_size=block_size)
        return temp_coh

def emperical_co_emi_temp_coh_pc(
    rslc:np.ndarray,
    idx:np.ndarray,
    pc_is_shp:np.ndarray,
    batch_size:int=1000,
    regularize:bool=False,
):
    """Parameters
    ----------
    rslc : np.ndarray
        rslc stack, dtype:'np.complex64'
    idx : np.ndarray
        index of point target (azimuth_index, range_index), dtype: `np.int32`, shape: (n_pc, 2)
    pc_is_shp : np.ndarray
        shp bool, dtype:'np.bool'
    batch_size : int, default: 1000
    regularize : bool, default: False
        regularize the coherence matrix in the phase linking as in `emi`; the temporal coherence is
        computed with the coherence matrix as estimated
    """
    xp = get_array_module(rslc)
    n_pc = idx.shape[0]
    nimages = rslc.shape[2]
    ph = xp.empty((n_pc, nimages),dtype=xp.complex64)
    emi_quality = xp.empty(n_pc,dtype=np.float32)
    t_coh = xp.empty(n_pc,dtype=np.float32)
    batch_bounds = np.arange(0,n_pc+batch_size,batch_size)
    # I forgot why I have to split data into batches, probably due to memory issue.
    if batch_bounds[-1]>n_pc: batch_bounds[-1]=n_pc
    for i in range(batch_bounds.shape[0]-1):
        start = batch_bounds[i]; stop = batch_bounds[i+1]
        _coh = mr.emperical_co_pc(rslc,idx[start:stop],pc_is_shp[start:stop])
        ph[start:stop],emi_quality[start:stop] = emi(_coh,regularize=regularize)
        t_coh[start:stop] = ds_temp_coh(_coh,ph[start:stop])
    return ph, emi_quality, t_coh
