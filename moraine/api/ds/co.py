"""Covariance and coherence matrix estimation"""

__all__ = ['emperical_co', 'emperical_co_pc', 'uncompress_single_coh_numba', 'uncompress_coh']

import math
import numpy as np
from ..utils_.device import is_cuda_available, get_array_module
from ..utils_.jit import mcuda_jit
if is_cuda_available():
    import cupy as cp
    from numba import cuda
from ..utils_.jit import ngpjit, ngjit
from ..tnet import TempNet, nimage_from_npair
from numba import prange, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir

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

    cov = cp.empty((nlines*width,nimages,nimages),dtype=rslc.dtype)
    coh = cp.empty((nlines*width,nimages,nimages),dtype=rslc.dtype)
    az_idx, r_idx = (cp.asarray(i.ravel(), dtype=cp.int32) for i in np.meshgrid(np.arange(nlines), np.arange(width), indexing='ij'))
    for start, stop, c, n in _shp_products_cp(rslc, az_idx, r_idx, is_shp.reshape(-1, az_win, r_win)):
        c = c.conj()     # sum x_m conj(x_j)
        cov[start:stop] = c/n[:,None,None].astype(cp.float32)
        d = 1/cp.sqrt(cp.diagonal(c, axis1=1, axis2=2).real)
        coh[start:stop] = c*(d[:,:,None]*d[:,None,:])
    return cov.reshape(nlines,width,nimages,nimages), coh.reshape(nlines,width,nimages,nimages)

# Coherence of the SHP samples X (n_shp, nimages) of a point on the CPU: the upper triangle of X^H X normalized by its
# diagonal. No BLAS inside the numba parallel loop (decision 0043): a threaded BLAS (MKL with its OpenMP runtime) called
# from numba's TBB threads makes them threads of the OpenMP runtime too, which oversubscribes the cores and makes any
# later fork of the process hang (the fork handlers of TBB and of OpenMP wait for each other). The product is a register
# blocked micro kernel instead, as in BLAS: a block of 4 rows x 8 columns of X^H X stays in 8 vector registers over all
# the samples, every sample is read once per block. LLVM does not keep such accumulators in registers when the block is
# written in numba (it vectorizes over the samples, or not at all), so it is written in LLVM vector instructions. Only
# the blocks of the upper triangle are computed (half the work of X^H X), and the coherence of the full network of image
# pairs, the upper triangle in row order, is written row by row without looking up the pairs. On the GPU: batched matrix
# products; every image pair of the point reads the same samples, so all pairs come from one product.
@intrinsic
def _conj_xt_x_4x8(typingctx, xr, xi, cr, ci, n, m0, j0):
    """cr + i ci [m0:m0+4, j0:j0+8] = sum over k < n of conj(x[k, m0:m0+4]) x[k, j0:j0+8], x = xr + i xi; float32 C
    contiguous 2D arrays, x with the columns m0:m0+4 and j0:j0+8, c with the rows m0:m0+4 and columns j0:j0+8"""
    sig = types.void(xr, xi, cr, ci, n, m0, j0)

    def codegen(context, builder, signature, args):
        f32 = ir.FloatType(); v8 = ir.VectorType(f32, 8); i64 = ir.IntType(64); i32 = ir.IntType(32)

        def array(i):
            a = context.make_array(signature.args[i])(context, builder, args[i])
            return a.data, builder.extract_value(a.shape, 1)
        xr_p, ncol = array(0); xi_p, _ = array(1); cr_p, ccol = array(2); ci_p, _ = array(3)
        n_v, m0_v, j0_v = (context.cast(builder, args[i], signature.args[i], types.int64) for i in (4, 5, 6))
        fma = cgutils.get_or_insert_function(builder.module, ir.FunctionType(v8, [v8, v8, v8]), 'llvm.fma.v8f32')
        undef = ir.Constant(v8, ir.Undefined)
        splat_mask = ir.Constant(ir.VectorType(i32, 8), [ir.Constant(i32, 0)]*8)
        zero = ir.Constant(v8, [ir.Constant(f32, 0.0)]*8)
        acc_r = [cgutils.alloca_once_value(builder, zero) for _ in range(4)]     # promoted to registers by LLVM
        acc_i = [cgutils.alloca_once_value(builder, zero) for _ in range(4)]

        def vector_at(ptr, offset):
            return builder.bitcast(builder.gep(ptr, [offset]), v8.as_pointer())

        def splat(x):
            return builder.shuffle_vector(builder.insert_element(undef, x, ir.Constant(i32, 0)), undef, splat_mask)

        with cgutils.for_range(builder, n_v) as loop:
            row = builder.mul(loop.index, ncol)
            br = builder.load(vector_at(xr_p, builder.add(row, j0_v)), align=4)
            bi = builder.load(vector_at(xi_p, builder.add(row, j0_v)), align=4)
            for r in range(4):
                o = builder.add(row, builder.add(m0_v, ir.Constant(i64, r)))
                ar = splat(builder.load(builder.gep(xr_p, [o])))
                ai = splat(builder.load(builder.gep(xi_p, [o])))
                # conj(a) b = (ar br + ai bi) + i (ar bi - ai br)
                c = builder.call(fma, [ar, br, builder.load(acc_r[r])])
                builder.store(builder.call(fma, [ai, bi, c]), acc_r[r])
                c = builder.call(fma, [ar, bi, builder.load(acc_i[r])])
                builder.store(builder.call(fma, [builder.fneg(ai), br, c]), acc_i[r])
        for r in range(4):
            o = builder.add(builder.mul(builder.add(m0_v, ir.Constant(i64, r)), ccol), j0_v)
            builder.store(builder.load(acc_r[r]), vector_at(cr_p, o), align=4)
            builder.store(builder.load(acc_i[r]), vector_at(ci_p, o), align=4)
        return context.get_dummy_value()

    return sig, codegen

@ngjit
def _shp_samples(rslc, az_idx_i, r_idx_i, shp_i, xr, xi):
    """the samples of the SHPs of a point in the rows of xr, xi (real and imaginary parts; the columns from nimages
    on are zero); returns their number"""
    nlines, width, nimages = rslc.shape
    az_win, r_win = shp_i.shape
    az_half_win, r_half_win = az_win//2, r_win//2
    ncol = xr.shape[1]
    n = 0
    for k in range(az_win):
        a = az_idx_i - az_half_win + k
        if a < 0 or a >= nlines:
            continue
        for l in range(r_win):
            r = r_idx_i - r_half_win + l
            if r >= 0 and r < width and shp_i[k, l]:
                for m in range(nimages):
                    v = rslc[a, r, m]
                    xr[n, m] = v.real
                    xi[n, m] = v.imag
                for m in range(nimages, ncol):
                    xr[n, m] = 0
                    xi[n, m] = 0
                n += 1
    return n

@ngjit
def _is_triu(image_pairs, nimages):
    """whether the image pairs are all pairs, in the order of the upper triangle by rows"""
    if image_pairs.shape[0] != nimages*(nimages-1)//2:
        return False
    p = 0
    for m in range(nimages):
        for j in range(m+1, nimages):
            if image_pairs[p, 0] != m or image_pairs[p, 1] != j:
                return False
            p += 1
    return True

@ngjit
def _coh_triu(cr, ci, d, coh, i):
    """coh[i] = conj(C) d_m d_j of the upper triangle of C = cr + i ci, rows m and columns j > m, in row order"""
    nimages = d.shape[0]
    p = 0
    for m in range(nimages):
        dm = d[m]
        crm = cr[m]; cim = ci[m]
        for jj in range(nimages-m-1):
            j = m + 1 + jj
            s = dm*d[j]
            coh[i, p+jj] = complex(crm[j]*s, -cim[j]*s)
        p += nimages - m - 1

@ngjit
def _coh_pairs(cr, ci, d, image_pairs, coh, i):
    """coh[i, p] = conj(C[m, j]) d_m d_j of the image pairs (m, j) from the upper triangle of C = cr + i ci"""
    for p in range(image_pairs.shape[0]):
        m = image_pairs[p, 0]; j = image_pairs[p, 1]
        s = d[m]*d[j]
        if m <= j:
            coh[i, p] = complex(cr[m, j]*s, -ci[m, j]*s)
        else:
            coh[i, p] = complex(cr[j, m]*s, ci[j, m]*s)

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
    npairs = image_pairs.shape[0]
    ncol = (nimages+7)//8*8     # columns of the samples and of C: whole vectors of 8
    nrow = (nimages+3)//4*4     # rows of C: whole blocks of 4
    triu = _is_triu(image_pairs, nimages)
    coh = np.empty((n_pc, npairs), dtype=rslc.dtype)
    chunk = 16                  # points of a task of the parallel loop, which allocates its buffers once
    for c in prange((n_pc+chunk-1)//chunk):
        xr = np.empty((az_win*r_win, ncol), dtype=np.float32)
        xi = np.empty((az_win*r_win, ncol), dtype=np.float32)
        cr = np.empty((nrow, ncol), dtype=np.float32)
        ci = np.empty((nrow, ncol), dtype=np.float32)
        d = np.empty(nimages, dtype=np.float32)
        for i in range(c*chunk, min((c+1)*chunk, n_pc)):
            n = _shp_samples(rslc, az_idx[i], r_idx[i], pc_is_shp[i], xr, xi)
            if n == 0:
                for p in range(npairs):
                    coh[i, p] = np.nan
                continue
            for m0 in range(0, nimages, 4):
                for j0 in range((m0//8)*8, ncol, 8):
                    _conj_xt_x_4x8(xr, xi, cr, ci, n, m0, j0)
            for m in range(nimages):
                d[m] = 1/math.sqrt(cr[m, m])
            if triu:
                _coh_triu(cr, ci, d, coh, i)
            else:
                _coh_pairs(cr, ci, d, image_pairs, coh, i)
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
    idx = idx.astype(np.int32)
    az_idx = idx[:,0]; r_idx = idx[:,1]

    if image_pairs is None:
        image_pairs = TempNet.from_bandwidth(nimages).image_pairs
    image_pairs = image_pairs.astype(np.int32)

    if xp is np:
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
        nimages = nimage_from_npair(coh.shape[-1])
        image_pairs = TempNet.from_bandwidth(nimages).image_pairs
    else:
        nimages = image_pairs[-1,-1]+1
    uncompressed_coh = xp.zeros((*coh.shape[:-1],nimages,nimages),dtype=coh.dtype)
    ref_images, sec_images = image_pairs[:,0], image_pairs[:,1]
    uncompressed_coh[...,ref_images,sec_images] = coh
    uncompressed_coh[...,sec_images,ref_images] = coh.conj()
    uncompressed_coh[...,np.arange(nimages),np.arange(nimages)] = 1
    return uncompressed_coh

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
