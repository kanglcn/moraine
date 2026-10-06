# EMI phase linking with regularization on the GPU (decision 0031). Imported by `pl.py` only when a GPU is available.
#
# One thread block per point; the matrices of the point stay in shared memory, as packed lower triangles (element
# (r, c), r >= c, at r(r+1)/2 + c) in two planes of n(n+1)/2 float32:
#   1. |coh| -> Householder tridiagonalization -> lambda_min, lambda_max by Sturm multisection -> beta as on the CPU
#      (decision 0023);
#   2. -|coh_beta|^-1 by the sweep operator on the lower triangle, no pivoting (positive definite after step 1);
#   3. A = |coh_beta|^-1 o coh_beta (real part in plane 1, imaginary part in plane 0), complex Householder
#      tridiagonalization with the vector of step k kept in column k below the diagonal, lambda_1 by multisection,
#      its eigenvector of the real tridiagonal matrix by inverse iteration (one thread), back transformation of that
#      vector only.
# Only the needed eigenpairs are computed and nothing but the input and the output is in global memory. The work of
# a point is a chain of ~n small steps with block barriers: the speed is bound by the latency of shared memory and
# the occupancy, which falls with n (A100: 4 blocks per SM up to n ~ 95, 1 from n ~ 138).
# numba types int32 arithmetic as int64 and checks negative indices of signed integers: array indices are uint32,
# and float constants are float32 (np.float32 globals) so that no float64 arithmetic enters the loops.
import math

import numpy as np
import cupy as cp
from numba import cuda, float32, float64, int32, uint32

from .pl import _EMI_MAX_COND, _emi_reg_beta_numba

_NT = 128                         # threads per block; also the number of points of each multisection step
_STATIC_SHARED_BYTES = 2048       # bound of the static shared arrays below (1.3 kB)
_FULL = 0xffffffff
_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_F2 = np.float32(2.0)
_F_EPS = np.float32(1.2e-7)

_reg_beta = cuda.jit(device=True)(_emi_reg_beta_numba.py_func)


@cuda.jit(device=True, inline=True)
def _tri(r, c):
    """packed index of (r, c), r >= c"""
    return uint32(((r * (r + 1)) >> 1) + c)


@cuda.jit(device=True, inline=True)
def _coh_at(coh, pt, n, r, c, s):
    """element (r, c) of the coherence matrix of point pt with the off-diagonal scaled by s; the image pairs are the
    upper triangle row by row (`TempNet.from_bandwidth(n).image_pairs`)"""
    if r == c:
        return _F1, _F0
    if r < c:
        z = coh[pt, (r * (2 * n - r - 1)) // 2 + c - r - 1]
        return s * z.real, s * z.imag
    z = coh[pt, (c * (2 * n - c - 1)) // 2 + r - c - 1]
    return s * z.real, -s * z.imag


@cuda.jit(device=True, inline=True)
def _block_sum2(a, b, red, tid, nt):
    for off in (16, 8, 4, 2, 1):
        a += cuda.shfl_down_sync(_FULL, a, off)
        b += cuda.shfl_down_sync(_FULL, b, off)
    w = tid // 32
    if tid % 32 == 0:
        red[uint32(w)] = a
        red[uint32(32 + w)] = b
    cuda.syncthreads()
    if tid < 32:
        nw = nt // 32
        a = red[uint32(tid)] if tid < nw else _F0
        b = red[uint32(32 + tid)] if tid < nw else _F0
        for off in (16, 8, 4, 2, 1):
            a += cuda.shfl_down_sync(_FULL, a, off)
            b += cuda.shfl_down_sync(_FULL, b, off)
        if tid == 0:
            red[uint32(64)] = a
            red[uint32(65)] = b
    cuda.syncthreads()
    return red[uint32(64)], red[uint32(65)]


@cuda.jit(device=True)
def _tridiag(Ar, Ai, cplx, n, d, er, ei, vr, vi, pr, pi, red, tid, nt):
    """Householder reduction of the Hermitian (real if not cplx) matrix (Ar, Ai), packed lower triangle, to
    tridiagonal form: diagonal d, subdiagonal (er, ei) with T[k+1, k] = e_k. The Householder vector of step k
    (unit norm, H_k = I - 2 v v^H on indices k+1..n-1) is kept in column k below the diagonal."""
    tx = tid % 32
    ty = tid // 32
    nty = nt // 32
    for k in range(n - 2):
        m0 = k + 1
        acc = _F0
        for r in range(m0 + tid, n, nt):            # x_r = A[r, k]
            i = _tri(r, k)
            a = Ar[i]
            acc += a * a
            if cplx:
                b = Ai[i]
                acc += b * b
        s2, _ = _block_sum2(acc, _F0, red, tid, nt)
        xnorm = math.sqrt(s2)
        if xnorm == _F0:
            if tid == 0:
                er[uint32(k)] = _F0
                ei[uint32(k)] = _F0
                d[uint32(k)] = Ar[_tri(k, k)]
            for r in range(m0 + tid, n, nt):
                i = _tri(r, k)
                Ar[i] = _F0
                if cplx:
                    Ai[i] = _F0
            cuda.syncthreads()
            continue
        i0 = _tri(m0, k)
        x0r = Ar[i0]
        x0i = Ai[i0] if cplx else _F0
        x0abs = math.sqrt(x0r * x0r + x0i * x0i)
        if x0abs > _F0:
            phr = x0r / x0abs
            phi = x0i / x0abs
        else:
            phr = _F1
            phi = _F0
        alr = -phr * xnorm
        ali = -phi * xnorm
        vn = math.sqrt(_F2 * xnorm * (xnorm + x0abs))
        for r in range(m0 + tid, n, nt):
            i = _tri(r, k)
            a = Ar[i]
            b = Ai[i] if cplx else _F0
            if r == m0:
                a -= alr
                b -= ali
            vr[uint32(r)] = a / vn
            vi[uint32(r)] = b / vn
        if tid == 0:
            er[uint32(k)] = alr
            ei[uint32(k)] = ali
            d[uint32(k)] = Ar[_tri(k, k)]
        cuda.syncthreads()
        # p = A v on the trailing block; A[r, c] = L[r, c] (r >= c) or conj(L[c, r]) (r < c), selected without
        # branching so that the lanes of a warp stay together
        for r in range(m0 + tid, n, nt):
            sr = _F0
            si = _F0
            rbr = _tri(r, 0)
            rbc = _tri(m0, 0)
            for c in range(m0, n):
                low = r >= c
                i = uint32(rbr + c) if low else uint32(rbc + r)
                a = Ar[i]
                vc_r = vr[uint32(c)]
                vc_i = vi[uint32(c)]
                if cplx:
                    b = Ai[i]
                    b = b if low else -b
                    sr += a * vc_r - b * vc_i
                    si += a * vc_i + b * vc_r
                else:
                    sr += a * vc_r
                rbc = uint32(rbc + c + 1)
            pr[uint32(r)] = sr
            pi[uint32(r)] = si
        cuda.syncthreads()
        acc = _F0
        for r in range(m0 + tid, n, nt):
            acc += vr[uint32(r)] * pr[uint32(r)] + vi[uint32(r)] * pi[uint32(r)]
        K, _ = _block_sum2(acc, _F0, red, tid, nt)
        for r in range(m0 + tid, n, nt):
            pr[uint32(r)] -= K * vr[uint32(r)]
            pi[uint32(r)] -= K * vi[uint32(r)]
        cuda.syncthreads()
        # A -= 2 (v w^H + w v^H) with w = p - (v^H p) v, lower triangle of the trailing block
        for r in range(m0 + ty, n, nty):
            a1 = vr[uint32(r)]
            a2 = vi[uint32(r)]
            a3 = pr[uint32(r)]
            a4 = pi[uint32(r)]
            rbr = _tri(r, 0)
            for c in range(m0 + tx, r + 1, 32):
                i = uint32(rbr + c)
                ur = a1 * pr[uint32(c)] + a2 * pi[uint32(c)] + a3 * vr[uint32(c)] + a4 * vi[uint32(c)]
                Ar[i] -= _F2 * ur
                if cplx:
                    ui = a2 * pr[uint32(c)] - a1 * pi[uint32(c)] + a4 * vr[uint32(c)] - a3 * vi[uint32(c)]
                    Ai[i] -= _F2 * ui
        for r in range(m0 + tid, n, nt):
            i = _tri(r, k)
            Ar[i] = vr[uint32(r)]
            if cplx:
                Ai[i] = vi[uint32(r)]
        cuda.syncthreads()
    if tid == 0:
        d[uint32(n - 2)] = Ar[_tri(n - 2, n - 2)]
        d[uint32(n - 1)] = Ar[_tri(n - 1, n - 1)]
        er[uint32(n - 2)] = Ar[_tri(n - 1, n - 2)]
        ei[uint32(n - 2)] = Ai[_tri(n - 1, n - 2)] if cplx else _F0
    cuda.syncthreads()


@cuda.jit(device=True)
def _sturm_count(d, b2, n, x):
    """number of eigenvalues < x of the symmetric tridiagonal matrix (diagonal d, squared off-diagonal b2)"""
    pivmin = float32(1e-30)
    q = d[uint32(0)] - x
    cnt = 0
    if q < _F0:
        cnt += 1
    for i in range(1, n):
        if abs(q) < pivmin:
            q = -pivmin
        q = d[uint32(i)] - x - b2[uint32(i - 1)] / q
        if q < _F0:
            cnt += 1
    return cnt


@cuda.jit(device=True)
def _gershgorin(d, b2, n):
    lo = float32(3e38)
    hi = float32(-3e38)
    for i in range(n):
        r = _F0
        if i > 0:
            r += math.sqrt(b2[uint32(i - 1)])
        if i < n - 1:
            r += math.sqrt(b2[uint32(i)])
        lo = min(lo, d[uint32(i)] - r)
        hi = max(hi, d[uint32(i)] + r)
    return lo - float32(1e-5) * (abs(lo) + _F1), hi + float32(1e-5) * (abs(hi) + _F1)


@cuda.jit(device=True)
def _eig_index(d, b2, n, j, lo, hi, cnt_sh, x_sh, tid, nt):
    """j-th smallest eigenvalue (0-based) of the symmetric tridiagonal matrix by multisection, all threads"""
    for it in range(6):
        x = lo + (hi - lo) * float32(tid + 1) / float32(nt + 1)
        x_sh[uint32(tid)] = x
        cnt_sh[uint32(tid)] = _sturm_count(d, b2, n, x)
        cuda.syncthreads()
        # counts grow with the thread index: the first index with a count > j
        a = 0
        b = nt
        while a < b:
            m = (a + b) // 2
            if cnt_sh[uint32(m)] > j:
                b = m
            else:
                a = m + 1
        if a > 0:
            lo = x_sh[uint32(a - 1)]
        if a < nt:
            hi = x_sh[uint32(a)]
        cuda.syncthreads()
    return float32(0.5) * (lo + hi)


@cuda.jit
def _emi_kernel(coh, n, ref, ph, quality):
    pt = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nt = cuda.blockDim.x
    sh = cuda.shared.array(0, dtype=float32)
    T = (n * (n + 1)) // 2
    P0 = sh[0:T]
    P1 = sh[T:2 * T]
    o = 2 * T
    vr = sh[o:o + n]; o += n
    vi = sh[o:o + n]; o += n
    pr = sh[o:o + n]; o += n
    pi = sh[o:o + n]; o += n
    d = sh[o:o + n]; o += n
    er = sh[o:o + n]; o += n
    ei = sh[o:o + n]; o += n
    b2 = sh[o:o + n]; o += n
    zr = sh[o:o + n]; o += n
    zi = sh[o:o + n]; o += n
    red = cuda.shared.array(66, dtype=float32)
    x_sh = cuda.shared.array(_NT, dtype=float32)
    cnt_sh = cuda.shared.array(_NT, dtype=int32)
    scal = cuda.shared.array(1, dtype=float32)
    tx0 = tid % 32
    ty0 = tid // 32
    nty0 = nt // 32

    # 1. regularization factor 1 - beta from the extreme eigenvalues of |coh|
    for r in range(ty0, n, nty0):
        for c in range(tx0, r + 1, 32):
            gr, gi = _coh_at(coh, pt, n, r, c, _F1)
            P0[_tri(r, c)] = math.sqrt(gr * gr + gi * gi)
    cuda.syncthreads()
    _tridiag(P0, P1, False, n, d, er, ei, vr, vi, pr, pi, red, tid, nt)
    for i in range(tid, n - 1, nt):
        b2[uint32(i)] = er[uint32(i)] * er[uint32(i)]
    cuda.syncthreads()
    lo, hi = _gershgorin(d, b2, n)
    lam_min = float64(_eig_index(d, b2, n, 0, lo, hi, cnt_sh, x_sh, tid, nt))
    lam_max = float64(_eig_index(d, b2, n, n - 1, lo, hi, cnt_sh, x_sh, tid, nt))
    if tid == 0:
        scal[uint32(0)] = float32(1.0 - _reg_beta(lam_min, lam_max))
    cuda.syncthreads()
    s = scal[uint32(0)]

    # 2. -|coh_beta|^-1 in P0 by the sweep operator; sweeping on k:
    #    a_kk <- -1/a_kk, a_ik <- a_ik/a_kk, a_ij <- a_ij - a_ik a_kj/a_kk
    for r in range(ty0, n, nty0):
        for c in range(tx0, r + 1, 32):
            gr, gi = _coh_at(coh, pt, n, r, c, s)
            P0[_tri(r, c)] = math.sqrt(gr * gr + gi * gi)
    cuda.syncthreads()
    for k in range(n):
        for i in range(tid, n, nt):
            pr[uint32(i)] = P0[_tri(i, k)] if i >= k else P0[_tri(k, i)]
        cuda.syncthreads()
        dinv = _F1 / pr[uint32(k)]
        for r in range(ty0, n, nty0):
            f = pr[uint32(r)] * dinv
            rbr = _tri(r, 0)
            for c in range(tx0, r + 1, 32):
                i = uint32(rbr + c)
                if r == k:
                    P0[i] = -dinv if c == k else pr[uint32(c)] * dinv
                elif c == k:
                    P0[i] = f
                else:
                    P0[i] -= f * pr[uint32(c)]
        cuda.syncthreads()

    # 3. smallest eigenpair of A = |coh_beta|^-1 o coh_beta
    for r in range(ty0, n, nty0):
        for c in range(tx0, r + 1, 32):
            gr, gi = _coh_at(coh, pt, n, r, c, s)
            i = _tri(r, c)
            w = -P0[i]
            P1[i] = w * gr
            P0[i] = w * gi
    cuda.syncthreads()
    _tridiag(P1, P0, True, n, d, er, ei, vr, vi, pr, pi, red, tid, nt)
    for i in range(tid, n - 1, nt):
        b2[uint32(i)] = er[uint32(i)] * er[uint32(i)] + ei[uint32(i)] * ei[uint32(i)]
    cuda.syncthreads()
    lo, hi = _gershgorin(d, b2, n)
    lam1 = _eig_index(d, b2, n, 0, lo, hi, cnt_sh, x_sh, tid, nt)
    if tid == 0:
        # inverse iteration on T - lam1 I (tridiagonal LU with partial pivoting as LAPACK gtsv), two steps;
        # work arrays: y = zr, dd = vr, du = vi, du2 = pr, dl = pi
        y = zr
        dd = vr
        du = vi
        du2 = pr
        dl = pi
        for i in range(n):
            y[uint32(i)] = _F1
        eps = _F_EPS * (abs(hi) + abs(lo) + _F1)
        for it in range(2):
            for i in range(n):
                dd[uint32(i)] = d[uint32(i)] - lam1
                if i < n - 1:
                    du[uint32(i)] = math.sqrt(b2[uint32(i)])
                    dl[uint32(i)] = du[uint32(i)]
                du2[uint32(i)] = _F0
            for i in range(n - 1):
                if abs(dd[uint32(i)]) >= abs(dl[uint32(i)]):
                    if abs(dd[uint32(i)]) < eps:
                        dd[uint32(i)] = eps
                    f = dl[uint32(i)] / dd[uint32(i)]
                    dd[uint32(i + 1)] -= f * du[uint32(i)]
                    y[uint32(i + 1)] -= f * y[uint32(i)]
                else:
                    f = dd[uint32(i)] / dl[uint32(i)]
                    dd[uint32(i)] = dl[uint32(i)]
                    t = dd[uint32(i + 1)]
                    dd[uint32(i + 1)] = du[uint32(i)] - f * t
                    if i < n - 2:
                        du2[uint32(i)] = du[uint32(i + 1)]
                        du[uint32(i + 1)] = -f * du2[uint32(i)]
                    du[uint32(i)] = t
                    t = y[uint32(i)]
                    y[uint32(i)] = y[uint32(i + 1)]
                    y[uint32(i + 1)] = t - f * y[uint32(i + 1)]
            if abs(dd[uint32(n - 1)]) < eps:
                dd[uint32(n - 1)] = eps
            y[uint32(n - 1)] /= dd[uint32(n - 1)]
            y[uint32(n - 2)] = (y[uint32(n - 2)] - du[uint32(n - 2)] * y[uint32(n - 1)]) / dd[uint32(n - 2)]
            for i in range(n - 3, -1, -1):
                y[uint32(i)] = ((y[uint32(i)] - du[uint32(i)] * y[uint32(i + 1)] - du2[uint32(i)] * y[uint32(i + 2)])
                                / dd[uint32(i)])
            nrm = _F0
            for i in range(n):
                nrm += y[uint32(i)] * y[uint32(i)]
            nrm = _F1 / math.sqrt(nrm)
            for i in range(n):
                y[uint32(i)] *= nrm
        # eigenvector of the complex T: D y, D = diag(delta), delta_0 = 1, delta_{k+1} = delta_k e_k / |e_k|
        dr = _F1
        di = _F0
        for i in range(n):
            yi = y[uint32(i)]
            zr[uint32(i)] = dr * yi
            zi[uint32(i)] = di * yi
            if i < n - 1:
                ab = math.sqrt(b2[uint32(i)])
                if ab > _F0:
                    nr = (dr * er[uint32(i)] - di * ei[uint32(i)]) / ab
                    ni = (dr * ei[uint32(i)] + di * er[uint32(i)]) / ab
                    dr = nr
                    di = ni
        quality[pt] = lam1
    cuda.syncthreads()
    # z = H_0 H_1 ... H_{n-3} z
    for k in range(n - 3, -1, -1):
        m0 = k + 1
        acc_r = _F0
        acc_i = _F0
        for r in range(m0 + tid, n, nt):
            a = P1[_tri(r, k)]
            b = P0[_tri(r, k)]
            acc_r += a * zr[uint32(r)] + b * zi[uint32(r)]
            acc_i += a * zi[uint32(r)] - b * zr[uint32(r)]
        sr, si = _block_sum2(acc_r, acc_i, red, tid, nt)
        for r in range(m0 + tid, n, nt):
            a = P1[_tri(r, k)]
            b = P0[_tri(r, k)]
            zr[uint32(r)] -= _F2 * (a * sr - b * si)
            zi[uint32(r)] -= _F2 * (a * si + b * sr)
        cuda.syncthreads()
    rr = zr[uint32(ref)]
    ri = zi[uint32(ref)]
    for i in range(tid, n, nt):
        a = zr[uint32(i)] * rr + zi[uint32(i)] * ri
        b = zi[uint32(i)] * rr - zr[uint32(i)] * ri
        m = math.sqrt(a * a + b * b)
        ph[pt, i] = complex(a / m, b / m)


def _shared_bytes(n_images):
    return (n_images * (n_images + 1) + 10 * n_images) * 4


_max_dynamic_shared = {}          # device id -> dynamic shared memory per block the kernel is allowed to use


def _allow_dynamic_shared(kernel):
    """Let `kernel` use all shared memory of the current device beyond 48 kB; returns the bytes it may use.
    Kernels may use more than 48 kB only after an explicit opt-in (cuFuncSetAttribute); numba has no public API for
    it. If the opt-in fails (e.g. the CUDA target built into numba instead of numba-cuda), 48 kB."""
    dev = cp.cuda.Device()
    if dev.id in _max_dynamic_shared:
        return _max_dynamic_shared[dev.id]
    limit = 48 * 1024 - _STATIC_SHARED_BYTES
    want = dev.attributes['MaxSharedMemoryPerBlockOptin'] - _STATIC_SHARED_BYTES
    try:
        from numba.cuda.cudadrv.driver import binding, driver
        attr = binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        for ov in kernel.overloads.values():
            f = ov._codelibrary.get_cufunc()
            driver.cuKernelSetAttribute(attr, want, f.handle, f.device.id)
        limit = want
    except Exception:
        pass
    _max_dynamic_shared[dev.id] = limit
    return limit


def emi_cuda_supported(n_images):
    """whether `emi_cuda` can process `n_images` images on the current device"""
    if n_images < 3:
        return False
    if _shared_bytes(n_images) <= 48 * 1024 - _STATIC_SHARED_BYTES:
        return True
    _compile()
    return _shared_bytes(n_images) <= _allow_dynamic_shared(_emi_kernel)


def _compile():
    if not _emi_kernel.overloads:
        _emi_kernel.compile('void(complex64[:, ::1], int32, int32, complex64[:, ::1], float32[::1])')


def emi_cuda(coh, n_images, ref=0):
    """Regularized EMI of `pl._emi` for cupy arrays (`emi_cuda_supported(n_images)` must be true).

    coh: complex64 (n_points, n_image_pairs) cupy array -> (ph complex64 (n_points, n_images), quality float32
    (n_points,))."""
    n_points = coh.shape[0]
    ph = cp.empty((n_points, n_images), dtype=cp.complex64)
    quality = cp.empty(n_points, dtype=cp.float32)
    if n_points == 0:
        return ph, quality
    if coh.dtype != cp.complex64 or not coh.flags.c_contiguous:
        coh = cp.ascontiguousarray(coh, dtype=cp.complex64)
    _compile()
    shared = _shared_bytes(n_images)
    if shared > 48 * 1024 - _STATIC_SHARED_BYTES:
        _allow_dynamic_shared(_emi_kernel)
    _emi_kernel[n_points, _NT, 0, shared](coh, np.int32(n_images), np.int32(ref), ph, quality)
    return ph, quality
