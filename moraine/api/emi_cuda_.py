# EMI phase linking with regularization on the GPU (decision 0031). Imported by `pl.py` only when a GPU is available.
#
# One thread block per point; the matrices of the point stay in shared memory, as packed lower triangles (element
# (r, c), r >= c, at r(r+1)/2 + c) in two planes of n(n+1)/2 float32:
#   1. |coh| -> Householder tridiagonalization -> lambda_min, lambda_max by Sturm multisection -> beta as on the CPU
#      (decision 0023); |coh| is kept in the second plane for step 2;
#   2. -|coh_beta|^-1 by the sweep operator, four pivots per step (2 x 2 block sweep and single pivots for the
#      remainder), no pivoting (positive definite after step 1);
#   3. A = |coh_beta|^-1 o coh_beta (real part in plane 1, imaginary part in plane 0), complex Householder
#      tridiagonalization with the vector of step k kept in column k below the diagonal, lambda_1 by multisection,
#      its eigenvector of the real tridiagonal matrix by inverse iteration (one thread), back transformation of that
#      vector only.
# Only the needed eigenpairs are computed and nothing but the input and the output is in global memory. The work of
# a point is a chain of ~n small steps with block barriers: the speed is bound by the latency of shared memory and
# the occupancy, which falls with n (A100: 4 blocks per SM up to n ~ 95, 1 from n ~ 138).
# Code shape, measured on an A100 (2026-10-08): numba types int32 arithmetic as int64 and checks negative indices of
# signed integers, so array indices are uint32 and loop bounds and index arithmetic are cast back to int32 (64-bit
# loop control doubled the instructions of the loops); float constants are float32 (np.float32 globals). The loops
# over the triangle have a uniform trip count (column chunks of 32 lanes) and take the rows r and n-1-r together:
# short per-row loops of 1-3 iterations were dominated by their loop control. The matrix-vector product of a
# Householder step is one thread per row, with 2 / 4 threads per row (parts of the columns) for trailing blocks of
# at most 64 / 32 rows, and two accumulators per thread. Approximate division (fast_fdividef) in the inverse
# iteration (it corrects itself); the Sturm counts use correctly rounded division (monotone in x). max_registers=72
# keeps the kernel without spills and lets 7 blocks per SM run for small n. The block has 128, 256 or 512 threads:
# the size whose blocks fit the most times into a multiprocessor (4 blocks of 128 up to n ~ 100 on an A100, 256
# where 2 blocks fit, 512 where only one does): the threads of a block wait for each other at the barriers, blocks do
# not, and with one block per multiprocessor a wider block divides the work of a step among more warps.
import math

import numpy as np
import cupy as cp
from numba import cuda, float32, float64, int32, uint32
from numba.cuda.libdevice import fast_fdividef

from .pl import _EMI_MAX_COND, _emi_reg_beta_numba

_BLOCK_SIZES = (128, 256, 512)    # threads per block (also the points of a multisection step), chosen per number of images
_STATIC_SHARED_BYTES = 2048       # bound of the static shared arrays of the kernel
_FULL = 0xffffffff
_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_F2 = np.float32(2.0)
_F_EPS = np.float32(1.2e-7)
_I0 = np.int32(0)
_I1 = np.int32(1)
_I2 = np.int32(2)
_I3 = np.int32(3)
_I31 = np.int32(31)
_I32 = np.int32(32)
_IM1 = np.int32(-1)

_reg_beta = cuda.jit(device=True)(_emi_reg_beta_numba.py_func)


@cuda.jit(device=True, inline=True)
def _tri(r, c):
    """packed index of (r, c), r >= c"""
    return uint32(((r * (r + 1)) >> 1) + c)


@cuda.jit(device=True, inline=True)
def _rowbase(r):
    """packed index of (r, 0) as int32"""
    return int32((r * (r + _I1)) >> 1)


@cuda.jit(device=True, inline=True)
def _coh_at(coh, pt, n, r, c, s):
    """element (r, c) of the coherence matrix of point pt with the off-diagonal scaled by s; the image pairs are the
    upper triangle row by row (`TempNet.from_bandwidth(n).image_pairs`)"""
    if r == c:
        return _F1, _F0
    if r < c:
        z = coh[pt, int32((r * (2 * n - r - 1)) // 2 + c - r - 1)]
        return s * z.real, s * z.imag
    z = coh[pt, int32((c * (2 * n - c - 1)) // 2 + r - c - 1)]
    return s * z.real, -s * z.imag


@cuda.jit(device=True, inline=True)
def _block_sum2(a, b, red, tid, nt):
    """sums of a and b over the block of 4, 8 or 16 warps: one barrier, every thread adds the warp sums in the order
    of a shuffle-down tree"""
    for off in (16, 8, 4, 2, 1):
        a += cuda.shfl_down_sync(_FULL, a, off)
        b += cuda.shfl_down_sync(_FULL, b, off)
    w = uint32(tid // _I32)
    if tid % _I32 == 0:
        red[w] = a
        red[uint32(16) + w] = b
    cuda.syncthreads()
    nw = int32(nt // _I32)
    if nw == 4:
        a = (red[uint32(0)] + red[uint32(2)]) + (red[uint32(1)] + red[uint32(3)])
        b = (red[uint32(16)] + red[uint32(18)]) + (red[uint32(17)] + red[uint32(19)])
    elif nw == 8:
        a = ((red[uint32(0)] + red[uint32(4)]) + (red[uint32(2)] + red[uint32(6)])) + ((red[uint32(1)] + red[uint32(5)]) + (red[uint32(3)] + red[uint32(7)]))
        b = ((red[uint32(16)] + red[uint32(20)]) + (red[uint32(18)] + red[uint32(22)])) + ((red[uint32(17)] + red[uint32(21)]) + (red[uint32(19)] + red[uint32(23)]))
    else:
        a = ((((red[uint32(0)] + red[uint32(8)]) + (red[uint32(4)] + red[uint32(12)])) + ((red[uint32(2)] + red[uint32(10)]) + (red[uint32(6)] + red[uint32(14)])))
             + (((red[uint32(1)] + red[uint32(9)]) + (red[uint32(5)] + red[uint32(13)])) + ((red[uint32(3)] + red[uint32(11)]) + (red[uint32(7)] + red[uint32(15)]))))
        b = ((((red[uint32(16)] + red[uint32(24)]) + (red[uint32(20)] + red[uint32(28)])) + ((red[uint32(18)] + red[uint32(26)]) + (red[uint32(22)] + red[uint32(30)])))
             + (((red[uint32(17)] + red[uint32(25)]) + (red[uint32(21)] + red[uint32(29)])) + ((red[uint32(19)] + red[uint32(27)]) + (red[uint32(23)] + red[uint32(31)]))))
    return a, b


@cuda.jit(device=True, inline=True)
def _row_dot(Ar, Ai, cplx, r, c0, c1, vr, vi):
    """sum over c in [c0, c1) of A[r, c] v_c for the Hermitian (real if not cplx) packed matrix: A[r, c] = L[r, c] for
    c <= r (contiguous), conj(L[c, r]) for c > r (down the column); two columns per iteration, two accumulators"""
    sr0 = _F0
    si0 = _F0
    sr1 = _F0
    si1 = _F0
    rbr = _rowbase(r)
    c = c0
    cend = min(c1, int32(r + _I1))
    while int32(c + _I1) < cend:
        i0 = uint32(int32(rbr + c))
        i1 = uint32(int32(rbr + c + _I1))
        a0 = Ar[i0]
        a1 = Ar[i1]
        v0r = vr[uint32(c)]
        v0i = vi[uint32(c)]
        v1r = vr[uint32(c + _I1)]
        v1i = vi[uint32(c + _I1)]
        if cplx:
            b0 = Ai[i0]
            b1 = Ai[i1]
            sr0 += a0 * v0r - b0 * v0i
            si0 += a0 * v0i + b0 * v0r
            sr1 += a1 * v1r - b1 * v1i
            si1 += a1 * v1i + b1 * v1r
        else:
            sr0 += a0 * v0r
            sr1 += a1 * v1r
        c = int32(c + _I2)
    if c < cend:
        i0 = uint32(int32(rbr + c))
        a0 = Ar[i0]
        v0r = vr[uint32(c)]
        v0i = vi[uint32(c)]
        if cplx:
            b0 = Ai[i0]
            sr0 += a0 * v0r - b0 * v0i
            si0 += a0 * v0i + b0 * v0r
        else:
            sr0 += a0 * v0r
    c = max(c0, int32(r + _I1))
    rbc = _rowbase(c)
    while int32(c + _I1) < c1:
        i0 = uint32(int32(rbc + r))
        i1 = uint32(int32(rbc + c + _I1 + r))
        a0 = Ar[i0]
        a1 = Ar[i1]
        v0r = vr[uint32(c)]
        v0i = vi[uint32(c)]
        v1r = vr[uint32(c + _I1)]
        v1i = vi[uint32(c + _I1)]
        if cplx:
            b0 = -Ai[i0]
            b1 = -Ai[i1]
            sr0 += a0 * v0r - b0 * v0i
            si0 += a0 * v0i + b0 * v0r
            sr1 += a1 * v1r - b1 * v1i
            si1 += a1 * v1i + b1 * v1r
        else:
            sr0 += a0 * v0r
            sr1 += a1 * v1r
        rbc = int32(rbc + c + c + _I3)
        c = int32(c + _I2)
    if c < c1:
        i0 = uint32(int32(rbc + r))
        a0 = Ar[i0]
        v0r = vr[uint32(c)]
        v0i = vi[uint32(c)]
        if cplx:
            b0 = -Ai[i0]
            sr0 += a0 * v0r - b0 * v0i
            si0 += a0 * v0i + b0 * v0r
        else:
            sr0 += a0 * v0r
    return sr0 + sr1, si0 + si1


@cuda.jit(device=True, inline=True)
def _inv4(c0, c1, c2, c3, k):
    """inverse of the symmetric positive definite 4 x 4 block B[a, b] = c_b[k + a] (sweep operator); the 10 entries
    of its lower triangle, row by row"""
    B = cuda.local.array((4, 4), dtype=float32)
    B[0, 0] = c0[uint32(k)]
    B[1, 0] = c0[uint32(k + 1)]
    B[2, 0] = c0[uint32(k + 2)]
    B[3, 0] = c0[uint32(k + 3)]
    B[1, 1] = c1[uint32(k + 1)]
    B[2, 1] = c1[uint32(k + 2)]
    B[3, 1] = c1[uint32(k + 3)]
    B[2, 2] = c2[uint32(k + 2)]
    B[3, 2] = c2[uint32(k + 3)]
    B[3, 3] = c3[uint32(k + 3)]
    for a in range(4):
        for b in range(a):
            B[b, a] = B[a, b]
    for p in range(4):
        dinv = _F1 / B[p, p]
        for a in range(4):
            if a != p:
                f = B[a, p] * dinv
                for b in range(4):
                    if b != p:
                        B[a, b] -= f * B[p, b]
        for a in range(4):
            if a != p:
                B[a, p] = B[a, p] * dinv
                B[p, a] = B[a, p]
        B[p, p] = -dinv
    return (-B[0, 0], -B[1, 0], -B[1, 1], -B[2, 0], -B[2, 1], -B[2, 2], -B[3, 0], -B[3, 1], -B[3, 2], -B[3, 3])


@cuda.jit(device=True)
def _tridiag(Ar, Ai, cplx, n, d, er, ei, vr, vi, pr, pi, red, tid, nt):
    """Householder reduction of the Hermitian (real if not cplx) matrix (Ar, Ai), packed lower triangle, to
    tridiagonal form: diagonal d, subdiagonal (er, ei) with T[k+1, k] = e_k. The Householder vector of step k
    (unit norm, H_k = I - 2 v v^H on indices k+1..n-1) is kept in column k below the diagonal."""
    tx = int32(tid % _I32)
    ty = int32(tid // _I32)
    nty = int32(nt // _I32)
    for k in range(int32(n - _I2)):
        m0 = int32(k + _I1)
        m = int32(n - m0)                          # order of the trailing block
        nchunks = int32((m + _I31) // _I32)        # column chunks of 32 lanes
        half = int32((m + _I1) // _I2)             # row pairs (m0 + j, n - 1 - j)
        acc = _F0
        for q in range(2):                          # x_r = A[r, k]; at most 2 rows per thread (n <= 2 nt)
            r = int32(m0 + tid + q * nt)
            if r < n:
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
            for q in range(2):
                r = int32(m0 + tid + q * nt)
                if r < n:
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
            iab = _F1 / x0abs
            phr = x0r * iab
            phi = x0i * iab
        else:
            phr = _F1
            phi = _F0
        alr = -phr * xnorm
        ali = -phi * xnorm
        ivn = _F1 / math.sqrt(_F2 * xnorm * (xnorm + x0abs))
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
                i = _tri(r, k)
                a = Ar[i]
                b = Ai[i] if cplx else _F0
                if r == m0:
                    a -= alr
                    b -= ali
                vr[uint32(r)] = a * ivn
                vi[uint32(r)] = b * ivn
        if tid == 0:
            er[uint32(k)] = alr
            ei[uint32(k)] = ali
            d[uint32(k)] = Ar[_tri(k, k)]
        cuda.syncthreads()
        # p = A v on the trailing block: one thread per row (two rows per thread when m > nt), or 2 / 4 threads per
        # row when the block has at most 64 / 32 rows, each over a part of the columns, combined by shuffles
        if m > (nt >> 1):
            for q in range(2):
                r = int32(m0 + tid + q * nt)
                if r < n:
                    sr, si = _row_dot(Ar, Ai, cplx, r, m0, n, vr, vi)
                    pr[uint32(r)] = sr
                    pi[uint32(r)] = si
        else:
            four = m <= (nt >> 2)
            rowi = int32(tid >> 2) if four else int32(tid >> 1)
            h = int32(tid & _I3) if four else int32(tid & _I1)
            span = int32((m + _I3) >> 2) if four else int32((m + _I1) >> 1)
            sr = _F0
            si = _F0
            if rowi < m:
                r = int32(m0 + rowi)
                cs = int32(m0 + h * span)
                ce = min(int32(cs + span), n)
                if cs < ce:
                    sr, si = _row_dot(Ar, Ai, cplx, r, cs, ce, vr, vi)
            sr += cuda.shfl_xor_sync(_FULL, sr, 1)
            si += cuda.shfl_xor_sync(_FULL, si, 1)
            if four:
                sr += cuda.shfl_xor_sync(_FULL, sr, 2)
                si += cuda.shfl_xor_sync(_FULL, si, 2)
            if h == 0 and rowi < m:
                pr[uint32(m0 + rowi)] = sr
                pi[uint32(m0 + rowi)] = si
        cuda.syncthreads()
        acc = _F0
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
                acc += vr[uint32(r)] * pr[uint32(r)] + vi[uint32(r)] * pi[uint32(r)]
        K, _ = _block_sum2(acc, _F0, red, tid, nt)
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
                pr[uint32(r)] -= K * vr[uint32(r)]
                pi[uint32(r)] -= K * vi[uint32(r)]
        cuda.syncthreads()
        # A -= 2 (v w^H + w v^H) with w = p - (v^H p) v, lower triangle of the trailing block: a warp takes the
        # rows r and r2 = n - 1 - (r - m0) together (m + 1 elements), its lanes a column chunk of 32
        for j in range(ty, half, nty):
            r = int32(m0 + j)
            r2 = int32(n - _I1 - j)
            a1 = vr[uint32(r)]
            a2 = vi[uint32(r)]
            a3 = pr[uint32(r)]
            a4 = pi[uint32(r)]
            b1 = vr[uint32(r2)]
            b2 = vi[uint32(r2)]
            b3 = pr[uint32(r2)]
            b4 = pi[uint32(r2)]
            rbr = _rowbase(r)
            rbr2 = _rowbase(r2)
            c = int32(m0 + tx)
            for q in range(nchunks):
                pc_r = pr[uint32(c)]
                pc_i = pi[uint32(c)]
                vc_r = vr[uint32(c)]
                vc_i = vi[uint32(c)]
                if c <= r:
                    i = uint32(int32(rbr + c))
                    ur = a1 * pc_r + a2 * pc_i + a3 * vc_r + a4 * vc_i
                    Ar[i] -= _F2 * ur
                    if cplx:
                        ui = a2 * pc_r - a1 * pc_i + a4 * vc_r - a3 * vc_i
                        Ai[i] -= _F2 * ui
                if r2 > r and c <= r2:
                    i = uint32(int32(rbr2 + c))
                    ur = b1 * pc_r + b2 * pc_i + b3 * vc_r + b4 * vc_i
                    Ar[i] -= _F2 * ur
                    if cplx:
                        ui = b2 * pc_r - b1 * pc_i + b4 * vc_r - b3 * vc_i
                        Ai[i] -= _F2 * ui
                c = int32(c + _I32)
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
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
    """number of eigenvalues < x of the symmetric tridiagonal matrix (diagonal d, squared off-diagonal b2): the
    negative pivots of the LDL^T factorization of T - xI. A pivot that is tiny in magnitude is replaced by -pivmin
    before it is counted (LAPACK dlaebz): the sign counted must be the sign used in the next pivot. Counting first
    (q < 0) and replacing afterwards treats a pivot that rounds to exactly 0 as non-negative while it acts as
    negative: the count is off by one at that x and the multisection converges to a wrong eigenvalue (phase errors
    of pi for 2 of 4.1 M candidates of the sample data). With correctly rounded division the count is monotone in x
    (Demmel, Dhillon, Ren 1995)."""
    pivmin = float32(1e-30)
    q = d[uint32(0)] - x
    if abs(q) < pivmin:
        q = -pivmin
    cnt = _I0
    if q < _F0:
        cnt = _I1
    for i in range(_I1, n):
        q = d[uint32(i)] - x - b2[uint32(i - 1)] / q
        if abs(q) < pivmin:
            q = -pivmin
        if q < _F0:
            cnt = int32(cnt + _I1)
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
        a = _I0
        b = nt
        while a < b:
            m = int32((a + b) // _I2)
            if cnt_sh[uint32(m)] > j:
                b = m
            else:
                a = int32(m + _I1)
        if a > 0:
            lo = x_sh[uint32(a - 1)]
        if a < nt:
            hi = x_sh[uint32(a)]
        cuda.syncthreads()
    return float32(0.5) * (lo + hi)


@cuda.jit(max_registers=72)
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
    red = cuda.shared.array(32, dtype=float32)
    x_sh = sh[o:o + nt]; o += nt
    sh_i32 = cuda.shared.array(0, dtype=int32)
    cnt_sh = sh_i32[o:o + nt]; o += nt
    scal = cuda.shared.array(1, dtype=float32)
    tx0 = int32(tid % _I32)
    ty0 = int32(tid // _I32)
    nty0 = int32(nt // _I32)
    nchunks = int32((n + _I31) // _I32)           # column chunks of 32 lanes of a full row
    half = int32((n + _I1) // _I2)                # row pairs (j, n - 1 - j) of the whole triangle

    # 1. regularization factor 1 - beta from the extreme eigenvalues of |coh|
    for j in range(ty0, half, nty0):
        r = j
        r2 = int32(n - _I1 - j)
        c = tx0
        for q in range(nchunks):
            if c <= r:
                gr, gi = _coh_at(coh, pt, n, r, c, _F1)
                i = _tri(r, c)
                v = math.sqrt(gr * gr + gi * gi)
                P0[i] = v
                P1[i] = v                                # kept for stage 2 (the real tridiagonalization uses P0 only)
            if r2 > r and c <= r2:
                gr, gi = _coh_at(coh, pt, n, r2, c, _F1)
                i = _tri(r2, c)
                v = math.sqrt(gr * gr + gi * gi)
                P0[i] = v
                P1[i] = v
            c = int32(c + _I32)
    cuda.syncthreads()
    _tridiag(P0, P1, False, n, d, er, ei, vr, vi, pr, pi, red, tid, nt)
    for i in range(tid, int32(n - _I1), nt):
        b2[uint32(i)] = er[uint32(i)] * er[uint32(i)]
    cuda.syncthreads()
    lo, hi = _gershgorin(d, b2, n)
    lam_min = float64(_eig_index(d, b2, n, _I0, lo, hi, cnt_sh, x_sh, tid, nt))
    lam_max = float64(_eig_index(d, b2, n, int32(n - _I1), lo, hi, cnt_sh, x_sh, tid, nt))
    if tid == 0:
        scal[uint32(0)] = float32(1.0 - _reg_beta(lam_min, lam_max))
    cuda.syncthreads()
    s = scal[uint32(0)]

    # 2. -|coh_beta|^-1 in P0 by the sweep operator; sweeping on k:
    #    a_kk <- -1/a_kk, a_ik <- a_ik/a_kk, a_ij <- a_ij - a_ik a_kj/a_kk
    for j in range(ty0, half, nty0):
        r = j
        r2 = int32(n - _I1 - j)
        c = tx0
        for q in range(nchunks):
            if c <= r:
                i = _tri(r, c)
                P0[i] = P1[i] * s if c != r else _F1
            if r2 > r and c <= r2:
                i = _tri(r2, c)
                P0[i] = P1[i] * s if c != r2 else _F1
            c = int32(c + _I32)
    cuda.syncthreads()
    k = _I0
    while int32(k + _I3) < n:                              # four pivots k..k+3 per step (4 x 4 block sweep)
        k1 = int32(k + _I1)
        k2 = int32(k + _I2)
        k3 = int32(k + _I3)
        for i in range(tid, n, nt):
            pr[uint32(i)] = P0[_tri(i, k)] if i >= k else P0[_tri(k, i)]
            pi[uint32(i)] = P0[_tri(i, k1)] if i >= k1 else P0[_tri(k1, i)]
            vr[uint32(i)] = P0[_tri(i, k2)] if i >= k2 else P0[_tri(k2, i)]
            vi[uint32(i)] = P0[_tri(i, k3)] if i >= k3 else P0[_tri(k3, i)]
        cuda.syncthreads()
        q00, q10, q11, q20, q21, q22, q30, q31, q32, q33 = _inv4(pr, pi, vr, vi, k)
        # generic update a_ij -= sum_p g_i^p a_{k+p, j}, g_i = (a_ik .. a_ik3) P^-1, on the whole triangle (rows and
        # columns k..k3 are overwritten below)
        for j in range(ty0, half, nty0):
            r = j
            r2 = int32(n - _I1 - j)
            f0 = pr[uint32(r)]
            f1 = pi[uint32(r)]
            f2 = vr[uint32(r)]
            f3 = vi[uint32(r)]
            g0 = f0 * q00 + f1 * q10 + f2 * q20 + f3 * q30
            g1 = f0 * q10 + f1 * q11 + f2 * q21 + f3 * q31
            g2 = f0 * q20 + f1 * q21 + f2 * q22 + f3 * q32
            g3 = f0 * q30 + f1 * q31 + f2 * q32 + f3 * q33
            f0 = pr[uint32(r2)]
            f1 = pi[uint32(r2)]
            f2 = vr[uint32(r2)]
            f3 = vi[uint32(r2)]
            e0 = f0 * q00 + f1 * q10 + f2 * q20 + f3 * q30
            e1 = f0 * q10 + f1 * q11 + f2 * q21 + f3 * q31
            e2 = f0 * q20 + f1 * q21 + f2 * q22 + f3 * q32
            e3 = f0 * q30 + f1 * q31 + f2 * q32 + f3 * q33
            rbr = _rowbase(r)
            rbr2 = _rowbase(r2)
            c = tx0
            for q in range(nchunks):
                u0 = pr[uint32(c)]
                u1 = pi[uint32(c)]
                u2 = vr[uint32(c)]
                u3 = vi[uint32(c)]
                if c <= r:
                    P0[uint32(int32(rbr + c))] -= g0 * u0 + g1 * u1 + g2 * u2 + g3 * u3
                if r2 > r and c <= r2:
                    P0[uint32(int32(rbr2 + c))] -= e0 * u0 + e1 * u1 + e2 * u2 + e3 * u3
                c = int32(c + _I32)
        cuda.syncthreads()
        for i in range(tid, n, nt):
            f0 = pr[uint32(i)]
            f1 = pi[uint32(i)]
            f2 = vr[uint32(i)]
            f3 = vi[uint32(i)]
            g0 = f0 * q00 + f1 * q10 + f2 * q20 + f3 * q30
            g1 = f0 * q10 + f1 * q11 + f2 * q21 + f3 * q31
            g2 = f0 * q20 + f1 * q21 + f2 * q22 + f3 * q32
            g3 = f0 * q30 + f1 * q31 + f2 * q32 + f3 * q33
            if i > k3:
                P0[_tri(i, k)] = g0
                P0[_tri(i, k1)] = g1
                P0[_tri(i, k2)] = g2
                P0[_tri(i, k3)] = g3
            elif i < k:
                P0[_tri(k, i)] = g0
                P0[_tri(k1, i)] = g1
                P0[_tri(k2, i)] = g2
                P0[_tri(k3, i)] = g3
            elif i == k:
                P0[_tri(k, k)] = -q00
            elif i == k1:
                P0[_tri(k1, k)] = -q10
                P0[_tri(k1, k1)] = -q11
            elif i == k2:
                P0[_tri(k2, k)] = -q20
                P0[_tri(k2, k1)] = -q21
                P0[_tri(k2, k2)] = -q22
            else:
                P0[_tri(k3, k)] = -q30
                P0[_tri(k3, k1)] = -q31
                P0[_tri(k3, k2)] = -q32
                P0[_tri(k3, k3)] = -q33
        cuda.syncthreads()
        k = int32(k + _I3 + _I1)
    while int32(k + _I1) < n:                              # two pivots k, k + 1 per step (2 x 2 block sweep)
        k1 = int32(k + _I1)
        for i in range(tid, n, nt):
            pr[uint32(i)] = P0[_tri(i, k)] if i >= k else P0[_tri(k, i)]
            pi[uint32(i)] = P0[_tri(i, k1)] if i >= k1 else P0[_tri(k1, i)]
        cuda.syncthreads()
        akk = pr[uint32(k)]
        ak1 = pr[uint32(k1)]
        a11 = pi[uint32(k1)]
        idet = _F1 / (akk * a11 - ak1 * ak1)
        p11 = a11 * idet
        p22 = akk * idet
        p12 = -ak1 * idet
        # generic update a_ij -= g1_i a_kj + g2_i a_k1,j with (g1, g2)_i = (a_ik, a_ik1) P^-1 on the whole triangle
        # (rows and columns k, k1 are overwritten below)
        for j in range(ty0, half, nty0):
            r = j
            r2 = int32(n - _I1 - j)
            f1 = pr[uint32(r)]
            f2 = pi[uint32(r)]
            g1 = f1 * p11 + f2 * p12
            g2 = f1 * p12 + f2 * p22
            h1 = pr[uint32(r2)]
            h2 = pi[uint32(r2)]
            e1 = h1 * p11 + h2 * p12
            e2 = h1 * p12 + h2 * p22
            rbr = _rowbase(r)
            rbr2 = _rowbase(r2)
            c = tx0
            for q in range(nchunks):
                uc = pr[uint32(c)]
                wc = pi[uint32(c)]
                if c <= r:
                    P0[uint32(int32(rbr + c))] -= g1 * uc + g2 * wc
                if r2 > r and c <= r2:
                    P0[uint32(int32(rbr2 + c))] -= e1 * uc + e2 * wc
                c = int32(c + _I32)
        cuda.syncthreads()
        for i in range(tid, n, nt):
            f1 = pr[uint32(i)]
            f2 = pi[uint32(i)]
            g1 = f1 * p11 + f2 * p12
            g2 = f1 * p12 + f2 * p22
            if i > k1:
                P0[_tri(i, k)] = g1
                P0[_tri(i, k1)] = g2
            elif i < k:
                P0[_tri(k, i)] = g1
                P0[_tri(k1, i)] = g2
            elif i == k:
                P0[_tri(k, k)] = -p11
                P0[_tri(k1, k)] = -p12
            else:
                P0[_tri(k1, k1)] = -p22
        cuda.syncthreads()
        k = int32(k + _I2)
    if k < n:                                               # odd n: the last pivot alone
        for i in range(tid, n, nt):
            pr[uint32(i)] = P0[_tri(i, k)] if i >= k else P0[_tri(k, i)]
        cuda.syncthreads()
        dinv = _F1 / pr[uint32(k)]
        for j in range(ty0, half, nty0):
            r = j
            r2 = int32(n - _I1 - j)
            f = pr[uint32(r)] * dinv
            f2 = pr[uint32(r2)] * dinv
            rbr = _rowbase(r)
            rbr2 = _rowbase(r2)
            c = tx0
            for q in range(nchunks):
                if c <= r:
                    P0[uint32(int32(rbr + c))] -= f * pr[uint32(c)]
                if r2 > r and c <= r2:
                    P0[uint32(int32(rbr2 + c))] -= f2 * pr[uint32(c)]
                c = int32(c + _I32)
        cuda.syncthreads()
        for i in range(tid, n, nt):
            if i > k:
                P0[_tri(i, k)] = pr[uint32(i)] * dinv
            elif i < k:
                P0[_tri(k, i)] = pr[uint32(i)] * dinv
            else:
                P0[_tri(k, k)] = -dinv
        cuda.syncthreads()

    # 3. smallest eigenpair of A = |coh_beta|^-1 o coh_beta
    for j in range(ty0, half, nty0):
        r = j
        r2 = int32(n - _I1 - j)
        c = tx0
        for q in range(nchunks):
            if c <= r:
                gr, gi = _coh_at(coh, pt, n, r, c, s)
                i = _tri(r, c)
                w = -P0[i]
                P1[i] = w * gr
                P0[i] = w * gi
            if r2 > r and c <= r2:
                gr, gi = _coh_at(coh, pt, n, r2, c, s)
                i = _tri(r2, c)
                w = -P0[i]
                P1[i] = w * gr
                P0[i] = w * gi
            c = int32(c + _I32)
    cuda.syncthreads()
    _tridiag(P1, P0, True, n, d, er, ei, vr, vi, pr, pi, red, tid, nt)
    for i in range(tid, int32(n - _I1), nt):
        b2[uint32(i)] = er[uint32(i)] * er[uint32(i)] + ei[uint32(i)] * ei[uint32(i)]
    cuda.syncthreads()
    lo, hi = _gershgorin(d, b2, n)
    lam1 = _eig_index(d, b2, n, _I0, lo, hi, cnt_sh, x_sh, tid, nt)
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
            for i in range(int32(n - _I1)):
                if abs(dd[uint32(i)]) >= abs(dl[uint32(i)]):
                    if abs(dd[uint32(i)]) < eps:
                        dd[uint32(i)] = eps
                    f = fast_fdividef(dl[uint32(i)], dd[uint32(i)])
                    dd[uint32(i + 1)] -= f * du[uint32(i)]
                    y[uint32(i + 1)] -= f * y[uint32(i)]
                else:
                    f = fast_fdividef(dd[uint32(i)], dl[uint32(i)])
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
            y[uint32(n - 1)] = fast_fdividef(y[uint32(n - 1)], dd[uint32(n - 1)])
            y[uint32(n - 2)] = fast_fdividef(y[uint32(n - 2)] - du[uint32(n - 2)] * y[uint32(n - 1)], dd[uint32(n - 2)])
            for i in range(int32(n - _I3), _IM1, _IM1):
                y[uint32(i)] = fast_fdividef(y[uint32(i)] - du[uint32(i)] * y[uint32(i + 1)] - du2[uint32(i)] * y[uint32(i + 2)],
                                             dd[uint32(i)])
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
    for k in range(int32(n - _I3), _IM1, _IM1):
        m0 = int32(k + _I1)
        acc_r = _F0
        acc_i = _F0
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
                a = P1[_tri(r, k)]
                b = P0[_tri(r, k)]
                acc_r += a * zr[uint32(r)] + b * zi[uint32(r)]
                acc_i += a * zi[uint32(r)] - b * zr[uint32(r)]
        sr, si = _block_sum2(acc_r, acc_i, red, tid, nt)
        for q in range(2):
            r = int32(m0 + tid + q * nt)
            if r < n:
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


def _shared_bytes(n_images, nt):
    """dynamic shared memory of a block: two packed planes, ten vectors, the multisection grid"""
    return (n_images * (n_images + 1) + 10 * n_images + 2 * nt) * 4


_max_dynamic_shared = {}          # device id -> dynamic shared memory per block the kernel is allowed to use


def _allow_dynamic_shared():
    """Let the kernel use all shared memory of the current device beyond 48 kB; returns the bytes it may use.
    Kernels may use more than 48 kB only after an explicit opt-in (cuFuncSetAttribute); numba has no public API for
    it. If the opt-in fails (e.g. the CUDA target built into numba instead of numba-cuda), 48 kB."""
    dev = cp.cuda.Device()
    if dev.id in _max_dynamic_shared:
        return _max_dynamic_shared[dev.id]
    _compile()
    limit = 48 * 1024 - _STATIC_SHARED_BYTES
    want = dev.attributes['MaxSharedMemoryPerBlockOptin'] - _STATIC_SHARED_BYTES
    try:
        from numba.cuda.cudadrv.driver import binding, driver
        attr = binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        for ov in _emi_kernel.overloads.values():
            f = ov._codelibrary.get_cufunc()
            driver.cuKernelSetAttribute(attr, want, f.handle, f.device.id)
        limit = want
    except Exception:
        pass
    _max_dynamic_shared[dev.id] = limit
    return limit


_block_sizes = {}                 # (device id, n_images) -> threads per block, 0 when no block size fits


def _block_size(n_images):
    """Threads per block for n_images: of 128, 256 and 512 the size whose blocks fit the most times into a
    multiprocessor (shared memory and registers; blocks of one point do not wait for each other), the largest at a
    tie; 0 when no block fits. Measured on an A100 (2026-10-08): 4 blocks of 128 threads beat 3 of 256, 2 of 128
    beat 1 of 256, and 1 block of 512 beats 1 of 256."""
    dev = cp.cuda.Device()
    key = (dev.id, n_images)
    if key not in _block_sizes:
        limit = _allow_dynamic_shared()
        f = next(iter(_emi_kernel.overloads.values()))._codelibrary.get_cufunc()
        attrs = dev.attributes
        per_sm = attrs['MaxSharedMemoryPerMultiprocessor']
        reserved = attrs.get('ReservedSharedMemoryPerBlock', 1024)
        regs_sm = attrs.get('MaxRegistersPerMultiprocessor', 65536)
        best = (0, 0)
        for nt in _BLOCK_SIZES:
            dynamic = _shared_bytes(n_images, nt)
            if dynamic > limit:
                continue
            blocks = min(per_sm // (dynamic + f.attrs.shared + reserved), regs_sm // (f.attrs.regs * nt),
                         attrs['MaxThreadsPerMultiProcessor'] // nt)
            if blocks >= best[0]:
                best = (blocks, nt)
        _block_sizes[key] = best[1] if best[0] > 0 else 0
    return _block_sizes[key]


def emi_cuda_supported(n_images):
    """whether `emi_cuda` can process `n_images` images on the current device"""
    return n_images >= 3 and _block_size(n_images) > 0


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
    nt = _block_size(n_images)
    _emi_kernel[n_points, nt, 0, _shared_bytes(n_images, nt)](coh, np.int32(n_images), np.int32(ref), ph, quality)
    return ph, quality
