"""Small dense linear algebra of the phase linking on the CPU, without LAPACK: Householder tridiagonalization,
eigenvalues by Sturm multisection, eigenvectors by inverse iteration, Gauss-Jordan inverse"""

# No BLAS or LAPACK here: called from numba's parallel loops, a threaded library (MKL with its OpenMP runtime)
# oversubscribes the cores and makes any later fork of the process hang (decision 0043). The algorithms are those of
# the GPU kernel (decision 0031), on full matrices (both triangles) whose rows are padded with zero columns to whole
# blocks of 32: the loops over a row then run from a 32 aligned column to the end of the padded row, so that LLVM
# vectorizes them without a scalar remainder (columns left of the trailing block are no longer used). The matrix
# vector products of the tridiagonalization are written as sums of rows (axpy) of the Hermitian matrix, which need no
# reductions. FMA contraction is allowed in the matrix kernels; the Sturm counts use correctly rounded division.

import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir
from ..utils_.jit import mjit, ngjit

F0 = np.float32(0.0)
F1 = np.float32(1.0)
F2 = np.float32(2.0)
HALF = np.float32(0.5)
F_EPS = np.float32(1.2e-7)
LANES = 16                  # shifts of a multisection step (2 vectors of 8)
_fma_jit = mjit(nopython=True, nogil=True, fastmath={'contract'})

@_fma_jit
def _inv_pivot(M, n, ipiv):
    """M = M^-1 in place by Gauss-Jordan elimination with partial pivoting (row interchanges), (n, npad)"""
    npad = M.shape[1]
    for k in range(n):
        p = k
        amax = abs(M[k, k])
        for i in range(k + 1, n):
            if abs(M[i, k]) > amax:
                amax = abs(M[i, k])
                p = i
        ipiv[k] = p
        if p != k:
            for jj in range(npad):
                t = M[k, jj]
                M[k, jj] = M[p, jj]
                M[p, jj] = t
        Mk = M[k]
        dinv = F1/Mk[k]
        Mk[k] = F1
        for jj in range(npad):
            Mk[jj] *= dinv
        for i in range(n):
            if i == k:
                continue
            Mi = M[i]
            f = Mi[k]
            Mi[k] = F0
            for jj in range(npad):
                Mi[jj] -= f*Mk[jj]
    for k in range(n - 1, -1, -1):          # the inverse of the row interchanged matrix: columns back in order
        p = ipiv[k]
        if p != k:
            for i in range(n):
                t = M[i, k]
                M[i, k] = M[i, p]
                M[i, p] = t

@_fma_jit
def _tridiag_real(A, n, d, e, v, p):
    """Householder reduction of the symmetric A (n, npad) to tridiagonal form: diagonal d, subdiagonal e; v, p work
    vectors of npad zeros. The vector of step k (unit norm, H_k = I - 2 v v^T) is kept in row k right of the
    diagonal."""
    for k in range(n - 2):
        m0 = k + 1
        m = n - m0
        Ak = A[k]
        d[k] = Ak[k]
        s2 = F0
        for jj in range(m):                  # x = A[m0:, k] = A[k, m0:]
            x = Ak[m0 + jj]
            s2 += x*x
        xnorm = np.sqrt(s2)
        if xnorm == F0:
            e[k] = F0
            for jj in range(m):
                Ak[m0 + jj] = F0
            continue
        x0 = Ak[m0]
        sign = F1 if x0 >= F0 else -F1
        al = -sign*xnorm
        ivn = F1/np.sqrt(F2*xnorm*(xnorm + abs(x0)))
        for jj in range(m):
            v[m0 + jj] = Ak[m0 + jj]*ivn
        v[m0] = (x0 - al)*ivn
        e[k] = al
        c0 = (m0//32)*32
        w = A.shape[1] - c0
        for jj in range(w):
            p[c0 + jj] = F0
        for c in range(m0, n):               # p = A v: rows of the symmetric block
            vc = v[c]
            Ac = A[c]
            for jj in range(w):
                p[c0 + jj] += Ac[c0 + jj]*vc
        K = F0
        for jj in range(m):
            K += v[m0 + jj]*p[m0 + jj]
        for jj in range(m):
            p[m0 + jj] -= K*v[m0 + jj]
        for r in range(m0, n):               # A -= 2 (v p^T + p v^T)
            vr = F2*v[r]
            pr = F2*p[r]
            Ar = A[r]
            for jj in range(w):
                Ar[c0 + jj] -= vr*p[c0 + jj] + pr*v[c0 + jj]
        for jj in range(m):
            Ak[m0 + jj] = v[m0 + jj]
        for jj in range(m0 - c0):            # the work vectors keep zeros outside the trailing block
            p[c0 + jj] = F0
    d[n - 2] = A[n - 2, n - 2]
    d[n - 1] = A[n - 1, n - 1]
    e[n - 2] = A[n - 1, n - 2]

@_fma_jit
def _tridiag_cplx(Ar, Ai, n, d, er, ei, vr, vi, pr, pi):
    """Householder reduction of the Hermitian Ar + i Ai (n, npad) to tridiagonal form: diagonal d, subdiagonal er + i ei
    (T[k+1, k]); vr, vi, pr, pi work vectors of npad zeros. The vector of step k (unit norm, H_k = I - 2 v v^H) is kept
    in row k right of the diagonal."""
    for k in range(n - 2):
        m0 = k + 1
        m = n - m0
        Ark = Ar[k]
        Aik = Ai[k]
        d[k] = Ark[k]
        s2 = F0
        for jj in range(m):                  # x = A[m0:, k] = conj(A[k, m0:])
            a = Ark[m0 + jj]
            b = Aik[m0 + jj]
            s2 += a*a + b*b
        xnorm = np.sqrt(s2)
        if xnorm == F0:
            er[k] = F0
            ei[k] = F0
            for jj in range(m):
                Ark[m0 + jj] = F0
                Aik[m0 + jj] = F0
            continue
        x0r = Ark[m0]
        x0i = -Aik[m0]
        x0abs = np.sqrt(x0r*x0r + x0i*x0i)
        if x0abs > F0:
            phr = x0r/x0abs
            phi = x0i/x0abs
        else:
            phr = F1
            phi = F0
        alr = -phr*xnorm
        ali = -phi*xnorm
        ivn = F1/np.sqrt(F2*xnorm*(xnorm + x0abs))
        for jj in range(m):
            vr[m0 + jj] = Ark[m0 + jj]*ivn
            vi[m0 + jj] = -Aik[m0 + jj]*ivn
        vr[m0] = (x0r - alr)*ivn
        vi[m0] = (x0i - ali)*ivn
        er[k] = alr
        ei[k] = ali
        c0 = (m0//32)*32
        w = Ar.shape[1] - c0
        for jj in range(w):
            pr[c0 + jj] = F0
            pi[c0 + jj] = F0
        for c in range(m0, n):               # p_r = sum_c conj(A[c, r]) v_c: rows c of the block
            vcr = vr[c]
            vci = vi[c]
            Arc = Ar[c]
            Aic = Ai[c]
            for jj in range(w):
                a = Arc[c0 + jj]
                b = Aic[c0 + jj]
                pr[c0 + jj] += a*vcr + b*vci
                pi[c0 + jj] += a*vci - b*vcr
        K = F0
        for jj in range(m):
            K += vr[m0 + jj]*pr[m0 + jj] + vi[m0 + jj]*pi[m0 + jj]
        for jj in range(m):
            pr[m0 + jj] -= K*vr[m0 + jj]
            pi[m0 + jj] -= K*vi[m0 + jj]
        for r in range(m0, n):               # A -= 2 (v p^H + p v^H)
            a1 = F2*vr[r]
            a2 = F2*vi[r]
            a3 = F2*pr[r]
            a4 = F2*pi[r]
            Arr = Ar[r]
            Air = Ai[r]
            for jj in range(w):
                c = c0 + jj
                Arr[c] -= a1*pr[c] + a2*pi[c] + a3*vr[c] + a4*vi[c]
                Air[c] -= a2*pr[c] - a1*pi[c] + a4*vr[c] - a3*vi[c]
        for jj in range(m):
            Ark[m0 + jj] = vr[m0 + jj]
            Aik[m0 + jj] = vi[m0 + jj]
        for jj in range(m0 - c0):
            pr[c0 + jj] = F0
            pi[c0 + jj] = F0
    d[n - 2] = Ar[n - 2, n - 2]
    d[n - 1] = Ar[n - 1, n - 1]
    er[n - 2] = Ar[n - 1, n - 2]
    ei[n - 2] = Ai[n - 1, n - 2]

@intrinsic
def _sturm16(typingctx, d, b2, n, x, cnt):
    """cnt[l] (float32) = number of eigenvalues < x[l] of the symmetric tridiagonal matrix (diagonal d, squared
    subdiagonal b2) for the 16 shifts x[l]: the negative pivots of the LDL^T factorization of T - x I. A pivot smaller
    than pivmin in magnitude is replaced by -pivmin before it is counted, as on the GPU (LAPACK dlaebz)."""
    sig = types.void(d, b2, n, x, cnt)

    def codegen(context, builder, signature, args):
        f32 = ir.FloatType(); v8 = ir.VectorType(f32, 8); i32 = ir.IntType(32); i64 = ir.IntType(64)

        def data(i):
            return context.make_array(signature.args[i])(context, builder, args[i]).data
        d_p, b2_p, x_p, cnt_p = data(0), data(1), data(3), data(4)
        n_v = context.cast(builder, args[2], signature.args[2], types.int64)
        fabs = cgutils.get_or_insert_function(builder.module, ir.FunctionType(v8, [v8]), 'llvm.fabs.v8f32')
        undef = ir.Constant(v8, ir.Undefined)
        mask0 = ir.Constant(ir.VectorType(i32, 8), [ir.Constant(i32, 0)]*8)

        def splat(s):
            return builder.shuffle_vector(builder.insert_element(undef, s, ir.Constant(i32, 0)), undef, mask0)

        def const(c):
            return ir.Constant(v8, [ir.Constant(f32, c)]*8)
        pivmin, npivmin, one, zero = const(1e-30), const(-1e-30), const(1.0), const(0.0)

        def vector_at(ptr, offset):
            return builder.bitcast(builder.gep(ptr, [ir.Constant(i64, offset)]), v8.as_pointer())

        def guarded(q):
            return builder.select(builder.fcmp_ordered('<', builder.call(fabs, [q]), pivmin), npivmin, q)

        def negative(q):
            return builder.select(builder.fcmp_ordered('<', q, zero), one, zero)
        xs = [builder.load(vector_at(x_p, 0), align=4), builder.load(vector_at(x_p, 8), align=4)]
        d0 = splat(builder.load(d_p))
        qs = [cgutils.alloca_once_value(builder, guarded(builder.fsub(d0, xv))) for xv in xs]   # in registers
        cs = [cgutils.alloca_once_value(builder, negative(builder.load(q))) for q in qs]
        with cgutils.for_range(builder, builder.sub(n_v, ir.Constant(i64, 1))) as loop:
            di = splat(builder.load(builder.gep(d_p, [builder.add(loop.index, ir.Constant(i64, 1))])))
            bi = splat(builder.load(builder.gep(b2_p, [loop.index])))
            for l in range(2):
                q = guarded(builder.fsub(builder.fsub(di, xs[l]), builder.fdiv(bi, builder.load(qs[l]))))
                builder.store(q, qs[l])
                builder.store(builder.fadd(builder.load(cs[l]), negative(q)), cs[l])
        for l in range(2):
            builder.store(builder.load(cs[l]), vector_at(cnt_p, 8*l), align=4)
        return context.get_dummy_value()

    return sig, codegen

@ngjit
def _gershgorin(d, b2, n):
    """interval of the eigenvalues of the symmetric tridiagonal matrix, slightly widened"""
    lo = np.float32(3e38)
    hi = np.float32(-3e38)
    for i in range(n):
        r = F0
        if i > 0:
            r += np.sqrt(b2[i - 1])
        if i < n - 1:
            r += np.sqrt(b2[i])
        lo = min(lo, d[i] - r)
        hi = max(hi, d[i] + r)
    return lo - np.float32(1e-5)*(abs(lo) + F1), hi + np.float32(1e-5)*(abs(hi) + F1)

@ngjit
def _eig_index(d, b2, n, j, lo, hi, x, cnt):
    """j-th smallest eigenvalue (0 based) of the symmetric tridiagonal matrix by multisection over 16 shifts per
    step, until the interval does not shrink any more; x, cnt work vectors of 16"""
    for it in range(12):
        w = hi - lo
        for l in range(LANES):
            x[l] = lo + w*np.float32(l + 1)/np.float32(LANES + 1)
        _sturm16(d, b2, n, x, cnt)
        a = 0
        while a < LANES and cnt[a] <= np.float32(j):
            a += 1
        new_lo = x[a - 1] if a > 0 else lo
        new_hi = x[a] if a < LANES else hi
        if new_lo == lo and new_hi == hi:
            break
        lo = new_lo
        hi = new_hi
    return HALF*(lo + hi)

@ngjit
def _inverse_iteration(d, b2, n, lam, lo, hi, y, dd, du, du2, dl):
    """y = eigenvector (unit norm) of the real tridiagonal matrix tridiag(sqrt(b2), d, sqrt(b2)) for the eigenvalue
    lam in [lo, hi]: two steps of inverse iteration with the LU factorization of T - lam I with partial pivoting, as on
    the GPU (LAPACK gtsv)"""
    for i in range(n):
        y[i] = F1
    eps = F_EPS*(abs(hi) + abs(lo) + F1)
    for it in range(2):
        for i in range(n):
            dd[i] = d[i] - lam
            if i < n - 1:
                du[i] = np.sqrt(b2[i])
                dl[i] = du[i]
            du2[i] = F0
        for i in range(n - 1):
            if abs(dd[i]) >= abs(dl[i]):
                if abs(dd[i]) < eps:
                    dd[i] = eps
                f = dl[i]/dd[i]
                dd[i + 1] -= f*du[i]
                y[i + 1] -= f*y[i]
            else:
                f = dd[i]/dl[i]
                dd[i] = dl[i]
                t = dd[i + 1]
                dd[i + 1] = du[i] - f*t
                if i < n - 2:
                    du2[i] = du[i + 1]
                    du[i + 1] = -f*du2[i]
                du[i] = t
                t = y[i]
                y[i] = y[i + 1]
                y[i + 1] = t - f*y[i + 1]
        if abs(dd[n - 1]) < eps:
            dd[n - 1] = eps
        y[n - 1] = y[n - 1]/dd[n - 1]
        y[n - 2] = (y[n - 2] - du[n - 2]*y[n - 1])/dd[n - 2]
        for i in range(n - 3, -1, -1):
            y[i] = (y[i] - du[i]*y[i + 1] - du2[i]*y[i + 2])/dd[i]
        nrm = F0
        for i in range(n):
            nrm += y[i]*y[i]
        nrm = F1/np.sqrt(nrm)
        for i in range(n):
            y[i] *= nrm

@ngjit
def _eigenvector_cplx(Ar, Ai, n, er, ei, b2, y, zr, zi):
    """z = eigenvector of the Hermitian matrix reduced by `_tridiag_cplx` from the eigenvector y of the real
    tridiagonal matrix: z = H_0 ... H_{n-3} D y, D = diag(delta), delta_0 = 1, delta_{k+1} = delta_k e_k/|e_k|"""
    dr = F1
    di = F0
    for k in range(n):
        zr[k] = dr*y[k]
        zi[k] = di*y[k]
        if k < n - 1:
            ab = np.sqrt(b2[k])
            if ab > F0:
                nr = (dr*er[k] - di*ei[k])/ab
                ni = (dr*ei[k] + di*er[k])/ab
                dr = nr
                di = ni
    for k in range(n - 3, -1, -1):
        m0 = k + 1
        Ark = Ar[k]
        Aik = Ai[k]
        sr = F0
        si = F0
        for jj in range(n - m0):             # s = v^H z
            r = m0 + jj
            a = Ark[r]
            b = Aik[r]
            sr += a*zr[r] + b*zi[r]
            si += a*zi[r] - b*zr[r]
        for jj in range(n - m0):             # z -= 2 v s
            r = m0 + jj
            a = Ark[r]
            b = Aik[r]
            zr[r] -= F2*(a*sr - b*si)
            zi[r] -= F2*(a*si + b*sr)
