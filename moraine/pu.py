"""phase unwrapping"""


__all__ = ['gamma_mcf_pt', 'mcf_pc', 'emcf_pc']

import numpy as np
import tempfile
from pathlib import Path
import os
from numba import njit, prange

from .gamma_ import read_gamma_pdata, read_gamma_plist, write_gamma_image, write_gamma_plist
from .delaunay_ import delaunay_halfedges

def gamma_mcf_pt(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    ph_weight:np.ndarray=None,
    ref_point:int=0,
) -> np.ndarray:
    """A simple wrapper for mcf_pt in GAMMA software, only work if you have access to mcf_pt.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape of (N,)
    pc_y : np.ndarray
        y coordinate, shape of (N,)
    ph : np.ndarray
        wrapped phase, shape of (N,) or (N,M)
    ph_weight : np.ndarray, optional
        point weight, shape of (N,) or (N,M), optional
    ref_point : int, default: 0
        index of the reference point (from 0), the first point by default

    Returns
    -------
    np.ndarray
        unwrapped phase, shape of (N,) or (N,M)
    """
    pc_x = pc_x.astype(np.int32)
    pc_y = pc_y.astype(np.int32)
    pc_xy = np.stack((pc_x,pc_y),axis=-1)
    ph = ph.astype(np.complex64)

    with tempfile.TemporaryDirectory() as tempdir_str:
        temp_dir = Path(tempdir_str)
        pc_path = temp_dir/'pc'
        ph_path = temp_dir/'ph'
        unwrap_ph_path = temp_dir/'unwrap_ph'
        write_gamma_plist(pc_xy,pc_path)
        write_gamma_image(ph,ph_path)
        if ph_weight is None:
            ph_weight_path = '-'
        else:
            ph_weight_path = temp_dir/'ph_weight'
            ph_weight = ph_weight.astype(np.float32)   # mcf_pt reads FLOAT
            write_gamma_image(ph_weight,ph_weight_path)

        mcf_pt_command = f'mcf_pt {str(pc_path)} - {str(ph_path)} - {str(ph_weight_path)} - {str(unwrap_ph_path)} - - {ref_point} &> {temp_dir/"gamma.log"}'
        os.system(mcf_pt_command)

        unwrap_ph = read_gamma_pdata(unwrap_ph_path,dtype='float')
        unwrap_ph = unwrap_ph.reshape(ph.shape)
    return unwrap_ph

# ---------------------------------------------------------------- minimum cost flow unwrapping
# Everything works on the half-edges of the Delaunay triangulation (moraine.delaunay_): `tri` (3T,) start
# vertex of every half-edge, `half` (3T,) its twin or -1 on the convex hull. The dual graph of the MCF
# problem is read from them directly: node e // 3 is the triangle of half-edge e, half[e] // 3 the
# triangle across it, node T (the earth) is behind every hull half-edge.

_TWO_PI = 2 * np.pi


@njit(cache=True)
def _hull_halfedges(half):
    n = 0
    for e in range(half.shape[0]):
        n += half[e] < 0
    hull = np.empty(n, np.int32)
    n = 0
    for e in range(half.shape[0]):
        if half[e] < 0:
            hull[n] = e
            n += 1
    return hull


def _mcf_network(x, y):
    """Triangulation of the points and the hull half-edges; shared by all interferograms."""
    tri, half = delaunay_halfedges(x, y)
    return tri, half, _hull_halfedges(half)


@njit(cache=True, nogil=True)
def _mcf_residues(psi, tri):
    """Residue of every triangle (sum of wrapped phase differences along its half-edges / 2 pi); the
    last element is the earth, which balances the total."""
    T = tri.shape[0] // 3
    r = np.empty(T + 1, np.int32)
    tot = 0
    for t in range(T):
        s = 0.0
        for j in range(3):
            d = np.float64(psi[tri[3 * t + (j + 1) % 3]]) - np.float64(psi[tri[3 * t + j]])
            s += (d + np.pi) % _TWO_PI - np.pi
        k = int(np.round(s / _TWO_PI))
        r[t] = k
        tot += k
    r[T] = -tot
    return r


@njit(cache=True, nogil=True)
def _grow(a):
    b = np.empty(a.shape[0] * 2, a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True, nogil=True)
def _mcf_ssp(tri, half, hull, supply, earth_cost, cost=None):
    """Min cost flow by successive shortest paths.

    From every node with positive excess, Dijkstra on reduced costs (node potentials keep them non
    negative) until the nearest node with negative excess, then one unit is pushed along the path. The
    search stops at the first sink, so it stays local when residues pair up with close neighbours.
    Arc cost: `cost[e]` (1 if None) across half-edge e between triangles, times `earth_cost` to the earth;
    unlimited capacity. Returns the flow per
    half-edge, leaving triangle e // 3 across e (antisymmetric on twins).
    """
    T = tri.shape[0] // 3
    EARTH = T
    n_nodes = T + 1
    f = np.zeros(half.shape[0], np.int32)
    pi = np.zeros(n_nodes, np.int64)
    excess = supply.astype(np.int64)
    INF = np.int64(1) << 62
    dist = np.full(n_nodes, INF, np.int64)
    settled = np.zeros(n_nodes, np.bool_)
    pred = np.zeros(n_nodes, np.int64)      # e >= 0: from e // 3 across e; -(e + 1): from the earth across hull e
    touched = np.empty(n_nodes, np.int32)
    hd = np.empty(1024, np.int64)           # binary heap (distance, node); ties by node for determinism
    hv = np.empty(1024, np.int32)
    for s in range(n_nodes):
        while excess[s] > 0:
            nt = 1
            touched[0] = s
            dist[s] = 0
            hd[0] = 0
            hv[0] = s
            hn = 1
            t = -1
            D = np.int64(0)
            while hn > 0:
                d = hd[0]
                u = hv[0]
                hn -= 1
                hd[0] = hd[hn]
                hv[0] = hv[hn]
                i = 0
                while True:
                    l = 2 * i + 1
                    if l >= hn:
                        break
                    c = l
                    if l + 1 < hn and (hd[l + 1] < hd[l] or (hd[l + 1] == hd[l] and hv[l + 1] < hv[l])):
                        c = l + 1
                    if hd[i] < hd[c] or (hd[i] == hd[c] and hv[i] <= hv[c]):
                        break
                    hd[c], hd[i] = hd[i], hd[c]
                    hv[c], hv[i] = hv[i], hv[c]
                    i = c
                if settled[u] or d > dist[u]:
                    continue
                settled[u] = True
                if excess[u] < 0:
                    t = u
                    D = d
                    break
                kmax = 3 if u != EARTH else hull.shape[0]
                for k in range(kmax):
                    if u != EARTH:
                        e = 3 * u + k
                        h = half[e]
                        if h >= 0:
                            v = h // 3
                            c = 1 if cost is None else cost[e]
                            mc = c if f[e] >= 0 else -c
                        else:
                            v = EARTH
                            c = earth_cost if cost is None else earth_cost * cost[e]
                            mc = c if f[e] >= 0 else -c
                        code = e
                    else:
                        e = hull[k]
                        v = e // 3
                        c = earth_cost if cost is None else earth_cost * cost[e]
                        mc = c if f[e] <= 0 else -c
                        code = -(e + 1)
                    if settled[v]:
                        continue
                    nd = d + mc + pi[u] - pi[v]
                    if nd < dist[v]:
                        if dist[v] == INF:
                            touched[nt] = v
                            nt += 1
                        dist[v] = nd
                        pred[v] = code
                        if hn >= hd.shape[0]:
                            hd = _grow(hd)
                            hv = _grow(hv)
                        j = hn
                        hd[j] = nd
                        hv[j] = v
                        hn += 1
                        while j > 0:
                            p = (j - 1) >> 1
                            if hd[p] < hd[j] or (hd[p] == hd[j] and hv[p] <= hv[j]):
                                break
                            hd[p], hd[j] = hd[j], hd[p]
                            hv[p], hv[j] = hv[j], hv[p]
                            j = p
            if t < 0:
                raise RuntimeError('mcf: no sink reachable')
            for i in range(nt):     # potentials += min(dist, D) - D; a uniform shift does not matter
                v = touched[i]
                if settled[v] and dist[v] < D:
                    pi[v] += dist[v] - D
            v = t
            while v != s:
                code = pred[v]
                if code >= 0:
                    f[code] += 1
                    h = half[code]
                    if h >= 0:
                        f[h] -= 1
                    v = code // 3
                else:
                    f[-code - 1] -= 1
                    v = EARTH
            excess[s] -= 1
            excess[t] += 1
            for i in range(nt):
                v = touched[i]
                dist[v] = INF
                settled[v] = False
    return f


@njit(cache=True, nogil=True)
def _mcf_integrate(psi, tri, half, f, n_points):
    """Breadth first search over the triangles from the first point; the unwrapped gradient along
    half-edge e is wrap(psi[b] - psi[a]) - 2 pi f[e]. The first point keeps its wrapped phase."""
    T = tri.shape[0] // 3
    unw = np.empty(n_points, np.float64)
    done = np.zeros(n_points, np.bool_)
    seen = np.zeros(T, np.bool_)
    t0 = 0
    for e in range(tri.shape[0]):
        if tri[e] == 0:
            t0 = e // 3
            break
    unw[0] = psi[0]
    done[0] = True
    queue = np.empty(T, np.int32)
    queue[0] = t0
    seen[t0] = True
    qh = 0
    qt = 1
    while qh < qt:
        t = queue[qh]
        qh += 1
        for rep in range(2):
            for j in range(3):
                e = 3 * t + j
                a = tri[e]
                b = tri[3 * t + (j + 1) % 3]
                if done[a] and not done[b]:
                    d = np.float64(psi[b]) - np.float64(psi[a])
                    unw[b] = unw[a] + (d + np.pi) % _TWO_PI - np.pi - _TWO_PI * f[e]
                    done[b] = True
        for j in range(3):
            h = half[3 * t + j]
            if h >= 0 and not seen[h // 3]:
                seen[h // 3] = True
                queue[qt] = h // 3
                qt += 1
    return unw


@njit(cache=True, nogil=True)
def _mcf_solve(ph, tri, half, hull, earth_cost):
    psi = np.empty(ph.shape[0], np.float32)
    for i in range(ph.shape[0]):
        psi[i] = np.arctan2(ph[i].imag, ph[i].real)
    f = _mcf_ssp(tri, half, hull, _mcf_residues(psi, tri), earth_cost)
    return _mcf_integrate(psi, tri, half, f, psi.shape[0])


def _mcf_unwrap(ph, tri, half, hull, earth_cost=1):
    """Unwrap one interferogram on a network made by `_mcf_network`."""
    return _mcf_solve(np.ascontiguousarray(ph), tri, half, hull, int(earth_cost))


def mcf_pc(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    earth_cost:int=1,
)-> np.ndarray:
    """Minimum cost flow phase unwrapping of a point cloud interferogram.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape (N,); the coordinates must be unique and not all collinear
    pc_y : np.ndarray
        y coordinate, shape (N,)
    ph : np.ndarray
        wrapped phase (complex), shape (N,)
    earth_cost : int, default: 1
        cost of a phase jump across the convex hull of the points, relative to 1 inside; a larger value
        discourages discharging residues through the border

    Returns
    -------
    np.ndarray
        unwrapped phase, shape (N,), np.float64; the first point keeps its wrapped phase
    """
    # Delaunay triangulation, residues of the triangles, successive shortest path min cost flow between
    # them, integration from the first point; exactly optimal, independent of the point order (ADR 0013)
    return _mcf_unwrap(ph, *_mcf_network(pc_x, pc_y), earth_cost)


# ---------------------------------------------------------------- extended minimum cost flow (EMCF)
# Temporal step: for every spatial edge (p, q) of the point triangulation, the spatial gradients of all
# interferograms are corrected by a min cost flow on the triangulation of the images in the
# (time, perpendicular baseline) plane, so that they close around every temporal triangle. Spatial step:
# every interferogram is unwrapped with the corrected gradients by the spatial min cost flow of `mcf_pc`.
# Both flows use `_mcf_ssp` on half-edges; the value on a half-edge a -> b is the gradient from a to b.


def _temporal_network(t, bperp, t_scale=None, bperp_scale=None):
    """Delaunay triangulation of the images in the (time, perpendicular baseline) plane, each axis divided
    by its scale (default: its range). Returns (tri, half, hull, pairs, pair_of_half, sign_of_half):
    image pairs (reference < secondary) sorted, and for every half-edge the pair it belongs to and +1 if
    it goes from the reference to the secondary image, else -1."""
    t = np.asarray(t, dtype=np.float64)
    b = np.asarray(bperp, dtype=np.float64)
    ts = t_scale or np.ptp(t)
    bs = bperp_scale or np.ptp(b)
    if not ts or not bs:
        raise ValueError('the images do not span a plane of time and perpendicular baseline; '
                         'EMCF needs a two dimensional network')
    try:
        tri, half = delaunay_halfedges(t / ts, b / bs)
    except ValueError as e:
        raise ValueError(f'no network of the images in time and perpendicular baseline: {e}') from None
    e = np.arange(tri.shape[0])
    nxt = e - e % 3 + (e + 1) % 3
    a, c = tri, tri[nxt]
    ref, sec = np.minimum(a, c), np.maximum(a, c)
    key = ref.astype(np.int64) * len(t) + sec
    uniq, pair_of_half = np.unique(key, return_inverse=True)
    pairs = np.stack(np.divmod(uniq, len(t)), -1).astype(np.int32)
    sign_of_half = np.where(a == ref, 1, -1).astype(np.int8)
    return tri, half, _hull_halfedges(half), pairs, pair_of_half.astype(np.int32), sign_of_half


def _temporal_costs(t, bperp, pairs, t_scale, bperp_scale, mode):
    """(cost per pair, adaptive): 'constant'; 'length': shorter interferograms in the normalized
    (time, baseline) plane cost more to correct, relative to the median length, 1..20; 'gradient': constant
    per pair, adapted per spatial edge to the gradient; 'length+gradient': both."""
    t = np.asarray(t, float)
    b = np.asarray(bperp, float)
    ts = t_scale or np.ptp(t)
    bs = bperp_scale or np.ptp(b)
    if mode not in ('constant', 'length', 'gradient', 'length+gradient'):
        raise ValueError(f'unknown temporal_cost {mode!r}')
    if 'length' in mode:
        L = np.hypot((t[pairs[:, 1]] - t[pairs[:, 0]]) / ts, (b[pairs[:, 1]] - b[pairs[:, 0]]) / bs)
        cost = np.clip(np.round(4 * np.median(L) / L), 1, 20).astype(np.int64)
    else:
        cost = np.ones(pairs.shape[0], np.int64)
    return cost, 'gradient' in mode


def _spatial_edges(tri, half):
    """One half-edge per spatial edge, and for every half-edge its edge and +1 if it has the edge's
    direction, else -1."""
    e = np.arange(tri.shape[0])
    rep = np.flatnonzero((half < 0) | (e < half)).astype(np.int32)
    edge_of_half = np.empty(tri.shape[0], np.int32)
    sign_of_half = np.empty(tri.shape[0], np.int8)
    edge_of_half[rep] = np.arange(rep.shape[0])
    sign_of_half[rep] = 1
    twin = half[rep]
    has = twin >= 0
    edge_of_half[twin[has]] = np.flatnonzero(has)
    sign_of_half[twin[has]] = -1
    return rep, edge_of_half, sign_of_half


@njit(cache=True, nogil=True)
def _pair_gradients(ph, p, q, pairs, g):
    """Wrapped gradient from point p to point q of every interferogram, into g."""
    for k in range(pairs.shape[0]):
        a = ph[q, pairs[k, 0]] * np.conj(ph[p, pairs[k, 0]])
        b = ph[q, pairs[k, 1]] * np.conj(ph[p, pairs[k, 1]])
        z = a * np.conj(b)
        g[k] = np.arctan2(z.imag, z.real)


@njit(cache=True, parallel=True)
def _emcf_temporal(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign, earth_cost, pair_cost, adaptive):
    """Integer correction (number of cycles) of every interferogram on every spatial edge, (n_pairs, n_edges).

    Correcting interferogram k costs pair_cost[k]; if `adaptive`, times 1 + 9 * (1 - |gradient| / pi), so that
    gradients close to +-pi (the least reliable) are the cheapest to correct."""
    n_edges = s_rep.shape[0]
    n_pairs = pairs.shape[0]
    Tt = t_tri.shape[0] // 3
    cycles = np.zeros((n_pairs, n_edges), np.int8)
    rep_half = np.empty(n_pairs, np.int64)          # a half-edge of every pair
    for e in range(t_tri.shape[0]):
        rep_half[t_pair[e]] = e
    for s in prange(n_edges):
        e0 = s_rep[s]
        p = s_tri[e0]
        q = s_tri[e0 - e0 % 3 + (e0 + 1) % 3]
        g = np.empty(n_pairs, np.float64)
        _pair_gradients(ph, p, q, pairs, g)
        supply = np.empty(Tt + 1, np.int32)
        tot = 0
        for tt in range(Tt):
            v = 0.0
            for j in range(3):
                e = 3 * tt + j
                v += t_sign[e] * g[t_pair[e]]
            r = int(np.round(v / _TWO_PI))
            supply[tt] = r
            tot += r
        supply[Tt] = -tot
        if tot == 0:
            any_residue = False
            for tt in range(Tt):
                if supply[tt] != 0:
                    any_residue = True
                    break
            if not any_residue:
                continue
        hc = np.empty(t_tri.shape[0], np.int64)
        for e in range(t_tri.shape[0]):
            k = t_pair[e]
            c = pair_cost[k]
            if adaptive:
                c = c * (1 + int(np.round(9.0 * (1.0 - abs(g[k]) / np.pi))))
            hc[e] = c
        f = _mcf_ssp(t_tri, t_half, t_hull, supply, earth_cost, hc)
        for k in range(n_pairs):
            e = rep_half[k]
            # corrected value on half-edge e is sign * g - 2 pi f[e]; back to the pair's direction
            cycles[k, s] = -t_sign[e] * f[e]
    return cycles


def _emcf_cycles(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign, earth_cost, pair_cost, adaptive,
                 block=1 << 20):
    """Temporal step in blocks of spatial edges, kept sparse: (ptr, edge, value) with the corrections of pair k
    in edge[ptr[k]:ptr[k + 1]], value[...] (cycles, int8)."""
    ks, ss, vs = [], [], []
    for start in range(0, s_rep.shape[0], block):
        c = _emcf_temporal(ph, s_tri, s_rep[start:start + block], t_tri, t_half, t_hull, pairs, t_pair, t_sign,
                           earth_cost, pair_cost, adaptive)
        k, s = np.nonzero(c)
        ks.append(k)
        ss.append(s + start)
        vs.append(c[k, s])
    k = np.concatenate(ks) if ks else np.empty(0, np.int64)
    s = np.concatenate(ss) if ss else np.empty(0, np.int64)
    v = np.concatenate(vs) if vs else np.empty(0, np.int8)
    order = np.lexsort((s, k))
    k, s, v = k[order], s[order], v[order]
    ptr = np.searchsorted(k, np.arange(pairs.shape[0] + 1)).astype(np.int64)
    return ptr, s.astype(np.int32), v.astype(np.int8)


_SPATIAL_COST_FACTORS = {'gradient': 1, 'correction': 2, 'length': 4, 'weight': 8}
_SPATIAL_COST_RANGE = 99.0       # arc costs of the spatial step from 1 to 1 + this


def _spatial_cost_flags(mode, weight):
    if mode == 'constant':
        return 0
    flags = 0
    for name in mode.split('+'):
        if name not in _SPATIAL_COST_FACTORS:
            raise ValueError(f'unknown spatial_cost {mode!r}: combine {", ".join(_SPATIAL_COST_FACTORS)} with +, '
                             f'or constant')
        flags |= _SPATIAL_COST_FACTORS[name]
    if flags & 8 and weight is None:
        raise ValueError("spatial_cost 'weight' needs the point weights")
    return flags


@njit(cache=True, nogil=True)
def _emcf_spatial(ph, k_ref, k_sec, s_tri, s_half, s_hull, s_rep, edge_of_half, sign_of_half, corr_edge, corr_val,
                  earth_cost, flags, edge_length, weight):
    """Unwrap one interferogram from the temporally corrected gradients of the spatial edges. Returns the
    unwrapped phase and, per point, whether a phase jump of the spatial min cost flow touches it.

    Arc cost of a spatial edge 1 + round(_SPATIAL_COST_RANGE * r), r the product of the reliabilities chosen
    by `flags`:
    1 wrapped gradient (1 - |g| / pi), 2 temporal step (0.1 if it corrected the edge, else 1), 4 length
    (`edge_length`, median / length up to 1), 8 point weights (smaller of the two, 0..1)."""
    n_points = ph.shape[0]
    n_half = s_tri.shape[0]
    n_edges = s_rep.shape[0]
    T = n_half // 3
    G = np.empty(n_edges, np.float64)
    r = np.ones(n_edges, np.float64)
    for s in range(n_edges):
        e0 = s_rep[s]
        p = s_tri[e0]
        q = s_tri[e0 - e0 % 3 + (e0 + 1) % 3]
        a = ph[q, k_ref] * np.conj(ph[p, k_ref])
        b = ph[q, k_sec] * np.conj(ph[p, k_sec])
        z = a * np.conj(b)
        G[s] = np.arctan2(z.imag, z.real)
        if flags & 1:
            r[s] *= 1.0 - abs(G[s]) / np.pi
        if flags & 4:
            r[s] *= edge_length[s]
        if flags & 8:
            r[s] *= min(weight[p], weight[q])
    for i in range(corr_edge.shape[0]):
        G[corr_edge[i]] += _TWO_PI * corr_val[i]
        if flags & 2:
            r[corr_edge[i]] *= 0.1
    supply = np.empty(T + 1, np.int32)
    tot = 0
    for t in range(T):
        v = 0.0
        for j in range(3):
            e = 3 * t + j
            v += sign_of_half[e] * G[edge_of_half[e]]
        rr = int(np.round(v / _TWO_PI))
        supply[t] = rr
        tot += rr
    supply[T] = -tot
    if flags:
        cost = np.empty(n_half, np.int32)
        for e in range(n_half):
            cost[e] = 1 + int(np.round(_SPATIAL_COST_RANGE * min(max(r[edge_of_half[e]], 0.0), 1.0)))
        f = _mcf_ssp(s_tri, s_half, s_hull, supply, earth_cost, cost)
    else:
        f = _mcf_ssp(s_tri, s_half, s_hull, supply, earth_cost)
    touched = np.zeros(n_points, np.bool_)
    for e in range(n_half):
        if f[e] != 0:
            touched[s_tri[e]] = True
            touched[s_tri[e - e % 3 + (e + 1) % 3]] = True
    # integration over the triangles from the first point
    unw = np.empty(n_points, np.float64)
    done = np.zeros(n_points, np.bool_)
    seen = np.zeros(T, np.bool_)
    t0 = 0
    for e in range(n_half):
        if s_tri[e] == 0:
            t0 = e // 3
            break
    # the first point gets the difference of the image phases (not wrapped), so that the interferograms
    # of every temporal triangle already close at the start of the integration
    unw[0] = (np.arctan2(ph[0, k_ref].imag, ph[0, k_ref].real)
              - np.arctan2(ph[0, k_sec].imag, ph[0, k_sec].real))
    done[0] = True
    queue = np.empty(T, np.int32)
    queue[0] = t0
    seen[t0] = True
    qh = 0
    qt = 1
    while qh < qt:
        t = queue[qh]
        qh += 1
        for rep in range(2):
            for j in range(3):
                e = 3 * t + j
                a_ = s_tri[e]
                b_ = s_tri[3 * t + (j + 1) % 3]
                if done[a_] and not done[b_]:
                    unw[b_] = unw[a_] + sign_of_half[e] * G[edge_of_half[e]] - _TWO_PI * f[e]
                    done[b_] = True
        for j in range(3):
            h = s_half[3 * t + j]
            if h >= 0 and not seen[h // 3]:
                seen[h // 3] = True
                queue[qt] = h // 3
                qt += 1
    return unw, touched


@njit(cache=True, parallel=True)
def _emcf_repair(ph, unw, pairs, t_tri, t_half, t_hull, t_pair, t_sign, earth_cost, pair_cost, touched,
                 touched_cost):
    """Make the interferograms of every point close around every temporal triangle, changing as few
    interferograms by as few cycles as possible (min cost flow on the temporal network), in place. Returns
    the number of triangles that did not close at every point.

    With consistent image phases phi, unw[k] = phi_a - phi_b + 2 pi c_k with integer cycles c_k; a triangle
    closes if its cycles do. An interferogram shifted by a cycle in a whole area (spatial unwrapping of
    each interferogram alone) opens the triangles on both sides of it and is the cheapest to change.
    Changing interferogram k costs pair_cost[k], times `touched_cost` unless the spatial step placed a phase
    jump next to the point in it (touched[i, k])."""
    n_points = unw.shape[0]
    n_pairs = pairs.shape[0]
    Tt = t_tri.shape[0] // 3
    rep_half = np.empty(n_pairs, np.int64)
    for e in range(t_tri.shape[0]):
        rep_half[t_pair[e]] = e
    n_open = np.zeros(n_points, np.int16)
    for i in prange(n_points):
        c = np.empty(n_pairs, np.int64)
        for k in range(n_pairs):
            a = np.arctan2(ph[i, pairs[k, 0]].imag, ph[i, pairs[k, 0]].real)
            b = np.arctan2(ph[i, pairs[k, 1]].imag, ph[i, pairs[k, 1]].real)
            c[k] = int(np.round((unw[i, k] - (a - b)) / _TWO_PI))
        supply = np.empty(Tt + 1, np.int32)
        tot = 0
        n = 0
        for tt in range(Tt):
            r = 0
            for j in range(3):
                e = 3 * tt + j
                r += t_sign[e] * c[t_pair[e]]
            supply[tt] = r
            tot += r
            if r != 0:
                n += 1
        n_open[i] = n
        if n == 0:
            continue
        supply[Tt] = -tot
        hc = np.empty(t_tri.shape[0], np.int64)
        for e in range(t_tri.shape[0]):
            k = t_pair[e]
            hc[e] = pair_cost[k] * (1 if touched[i, k] else touched_cost)
        f = _mcf_ssp(t_tri, t_half, t_hull, supply, earth_cost, hc)
        for k in range(n_pairs):
            d = -t_sign[rep_half[k]] * f[rep_half[k]]
            if d != 0:
                unw[i, k] += _TWO_PI * d
    return n_open


def emcf_pc(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    t:np.ndarray,
    bperp:np.ndarray,
    weight:np.ndarray=None,
    earth_cost:int=1,
    t_scale:float=None,
    bperp_scale:float=None,
    temporal_cost:str='length+gradient',
    spatial_cost:str='gradient+correction+length',
    repair:bool=True,
    repair_cost:int=1,
    n_workers:int=None,
    exclude:list=None,
):
    """Extended minimum cost flow (EMCF) phase unwrapping of point cloud interferograms.

    The interferograms are unwrapped together: they are the edges of the Delaunay triangulation of the images
    in the plane of time and perpendicular baseline, and the redundancy between them is used against
    unwrapping errors. At every point the unwrapped phases of every triangle of images add up to zero:
    unw(a, b) + unw(b, c) = unw(a, c).

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape (n_points,), e.g. in meters; the coordinates must be unique and not all collinear
    pc_y : np.ndarray
        y coordinate, shape (n_points,), in the same unit as `pc_x`
    ph : np.ndarray
        wrapped phase history (complex), shape (n_points, nimages)
    t : np.ndarray
        acquisition time of the images, e.g. in days, shape (nimages,)
    bperp : np.ndarray
        perpendicular baseline of the images in meters, shape (nimages,)
    weight : np.ndarray, optional
        quality of the points from 0 (unreliable) to 1, e.g. the temporal coherence, shape (n_points,);
        used by `spatial_cost` 'weight'
    earth_cost : int, default: 1
        cost of a phase jump across the border of the network (of the points and of the images), relative
        to 1 inside; a larger value discourages discharging residues through the border
    t_scale : float, optional
        time distance that counts like `bperp_scale` of perpendicular baseline in the network of the
        images; the time span by default
    bperp_scale : float, optional
        perpendicular baseline distance matching `t_scale`; the baseline span by default
    temporal_cost : str, default: 'length+gradient'
        which interferograms are corrected first where the interferograms of a triangle of images disagree:
        'gradient' (those with the least reliable phase difference between the two points), 'length' (the
        longest in time and baseline), 'length+gradient', or 'constant' (all alike)
    spatial_cost : str, default: 'gradient+correction+length'
        where phase jumps are placed first in every interferogram: 'constant' (anywhere alike) or a
        combination with + of 'gradient' (phase differences close to +-pi), 'correction' (differences
        changed by the network of images), 'length' (long connections between points) and 'weight' (points
        of low `weight`)
    repair : bool, default: True
        make the interferograms of every triangle of images add up to zero at every point, changing as few
        interferograms as possible; without it they may not where unwrapping every interferogram in space
        disagrees
    repair_cost : int, default: 1
        how much more it costs the repair to change an interferogram at a point where it has no phase jump
    n_workers : int, optional
        number of interferograms unwrapped at the same time; up to 8 by default
    exclude : list, optional
        indices of images left out of the network, e.g. decorrelated by snow; no interferogram uses them

    Returns
    -------
    unw : np.ndarray
        unwrapped phase of the interferograms, shape (n_points, n_image_pairs), np.float32; the first
        point keeps its wrapped phase
    image_pairs : np.ndarray
        the interferograms (reference, secondary) image indices (of all images, excluded ones included),
        shape (n_image_pairs, 2), np.int32
    misclosure : np.ndarray
        fraction of the triangles of images whose interferograms did not add up to zero at every point before
        the repair, shape (n_points,), np.float32; 0 where the interferograms agreed
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    ph = np.ascontiguousarray(ph)
    if ph.ndim != 2 or ph.shape[1] != len(t) or len(t) != len(bperp):
        raise ValueError(f'ph must have shape (n_points, nimages) with nimages = len(t) = len(bperp), got '
                         f'{ph.shape}, {len(t)}, {len(bperp)}')
    keep = np.ones(len(t), bool)
    if exclude is not None and len(exclude):
        keep[np.asarray(exclude, dtype=int)] = False
        if keep.sum() < 3:
            raise ValueError('at least 3 images must be left after `exclude`')
    images = np.flatnonzero(keep)
    if not keep.all():
        ph = np.ascontiguousarray(ph[:, images])
        t = np.asarray(t, dtype=np.float64)[images]
        bperp = np.asarray(bperp, dtype=np.float64)[images]
    flags = _spatial_cost_flags(spatial_cost, weight)
    t_tri, t_half, t_hull, pairs, t_pair, t_sign = _temporal_network(t, bperp, t_scale, bperp_scale)
    pair_cost, adaptive = _temporal_costs(t, bperp, pairs, t_scale, bperp_scale, temporal_cost)
    x = np.asarray(pc_x, dtype=np.float64)
    y = np.asarray(pc_y, dtype=np.float64)
    s_tri, s_half, s_hull = _mcf_network(x, y)
    s_rep, edge_of_half, sign_of_half = _spatial_edges(s_tri, s_half)
    earth_cost = int(earth_cost)
    ptr, corr_edge, corr_val = _emcf_cycles(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign,
                                            earth_cost, pair_cost, adaptive)
    if flags & 4:
        p, q = s_tri[s_rep], s_tri[s_rep - s_rep % 3 + (s_rep + 1) % 3]
        length = np.hypot(x[q] - x[p], y[q] - y[p])
        edge_length = np.minimum(np.median(length) / np.maximum(length, 1e-12), 1.0)
    else:
        edge_length = np.ones(1)
    w = np.clip(np.asarray(weight, dtype=np.float64), 0, 1) if weight is not None else np.ones(1)
    n_pairs = pairs.shape[0]
    unw = np.empty((ph.shape[0], n_pairs), np.float32)
    touched = np.zeros((ph.shape[0], n_pairs), np.bool_)

    def spatial(k):
        u, tch = _emcf_spatial(ph, pairs[k, 0], pairs[k, 1], s_tri, s_half, s_hull, s_rep, edge_of_half,
                               sign_of_half, corr_edge[ptr[k]:ptr[k + 1]], corr_val[ptr[k]:ptr[k + 1]], earth_cost,
                               flags, edge_length, w)
        unw[:, k] = u
        touched[:, k] = tch

    n_workers = n_workers or min(8, n_pairs, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(spatial, range(n_pairs)))
    if repair:
        n_open = _emcf_repair(ph, unw, pairs, t_tri, t_half, t_hull, t_pair, t_sign, earth_cost,
                              pair_cost if 'length' in temporal_cost else np.ones(n_pairs, np.int64), touched,
                              int(repair_cost))
    else:
        n_open = _emcf_repair(ph, unw.copy(), pairs, t_tri, t_half, t_hull, t_pair, t_sign, earth_cost,
                              np.ones(n_pairs, np.int64), touched, 1)
    return unw, images[pairs].astype(np.int32), (n_open / (t_tri.shape[0] // 3)).astype(np.float32)
