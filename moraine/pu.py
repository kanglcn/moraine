"""phase unwrapping"""


__all__ = ['gamma_mcf_pt', 'mcf_pc']

import numpy as np
import tempfile
from pathlib import Path
import os
from numba import njit

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
def _mcf_ssp(tri, half, hull, supply, earth_cost):
    """Min cost flow by successive shortest paths.

    From every node with positive excess, Dijkstra on reduced costs (node potentials keep them non
    negative) until the nearest node with negative excess, then one unit is pushed along the path. The
    search stops at the first sink, so it stays local when residues pair up with close neighbours.
    Arc cost: 1 between triangles, `earth_cost` to the earth, unlimited capacity. Returns the flow per
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
                            mc = 1 if f[e] >= 0 else -1
                        else:
                            v = EARTH
                            mc = earth_cost if f[e] >= 0 else -earth_cost
                        code = e
                    else:
                        e = hull[k]
                        v = e // 3
                        mc = earth_cost if f[e] <= 0 else -earth_cost
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

    The points are triangulated (Delaunay); every triangle whose wrapped phase differences do not sum
    to zero is a residue; a minimum cost flow between the triangles (successive shortest paths) decides
    where the 2 pi jumps go, and the phase is integrated over the triangles from the first point. The
    result is exactly optimal for the costs below and does not depend on the order of the points.

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
        discourages discharging residues through the border (GAMMA mcf_pt behaves like 3)

    Returns
    -------
    np.ndarray
        unwrapped phase, shape (N,), np.float64; the first point keeps its wrapped phase
    """
    return _mcf_unwrap(ph, *_mcf_network(pc_x, pc_y), earth_cost)
