"""extended minimum cost flow (EMCF) unwrapping of point cloud interferogram networks"""

__all__ = ['emcf_pc']

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange

from ..tnet import TempNet
from ..utils_ import get_mem_avail, get_n_cpus_avail
from ..utils_ import mjit
from .closure import _l1_fit, _pair_forest
from .mcf import _SPATIAL_COST_RANGE, _TWO_PI, _edge_length, _mcf_edges, _mcf_ssp

# Two steps, each on its own unit so that they can run on data larger than memory (decision 0020):
# 1. temporal step, per block of spatial edges: on every edge (p, q) of the point triangulation, the wrapped
#    gradients of the interferograms are corrected by whole cycles so that they are differences of gradients
#    of the images, at the least cost (`_l1_fit` on the graph of image pairs);
# 2. spatial step, per interferogram: min cost flow on the triangulation with the corrected gradients
#    (`_mcf_ssp`), integration from the first point.
# Loops that still do not close are corrected afterwards by `unwrap_correct_closure_pc` (decision 0021).
# The value of image pair k = (a, b) is always image a - image b; the gradient of edge (p, q) goes from p to q.


def _pair_costs(t, pairs, mode):
    """Cost of correcting every pair: 'constant' 1; 'length': shorter interferograms in time cost more to
    correct, relative to the median time span, 1..20 (like the distance costs of spurt). Decision 0020."""
    if mode not in ('constant', 'length'):
        raise ValueError(f'unknown temporal_cost {mode!r}: constant or length')
    if mode == 'length':
        if t is None:
            raise ValueError(f'temporal_cost {mode!r} needs the acquisition times t')
        span = np.maximum(np.abs(t[pairs[:, 1]] - t[pairs[:, 0]]), 1e-12)
        cost = np.clip(np.round(4 * np.median(span) / span), 1, 20).astype(np.int64)
    else:
        cost = np.ones(pairs.shape[0], np.int64)
    return cost


@mjit(nopython=True, inline='always')
def _gradient(rp, rq, sp, sq):
    """Wrapped gradient from p to q of the interferogram of images r and s (phases rp, rq, sp, sq)."""
    a = rq * np.conj(rp)
    b = sq * np.conj(sp)
    z = a * np.conj(b)
    return np.arctan2(z.imag, z.real)


@mjit(nopython=True, parallel=True)
def _emcf_temporal(ph, p, q, pairs, pair_cost, order, parent):
    """Temporal step on a block of edges (p[i], q[i]), rows of `ph` (n_rows, nimages): whole cycles to add to
    the wrapped gradient of every interferogram, (n_edges, n_pairs) int8; correcting interferogram k costs
    pair_cost[k]."""
    n = p.shape[0]
    K = pairs.shape[0]
    V = ph.shape[1]
    out = np.zeros((n, K), np.int8)
    for i in prange(n):
        pi_ = p[i]
        qi = q[i]
        dimg = np.empty(V, np.float64)       # wrapped gradients of the images
        for j in range(V):
            zz = ph[qi, j] * np.conj(ph[pi_, j])
            dimg[j] = np.arctan2(zz.imag, zz.real)
        m = np.empty(K, np.int64)
        nonzero = False
        for k in range(K):
            a = pairs[k, 0]
            b = pairs[k, 1]
            g = _gradient(ph[pi_, a], ph[qi, a], ph[pi_, b], ph[qi, b])
            m[k] = int(np.round((dimg[a] - dimg[b] - g) / _TWO_PI))
            if m[k] != 0:
                nonzero = True
        if not nonzero:
            continue
        z = np.empty(K, np.int64)
        _l1_fit(pairs, m, pair_cost, order, parent, z)
        for k in range(K):
            out[i, k] = z[k]
    return out


_SPATIAL_COST_FACTORS = {'length': 1, 'weight': 2}


def _spatial_cost_flags(mode, weight):
    if mode == 'constant':
        return 0
    flags = 0
    for name in mode.split('+'):
        if name not in _SPATIAL_COST_FACTORS:
            raise ValueError(f'unknown spatial_cost {mode!r}: combine {", ".join(_SPATIAL_COST_FACTORS)} with +, '
                             f'or constant')
        flags |= _SPATIAL_COST_FACTORS[name]
    if flags & 2 and weight is None:
        raise ValueError("spatial_cost 'weight' needs the point weights")
    return flags


_MEMORY_FRACTION = 0.5          # share of the available memory the spatial workers may use by default


def _image_pairs(image_pairs, nimages):
    """The image pairs checked, (n_pairs, 2) int64."""
    pairs = np.asarray(image_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f'image_pairs must be integers of shape (n_pairs, 2), got {pairs.dtype} {pairs.shape}')
    if pairs.min() < 0 or pairs.max() >= nimages or (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError(f'image_pairs must pair two different images of 0 .. {nimages - 1}')
    return np.ascontiguousarray(pairs, dtype=np.int64)



def _spatial_workers(n_points, n_edges, n_tri, flags, n_pairs):
    """Default number of interferograms unwrapped at the same time: bounded by the available cores and
    `_MEMORY_FRACTION` of the available memory."""
    per_worker = _spatial_worker_bytes(n_points, n_edges, n_tri, flags)
    return max(1, min(n_pairs, get_n_cpus_avail(), int(_MEMORY_FRACTION * get_mem_avail() // per_worker)))


def _spatial_worker_bytes(n_points, n_edges, n_tri, flags):
    """Working memory of one `_emcf_spatial` call in bytes, about 200 per point."""
    b = n_points * (16 + 8 + 1 + 4)                  # image phases (2 complex64), unw float64, done, float32 result
    b += n_edges * (1 + 4 + 4)                       # cycles, G, r
    b += (n_tri + 1) * (4 + 1 + 4 + 37)              # supply, seen, queue; per node of `_mcf_ssp`: pi, excess,
                                                     # dist, pred (int64), settled, touched
    b += 3 * n_tri * (4 + (4 if flags else 0))       # flow and arc cost per half-edge
    return b


@mjit(nopython=True, nogil=True)
def _emcf_spatial(ph_ref, ph_sec, cycles, tri, half, hull, edges, edge_of_half, sign_of_half, earth_cost, flags,
                  edge_length, weight):
    """Spatial step of one interferogram (images `ph_ref`, `ph_sec`, (n_points,)) with the cycles of the temporal
    step on every edge, (n_edges,). Returns the unwrapped phase (n_points,) float32.

    Arc cost of a spatial edge 1 + round(_SPATIAL_COST_RANGE * r), r the product of the reliabilities chosen
    by `flags`: 1 length (`edge_length`, median / length up to 1, like the distance costs of spurt), 2 point
    weights (smaller of the two, 0..1, like coherence weights). Decision 0020."""
    n_points = ph_ref.shape[0]
    n_half = tri.shape[0]
    n_edges = edges.shape[0]
    T = n_half // 3
    G = np.empty(n_edges, np.float32)
    r = np.ones(n_edges, np.float32)
    for s in range(n_edges):
        p = edges[s, 0]
        q = edges[s, 1]
        G[s] = _gradient(ph_ref[p], ph_ref[q], ph_sec[p], ph_sec[q]) + _TWO_PI * cycles[s]
        if flags & 1:
            r[s] *= edge_length[s]
        if flags & 2:
            r[s] *= min(weight[p], weight[q])
    supply = np.empty(T + 1, np.int32)
    tot = 0
    for t in range(T):
        v = 0.0
        for j in range(3):
            e = 3 * t + j
            v += sign_of_half[e] * np.float64(G[edge_of_half[e]])
        rr = int(np.round(v / _TWO_PI))
        supply[t] = rr
        tot += rr
    supply[T] = -tot
    if flags:
        cost = np.empty(n_half, np.int32)
        for e in range(n_half):
            cost[e] = 1 + int(np.round(_SPATIAL_COST_RANGE * min(max(r[edge_of_half[e]], 0.0), 1.0)))
        f = _mcf_ssp(tri, half, hull, supply, earth_cost, cost)
    else:
        f = _mcf_ssp(tri, half, hull, supply, earth_cost)
    # integration over the triangles from the first point
    unw = np.empty(n_points, np.float64)
    done = np.zeros(n_points, np.bool_)
    seen = np.zeros(T, np.bool_)
    t0 = 0
    for e in range(n_half):
        if tri[e] == 0:
            t0 = e // 3
            break
    # the first point gets the difference of the image phases (not wrapped), so that the interferograms
    # of every loop of image pairs already close at the start of the integration
    unw[0] = (np.arctan2(ph_ref[0].imag, ph_ref[0].real)
              - np.arctan2(ph_sec[0].imag, ph_sec[0].real))
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
                a_ = tri[e]
                b_ = tri[3 * t + (j + 1) % 3]
                if done[a_] and not done[b_]:
                    unw[b_] = unw[a_] + sign_of_half[e] * np.float64(G[edge_of_half[e]]) - _TWO_PI * f[e]
                    done[b_] = True
        for j in range(3):
            h = half[3 * t + j]
            if h >= 0 and not seen[h // 3]:
                seen[h // 3] = True
                queue[qt] = h // 3
                qt += 1
    return unw.astype(np.float32)


def emcf_pc(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    t:np.ndarray=None,
    image_pairs:np.ndarray=None,
    weight:np.ndarray=None,
    earth_cost:int=1,
    temporal_cost:str='constant',
    spatial_cost:str='constant',
    n_workers:int=None,
):
    """Extended minimum cost flow (EMCF) phase unwrapping of point cloud interferograms.

    The interferograms of a network of image pairs are unwrapped together: the redundancy of the network
    (image pairs that close loops, e.g. (a, b), (b, c) and (a, c)) is used against unwrapping errors.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape (n_points,), e.g. in meters; the coordinates must be unique and not all collinear
    pc_y : np.ndarray
        y coordinate, shape (n_points,), in the same unit as `pc_x`
    ph : np.ndarray
        wrapped phase history (complex), shape (n_points, nimages)
    t : np.ndarray, optional
        acquisition time of the images, e.g. in days, shape (nimages,); needed by `temporal_cost` 'length'
    image_pairs : np.ndarray, optional
        the interferograms: (reference, secondary) image indices, shape (n_pairs, 2), int; the interferogram
        is reference * conj(secondary). By default every image is paired with the next three. Without any
        loop in the network every interferogram is unwrapped alone (a warning is given)
    weight : np.ndarray, optional
        quality of the points from 0 (unreliable) to 1, e.g. the temporal coherence, shape (n_points,);
        used by `spatial_cost` 'weight'
    earth_cost : int, default: 1
        cost of a phase jump across the border of the network of points, relative to 1 inside; a larger
        value discourages discharging residues through the border
    temporal_cost : str, default: 'constant'
        which interferograms are corrected first where the interferograms of a loop of image pairs disagree:
        'constant' (all alike) or 'length' (the longest in time)
    spatial_cost : str, default: 'constant'
        where phase jumps are placed first in every interferogram: 'constant' (anywhere alike), 'length'
        (long connections between points), 'weight' (points of low `weight`) or 'length+weight'
    n_workers : int, optional
        number of interferograms unwrapped at the same time, each needing about 200 bytes per point; by
        default as many as the available cores and half of the available memory allow

    Returns
    -------
    unw : np.ndarray
        unwrapped phase of the interferograms in radians, shape (n_points, n_pairs), np.float32; at the first
        point it is the difference of the wrapped phases of the two images. Loops of image pairs may still not
        add up to zero at some points: see `unwrap_correct_closure_pc`
    image_pairs : np.ndarray
        the interferograms (reference, secondary), as given or the default network, shape (n_pairs, 2),
        np.int32
    """
    ph = np.ascontiguousarray(ph)
    if ph.ndim != 2:
        raise ValueError(f'ph must have shape (n_points, nimages), got {ph.shape}')
    nimages = ph.shape[1]
    if image_pairs is None:
        if nimages < 2:
            raise ValueError('at least 2 images are needed')
        image_pairs = TempNet.from_bandwidth(nimages, min(3, nimages - 1)).image_pairs
    pairs = _image_pairs(image_pairs, nimages)
    if t is not None:
        t = np.asarray(t, dtype=np.float64)
        if t.shape != (nimages,):
            raise ValueError(f't must have shape ({nimages},), got {t.shape}')
    pair_cost = _pair_costs(t, pairs, temporal_cost)
    flags = _spatial_cost_flags(spatial_cost, weight)
    order, parent, n_loops = _pair_forest(pairs, nimages)
    if n_loops == 0:
        warnings.warn('the image pairs close no loop: every interferogram is unwrapped alone')

    x = np.asarray(pc_x, dtype=np.float64)
    y = np.asarray(pc_y, dtype=np.float64)
    tri, half, hull, edges, edge_of_half, sign_of_half = _mcf_edges(x, y)
    earth_cost = int(earth_cost)
    cycles = _emcf_temporal(ph, edges[:, 0], edges[:, 1], pairs, pair_cost, order, parent)

    edge_length = _edge_length(x, y, edges) if flags & 1 else np.ones(1, np.float32)
    w = np.clip(np.asarray(weight, dtype=np.float32), 0, 1) if weight is not None else np.ones(1, np.float32)
    n_pairs = pairs.shape[0]
    unw = np.empty((ph.shape[0], n_pairs), np.float32)

    def spatial(k):
        unw[:, k] = _emcf_spatial(np.ascontiguousarray(ph[:, pairs[k, 0]]), np.ascontiguousarray(ph[:, pairs[k, 1]]),
                                  np.ascontiguousarray(cycles[:, k]), tri, half, hull, edges, edge_of_half,
                                  sign_of_half, earth_cost, flags, edge_length, w)

    n_workers = n_workers or _spatial_workers(ph.shape[0], edges.shape[0], tri.shape[0] // 3, flags, n_pairs)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(spatial, range(n_pairs)))
    return unw, pairs.astype(np.int32)
