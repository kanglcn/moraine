"""correction of unwrapping errors of point cloud interferogram networks by phase closure"""

__all__ = ['unwrap_correct_closure_pc']

import warnings

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mcf import _TWO_PI, _mcf_edges

# Phase closure correction as in MintPy (Yunjun et al. 2019, decision 0021), on point clouds:
# 1. per block of points (`_closure_estimate`): with consistent image phases phi, unw[k] = phi_a - phi_b + 2 pi c_k
#    with integer cycles c_k; every loop of image pairs closes if c_k = N_a - N_b for integer cycles N of the
#    images. The cycles closest to c in L1 give the correction d_k = N_a - N_b - c_k of every interferogram
#    (`_l1_fit`, unit costs).
# 2. regions (`_closure_regions`): connected components of the point triangulation without its long edges (they
#    cross areas without points, like the connected components of SNAPHU); errors of spatial unwrapping shift
#    whole regions.
# 3. per interferogram (`_closure_apply`): every region large enough gets the median of d_k of its points; the points
#    of smaller regions (e.g. isolated points between clusters) have nothing to share with and get their own d_k.
#    MintPy leaves them out (and masks them later); moraine keeps them and returns the regions for masking.
# 4. per block of points (`_closure_ts`): `_l1_fit` once more on the corrected interferograms gives the cycles N of
#    the images, integrated along the spanning forest; the phase of image j is angle(ph_j) + 2 pi N_j, relative to
#    the reference image (decision 0022). Where the region median closes every loop, nothing changes; elsewhere the
#    fit decides per point. The median is taken over d, not N: d is constant in a region shifted by spatial
#    unwrapping, N holds the spatial unwrapping of every point. `change_fraction` compares the result with the input
#    (steps 2 and 4 together); on synthetic data it predicts wrong points better than the misclosure or the changes
#    of step 4 alone, but not errors that close every loop (decision 0022).
# The value of image pair k = (a, b) is always image a - image b.


def _pair_forest(pairs, n_images):
    """Spanning forest of the graph of image pairs: (order, parent, n_loops), the images in breadth first
    order, for every image the pair to its parent (-1 for a root), and the number of independent loops."""
    adj = [[] for _ in range(n_images)]
    for k, (a, b) in enumerate(pairs):
        adj[a].append((k, b))
        adj[b].append((k, a))
    parent = np.full(n_images, -1, np.int64)
    seen = np.zeros(n_images, bool)
    order = []
    for s in range(n_images):
        if seen[s]:
            continue
        seen[s] = True
        queue = [s]
        for v in queue:
            order.append(v)
            for k, w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    parent[w] = k
                    queue.append(w)
    return np.array(order, np.int64), parent, pairs.shape[0] - int((parent >= 0).sum())


@njit(cache=True, nogil=True)
def _pred_cycle(pairs, pred, mark):
    """An image on a cycle of the predecessor arcs, or -1."""
    mark[:] = -1
    for s in range(pred.shape[0]):
        if mark[s] != -1:
            continue
        v = s
        while v >= 0 and mark[v] == -1:
            mark[v] = s
            arc = pred[v]
            v = pairs[arc >> 1, arc & 1] if arc >= 0 else -1
        if v >= 0 and mark[v] == s:
            return v
    return -1


@njit(cache=True, nogil=True)
def _l1_fit(pairs, m, cost, order, parent, z):
    """Integer n of the images minimizing sum_k cost[k] |m[k] + n[a_k] - n[b_k]| on any graph of image pairs;
    z = m + n[a] - n[b] (the correction of every pair at the optimum), into z.

    The potentials of the spanning forest first make z zero on its pairs, so that only pairs closing a loop
    that does not add up remain. Then the dual, a circulation x with |x_k| <= cost_k of least cost
    sum -z_k x_k, is solved by cancelling negative cycles (Bellman-Ford; a cycle of the predecessor arcs is
    negative); the final distances d are optimal potentials, n = -d."""
    V = order.shape[0]
    K = m.shape[0]
    P = np.zeros(V, np.int64)
    for i in range(V):
        v = order[i]
        k = parent[v]
        if k >= 0:
            if pairs[k, 1] == v:
                P[v] = P[pairs[k, 0]] + m[k]
            else:
                P[v] = P[pairs[k, 1]] - m[k]
    open_loop = False
    for k in range(K):
        z[k] = m[k] + P[pairs[k, 0]] - P[pairs[k, 1]]
        if z[k] != 0:
            open_loop = True
    if not open_loop:
        return
    # residual arcs: 2k from a to b (x_k up, cost -z_k), 2k + 1 from b to a (x_k down, cost z_k); the tail of
    # arc r is pairs[r >> 1, r & 1]
    x = np.zeros(K, np.int64)
    d = np.empty(V, np.int64)
    pred = np.empty(V, np.int64)
    mark = np.empty(V, np.int64)
    while True:
        d[:] = 0
        pred[:] = -1
        cyc = -1
        upd = True
        for it in range(V + 1):
            upd = False
            for r in range(2 * K):
                k = r >> 1
                u = pairs[k, r & 1]
                v = pairs[k, 1 - (r & 1)]
                if r & 1 == 0:
                    cap = cost[k] - x[k]
                    w = -z[k]
                else:
                    cap = cost[k] + x[k]
                    w = z[k]
                if cap > 0 and d[u] + w < d[v]:
                    d[v] = d[u] + w
                    pred[v] = r
                    upd = True
            if not upd:
                break
            cyc = _pred_cycle(pairs, pred, mark)
            if cyc >= 0:
                break
        if cyc < 0:
            if upd:
                raise RuntimeError('l1 fit: no convergence')
            break
        amount = np.int64(1) << 62
        v = cyc
        while True:
            r = pred[v]
            k = r >> 1
            cap = cost[k] - x[k] if r & 1 == 0 else cost[k] + x[k]
            amount = min(amount, cap)
            v = pairs[k, r & 1]
            if v == cyc:
                break
        v = cyc
        while True:
            r = pred[v]
            k = r >> 1
            if r & 1 == 0:
                x[k] += amount
            else:
                x[k] -= amount
            v = pairs[k, r & 1]
            if v == cyc:
                break
    for k in range(K):
        z[k] = z[k] - d[pairs[k, 0]] + d[pairs[k, 1]]


@njit(cache=True, parallel=True)
def _closure_ts(ph, unw, unw_in, pairs, order, parent, ref):
    """Phase time series of a block of points (`ph` (n, nimages), `unw` (n, n_pairs) after the region correction,
    `unw_in` the input before it, same shape): unwrapped phase of every image relative to image `ref`,
    (n, nimages) float32, the fraction of the interferograms whose whole cycles in the result differ from
    `unw_in`, (n,) float32, and the largest distance of unw - (phi_a - phi_b) from whole cycles, in cycles. The
    image pairs must connect all images (one root in `parent`)."""
    n = unw.shape[0]
    K = pairs.shape[0]
    V = ph.shape[1]
    ts = np.zeros((n, V), np.float32)
    frac = np.zeros(n, np.float32)
    dev = np.zeros(n, np.float64)
    cost = np.ones(K, np.int64)
    for i in prange(n):
        ang = np.empty(V, np.float64)
        for j in range(V):
            ang[j] = np.arctan2(ph[i, j].imag, ph[i, j].real)
        m = np.empty(K, np.int64)
        worst = 0.0
        for k in range(K):
            c = (unw[i, k] - (ang[pairs[k, 0]] - ang[pairs[k, 1]])) / _TWO_PI
            m[k] = -int(np.round(c))
            worst = max(worst, abs(c + m[k]))
        dev[i] = worst
        z = np.empty(K, np.int64)
        _l1_fit(pairs, m, cost, order, parent, z)
        # N_a - N_b = z_k - m_k on every pair; integrated from the root along the spanning forest
        cyc = np.zeros(V, np.int64)
        for o in range(V):
            v = order[o]
            k = parent[v]
            if k >= 0:
                if pairs[k, 1] == v:
                    cyc[v] = cyc[pairs[k, 0]] - (z[k] - m[k])
                else:
                    cyc[v] = cyc[pairs[k, 1]] + (z[k] - m[k])
        for j in range(V):
            ts[i, j] = ang[j] - ang[ref] + _TWO_PI * (cyc[j] - cyc[ref])
        n_changed = 0
        for k in range(K):
            a = pairs[k, 0]
            b = pairs[k, 1]
            c_in = int(np.round((unw_in[i, k] - (ang[a] - ang[b])) / _TWO_PI))
            if cyc[a] - cyc[b] != c_in:
                n_changed += 1
        frac[i] = n_changed / K
    return ts, frac, dev.max() if n else 0.0


@njit(cache=True, parallel=True)
def _closure_estimate(ph, unw, pairs, order, parent):
    """Per point corrections on a block of points (`ph` (n, nimages), `unw` (n, n_pairs)): whole cycles to add to
    every interferogram so that every loop of image pairs closes, (n, n_pairs) int8, the fraction of the
    interferograms to correct per point, (n,) float32, and the largest distance of unw - (phi_a - phi_b) from
    whole cycles, in cycles (0 if `unw` rewraps to the interferograms of `ph`)."""
    n = unw.shape[0]
    K = pairs.shape[0]
    d = np.zeros((n, K), np.int8)
    frac = np.zeros(n, np.float32)
    dev = np.zeros(n, np.float64)
    cost = np.ones(K, np.int64)
    for i in prange(n):
        m = np.empty(K, np.int64)
        worst = 0.0
        for k in range(K):
            za = ph[i, pairs[k, 0]]
            zb = ph[i, pairs[k, 1]]
            c = (unw[i, k] - (np.arctan2(za.imag, za.real) - np.arctan2(zb.imag, zb.real))) / _TWO_PI
            m[k] = -int(np.round(c))
            worst = max(worst, abs(c + m[k]))
        dev[i] = worst
        z = np.empty(K, np.int64)
        _l1_fit(pairs, m, cost, order, parent, z)
        n_changed = 0
        for k in range(K):
            d[i, k] = z[k]
            if z[k] != 0:
                n_changed += 1
        frac[i] = n_changed / K
    return d, frac, dev.max() if n else 0.0


def _closure_regions(x, y, edges, max_edge_factor, min_region_points):
    """Regions of points connected by edges no longer than `max_edge_factor` times the median edge length:
    per point the index of its region, -1 in regions of less than `min_region_points` points, (n_points,) int32,
    and the number of regions."""
    p, q = edges[:, 0], edges[:, 1]
    length = np.hypot(x[q] - x[p], y[q] - y[p])
    keep = length <= max_edge_factor * np.median(length)
    n = x.shape[0]
    graph = coo_matrix((np.ones(int(keep.sum()), np.int8), (p[keep], q[keep])), shape=(n, n))
    lab = connected_components(graph, directed=False)[1]
    size = np.bincount(lab)
    big = size >= min_region_points
    index = np.full(size.shape[0], -1, np.int32)
    index[big] = np.arange(int(big.sum()), dtype=np.int32)
    return index[lab], int(big.sum())


@njit(cache=True, nogil=True)
def _closure_apply(unw, d, region, n_regions):
    """Correction of one interferogram (`unw` (n_points,), in place) by the median of the per point corrections
    `d` (n_points,) in every region (`region` (n_points,); -1: small region, every point gets its own correction).
    An even count takes the mean of the two middle values, rounded to the nearest integer (halves to even)."""
    lo = 0
    hi = 0
    for i in range(d.shape[0]):
        lo = min(lo, d[i])
        hi = max(hi, d[i])
    width = hi - lo + 1
    count = np.zeros((n_regions, width), np.int64)
    size = np.zeros(n_regions, np.int64)
    for i in range(d.shape[0]):
        r = region[i]
        if r >= 0:
            count[r, d[i] - lo] += 1
            size[r] += 1
    shift = np.zeros(n_regions, np.int64)
    for r in range(n_regions):
        # values at ranks (size - 1) // 2 and size // 2 of the sorted corrections
        r0 = (size[r] - 1) // 2
        r1 = size[r] // 2
        v0 = 0
        v1 = 0
        acc = 0
        found0 = False
        for j in range(width):
            acc += count[r, j]
            if not found0 and acc > r0:
                v0 = j + lo
                found0 = True
            if acc > r1:
                v1 = j + lo
                break
        shift[r] = np.int64(np.round((v0 + v1) / 2.0))
    for i in range(d.shape[0]):
        r = region[i]
        c = shift[r] if r >= 0 else d[i]
        if c != 0:
            unw[i] += _TWO_PI * c


@njit(cache=True, parallel=True)
def _closure_apply_all(unw, d, region, n_regions):
    for k in prange(unw.shape[1]):
        _closure_apply(unw[:, k], d[:, k], region, n_regions)


def unwrap_correct_closure_pc(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    unw:np.ndarray,
    image_pairs:np.ndarray,
    ref:int=0,
    max_edge_factor:float=4.0,
    min_region_points:int=30,
):
    """Correction of unwrapping errors of point cloud interferograms by phase closure, to the unwrapped phase of
    every image.

    The unwrapped phases of every loop of image pairs should add up to zero, e.g.
    unw(a, b) + unw(b, c) = unw(a, c). Where they do not, the interferograms are corrected by whole cycles,
    changing as few interferograms as possible, by the same amount for all points of a region: the points
    connected by edges of the point network no longer than `max_edge_factor` times the median edge length.
    Points of smaller regions than `min_region_points` are corrected one by one. The correction assumes
    that most interferograms of a region are right: where most are wrong, it makes them worse. Where a point
    still does not fit the loops after the correction of its region, the interferograms fitting most of its
    loops are kept. The result is the unwrapped phase of every image, whose differences close every loop.
    Errors that close every loop (e.g. the same whole cycles in every interferogram of one image) are neither
    corrected nor reported.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape (n_points,), e.g. in meters; the coordinates must be unique and not all collinear
    pc_y : np.ndarray
        y coordinate, shape (n_points,), in the same unit as `pc_x`
    ph : np.ndarray
        wrapped phase history (complex), shape (n_points, nimages)
    unw : np.ndarray
        unwrapped phase of the interferograms in radians, shape (n_points, n_pairs), e.g. from `mcf_pc` or
        `emcf_pc`; rewrapped, it must be the phase of ph[:, reference] * conj(ph[:, secondary])
    image_pairs : np.ndarray
        the interferograms: (reference, secondary) image indices of the columns of `unw`, shape (n_pairs, 2),
        int; the pairs must connect all images; the network needs loops (e.g. every image paired with the
        next three) to correct anything, otherwise only the image phases are computed (a warning is given)
    ref : int, default: 0
        index of the reference image, 0 .. nimages - 1: the image phases are relative to it
    max_edge_factor : float, default: 4.0
        edges of the point network longer than this times the median edge length do not connect a region,
        e.g. edges across water or decorrelated areas; larger than 1
    min_region_points : int, default: 30
        in regions of fewer points every point is corrected by its own loops alone, which is less reliable

    Returns
    -------
    ts : np.ndarray
        unwrapped phase of every image relative to image `ref` in radians, shape (n_points, nimages),
        np.float32; ts[:, ref] is 0; rewrapped, ts[:, j] is the phase of ph[:, j] * conj(ph[:, ref]); the
        corrected interferogram (a, b) is ts[:, a] - ts[:, b]
    misclosure_fraction : np.ndarray
        fraction of the interferograms of every point that do not fit the loops before the correction,
        shape (n_points,), np.float32, 0..1; 0 where every loop closes
    change_fraction : np.ndarray
        fraction of the interferograms of every point whose whole cycles in the result differ from `unw`,
        shape (n_points,), np.float32, 0..1; 0 where `unw` was kept; large values mark points whose input
        unwrapping was poor (e.g. to be masked)
    region : np.ndarray
        region of every point, shape (n_points,), np.int32: 0 .. n_regions - 1, or -1 for the points of
        regions smaller than `min_region_points`, corrected one by one (e.g. to be masked)
    """
    ph = np.ascontiguousarray(ph)
    unw = np.asarray(unw)
    pairs = np.asarray(image_pairs)
    if ph.ndim != 2 or unw.ndim != 2 or unw.shape[0] != ph.shape[0]:
        raise ValueError(f'ph (n_points, nimages) and unw (n_points, n_pairs) do not match: {ph.shape}, {unw.shape}')
    if pairs.shape != (unw.shape[1], 2) or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f'image_pairs must be integers of shape ({unw.shape[1]}, 2), got {pairs.dtype} {pairs.shape}')
    if pairs.min() < 0 or pairs.max() >= ph.shape[1] or (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError(f'image_pairs must pair two different images of 0 .. {ph.shape[1] - 1}')
    if not 0 <= int(ref) < ph.shape[1]:
        raise ValueError(f'ref must be an image index of 0 .. {ph.shape[1] - 1}, got {ref}')
    if not max_edge_factor > 1:
        raise ValueError(f'max_edge_factor must be larger than 1, got {max_edge_factor}')
    if int(min_region_points) < 1:
        raise ValueError(f'min_region_points must be at least 1, got {min_region_points}')
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    order, parent, n_loops = _pair_forest(pairs, ph.shape[1])
    _check_connected(parent)
    unw_cor = np.array(unw, dtype=np.float32)
    x = np.asarray(pc_x, dtype=np.float64)
    y = np.asarray(pc_y, dtype=np.float64)
    edges = _mcf_edges(x, y)[3]
    region, n_regions = _closure_regions(x, y, edges, float(max_edge_factor), int(min_region_points))
    if n_loops == 0:
        warnings.warn('the image pairs close no loop: nothing to correct')
        misclosure_fraction = np.zeros(ph.shape[0], np.float32)
    else:
        d, misclosure_fraction, dev = _closure_estimate(ph, unw_cor, pairs, order, parent)
        _check_rewrap(dev)
        _closure_apply_all(unw_cor, d, region, n_regions)
    ts, change_fraction, dev = _closure_ts(ph, unw_cor, np.asarray(unw, dtype=np.float32), pairs, order, parent,
                                           int(ref))
    _check_rewrap(dev)
    return ts, misclosure_fraction, change_fraction, region


def _check_connected(parent):
    n_parts = int((parent < 0).sum())
    if n_parts > 1:
        raise ValueError(f'image_pairs must connect all images, they make {n_parts} separate groups of images')


def _check_rewrap(dev, name='unw'):
    if dev > 0.1:
        raise ValueError(f'{name} does not rewrap to the interferograms of ph (up to {dev:.2f} cycles off)')
