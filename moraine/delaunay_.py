"""2D Delaunay triangulation (sweep-hull), used by the phase unwrapping of point clouds.

Points are added in order of distance from the circumcenter of a seed triangle; each new point is
connected to the part of the convex hull it sees, and edges are flipped until the Delaunay condition
holds again. The hull is a doubly linked list with an angular hash for finding a visible edge.

- Exact predicates for integer coordinates (int64 arithmetic, coordinate span < 2**15), so cocircular
  points, common on the integer grid of radar coordinates, are decided exactly; float64 otherwise.
- Independent of the input order: sort and seed ties are broken by the coordinates, not the index.
- Output as half-edges: `tri` (3 n_tri,) the start vertex of every half-edge (clockwise triangles,
  half-edge e goes from tri[e] to tri[e - e % 3 + (e + 1) % 3]), `half` (3 n_tri,) the twin half-edge,
  -1 on the convex hull. Triangle of half-edge e: e // 3.

The algorithm and the structure of this implementation follow Delaunator
(https://github.com/mapbox/delaunator), distributed under the ISC license:

    ISC License

    Copyright (c) 2026, Mapbox

    Permission to use, copy, modify, and/or distribute this software for any purpose
    with or without fee is hereby granted, provided that the above copyright notice
    and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
    FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
    OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
    TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
    THIS SOFTWARE.
"""

__all__ = ['delaunay_halfedges', 'delaunay']

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _orient(px, py, qx, qy, rx, ry):
    """True if p, q, r are clockwise in this convention (the seed triangle and all triangles are made so)."""
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0


@njit(cache=True, inline='always')
def _in_circle(ax, ay, bx, by, cx, cy, px, py):
    dx = ax - px
    dy = ay - py
    ex = bx - px
    ey = by - py
    fx = cx - px
    fy = cy - py
    ap = dx * dx + dy * dy
    bp = ex * ex + ey * ey
    cp = fx * fx + fy * fy
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0


@njit(cache=True, inline='always')
def _circumradius2(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay
    den = dx * ey - dy * ex
    if den == 0:
        return np.inf
    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    d = 0.5 / den
    x = (ey * bl - dy * cl) * d
    y = (dx * cl - ex * bl) * d
    return x * x + y * y


@njit(cache=True, inline='always')
def _circumcenter(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay
    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    d = 0.5 / (dx * ey - dy * ex)
    return ax + (ey * bl - dy * cl) * d, ay + (dx * cl - ex * bl) * d


@njit(cache=True, inline='always')
def _less(d1, x1, y1, d2, x2, y2):
    """Lexicographic (d, x, y) comparison: ties are decided by the coordinates, not the index."""
    if d1 != d2:
        return d1 < d2
    if x1 != x2:
        return x1 < x2
    return y1 < y2


@njit(cache=True)
def _seeds(xf, yf):
    n = xf.shape[0]
    cx = (xf.min() + xf.max()) / 2
    cy = (yf.min() + yf.max()) / 2
    i0 = 0
    best = np.inf
    for i in range(n):
        d = (xf[i] - cx) ** 2 + (yf[i] - cy) ** 2
        if i == 0 or _less(d, xf[i], yf[i], best, xf[i0], yf[i0]):
            best = d
            i0 = i
    i1 = -1
    best = np.inf
    for i in range(n):
        if i == i0:
            continue
        d = (xf[i] - xf[i0]) ** 2 + (yf[i] - yf[i0]) ** 2
        if d > 0 and (i1 == -1 or _less(d, xf[i], yf[i], best, xf[i1], yf[i1])):
            best = d
            i1 = i
    i2 = -1
    best = np.inf
    for i in range(n):
        if i == i0 or i == i1:
            continue
        r = _circumradius2(xf[i0], yf[i0], xf[i1], yf[i1], xf[i], yf[i])
        if r < np.inf and (i2 == -1 or _less(r, xf[i], yf[i], best, xf[i2], yf[i2])):
            best = r
            i2 = i
    return i0, i1, i2


@njit(cache=True, inline='always')
def _pseudo_angle(dx, dy):
    p = dx / (abs(dx) + abs(dy))
    return (3 - p if dy > 0 else 1 + p) / 4


@njit(cache=True, inline='always')
def _hash_key(x, y, cx, cy, size):
    return int(np.floor(_pseudo_angle(x - cx, y - cy) * size)) % size


@njit(cache=True, inline='always')
def _link(halfedges, a, b):
    halfedges[a] = b
    if b != -1:
        halfedges[b] = a


@njit(cache=True)
def _sweep(X, Y, xf, yf, ids, i0, i1, i2, cx, cy):
    n = X.shape[0]
    max_tri = max(2 * n - 5, 1)
    triangles = np.empty(max_tri * 3, np.int32)
    halfedges = np.empty(max_tri * 3, np.int32)
    hash_size = int(np.ceil(np.sqrt(n)))
    hull_prev = np.empty(n, np.int32)
    hull_next = np.empty(n, np.int32)
    hull_tri = np.empty(n, np.int32)
    hull_hash = np.full(hash_size, -1, np.int32)
    stack = np.empty(1 << 16, np.int64)

    hull_start = i0
    hull_next[i0] = i1; hull_prev[i2] = i1
    hull_next[i1] = i2; hull_prev[i0] = i2
    hull_next[i2] = i0; hull_prev[i1] = i0
    hull_tri[i0] = 0; hull_tri[i1] = 1; hull_tri[i2] = 2
    hull_hash[_hash_key(xf[i0], yf[i0], cx, cy, hash_size)] = i0
    hull_hash[_hash_key(xf[i1], yf[i1], cx, cy, hash_size)] = i1
    hull_hash[_hash_key(xf[i2], yf[i2], cx, cy, hash_size)] = i2

    tlen = 0
    # add_triangle(i0, i1, i2, -1, -1, -1)
    triangles[0] = i0; triangles[1] = i1; triangles[2] = i2
    halfedges[0] = -1; halfedges[1] = -1; halfedges[2] = -1
    tlen = 3
    skipped = 0

    for k in range(ids.shape[0]):
        i = ids[k]
        if i == i0 or i == i1 or i == i2:
            continue
        x, y = X[i], Y[i]
        # a visible hull edge through the angular hash
        start = 0
        key = _hash_key(xf[i], yf[i], cx, cy, hash_size)
        for j in range(hash_size):
            start = hull_hash[(key + j) % hash_size]
            if start != -1 and start != hull_next[start]:
                break
        start = hull_prev[start]
        e = start
        while True:
            q = hull_next[e]
            if _orient(x, y, X[e], Y[e], X[q], Y[q]):
                break
            e = q
            if e == start:
                e = -1
                break
        if e == -1:
            skipped += 1          # duplicate point (or inside the hull, impossible for exact input)
            continue

        # first triangle from the point
        t = tlen
        triangles[t] = e; triangles[t + 1] = i; triangles[t + 2] = hull_next[e]
        _link(halfedges, t, -1); _link(halfedges, t + 1, -1); _link(halfedges, t + 2, hull_tri[e])
        tlen += 3
        ar, hull_start = _legalize(t + 2, triangles, halfedges, X, Y, hull_tri, hull_prev, hull_start, stack)
        hull_tri[i] = ar
        hull_tri[e] = t

        # walk forward through the hull
        nn = hull_next[e]
        while True:
            q = hull_next[nn]
            if not _orient(x, y, X[nn], Y[nn], X[q], Y[q]):
                break
            t = tlen
            triangles[t] = nn; triangles[t + 1] = i; triangles[t + 2] = q
            _link(halfedges, t, hull_tri[i]); _link(halfedges, t + 1, -1); _link(halfedges, t + 2, hull_tri[nn])
            tlen += 3
            ar, hull_start = _legalize(t + 2, triangles, halfedges, X, Y, hull_tri, hull_prev, hull_start, stack)
            hull_tri[i] = ar
            hull_next[nn] = nn        # removed from the hull
            nn = q

        # walk backward from the other side
        if e == start:
            while True:
                q = hull_prev[e]
                if not _orient(x, y, X[q], Y[q], X[e], Y[e]):
                    break
                t = tlen
                triangles[t] = q; triangles[t + 1] = i; triangles[t + 2] = e
                _link(halfedges, t, -1); _link(halfedges, t + 1, hull_tri[e]); _link(halfedges, t + 2, hull_tri[q])
                tlen += 3
                ar, hull_start = _legalize(t + 2, triangles, halfedges, X, Y, hull_tri, hull_prev, hull_start, stack)
                hull_tri[q] = t
                hull_next[e] = e
                e = q

        hull_start = e
        hull_prev[i] = e
        hull_next[e] = i
        hull_prev[nn] = i
        hull_next[i] = nn
        hull_hash[_hash_key(xf[i], yf[i], cx, cy, hash_size)] = i
        hull_hash[_hash_key(xf[e], yf[e], cx, cy, hash_size)] = e
    return triangles[:tlen], halfedges[:tlen], skipped


@njit(cache=True)
def _legalize(a, triangles, halfedges, X, Y, hull_tri, hull_prev, hull_start, stack):
    i = 0
    ar = 0
    while True:
        b = halfedges[a]
        a0 = a - a % 3
        ar = a0 + (a + 2) % 3
        if b == -1:
            if i == 0:
                break
            i -= 1
            a = stack[i]
            continue
        b0 = b - b % 3
        al = a0 + (a + 1) % 3
        bl = b0 + (b + 2) % 3
        p0 = triangles[ar]
        pr = triangles[a]
        pl = triangles[al]
        p1 = triangles[bl]
        if _in_circle(X[p0], Y[p0], X[pr], Y[pr], X[pl], Y[pl], X[p1], Y[p1]):
            triangles[a] = p1
            triangles[b] = p0
            hbl = halfedges[bl]
            if hbl == -1:       # the flipped edge was on the hull: fix the hull triangle reference
                e = hull_start
                while True:
                    if hull_tri[e] == bl:
                        hull_tri[e] = a
                        break
                    e = hull_prev[e]
                    if e == hull_start:
                        break
            _link(halfedges, a, hbl)
            _link(halfedges, b, halfedges[ar])
            _link(halfedges, ar, bl)
            br = b0 + (b + 1) % 3
            if i >= stack.shape[0]:
                raise RuntimeError('delaunay: flip stack overflow')
            stack[i] = br
            i += 1
        else:
            if i == 0:
                break
            i -= 1
            a = stack[i]
    return ar, hull_start


@njit(cache=True)
def _coordinates(xf, yf):
    """Integer coordinates with a span < 2**15 get exact int64 predicates (relative to the minimum)."""
    n = xf.shape[0]
    xmin, xmax, ymin, ymax = xf[0], xf[0], yf[0], yf[0]
    integer = True
    for i in range(n):
        xmin = min(xmin, xf[i]); xmax = max(xmax, xf[i])
        ymin = min(ymin, yf[i]); ymax = max(ymax, yf[i])
        if integer and (xf[i] != np.floor(xf[i]) or yf[i] != np.floor(yf[i])):
            integer = False
    integer = integer and max(xmax - xmin, ymax - ymin) < 2 ** 15
    X = np.empty(n if integer else 0, np.int64)
    Y = np.empty(n if integer else 0, np.int64)
    if integer:
        for i in range(n):
            X[i] = np.int64(xf[i] - xmin)
            Y[i] = np.int64(yf[i] - ymin)
    return integer, X, Y


@njit(cache=True)
def _distances(xf, yf, cx, cy):
    d = np.empty(xf.shape[0], np.float64)
    for i in range(xf.shape[0]):
        d[i] = (xf[i] - cx) ** 2 + (yf[i] - cy) ** 2
    return d


@njit(cache=True)
def _break_ties(ids, d, xf, yf):
    """Within runs of equal distance, order by (x, y), so the order does not depend on the input order
    (insertion sort: runs are short)."""
    n = ids.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and d[ids[j]] == d[ids[i]]:
            j += 1
        for a in range(i + 1, j):
            v = ids[a]
            b = a - 1
            while b >= i and (xf[ids[b]] > xf[v] or (xf[ids[b]] == xf[v] and yf[ids[b]] > yf[v])):
                ids[b + 1] = ids[b]
                b -= 1
            ids[b + 1] = v
        i = j
    return ids


def delaunay_halfedges(x, y):
    """Delaunay triangulation of unique points as half-edges.

    Parameters
    ----------
    x : np.ndarray
        x coordinates, shape (n,)
    y : np.ndarray
        y coordinates, shape (n,)

    Returns
    -------
    tri : np.ndarray
        start vertex of every half-edge, shape (3 * n_tri,), np.int32; triangles are clockwise
    half : np.ndarray
        twin half-edge, shape (3 * n_tri,), np.int32; -1 on the convex hull
    """
    xf = np.ascontiguousarray(x, dtype=np.float64)
    yf = np.ascontiguousarray(y, dtype=np.float64)
    n = xf.shape[0]
    if n < 3:
        raise ValueError('need at least 3 points')
    if 6 * n >= 2 ** 31:
        raise ValueError(f'{n} points: too many for the int32 half-edge indices')
    integer, X, Y = _coordinates(xf, yf)
    if not integer:
        X, Y = xf, yf
    i0, i1, i2 = _seeds(xf, yf)
    if i1 == -1 or i2 == -1:
        raise ValueError('points are collinear or duplicated')
    if _orient(X[i0], Y[i0], X[i1], Y[i1], X[i2], Y[i2]):
        i1, i2 = i2, i1
    cx, cy = _circumcenter(xf[i0], yf[i0], xf[i1], yf[i1], xf[i2], yf[i2])
    d = _distances(xf, yf, cx, cy)
    ids = _break_ties(np.argsort(d), d, xf, yf)     # by distance, ties by coordinates; numpy's argsort is fastest
    del d
    tri, half, skipped = _sweep(X, Y, xf, yf, ids, i0, i1, i2, cx, cy)
    if skipped:
        raise ValueError(f'{skipped} points were not triangulated (duplicated coordinates?)')
    return tri, half


def delaunay(x, y):
    """Delaunay triangulation of unique points in the format of scipy.spatial.Delaunay.

    Parameters
    ----------
    x : np.ndarray
        x coordinates, shape (n,)
    y : np.ndarray
        y coordinates, shape (n,)

    Returns
    -------
    simplices : np.ndarray
        vertices of the triangles, counter clockwise, shape (n_tri, 3)
    neighbors : np.ndarray
        neighbors[t, k] is the triangle opposite vertex k of triangle t, -1 on the hull, shape (n_tri, 3)
    """
    tri, half = delaunay_halfedges(x, y)
    simplices = tri.reshape(-1, 3)
    opp = half.reshape(-1, 3)
    neighbors = np.full(simplices.shape, -1, np.int64)
    for j in range(3):   # half-edge j goes from vertex j to vertex j+1; the vertex opposite to it is j+2
        neighbors[:, (j + 2) % 3] = np.where(opp[:, j] >= 0, opp[:, j] // 3, -1)
    # clockwise -> counter clockwise: swap vertices 1 and 2 and the neighbors opposite them
    return np.ascontiguousarray(simplices[:, [0, 2, 1]]), np.ascontiguousarray(neighbors[:, [0, 2, 1]])
