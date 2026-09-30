"""Delaunay triangulation: identical to scipy (Qhull) where unique, Delaunay property where points are
cocircular, independent of the point order."""
import numpy as np
import pytest
from scipy.spatial import Delaunay

from moraine.api.unwrap.delaunay_ import delaunay


def tri_set(simplices):
    return {tuple(sorted(t)) for t in simplices.tolist()}


def edge_set(simplices):
    s = set()
    for a, b, c in simplices.tolist():
        s |= {tuple(sorted(e)) for e in ((a, b), (b, c), (c, a))}
    return s


def check_structure(x, y, simp, nb):
    n = len(x)
    a, b, c = simp[:, 0], simp[:, 1], simp[:, 2]
    area2 = (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])
    assert (area2 > 0).all(), 'not all counter clockwise / degenerate triangle'
    assert len(np.unique(simp)) == n, 'not every point is used'
    hull_edges = int((nb == -1).sum())
    assert len(simp) == 2 * n - 2 - hull_edges, 'Euler count'
    for t in range(len(simp)):            # neighbour symmetry and shared edge
        for k in range(3):
            u = nb[t, k]
            if u >= 0:
                assert t in nb[u]
                shared = set(simp[t]) - {simp[t, k]}
                assert shared <= set(simp[u])


def check_delaunay(x, y, simp, nb):
    """No point of a neighbouring triangle strictly inside the circumcircle (exact for integers)."""
    X = np.asarray(x).astype(np.int64) if np.all(np.asarray(x) == np.round(x)) else np.asarray(x, float)
    Y = np.asarray(y).astype(X.dtype)
    bad = 0
    for t in range(len(simp)):
        a, b, c = simp[t]
        for k in range(3):
            u = nb[t, k]
            if u < 0:
                continue
            d = [v for v in simp[u] if v not in simp[t]][0]
            m = [[X[p] - X[d], Y[p] - Y[d], (X[p] - X[d]) ** 2 + (Y[p] - Y[d]) ** 2] for p in (a, b, c)]
            det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            bad += det > 0          # d strictly inside the circumcircle of the ccw triangle (a, b, c)
    assert bad == 0, f'{bad} edges violate the Delaunay condition'


@pytest.mark.parametrize('seed', range(10))
def test_random_float_equals_scipy(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 3000))
    p = rng.uniform(-1000, 1000, (n, 2))
    simp, nb = delaunay(p[:, 0], p[:, 1])
    check_structure(p[:, 0], p[:, 1], simp, nb)
    assert tri_set(simp) == tri_set(Delaunay(p).simplices)


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('span', [8, 40, 300])
def test_lattice(seed, span):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, min(span * span, 1500)))
    cells = rng.choice(span * span, n, replace=False)
    y, x = np.divmod(cells, span)
    x, y = x.astype(float) - span // 3, y.astype(float) + 7       # negative and offset coordinates
    if np.linalg.matrix_rank(np.stack((x - x[0], y - y[0]), -1)) < 2:
        pytest.skip('collinear')
    simp, nb = delaunay(x, y)
    check_structure(x, y, simp, nb)
    check_delaunay(x, y, simp, nb)
    # differences with Qhull only where the triangulation is ambiguous (both are valid Delaunay)
    ref = Delaunay(np.stack((x, y), -1))
    check_delaunay(x, y, ref.simplices, ref.neighbors)


@pytest.mark.parametrize('seed', range(10))
def test_order_invariance(seed):
    rng = np.random.default_rng(seed)
    span = 30
    cells = rng.choice(span * span, 400, replace=False)
    y, x = np.divmod(cells, span)
    simp, _ = delaunay(x, y)
    perm = rng.permutation(len(x))
    simp2, _ = delaunay(x[perm], y[perm])
    assert tri_set(perm[simp2]) == tri_set(simp)


def test_full_grid_and_small_cases():
    g = np.stack(np.meshgrid(np.arange(50), np.arange(40)), -1).reshape(-1, 2)
    simp, nb = delaunay(g[:, 0], g[:, 1])
    check_structure(g[:, 0], g[:, 1], simp, nb)
    check_delaunay(g[:, 0], g[:, 1], simp, nb)
    simp, nb = delaunay([0, 1, 0], [0, 0, 1])
    assert len(simp) == 1
    for x, y in [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 1, 1, 0], [0, 0, 0, 1]), ([0, 1], [0, 1])]:
        with pytest.raises(ValueError):
            delaunay(x, y)
