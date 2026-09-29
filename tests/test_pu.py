"""Phase unwrapping: MCF correctness by independent checks, not by comparison with another program.

- optimality certificate: no negative cycle in the residual network of the flow
- flow conservation: the net outflow of every node is its residue
- brute force on tiny point sets: the point formulation min sum w |c_e + k_j - k_i| over integer k
- known truth, consistency with the wrapped phase, independence of the point order
"""
import itertools

import numpy as np
import pytest
from numba import njit

import moraine as mr
from moraine.pu import mcf_pc, gamma_mcf_pt, _mcf_network, _mcf_residues, _mcf_ssp, _mcf_integrate


def wrap(unw):
    return np.mod(unw + np.pi, 2 * np.pi) - np.pi


@njit(cache=True)
def _parent_cycle(parent, n):
    state = np.zeros(n, np.int8)
    for s in range(n):
        v = s
        while v != -1 and state[v] == 0:
            state[v] = 1
            v = parent[v]
        if v != -1 and state[v] == 1:
            return True
        v = s
        while v != -1 and state[v] == 1:
            state[v] = 2
            v = parent[v]
    return False


@njit(cache=True)
def _has_negative_cycle(n_nodes, arcs, cost, f):
    """Bellman-Ford on the residual network from a virtual source; a cycle in the predecessors proves
    a negative cycle, i.e. a cheaper flow exists."""
    dist = np.zeros(n_nodes, np.int64)
    parent = np.full(n_nodes, -1, np.int64)
    for it in range(n_nodes + 1):
        changed = False
        for e in range(arcs.shape[0]):
            a, b = arcs[e, 0], arcs[e, 1]
            cab = cost[e] if f[e] >= 0 else -cost[e]
            cba = cost[e] if f[e] <= 0 else -cost[e]
            if dist[a] + cab < dist[b]:
                dist[b] = dist[a] + cab
                parent[b] = a
                changed = True
            if dist[b] + cba < dist[a]:
                dist[a] = dist[b] + cba
                parent[a] = b
                changed = True
        if not changed:
            return False
        if _parent_cycle(parent, n_nodes):
            return True
    return True


def solve(x, y, ph, earth_cost=1):
    tri, half, hull = _mcf_network(x, y)
    psi = np.angle(ph).astype(np.float32)
    sup = _mcf_residues(psi, tri)
    f = _mcf_ssp(tri, half, hull, sup, earth_cost)
    unw = _mcf_integrate(psi, tri, half, f, len(psi))
    T = len(tri) // 3
    e = np.arange(len(tri))
    e = e[(half < 0) | (e < half)]                     # every dual arc once
    arcs = np.stack((e // 3, np.where(half[e] >= 0, half[e] // 3, T)), -1)
    cost = np.where(half[e] >= 0, 1, earth_cost).astype(np.int64)
    edges = np.stack((tri[e], tri[e - e % 3 + (e + 1) % 3]), -1)
    return dict(T=T, sup=sup, f=f[e].astype(np.int64), arcs=arcs, cost=cost, edges=edges, psi=psi, unw=unw,
                total=int((np.abs(f[e]) * cost).sum()))


def check(r, ph):
    assert not _has_negative_cycle(r['T'] + 1, r['arcs'], r['cost'], r['f']), 'flow is not optimal'
    div = np.zeros(r['T'] + 1, np.int64)
    np.add.at(div, r['arcs'][:, 0], r['f'])
    np.add.at(div, r['arcs'][:, 1], -r['f'])
    np.testing.assert_array_equal(div, r['sup'])
    assert np.abs(wrap(r['unw'] - np.angle(ph))).max() < 1e-3
    e, psi = r['edges'], r['psi'].astype(float)
    k = np.rint((r['unw'][e[:, 1]] - r['unw'][e[:, 0]] - wrap(psi[e[:, 1]] - psi[e[:, 0]])) / (2 * np.pi))
    assert int((np.abs(k) * r['cost']).sum()) == r['total'], 'integrated phase does not follow the flow'


def synthetic(n, span, noise, rng):
    xy = np.unique(rng.integers(0, span, size=(n, 2)), axis=0)
    x, y = xy[:, 0], xy[:, 1]
    s = 5000 / span
    true = s * (0.004 * x + 0.003 * y + 3 * np.sin(s * x / 700)) + rng.normal(0, noise, len(x))
    return x, y, true, np.exp(1j * true).astype(np.complex64)


@pytest.mark.parametrize('earth_cost', [1, 3])
@pytest.mark.parametrize('span', [60, 5000], ids=['lattice', 'sparse'])
def test_mcf_optimal(earth_cost, span):
    rng = np.random.default_rng(span + earth_cost)
    for _ in range(25):
        x, y, _, ph = synthetic(int(rng.integers(10, 400)), span, float(rng.uniform(0.2, 2.0)), rng)
        check(solve(x, y, ph, earth_cost), ph)


@pytest.mark.parametrize('seed', range(30))
def test_mcf_brute_force(seed):
    rng = np.random.default_rng(seed)
    xy = np.unique(rng.integers(0, 20, size=(int(rng.integers(4, 8)), 2)), axis=0)
    if len(xy) < 4 or np.linalg.matrix_rank(xy - xy[0]) < 2:
        pytest.skip('degenerate points')
    ph = np.exp(1j * rng.uniform(-np.pi, np.pi, len(xy))).astype(np.complex64)   # many residues
    for earth_cost in (1, 3):
        r = solve(xy[:, 0], xy[:, 1], ph, earth_cost)
        e, psi = r['edges'], r['psi'].astype(float)
        c = np.rint((psi[e[:, 1]] - psi[e[:, 0]] - wrap(psi[e[:, 1]] - psi[e[:, 0]])) / (2 * np.pi))
        best = min(int((np.abs(c + np.r_[0, ks][e[:, 1]] - np.r_[0, ks][e[:, 0]]) * r['cost']).sum())
                   for ks in itertools.product(range(-2, 3), repeat=len(xy) - 1))
        assert r['total'] == best


def test_mcf_known_truth():
    g = np.stack(np.meshgrid(np.arange(40), np.arange(30)), -1).reshape(-1, 2)
    true = 1.0 * g[:, 0] + 0.5 * g[:, 1]               # steep, residue free ramp
    unw = mcf_pc(g[:, 0], g[:, 1], np.exp(1j * true))
    np.testing.assert_allclose(unw - unw[0], true - true[0], atol=1e-4)
    x, y, true, ph = synthetic(3000, 3000, 0.3, np.random.default_rng(1))
    r = solve(x, y, ph)
    check(r, ph)
    assert (np.rint((r['unw'] - r['unw'][0] - (true - true[0])) / (2 * np.pi)) != 0).mean() < 0.01


def test_mcf_order_independent():
    rng = np.random.default_rng(5)
    for span in (60, 5000):
        x, y, _, ph = synthetic(800, span, 1.2, rng)
        unw = mcf_pc(x, y, ph)
        perm = np.concatenate(([0], 1 + rng.permutation(len(x) - 1)))   # the first point is the reference
        np.testing.assert_array_equal(mcf_pc(x[perm], y[perm], ph[perm]), unw[perm])


def test_mcf_edge_cases():
    for phase in ([0, 2, -2], [0, 3, 3.1], [3, -3, 0]):
        ph = np.exp(1j * np.array(phase)).astype(np.complex64)
        check(solve(np.array([0, 10, 0]), np.array([0, 0, 10]), ph), ph)
    g = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)), -1).reshape(-1, 2).astype(float)
    ph = np.exp(1j * np.arctan2(g[:, 1] - 0.2, g[:, 0] - 0.3)).astype(np.complex64)   # one vortex
    for earth_cost in (1, 3, 100):
        check(solve(g[:, 0], g[:, 1], ph, earth_cost), ph)
    for x, y in [(np.arange(10.), np.arange(10.)), (np.array([0., 1, 1, 0]), np.array([0., 0, 0, 1]))]:
        with pytest.raises(ValueError):
            mcf_pc(x, y, np.ones(len(x), np.complex64))


@pytest.fixture(scope='module')
def ds_ph(ds_can):
    ph, quality, t_coh = mr.emperical_co_emi_temp_coh_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000)
    keep = (quality >= 1.0) & (quality < 1.2) & (t_coh > 0.7) & (t_coh <= 1.0)
    gix, ph = ds_can['gix'][keep], ph[keep]
    key = mr.pc_sort(mr.pc_hix(gix, shape=ds_can['rslc'].shape[:2]))
    gix, ph = gix[key], ph[key]
    ph = ph * ph[:, 0:1].conj()
    return gix, ph[:, 14] * ph[:, 9].conj()


def test_mcf_pc_sample_data(ds_ph):
    gix, ph = ds_ph
    check(solve(gix[:, 1], gix[:, 0], ph), ph)


def test_mcf_pc_not_worse_than_gamma(ds_ph):
    """GAMMA mcf_pt discharges residues through the border like earth_cost=3; on the same network
    moraine's flow must be at least as cheap. Point by point equality is not expected: the optimum is
    not unique and cocircular points allow different triangulations."""
    pytest.importorskip('py_gamma')
    gix, ph = ds_ph
    r = solve(gix[:, 1], gix[:, 0], ph, earth_cost=3)
    g = gamma_mcf_pt(gix[:, 1], gix[:, 0], ph)
    assert np.abs(wrap(g - np.angle(ph))).max() < 1e-3
    e, psi = r['edges'], r['psi'].astype(float)
    k = np.rint((g[e[:, 1]] - g[e[:, 0]] - wrap(psi[e[:, 1]] - psi[e[:, 0]])) / (2 * np.pi))
    assert r['total'] <= int((np.abs(k) * r['cost']).sum())


needs_gamma = pytest.mark.skipif(__import__('shutil').which('mcf_pt') is None, reason='GAMMA mcf_pt not found')


@pytest.fixture(scope='module')
def ramp_ph(ds_ph):
    """Sample interferogram plus a ramp of about 3 cycles, and a point that is not in the same cycle as
    the first point: with it as reference, the result differs from the default reference."""
    gix, ph = ds_ph
    ph = (ph * np.exp(1j * 0.05 * gix[:, 1])).astype(np.complex64)
    unw0 = gamma_mcf_pt(gix[:, 1], gix[:, 0], ph)
    k = int(np.flatnonzero(np.abs(unw0 - np.angle(ph)) > 1)[0])
    return gix, ph, k


@needs_gamma
def test_gamma_mcf_pt_weights_and_reference(ramp_ph):
    gix, ph, k = ramp_ph
    x, y = gix[:, 1], gix[:, 0]
    w = np.random.default_rng(0).uniform(0.2, 1.0, len(ph))
    # mcf_pt reads FLOAT weights: float64 input must give the same result as float32
    np.testing.assert_array_equal(gamma_mcf_pt(x, y, ph, ph_weight=w), gamma_mcf_pt(x, y, ph, ph_weight=w.astype(np.float32)))
    unw = gamma_mcf_pt(x, y, ph, ref_point=k)
    assert abs(unw[k] - np.angle(ph[k])) < 1e-5      # the reference point keeps its wrapped phase


@needs_gamma
def test_cli_gamma_mcf_pt_reference(ramp_ph, tmp_path):
    import zarr
    import moraine.cli as mc
    gix, ph, k = ramp_ph
    for name, data in [('x.zarr', gix[:, 1].astype(np.float64)), ('y.zarr', gix[:, 0].astype(np.float64)),
                       ('ph.zarr', np.stack((np.ones_like(ph), ph), -1).astype(np.complex64))]:
        z = zarr.open(str(tmp_path / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=data.shape)
        z[:] = data
    mc.gamma_mcf_pt(str(tmp_path / 'x.zarr'), str(tmp_path / 'y.zarr'), str(tmp_path / 'ph.zarr'),
                    str(tmp_path / 'unw.zarr'), np.array([[1, 0]]), ref_point=k)
    unw = zarr.open(str(tmp_path / 'unw.zarr'), mode='r')[:, 0]
    assert abs(unw[k] - np.angle(ph[k])) < 1e-4
