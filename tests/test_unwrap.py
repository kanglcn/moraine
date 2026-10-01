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
from moraine.api.unwrap.mcf import mcf_pc, _mcf_network, _mcf_residues, _mcf_ssp, _mcf_integrate
from moraine.api.unwrap.gamma import gamma_mcf_pt


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


def test_mcf_pc_length_cost():
    """Arc costs from the edge length: consistent with the wrapped phase, independent of the point order,
    long edges cheaper (a cost of 1 + 99 x median / length up to 1)."""
    from moraine.api.unwrap.mcf import _mcf_cost_network, _mcf_edges
    x, y, t, b, true, ph = emcf_synthetic(noise=0.9, n=1500)
    intf = ph[:, 3] * ph[:, 6].conj()
    unw = mcf_pc(x, y, intf, spatial_cost='length')
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3
    perm = np.concatenate(([0], 1 + np.random.default_rng(1).permutation(len(x) - 1)))
    np.testing.assert_array_equal(mcf_pc(x[perm], y[perm], intf[perm], spatial_cost='length'), unw[perm])
    tri, half, hull, cost = _mcf_cost_network(x, y, 'length')
    e = np.arange(len(tri))
    length = np.hypot(*(np.stack((x, y))[:, tri[e - e % 3 + (e + 1) % 3]] - np.stack((x, y))[:, tri]))
    med = np.median(np.hypot(*(np.stack((x, y))[:, _mcf_edges(x, y)[3][:, 1]] -
                               np.stack((x, y))[:, _mcf_edges(x, y)[3][:, 0]])))
    np.testing.assert_array_equal(cost, 1 + np.rint(99 * np.minimum(med / length, 1)).astype(np.int32))
    assert _mcf_cost_network(x, y, 'constant')[3] is None
    with pytest.raises(ValueError, match='spatial_cost'):
        mcf_pc(x, y, intf, spatial_cost='weight')


# ---------------------------------------------------------------- EMCF and closure correction

from moraine.api.unwrap.closure import _l1_fit, _pair_forest, _closure_estimate
from unwrap_benchmark import make, make_islands, scores, wrong_cycles


def emcf_synthetic(n=3000, nimg=12, noise=0.5, seed=0):
    """Points on a 10 pixel grid with 10 % gaps, images over two years with baselines; deformation bowl,
    ramp and DEM error, scaled so that no noise free interferogram aliases (neighbour differences below
    2.5 rad), plus noise."""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n / 0.9))
    g = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)
    g = g[rng.random(len(g)) < 0.9]
    x, y = 10.0 * g[:, 0], 10.0 * g[:, 1]
    t = np.sort(rng.uniform(0, 700, nimg))
    t -= t[0]
    b = rng.normal(0, 80, nimg)
    u, v = g[:, 0] / side, g[:, 1] / side
    vel = 10 * np.exp(-((u - 0.4) ** 2 + (v - 0.5) ** 2) / 0.02) + 3 * u     # rad / year
    dem = 0.03 * np.sin(6 * u) * np.cos(5 * v)                               # rad / m of baseline
    clean = vel[:, None] * t[None, :] / 365 + dem[:, None] * b[None, :]
    tri, half, _ = _mcf_network(x, y)
    e = np.arange(len(tri))
    d = clean[tri[e - e % 3 + (e + 1) % 3]] - clean[tri]                      # image gradients along edges
    worst = np.abs(d[:, :, None] - d[:, None, :]).max()                       # interferogram gradients
    clean *= min(1.0, 2.5 / worst)
    true = clean + rng.normal(0, noise, clean.shape)
    return x, y, t, b, true, np.exp(1j * true).astype(np.complex64)


def hop3(nimg):
    return mr.TempNet.from_bandwidth(nimg, min(3, nimg - 1)).image_pairs


def networks(nimg, rng):
    """Networks of image pairs of different kinds, (name, pairs)."""
    yield 'hop3', hop3(nimg)
    yield 'no triangle', np.array([(i, i + 1) for i in range(nimg - 1)] + [(i, i + 3) for i in range(nimg - 3)])
    yield 'tree', np.c_[np.arange(1, nimg), rng.integers(0, np.arange(1, nimg))]
    h = hop3(nimg // 2)
    yield 'two parts', np.r_[h, h + nimg // 2]
    a, b = rng.integers(0, nimg, 3 * nimg), rng.integers(0, nimg, 3 * nimg)
    yield 'random directions', np.c_[a[a != b], b[a != b]]


def loops_close(unw, ph, pairs):
    """Per point, whether the unwrapped interferograms of every loop of image pairs add up to zero."""
    p64 = np.ascontiguousarray(pairs, dtype=np.int64)
    order, parent, _ = _pair_forest(p64, ph.shape[1])
    _, frac, _ = _closure_estimate(np.ascontiguousarray(ph), np.asarray(unw, np.float32), p64, order, parent)
    return frac == 0


def l1_fit_lp(pairs, m, c, n_images):
    """min over n of sum c |m + n_a - n_b| by linear programming (the constraint matrix is totally unimodular)."""
    from scipy.optimize import linprog
    K = len(m)
    A = np.zeros((K, n_images + 2 * K))
    A[np.arange(K), pairs[:, 0]] = 1
    A[np.arange(K), pairs[:, 1]] -= 1
    A[:, n_images:n_images + K] = -np.eye(K)
    A[:, n_images + K:] = np.eye(K)
    r = linprog(np.r_[np.zeros(n_images), c, c], A_eq=A, b_eq=-m,
                bounds=[(None, None)] * n_images + [(0, None)] * 2 * K)
    return r.fun


def test_l1_fit_optimal_on_any_network():
    """The L1 fit of whole cycles is optimal (same cost as a linear program) on any graph of image pairs, and
    its correction is a difference of image cycles."""
    rng = np.random.default_rng(0)
    for nimg in (4, 9, 20):
        for name, pairs in networks(nimg, rng):
            pairs = np.ascontiguousarray(pairs, dtype=np.int64)
            order, parent, loops = _pair_forest(pairs, nimg)
            for _ in range(15):
                m = (rng.integers(-2, 3, len(pairs)) * (rng.random(len(pairs)) < 0.3)).astype(np.int64)
                c = rng.integers(1, 20, len(pairs)).astype(np.int64)
                z = np.empty(len(pairs), np.int64)
                _l1_fit(pairs, m, c, order, parent, z)
                assert np.isclose(np.sum(c * np.abs(z)), l1_fit_lp(pairs, m, c, nimg)), (name, nimg)
                check = np.empty(len(pairs), np.int64)
                _l1_fit(pairs, -(z - m), np.ones(len(pairs), np.int64), order, parent, check)
                assert not check.any(), (name, nimg)              # z - m = n_a - n_b closes every loop
                if loops == 0:
                    assert not z.any()


@pytest.mark.parametrize('mode', ['constant', 'length'])
def test_emcf_temporal_step_closes_and_is_optimal(mode):
    """On every edge of the points, the corrected gradients are differences of image gradients (every loop of
    image pairs closes), at the least cost."""
    from moraine.api.unwrap.emcf import _emcf_temporal, _pair_costs
    from moraine.api.unwrap.mcf import _mcf_edges
    x, y, t, b, true, ph = emcf_synthetic(noise=1.0, n=800)
    pairs = np.ascontiguousarray(hop3(len(t)), dtype=np.int64)
    order, parent, _ = _pair_forest(pairs, len(t))
    cost = _pair_costs(t, pairs, mode)
    edges = _mcf_edges(x, y)[3]
    cycles = _emcf_temporal(ph, edges[:, 0], edges[:, 1], pairs, cost, order, parent)
    assert cycles.shape == (len(edges), len(pairs)) and cycles.dtype == np.int8
    corrected = np.flatnonzero(cycles.any(1))
    assert len(corrected) > 0
    a, b_ = pairs[:, 0], pairs[:, 1]
    for s in corrected[:200]:
        p, q = edges[s]
        dimg = np.angle(ph[q] * ph[p].conj())
        g = np.angle((ph[q, a] * ph[p, a].conj()) * (ph[q, b_] * ph[p, b_].conj()).conj())
        m = np.rint((dimg[a] - dimg[b_] - g) / (2 * np.pi)).astype(np.int64)
        z = cycles[s].astype(np.int64)
        check = np.empty(len(pairs), np.int64)
        _l1_fit(pairs, -(z - m), np.ones(len(pairs), np.int64), order, parent, check)
        assert not check.any()
        assert np.isclose(np.sum(cost * np.abs(z)), l1_fit_lp(pairs, m, cost.astype(float), len(t)))


def test_emcf_noise_free_is_exact():
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0)
    unw, pairs = mr.emcf_pc(x, y, ph, t)
    assert unw.shape == (len(x), len(pairs)) and unw.dtype == np.float32
    assert pairs.dtype == np.int32
    np.testing.assert_array_equal(pairs, hop3(len(t)))                     # default network
    assert not wrong_cycles(unw, pairs, true).any()
    assert loops_close(unw, ph, pairs).all()


def test_emcf_consistent_and_better_than_mcf_pc():
    """Consistent with the wrapped phase, and fewer wrong cycles than unwrapping every interferogram alone
    (measured 2026-09-30: 0.0012 against 0.0053)."""
    x, y, t, b, true, ph = emcf_synthetic(noise=0.7)
    unw, pairs = mr.emcf_pc(x, y, ph, t)
    intf = ph[:, pairs[:, 0]] * ph[:, pairs[:, 1]].conj()
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3
    unw_mcf = np.stack([mcf_pc(x, y, intf[:, k]) for k in range(len(pairs))], -1)
    assert np.mean(wrong_cycles(unw, pairs, true) != 0) < np.mean(wrong_cycles(unw_mcf, pairs, true) != 0) / 2


def test_emcf_any_network():
    """Any network of image pairs, also one without triangles (loops of four images) or in reversed direction;
    the pairs are returned as given."""
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0, n=1500)
    nimg = len(t)
    square = np.array([(i, i + 1) for i in range(nimg - 1)] + [(i + 3, i) for i in range(nimg - 3)])
    unw, pairs = mr.emcf_pc(x, y, ph, t, image_pairs=square)
    np.testing.assert_array_equal(pairs, square)
    assert not wrong_cycles(unw, pairs, true).any()
    intf = ph[:, pairs[:, 0]] * ph[:, pairs[:, 1]].conj()
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3


def test_emcf_without_loops_is_mcf():
    """Without any loop (sequential pairs) the temporal step changes nothing: with constant costs the result
    is the one of `mcf_pc` of every interferogram (relative to the first point), and a warning is given."""
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    seq = np.c_[np.arange(len(t) - 1), np.arange(1, len(t))]
    with pytest.warns(UserWarning, match='no loop'):
        unw, pairs = mr.emcf_pc(x, y, ph, t, image_pairs=seq, spatial_cost='constant')
    intf = ph[:, seq[:, 0]] * ph[:, seq[:, 1]].conj()
    for k in range(len(seq)):
        u = mcf_pc(x, y, intf[:, k])
        np.testing.assert_allclose(unw[:, k] - unw[0, k], u - u[0], atol=1e-3)


def test_emcf_order_independent():
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    unw, pairs = mr.emcf_pc(x, y, ph, t)
    perm = np.concatenate(([0], 1 + np.random.default_rng(1).permutation(len(x) - 1)))
    unw2, pairs2 = mr.emcf_pc(x[perm], y[perm], ph[perm], t)
    np.testing.assert_array_equal(pairs2, pairs)
    np.testing.assert_array_equal(unw2, unw[perm])


def test_emcf_parallel_is_exact():
    x, y, t, b, true, ph = emcf_synthetic(noise=0.9, n=1500)
    u1, _ = mr.emcf_pc(x, y, ph, t, n_workers=1)
    u4, _ = mr.emcf_pc(x, y, ph, t, n_workers=4)
    np.testing.assert_array_equal(u1, u4)


@pytest.mark.parametrize('spatial_cost', ['constant', 'length', 'weight', 'length+weight', 'weight+length'])
def test_emcf_spatial_costs(spatial_cost):
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    weight = np.random.default_rng(0).uniform(0, 1, len(x))
    unw, pairs = mr.emcf_pc(x, y, ph, t, weight=weight, spatial_cost=spatial_cost)
    intf = ph[:, pairs[:, 0]] * ph[:, pairs[:, 1]].conj()
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3


def test_emcf_errors():
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0, n=300)
    with pytest.raises(ValueError, match='temporal_cost'):
        mr.emcf_pc(x, y, ph, t, temporal_cost='cheap')
    with pytest.raises(ValueError, match='acquisition times'):
        mr.emcf_pc(x, y, ph, temporal_cost='length')
    mr.emcf_pc(x, y, ph)                                                      # t only for 'length'
    with pytest.raises(ValueError, match='temporal_cost'):
        mr.emcf_pc(x, y, ph, t, temporal_cost='length+gradient')              # removed (decision 0020)
    with pytest.raises(ValueError, match='spatial_cost'):
        mr.emcf_pc(x, y, ph, t, spatial_cost='gradient+length')               # removed (decision 0020)
    with pytest.raises(ValueError, match='weights'):
        mr.emcf_pc(x, y, ph, t, spatial_cost='weight')
    with pytest.raises(ValueError, match='two different images'):
        mr.emcf_pc(x, y, ph, t, image_pairs=np.array([[0, 0], [0, 1]]))
    with pytest.raises(ValueError, match='shape'):
        mr.emcf_pc(x, y, ph, t[:-1])


def islands_exact(seed=0, side=200):
    """Islands, their exact unwrapped interferograms (true phase differences) on a Hop-3 network."""
    d = make_islands(seed=seed, side=side, nimg=12)
    pairs = hop3(len(d['t']))
    exact = (d['true'][:, pairs[:, 0]] - d['true'][:, pairs[:, 1]]).astype(np.float32)
    return d, pairs, exact


def test_closure_restores_shifted_regions():
    """Whole islands shifted by a cycle in some interferograms (what spatial unwrapping across water does) are
    shifted back; the islands are the regions."""
    d, pairs, exact = islands_exact()
    island = d['island']
    assert island.max() >= 2
    wrong = exact.copy()
    wrong[island == 1, 4] += 2 * np.pi
    wrong[island == 2, 9] -= 2 * np.pi
    wrong[island == 2, 20] += 4 * np.pi
    cor, mis, region = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], wrong, pairs)
    assert cor.dtype == np.float32 and mis.dtype == np.float32 and region.dtype == np.int32
    np.testing.assert_allclose(cor, exact, atol=1e-4)
    assert (mis[(island == 1) | (island == 2)] > 0).all() and (mis[(island != 1) & (island != 2)] == 0).all()
    for i in np.unique(island):                                            # one region per island
        assert len(np.unique(region[island == i])) == 1


def test_closure_no_salt_and_pepper():
    """A single wrong point in a large region is left as it is (the region decides, not its own loops), and
    the rest of the region is not changed."""
    d, pairs, exact = islands_exact()
    i = np.flatnonzero(d['island'] == 0)[10]
    wrong = exact.copy()
    wrong[i, [3, 7]] += 2 * np.pi
    cor, mis, region = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], wrong, pairs)
    assert region[i] >= 0
    np.testing.assert_array_equal(cor, wrong)
    assert mis[i] > 0 and (np.delete(mis, i) == 0).all()


def test_closure_small_regions_point_by_point():
    """Points of regions smaller than min_region_points (an isolated point far away) are corrected by their
    own loops and marked -1."""
    d, pairs, exact = islands_exact()
    x, y = np.r_[d['x'], -500.0], np.r_[d['y'], -500.0]
    ph = np.r_[d['ph'], d['ph'][:1]]
    ex = np.r_[exact, exact[:1]]
    wrong = ex.copy()
    wrong[-1, 5] += 2 * np.pi
    cor, mis, region = mr.unwrap_correct_closure_pc(x, y, ph, wrong, pairs)
    assert region[-1] == -1 and (region[:-1] >= 0).all()
    np.testing.assert_allclose(cor, ex, atol=1e-4)
    cor, mis, region = mr.unwrap_correct_closure_pc(x, y, ph, wrong, pairs, min_region_points=1)
    assert region[-1] >= 0


def test_closure_after_unwrapping_islands():
    """After spatial unwrapping of the islands, the correction removes wrong cycles and keeps the result
    consistent with the wrapped phase."""
    d = make_islands(seed=0)
    pairs = hop3(len(d['t']))
    intf = d['ph'][:, pairs[:, 0]] * d['ph'][:, pairs[:, 1]].conj()
    unw = np.stack([mcf_pc(d['x'], d['y'], intf[:, k]) for k in range(len(pairs))], -1)
    cor, mis, region = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], unw, pairs)
    assert np.abs(wrap(cor - np.angle(intf))).max() < 1e-3
    w0, e0 = scores(unw, pairs, d['true'], d['x'], d['y'])
    w1, e1 = scores(cor, pairs, d['true'], d['x'], d['y'])
    assert w1 < 0.9 * w0 and e1 <= e0 * 1.01                             # measured 2026-09-30: 0.033 -> 0.027


def test_closure_without_loops_and_errors():
    d, pairs, exact = islands_exact()
    seq = np.c_[np.arange(11), np.arange(1, 12)]
    unw = (d['true'][:, seq[:, 0]] - d['true'][:, seq[:, 1]]).astype(np.float32)
    with pytest.warns(UserWarning, match='no loop'):
        cor, mis, region = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], unw, seq)
    np.testing.assert_array_equal(cor, unw)
    assert (mis == 0).all()
    x, y, ph = d['x'], d['y'], d['ph']
    with pytest.raises(ValueError, match='rewrap'):
        mr.unwrap_correct_closure_pc(x, y, ph, exact + 1.0, pairs)
    with pytest.raises(ValueError, match='image_pairs'):
        mr.unwrap_correct_closure_pc(x, y, ph, exact[:, :-1], pairs)
    with pytest.raises(ValueError, match='two different images'):
        mr.unwrap_correct_closure_pc(x, y, ph, exact, np.where(pairs == 11, 12, pairs))
    with pytest.raises(ValueError, match='max_edge_factor'):
        mr.unwrap_correct_closure_pc(x, y, ph, exact, pairs, max_edge_factor=1.0)
    with pytest.raises(ValueError, match='min_region_points'):
        mr.unwrap_correct_closure_pc(x, y, ph, exact, pairs, min_region_points=0)


def _write_zarr(path, data, chunks):
    import zarr
    z = zarr.open(str(path), mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks)
    z[:] = data
    return str(path)


@pytest.mark.parametrize('chunk', [97, 2000])
def test_cli_emcf_pc(tmp_path, chunk):
    """The command equals the API (coordinates in meters from the pixel spacings), with blocks of the temporal
    step much smaller than the point cloud (rows of far points read) and with one block."""
    import zarr
    import moraine.cli as mc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    dates = [(np.datetime64('2021-01-01') + int(d)).astype(str).replace('-', '') for d in t]
    t = np.array([(np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') - np.datetime64('2021-01-01')).astype(int)
                  for d in dates], float)
    gix = np.stack((y / 10, x / 10), -1).astype(np.int32)
    weight = np.random.default_rng(0).uniform(0, 1, len(x)).astype(np.float32)
    pairs = hop3(len(t))
    paths = dict(gix=_write_zarr(tmp_path / 'gix.zarr', gix, (chunk, 2)),
                 ph=_write_zarr(tmp_path / 'ph.zarr', ph, (chunk, 1)),
                 weight=_write_zarr(tmp_path / 'w.zarr', weight, (chunk,)))
    for tc, sc in (('constant', 'length'), ('length', 'length+weight')):
        out = str(tmp_path / f'unw_{tc}.zarr')
        mc.emcf_pc(paths['gix'], paths['ph'], pairs, out, range_pixel_spacing=4.0, azimuth_pixel_spacing=3.0,
                   dates=dates, weight=paths['weight'], temporal_cost=tc, spatial_cost=sc)
        unw, _ = mr.emcf_pc(gix[:, 1] * 4.0, gix[:, 0] * 3.0, ph, t, image_pairs=pairs, weight=weight,
                            temporal_cost=tc, spatial_cost=sc)
        z = zarr.open(out, mode='r')
        assert z.chunks == (chunk, 1) and z.dtype == np.float32
        np.testing.assert_array_equal(z[:], unw)
        assert not (tmp_path / f'unw_{tc}.zarr.tmp').exists()


def test_cli_emcf_pc_errors(tmp_path):
    import moraine.cli as mc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0, n=300)
    gix = _write_zarr(tmp_path / 'gix.zarr', np.stack((y / 10, x / 10), -1).astype(np.int32), (100, 2))
    pairs = hop3(len(t))
    args = dict(range_pixel_spacing=1.0, azimuth_pixel_spacing=1.0)
    by_point = _write_zarr(tmp_path / 'ph_rows.zarr', ph, (100, len(t)))
    with pytest.raises(ValueError, match='one image per chunk'):
        mc.emcf_pc(gix, by_point, pairs, str(tmp_path / 'u.zarr'), **args)
    ph_path = _write_zarr(tmp_path / 'ph.zarr', ph, (100, 1))
    with pytest.raises(ValueError, match='acquisition times'):
        mc.emcf_pc(gix, ph_path, pairs, str(tmp_path / 'u.zarr'), temporal_cost='length', **args)
    with pytest.raises(ValueError, match='dates'):
        mc.emcf_pc(gix, ph_path, pairs, str(tmp_path / 'u.zarr'), dates=['20210101', '20210113'], **args)
    with pytest.raises(ValueError, match='two different images'):
        mc.emcf_pc(gix, ph_path, np.array([[0, 0]]), str(tmp_path / 'u.zarr'), **args)


def _islands_files(tmp_path, chunk):
    """Islands after spatial unwrapping of every interferogram, written to zarr (grid index from the
    coordinates, pixel spacings 1)."""
    d = make_islands(seed=0, side=200, nimg=12)
    pairs = hop3(len(d['t']))
    intf = d['ph'][:, pairs[:, 0]] * d['ph'][:, pairs[:, 1]].conj()
    unw = np.stack([mcf_pc(d['x'], d['y'], intf[:, k]) for k in range(len(pairs))], -1).astype(np.float32)
    gix = np.stack((d['y'], d['x']), -1).astype(np.int32)
    paths = dict(gix=_write_zarr(tmp_path / 'gix.zarr', gix, (chunk, 2)),
                 ph=_write_zarr(tmp_path / 'ph.zarr', d['ph'], (chunk, 1)),
                 unw=_write_zarr(tmp_path / 'unw.zarr', unw, (chunk, 1)))
    return d, pairs, unw, paths


@pytest.mark.parametrize('chunk', [97, 100000])
def test_cli_unwrap_correct_closure_pc(tmp_path, chunk):
    """The command equals the API, with blocks much smaller than the point cloud and with one block; the
    optional outputs are written and the temporary zarr removed."""
    import zarr
    import moraine.cli as mc
    d, pairs, unw, paths = _islands_files(tmp_path, chunk)
    out = str(tmp_path / 'cor.zarr')
    mc.unwrap_correct_closure_pc(paths['gix'], paths['ph'], paths['unw'], pairs, out, range_pixel_spacing=1.0,
                                 azimuth_pixel_spacing=1.0, misclosure_fraction=str(tmp_path / 'mis.zarr'),
                                 region=str(tmp_path / 'region.zarr'))
    cor, mis, region = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], unw, pairs)
    z = zarr.open(out, mode='r')
    assert z.dtype == np.float32 and z.chunks == (chunk, 1)
    np.testing.assert_array_equal(z[:], cor)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'mis.zarr'), mode='r')[:], mis)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'region.zarr'), mode='r')[:], region)
    assert (cor != unw).any()                                             # something was corrected
    assert not (tmp_path / 'cor.zarr.tmp').exists()


def test_cli_unwrap_correct_closure_pc_errors(tmp_path):
    import zarr
    import moraine.cli as mc
    d, pairs, unw, paths = _islands_files(tmp_path, 500)
    args = dict(range_pixel_spacing=1.0, azimuth_pixel_spacing=1.0)
    out = str(tmp_path / 'cor.zarr')
    rows = _write_zarr(tmp_path / 'unw_rows.zarr', unw, (500, unw.shape[1]))
    with pytest.raises(ValueError, match='one image'):
        mc.unwrap_correct_closure_pc(paths['gix'], paths['ph'], rows, pairs, out, **args)
    shifted = _write_zarr(tmp_path / 'unw_shifted.zarr', unw + 1.0, (500, 1))
    with pytest.raises(ValueError, match='rewrap'):
        mc.unwrap_correct_closure_pc(paths['gix'], paths['ph'], shifted, pairs, out, **args)
    assert (tmp_path / 'cor.zarr.tmp').exists()                           # kept when the command fails
    with pytest.raises(ValueError, match='columns'):
        mc.unwrap_correct_closure_pc(paths['gix'], paths['ph'], paths['unw'], pairs[:-1], out, **args)
    seq = np.c_[np.arange(11), np.arange(1, 12)]
    seq_unw = _write_zarr(tmp_path / 'unw_seq.zarr', unw[:, :11].copy(), (500, 1))
    with pytest.warns(UserWarning, match='no loop'):
        mc.unwrap_correct_closure_pc(paths['gix'], paths['ph'], seq_unw, seq, out, **args)
    np.testing.assert_array_equal(zarr.open(out, mode='r')[:], unw[:, :11])


@pytest.mark.parametrize('chunk', [97, 2000])
def test_cli_mcf_pc(tmp_path, chunk):
    """The command equals the API on every interferogram (coordinates in meters from the pixel spacings)."""
    import zarr
    import moraine.cli as mc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    gix = np.stack((y / 10, x / 10), -1).astype(np.int32)
    paths = dict(gix=_write_zarr(tmp_path / 'gix.zarr', gix, (chunk, 2)),
                 ph=_write_zarr(tmp_path / 'ph.zarr', ph, (chunk, 1)))
    pairs = hop3(len(t))[:7]
    for sc in ('constant', 'length'):
        out = str(tmp_path / f'unw_{sc}.zarr')
        mc.mcf_pc(paths['gix'], paths['ph'], out, pairs, range_pixel_spacing=4.0, azimuth_pixel_spacing=3.0,
                  spatial_cost=sc, n_workers=3)
        z = zarr.open(out, mode='r')
        assert z.chunks == (chunk, 1) and z.dtype == np.float32
        for k, (a, b_) in enumerate(pairs):
            ref = mcf_pc(gix[:, 1] * 4.0, gix[:, 0] * 3.0, ph[:, a] * ph[:, b_].conj(), spatial_cost=sc)
            np.testing.assert_array_equal(z[:, k], ref.astype(np.float32))
    rows = _write_zarr(tmp_path / 'ph_rows.zarr', ph, (chunk, len(t)))
    with pytest.raises(ValueError, match='one image per chunk'):
        mc.mcf_pc(paths['gix'], rows, str(tmp_path / 'u.zarr'), pairs, range_pixel_spacing=1.0,
                  azimuth_pixel_spacing=1.0)


@pytest.mark.slow
def test_emcf_benchmark():
    """Realistic synthetic data (clusters of points linked by sparse points, a winter gap, seasonal deformation,
    DEM error, atmosphere, noise from coherence and snow), 8 realisations, Hop-3: EMCF, then the closure
    correction, must stay clearly better than unwrapping every interferogram alone. Reference (2026-09-30,
    default costs), share of wrong (point, interferogram), median / max: mcf_pc 0.0154 / 0.066, emcf_pc
    0.0083 / 0.066, emcf_pc + closure 0.0081 / 0.044; wrong edges median: 0.0023, 0.0014, 0.0015."""
    r = []
    for seed in range(8):
        d = make(seed=seed)
        unw, pairs = mr.emcf_pc(d['x'], d['y'], d['ph'], d['t'])
        cor = mr.unwrap_correct_closure_pc(d['x'], d['y'], d['ph'], unw, pairs)[0]
        r.append(scores(unw, pairs, d['true'], d['x'], d['y']) + scores(cor, pairs, d['true'], d['x'], d['y']))
    r = np.array(r)
    assert np.median(r[:, 0]) < 0.012 and np.median(r[:, 2]) < 0.012
    assert r[:, 2].max() < 0.06
    assert np.median(r[:, 1]) < 0.002 and np.median(r[:, 3]) < 0.002
