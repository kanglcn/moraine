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


# ---------------------------------------------------------------- EMCF

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


def _closure(unw, t, b):
    from moraine.pu import _temporal_network
    tri, half, hull, pairs, t_pair, t_sign = _temporal_network(t, b)
    T = np.arange(len(tri) // 3)
    return sum(t_sign[3 * T + j][None, :] * unw[:, t_pair[3 * T + j]] for j in range(3))


def _wrong_cycles(unw, pairs, true):
    tr = true[:, pairs[:, 0]] - true[:, pairs[:, 1]]
    return float(np.mean(np.rint((unw - unw[0] - (tr - tr[0])) / (2 * np.pi)) != 0))


def test_emcf_noise_free_is_exact_and_closes():
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0)
    unw, pairs, _m = emcf_pc(x, y, ph, t, b)
    assert unw.shape == (len(x), len(pairs)) and unw.dtype == np.float32
    assert _wrong_cycles(unw, pairs, true) == 0
    assert np.abs(_closure(unw, t, b)).max() < 1e-4


@pytest.mark.parametrize('noise, fewer_errors', [(0.7, True), (0.9, False)])
def test_emcf_consistent_closes_and_better_than_mcf_pc(noise, fewer_errors):
    """Consistent with the wrapped phase, every triangle of interferograms closes at every point, and fewer
    wrong cycles than unwrapping every interferogram alone where that is reliable (at 0.9 rad image noise
    both fail in whole areas)."""
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=noise)
    unw, pairs, _m = emcf_pc(x, y, ph, t, b)
    intf = ph[:, pairs[:, 0]] * ph[:, pairs[:, 1]].conj()
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3
    assert np.abs(_closure(unw, t, b)).max() < 1e-3
    unw_mcf = np.stack([mcf_pc(x, y, intf[:, k]) for k in range(len(pairs))], -1)
    if fewer_errors:
        assert _wrong_cycles(unw, pairs, true) < _wrong_cycles(unw_mcf, pairs, true) / 3
    unw_free, _, _m = emcf_pc(x, y, ph, t, b, repair=False)
    assert _wrong_cycles(unw, pairs, true) <= _wrong_cycles(unw_free, pairs, true)


def test_emcf_repair_restores_a_shifted_interferogram():
    """An interferogram shifted by a cycle in a whole area (what unwrapping it alone in space can do) is
    found by the triangles on its two sides and shifted back; nothing else changes."""
    from moraine.pu import emcf_pc, _temporal_network, _emcf_repair
    x, y, t, b, true, ph = emcf_synthetic(noise=0.3)
    unw, pairs, _m = emcf_pc(x, y, ph, t, b)
    tri, half, hull, P, t_pair, t_sign = _temporal_network(t, b)
    interior = np.bincount(t_pair, minlength=len(pairs)) == 2        # pairs between two triangles
    k = int(np.flatnonzero(interior)[len(pairs) // 3])
    area = (x > x.mean()) & (y > y.mean())
    shifted = unw.copy()
    shifted[area, k] += 2 * np.pi
    assert np.abs(_closure(shifted, t, b)).max() > 6
    n_open = _emcf_repair(ph, shifted, pairs, tri, half, hull, t_pair, t_sign, 1, np.ones(len(pairs), np.int64),
                          np.zeros(shifted.shape, bool), 1)
    assert (n_open > 0).sum() == area.sum()
    np.testing.assert_allclose(shifted, unw, atol=1e-4)


@pytest.mark.parametrize('mode', ['constant', 'length', 'gradient', 'length+gradient'])
def test_emcf_temporal_step_closes_every_temporal_triangle(mode):
    from moraine.pu import (_temporal_network, _spatial_edges, _emcf_temporal, _pair_gradients, _temporal_costs)
    x, y, t, b, true, ph = emcf_synthetic(noise=1.0, n=800)
    t_tri, t_half, t_hull, pairs, t_pair, t_sign = _temporal_network(t, b)
    s_tri, s_half, s_hull = _mcf_network(x, y)
    s_rep, _, _ = _spatial_edges(s_tri, s_half)
    for earth_cost in (1, 3):
        pair_cost, adaptive = _temporal_costs(t, b, pairs, None, None, mode)
        cycles = _emcf_temporal(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign, earth_cost,
                                pair_cost, adaptive)
        g = np.empty(len(pairs))
        n_corrected = 0
        for s, e0 in enumerate(s_rep):
            p, q = s_tri[e0], s_tri[e0 - e0 % 3 + (e0 + 1) % 3]
            _pair_gradients(ph, p, q, pairs, g)
            G = g + 2 * np.pi * cycles[:, s]
            T = np.arange(len(t_tri) // 3)
            clos = sum(t_sign[3 * T + j] * G[t_pair[3 * T + j]] for j in range(3))
            assert np.abs(clos).max() < 1e-6, (s, clos)
            n_corrected += bool(cycles[:, s].any())
        assert n_corrected > 0          # the noise made residues that the temporal step removed


def test_emcf_order_independent():
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    unw, pairs, _m = emcf_pc(x, y, ph, t, b)
    perm = np.concatenate(([0], 1 + np.random.default_rng(1).permutation(len(x) - 1)))
    unw2, pairs2, _m = emcf_pc(x[perm], y[perm], ph[perm], t, b)
    np.testing.assert_array_equal(pairs2, pairs)
    np.testing.assert_array_equal(unw2, unw[perm])


def test_emcf_network():
    from moraine.pu import _temporal_network
    rng = np.random.default_rng(0)
    t, b = np.sort(rng.uniform(0, 500, 10)), rng.normal(0, 50, 10)
    tri, half, hull, pairs, t_pair, t_sign = _temporal_network(t, b)
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert (np.lexsort(pairs.T[::-1]) == np.arange(len(pairs))).all()              # sorted
    assert set(pairs.ravel()) == set(range(10))
    assert len(pairs) == 3 * 10 - 3 - len(hull)                                     # triangulated
    with pytest.raises(ValueError, match='plane'):
        _temporal_network(t, np.zeros(10))                                           # no baselines
    from moraine.pu import emcf_pc
    with pytest.raises(ValueError, match='temporal_cost'):
        emcf_pc(np.array([0., 5, 0, 5]), np.array([0., 0, 5, 5]), np.ones((4, 10), np.complex64), t, b,
                temporal_cost='cheap')


def test_cli_emcf_pc(tmp_path):
    import toml
    import zarr
    import moraine.cli as mc
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.6, n=1500)
    dates = [(np.datetime64('2021-01-01') + int(d)).astype(str).replace('-', '') for d in t]
    t = np.array([(np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') - np.datetime64('2021-01-01')).astype(int) for d in dates], float)
    toml.dump({'dates': dates, 'perpendicular_baseline': b.tolist(), 'range_pixel_spacing': 4.0,
               'azimuth_pixel_spacing': 3.0}, open(tmp_path / 'meta.toml', 'w'))
    for name, data in [('gix.zarr', np.stack((y, x), -1).astype(np.int32)), ('ph.zarr', ph)]:
        z = zarr.open(str(tmp_path / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=(500, 1))
        z[:] = data
    mc.emcf_pc(str(tmp_path / 'gix.zarr'), str(tmp_path / 'ph.zarr'), str(tmp_path / 'meta.toml'),
               str(tmp_path / 'unw.zarr'), str(tmp_path / 'pairs.txt'), misclosure=str(tmp_path / 'mis.zarr'))
    unw, pairs, mis = emcf_pc(x * 4.0, y * 3.0, ph, t - t[0], b)      # coordinates in meters
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'mis.zarr'), mode='r')[:], mis)
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'pairs.txt', dtype=int).reshape(-1, 2), pairs)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'unw.zarr'), mode='r')[:], unw)


def test_emcf_parallel_and_sparse_are_exact():
    from moraine.pu import (emcf_pc, _temporal_network, _temporal_costs, _spatial_edges, _emcf_temporal,
                            _emcf_cycles)
    x, y, t, b, true, ph = emcf_synthetic(noise=0.9, n=1500)
    u1, p1, m1 = emcf_pc(x, y, ph, t, b, n_workers=1)
    u4, p4, m4 = emcf_pc(x, y, ph, t, b, n_workers=4)
    np.testing.assert_array_equal(u1, u4)
    np.testing.assert_array_equal(m1, m4)
    t_tri, t_half, t_hull, pairs, t_pair, t_sign = _temporal_network(t, b)
    cost, adaptive = _temporal_costs(t, b, pairs, None, None, 'length+gradient')
    s_tri, s_half, _ = _mcf_network(x, y)
    s_rep, _, _ = _spatial_edges(s_tri, s_half)
    dense = _emcf_temporal(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign, 1, cost, adaptive)
    ptr, edge, val = _emcf_cycles(ph, s_tri, s_rep, t_tri, t_half, t_hull, pairs, t_pair, t_sign, 1, cost, adaptive,
                                  block=97)
    rebuilt = np.zeros_like(dense)
    for k in range(len(pairs)):
        rebuilt[k, edge[ptr[k]:ptr[k + 1]]] = val[ptr[k]:ptr[k + 1]]
    np.testing.assert_array_equal(rebuilt, dense)
    assert dense.any()


@pytest.mark.parametrize('spatial_cost', ['gradient', 'correction', 'length', 'weight',
                                          'gradient+correction+length+weight'])
def test_emcf_spatial_costs(spatial_cost):
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.8, n=1500)
    weight = np.random.default_rng(0).uniform(0, 1, len(x))
    unw, pairs, mis = emcf_pc(x, y, ph, t, b, weight=weight, spatial_cost=spatial_cost)
    intf = ph[:, pairs[:, 0]] * ph[:, pairs[:, 1]].conj()
    assert np.abs(wrap(unw - np.angle(intf))).max() < 1e-3
    assert np.abs(_closure(unw, t, b)).max() < 1e-3
    assert ((mis >= 0) & (mis <= 1)).all()


def test_emcf_misclosure_and_errors():
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.0, n=1500)
    unw, pairs, mis = emcf_pc(x, y, ph, t, b)
    assert mis.shape == (len(x),) and mis.dtype == np.float32 and (mis == 0).all()
    with pytest.raises(ValueError, match='spatial_cost'):
        emcf_pc(x, y, ph, t, b, spatial_cost='gradient+cheap')
    with pytest.raises(ValueError, match='weights'):
        emcf_pc(x, y, ph, t, b, spatial_cost='weight')


@pytest.mark.slow
def test_emcf_benchmark():
    """Realistic synthetic data (clusters of points linked by sparse points, a winter gap, seasonal deformation,
    DEM error, atmosphere, noise from coherence and snow), 8 realisations: EMCF must stay clearly better than
    unwrapping every interferogram alone. Reference (2026-09-29): wrong cycles median 0.10 %, max 10.3 %,
    wrong edges 0.088 %; mcf_pc 1.08 %, 43.5 %, 0.174 %."""
    from moraine.pu import emcf_pc
    from unwrap_benchmark import make, scores
    r = []
    for seed in range(8):
        d = make(seed=seed)
        unw, pairs, _ = emcf_pc(d['x'], d['y'], d['ph'], d['t'], d['b'])
        r.append(scores(unw, pairs, d['true'], d['x'], d['y']))
    r = np.array(r)
    assert np.median(r[:, 0]) < 0.003
    assert r[:, 0].max() < 0.2
    assert r[:, 1].mean() < 0.0011


def test_emcf_exclude_images(tmp_path):
    """Excluded images are left out of the network; the pairs keep the image numbers of the full stack."""
    import toml
    import zarr
    import moraine.cli as mc
    from moraine.pu import emcf_pc
    x, y, t, b, true, ph = emcf_synthetic(noise=0.3, n=1500)
    unw, pairs, mis = emcf_pc(x, y, ph, t, b, exclude=[3, 7])
    assert not np.isin(pairs, [3, 7]).any() and set(pairs.ravel()) == set(range(12)) - {3, 7}
    keep = np.setdiff1d(np.arange(12), [3, 7])
    ref, ref_pairs, _ = emcf_pc(x, y, ph[:, keep], t[keep], b[keep])
    np.testing.assert_array_equal(pairs, keep[ref_pairs])
    np.testing.assert_array_equal(unw, ref)
    with pytest.raises(ValueError, match='at least 3'):
        emcf_pc(x, y, ph, t, b, exclude=list(range(10)))
    # command: dates
    dates = [(np.datetime64('2021-01-01') + int(d)).astype(str).replace('-', '') for d in t]
    toml.dump({'dates': dates, 'perpendicular_baseline': b.tolist()}, open(tmp_path / 'meta.toml', 'w'))
    for name, data in [('gix.zarr', np.stack((y, x), -1).astype(np.int32)), ('ph.zarr', ph)]:
        z = zarr.open(str(tmp_path / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=(500, 1))
        z[:] = data
    mc.emcf_pc(str(tmp_path / 'gix.zarr'), str(tmp_path / 'ph.zarr'), str(tmp_path / 'meta.toml'),
               str(tmp_path / 'unw.zarr'), str(tmp_path / 'pairs.txt'), exclude=[dates[3], dates[7]])
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'pairs.txt', dtype=int), pairs)
    with pytest.raises(ValueError, match='not in'):
        mc.emcf_pc(str(tmp_path / 'gix.zarr'), str(tmp_path / 'ph.zarr'), str(tmp_path / 'meta.toml'),
                   str(tmp_path / 'unw2.zarr'), str(tmp_path / 'pairs2.txt'), exclude='19990101')
