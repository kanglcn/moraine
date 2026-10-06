import os

import numpy as np
import pytest

import moraine as mr
from moraine.api.pl import emi, _emi, ds_temp_coh, emperical_co_emi_temp_coh_pc, _edge_pfa


@pytest.fixture(scope='module')
def ds_coh(ds_can):
    return mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'])


def _compress(full):
    """Coherence matrices (..., nimages, nimages) -> upper triangles (..., n_image_pairs) as used by `emi`."""
    pairs = mr.TempNet.from_bandwidth(full.shape[-1]).image_pairs
    return full[..., pairs[:, 0], pairs[:, 1]]


def _consistent_coh(theta, coh_mag):
    """Coherence matrices whose phases close: exp(j theta_i) coh_mag_ij exp(-j theta_j)."""
    return np.exp(1j * theta)[..., :, None] * coh_mag * np.exp(-1j * theta)[..., None, :]


def _sample_coh(rng, theta, coh_mag, n_points, n_looks):
    """Sample coherence matrices (n_points, nimages, nimages) of `n_looks` looks with true coherence
    `_consistent_coh(theta, coh_mag)`."""
    nimages = theta.shape[0]
    w = rng.standard_normal((n_points, nimages, n_looks)) + 1j * rng.standard_normal((n_points, nimages, n_looks))
    x = np.exp(1j * theta)[:, None] * (np.linalg.cholesky(coh_mag) @ w)
    cov = x @ np.swapaxes(x.conj(), -1, -2)
    amp = np.sqrt(np.einsum('...ii->...i', cov).real)
    return cov / (amp[..., :, None] * amp[..., None, :])


def _phase_error(ph, theta, ref=0):
    """Largest absolute phase error in radians of each point, relative to image `ref`."""
    return np.abs(np.angle(ph * np.exp(-1j * (theta - theta[..., [ref]])))).max(axis=-1)


def _well_conditioned_pd(coh):
    """Points whose coherence magnitude matrix is clearly positive definite and well conditioned."""
    lam = np.linalg.eigvalsh(np.abs(mr.uncompress_coh(coh)))
    return (lam[:, 0] > 1e-3) & (lam[:, -1] < 1e4 * lam[:, 0])


@pytest.fixture(scope='module')
def not_pd():
    """Fewer looks (12) than images (30): the coherence magnitude matrices are mostly not positive definite."""
    rng = np.random.default_rng(1)
    nimages = 30
    theta = rng.uniform(-np.pi, np.pi, nimages)
    coh_mag = 0.8 + 0.2 * np.eye(nimages)
    coh = _compress(_sample_coh(rng, theta, coh_mag, 200, 12)).astype(np.complex64)
    return coh, theta


def test_emi(ds_coh):
    ph = emi(ds_coh)
    assert ph.shape == (ds_coh.shape[0], 17) and ph.dtype == np.complex64
    np.testing.assert_allclose(np.abs(ph), 1, rtol=1e-5)
    np.testing.assert_allclose(ph[:, 0], 1, rtol=1e-5)          # reference image
    np.testing.assert_array_equal(ph, _emi(ds_coh)[0])


def test_emi_ill_conditioned_closing_phases(rng):
    """Phases that close, coherence magnitude 1 - 1e-3 (condition number 2e4): quality 1."""
    nimages = 20
    theta = rng.uniform(-np.pi, np.pi, (20, nimages))
    coh_mag = np.full((nimages, nimages), 1 - 1e-3) + 1e-3 * np.eye(nimages)
    ph, quality = _emi(_compress(_consistent_coh(theta, coh_mag)).astype(np.complex64), regularize=False)
    assert _phase_error(ph, theta).max() < 1e-3
    np.testing.assert_allclose(quality, 1, atol=1e-2)


def test_emi_regularize_not_positive_definite(not_pd):
    coh, theta = not_pd
    ph, quality = _emi(coh, regularize=False)
    assert np.mean(quality < 0) > 0.5                       # EMI fails where |coh| is not positive definite
    ph_reg, quality_reg = _emi(coh, regularize=True)
    assert (quality_reg > 0).all()
    np.testing.assert_allclose(np.abs(ph_reg), 1, rtol=1e-5)
    error, error_reg = _phase_error(ph, theta), _phase_error(ph_reg, theta)
    assert np.median(error_reg) < 0.5 < np.median(error)
    np.testing.assert_array_equal(emi(coh), ph_reg)          # regularized by default


def test_emi_regularize_keeps_positive_definite(ds_coh):
    pd = _well_conditioned_pd(ds_coh)
    assert pd.sum() > 100
    ph, quality = _emi(ds_coh, regularize=False)
    ph_reg, quality_reg = _emi(ds_coh, regularize=True)
    np.testing.assert_array_equal(ph_reg[pd], ph[pd])
    np.testing.assert_array_equal(quality_reg[pd], quality[pd])
    assert (quality_reg[quality < 0] > 0).all()


@pytest.mark.parametrize('beta', [0.0, 0.3, 0.9])
def test_emi_regularized_closing_phases_exact(rng, beta):
    """(1-beta)*coh + beta*I of a coherence matrix whose phases close gives its phases and quality 1."""
    nimages = 20
    theta = rng.uniform(-np.pi, np.pi, (5, nimages))
    t = np.arange(nimages)
    coh = _compress(_consistent_coh(theta, 0.9 ** np.abs(t[:, None] - t[None, :])))
    ph, quality = _emi((coh * (1 - beta)).astype(np.complex64), regularize=False)
    assert _phase_error(ph, theta).max() < 1e-4
    np.testing.assert_allclose(quality, 1, atol=1e-4)


def test_emi_regularize_numerically_singular(rng):
    """Phases that close, coherence magnitude 1 - 1e-6 (condition number ~1e7)."""
    nimages = 20
    theta = rng.uniform(-np.pi, np.pi, (5, nimages))
    coh_mag = np.full((nimages, nimages), 1 - 1e-6) + 1e-6 * np.eye(nimages)
    ph, quality = _emi(_compress(_consistent_coh(theta, coh_mag)).astype(np.complex64), regularize=True)
    assert _phase_error(ph, theta).max() < 1e-3
    np.testing.assert_allclose(quality, 1, atol=1e-2)


@pytest.mark.gpu
def test_emi_regularize_gpu(not_pd, ds_coh):
    import cupy as cp
    for c in (not_pd[0], ds_coh):
        ph, quality = _emi(c, regularize=True)
        ph_gpu, quality_gpu = (a.get() for a in _emi(cp.asarray(c), regularize=True))
        # CPU and GPU eigen decompositions round differently in float32
        np.testing.assert_allclose(ph_gpu, ph, atol=1e-2)
        np.testing.assert_allclose(quality_gpu, quality, rtol=1e-2, atol=1e-2)
    pd = _well_conditioned_pd(ds_coh)
    ph = emi(cp.asarray(ds_coh), regularize=False).get()
    ph_reg = emi(cp.asarray(ds_coh)).get()
    # batched GPU eigen solvers may round differently when other matrices of the batch change
    np.testing.assert_allclose(ph_reg[pd], ph[pd], atol=1e-5)


@pytest.mark.gpu
def test_ds_temp_coh_gpu(ds_coh):
    import cupy as cp
    ph = emi(ds_coh)
    np.testing.assert_array_almost_equal(ds_temp_coh(ds_coh, ph), ds_temp_coh(cp.asarray(ds_coh), cp.asarray(ph)).get())


def _ds_temp_coh_weighted_ref(coh, ph, n_looks):
    """float64 numpy version of the weighted temporal coherence and effective number of image pairs."""
    pairs = mr.TempNet.from_bandwidth(ph.shape[1]).image_pairs
    mag2 = np.abs(coh).astype(np.float64) ** 2
    inv = 1 / np.broadcast_to(np.asarray(n_looks, np.float64), (coh.shape[0],))[:, None]
    w = np.where(inv < 1, np.maximum(0, (mag2 - inv) / np.where(inv < 1, 1 - inv, 1)), 0)
    e = np.where(w > 0, coh / np.where(mag2 > 0, np.abs(coh), 1) * ph[:, pairs[:, 0]].conj() * ph[:, pairs[:, 1]], 0)
    sw = w.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        t_coh_w, eff_n_pairs = np.abs((w * e).sum(axis=1)) / sw, sw ** 2 / (w ** 2).sum(axis=1)
    t_coh_w[sw == 0], eff_n_pairs[sw == 0] = np.nan, 0
    return t_coh_w, eff_n_pairs


def _n_components_ref(coh, n_looks, n_images, alpha):
    """float64 python version of the number of connected components of the graph of the coherent image pairs."""
    pairs = mr.TempNet.from_bandwidth(n_images).image_pairs
    n_looks = np.broadcast_to(np.asarray(n_looks, np.float64), (coh.shape[0],))
    p = _edge_pfa(n_images, alpha)
    out = np.empty(coh.shape[0], np.int16)
    for i in range(coh.shape[0]):
        if n_looks[i] <= 1:
            out[i] = n_images
            continue
        thr = -np.expm1(np.log(p) / (n_looks[i] - 1))
        parent = list(range(n_images))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for k in np.flatnonzero(np.abs(coh[i]).astype(np.float64) ** 2 > thr):
            a, b = find(pairs[k, 0]), find(pairs[k, 1])
            if a != b: parent[max(a, b)] = min(a, b)
        out[i] = sum(parent[x] == x for x in range(n_images))
    return out


@pytest.fixture(scope='module')
def intermittent():
    """Coherence exp(-time span / 1.5 images), 60 looks: only short time spans are coherent."""
    rng = np.random.default_rng(2)
    nimages = 30
    theta = rng.uniform(-np.pi, np.pi, nimages)
    t = np.arange(nimages)
    coh = _compress(_sample_coh(rng, theta, np.exp(-np.abs(t[:, None] - t[None, :]) / 1.5), 500, 60)).astype(np.complex64)
    return coh, emi(coh)


@pytest.mark.gpu
def test_ds_temp_coh_gpu_image_pairs(intermittent):
    """A subset of the image pairs (every image with the next three)."""
    import cupy as cp
    coh, ph = intermittent
    all_pairs = mr.TempNet.from_bandwidth(ph.shape[1]).image_pairs
    pairs = mr.TempNet.from_bandwidth(ph.shape[1], bandwidth=3).image_pairs
    sub = coh[:, [np.flatnonzero((all_pairs == p).all(-1))[0] for p in pairs]]
    t_coh = ds_temp_coh(sub, ph, image_pairs=pairs)
    np.testing.assert_allclose(ds_temp_coh(cp.asarray(sub), cp.asarray(ph), image_pairs=pairs).get(), t_coh, rtol=1e-5, atol=1e-6)
    assert np.median(t_coh) > np.median(ds_temp_coh(coh, ph))      # short time spans are the coherent ones


@pytest.mark.parametrize('n_looks', [60.0, np.inf, 'per point'])
def test_ds_temp_coh_weighted(intermittent, n_looks):
    coh, ph = intermittent
    if n_looks == 'per point':
        n_looks = np.random.default_rng(3).uniform(30, 120, coh.shape[0]).astype(np.float32)
    t_coh, t_coh_w, eff_n_pairs, n_components = ds_temp_coh(coh, ph, n_looks=n_looks)
    assert t_coh_w.dtype == eff_n_pairs.dtype == np.float32 and t_coh_w.shape == eff_n_pairs.shape == (coh.shape[0],)
    assert n_components.dtype == np.int16 and n_components.shape == (coh.shape[0],)
    np.testing.assert_array_equal(t_coh, ds_temp_coh(coh, ph))    # the uniform one does not depend on n_looks
    t_ref, eff_ref = _ds_temp_coh_weighted_ref(coh, ph, n_looks)
    np.testing.assert_allclose(t_coh_w, t_ref, rtol=1e-4)
    np.testing.assert_allclose(eff_n_pairs, eff_ref, rtol=1e-4)
    np.testing.assert_array_equal(n_components, _n_components_ref(coh, n_looks, ph.shape[1], 1e-3))


def test_ds_temp_coh_weighted_intermittent(intermittent):
    """Incoherent long time spans lower the temporal coherence, not the weighted one."""
    coh, ph = intermittent
    t_coh, t_coh_w, eff_n_pairs, n_components = ds_temp_coh(coh, ph, n_looks=60.0)
    assert np.median(t_coh) < 0.5 < 0.8 < np.median(t_coh_w)
    assert np.median(eff_n_pairs) < 0.2 * coh.shape[1]
    assert np.median(n_components) == 1      # the short time spans link all images


def test_ds_temp_coh_weighted_no_information(intermittent):
    coh, ph = intermittent
    _, t_coh_w, eff_n_pairs, n_components = ds_temp_coh(coh * np.float32(0.5), ph, n_looks=2.0)     # |coh|^2 < 1/n_looks
    assert np.isnan(t_coh_w).all() and (eff_n_pairs == 0).all() and (n_components == ph.shape[1]).all()
    _, t_coh_w, eff_n_pairs, n_components = ds_temp_coh(coh, ph, n_looks=np.r_[1.0, 0.5, np.full(coh.shape[0] - 2, 60.0)])
    assert np.isnan(t_coh_w[:2]).all() and (eff_n_pairs[:2] == 0).all() and np.isfinite(t_coh_w[2:]).all()
    assert (n_components[:2] == ph.shape[1]).all()      # n_looks <= 1: every image is alone


def _graph_coh(n_images, magnitude):
    """Coherence of one point with the given symmetric magnitude (n_images, n_images) and phases that close."""
    pairs = mr.TempNet.from_bandwidth(n_images).image_pairs
    return magnitude[pairs[:, 0], pairs[:, 1]][None].astype(np.complex64), np.ones((1, n_images), np.complex64)


def _graph_cases():
    n = 10
    two_groups = np.full((n, n), 0.02); two_groups[:5, :5] = 0.9; two_groups[5:, 5:] = 0.9
    one_alone = np.full((n, n), 0.9); one_alone[9, :] = one_alone[:, 9] = 0.02
    two_pairs = np.full((n, n), 0.02); two_pairs[0, 1] = two_pairs[2, 3] = 0.9
    return n, [(np.full((n, n), 0.9), 1), (two_groups, 2), (one_alone, 2), (np.full((n, n), 0.02), n), (two_pairs, n - 2)]


def test_ds_temp_coh_components():
    """Images that are linked by coherent pairs form one component, whatever the phases of the pairs are."""
    n, cases = _graph_cases()
    for magnitude, expected in cases:
        coh, ph = _graph_coh(n, magnitude)
        assert ds_temp_coh(coh, ph, n_looks=50.0)[3][0] == expected


@pytest.mark.gpu
def test_ds_temp_coh_components_gpu():
    import cupy as cp
    n, cases = _graph_cases()
    for magnitude, expected in cases:
        coh, ph = _graph_coh(n, magnitude)
        assert ds_temp_coh(cp.asarray(coh), cp.asarray(ph), n_looks=cp.asarray(50.0))[3].get()[0] == expected


def test_edge_pfa():
    """p so that a random graph with n nodes and edge probability p is connected with probability alpha."""
    assert _edge_pfa(2, 1e-3) == pytest.approx(1e-3, rel=1e-6)       # one edge
    assert _edge_pfa(3, 0.5) == pytest.approx(0.5, rel=1e-6)         # 3 p^2 - 2 p^3 = 1/2
    assert _edge_pfa(5, 0.01) > _edge_pfa(17, 0.01) > _edge_pfa(92, 0.01) > _edge_pfa(400, 0.01)
    assert _edge_pfa(17, 1e-2) > _edge_pfa(17, 1e-3) > _edge_pfa(17, 1e-4)


@pytest.mark.parametrize('alpha', [0.0, 1e-7, 0.6])
def test_ds_temp_coh_alpha(intermittent, alpha):
    coh, ph = intermittent
    with pytest.raises(ValueError, match='alpha'):
        ds_temp_coh(coh, ph, n_looks=60.0, alpha=alpha)


def test_ds_temp_coh_components_noise():
    """A point without any coherent image pair is connected with probability alpha."""
    rng = np.random.default_rng(5)
    n_images, n_looks, n_points, alpha = 17, 46, 3000, 0.05
    x = rng.standard_normal((n_points, n_images, n_looks, 2)).astype(np.float32).view(np.complex64)[..., 0]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    pairs = mr.TempNet.from_bandwidth(n_images).image_pairs
    coh = np.einsum('pk,pk->p', x[:, pairs[:, 0]].reshape(-1, n_looks), x[:, pairs[:, 1]].conj().reshape(-1, n_looks))
    coh = coh.reshape(n_points, -1).astype(np.complex64)
    ph = np.ones((n_points, n_images), np.complex64)
    connected = (ds_temp_coh(coh, ph, n_looks=float(n_looks), alpha=alpha)[3] == 1).mean()
    assert abs(connected - alpha) < 4 * np.sqrt(alpha * (1 - alpha) / n_points)


@pytest.mark.gpu
def test_ds_temp_coh_weighted_gpu(intermittent, ds_coh):
    import cupy as cp
    coh, ph = intermittent
    n_looks = np.random.default_rng(3).uniform(1.5, 120, coh.shape[0]).astype(np.float32)
    for c, p, n in ((coh, ph, n_looks), (ds_coh, emi(ds_coh), 60.0)):
        cpu = ds_temp_coh(c, p, n_looks=n)
        gpu = [a.get() for a in ds_temp_coh(cp.asarray(c), cp.asarray(p), n_looks=cp.asarray(n))]
        for a, b in zip(gpu, cpu):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        assert gpu[3].dtype == np.int16
        np.testing.assert_array_equal(gpu[3], cpu[3])
        np.testing.assert_array_equal(gpu[0], ds_temp_coh(cp.asarray(c), cp.asarray(p)).get())


@pytest.mark.parametrize('regularize', [False, True])
def test_emperical_co_emi_temp_coh_pc(ds_can, ds_coh, regularize):
    ph, t_coh = emperical_co_emi_temp_coh_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000,
                                             regularize=regularize)
    ph_ = emi(ds_coh, regularize=regularize)
    np.testing.assert_array_equal(ph, ph_)
    np.testing.assert_array_equal(t_coh, ds_temp_coh(ds_coh, ph_))


def test_emperical_co_emi_temp_coh_pc_weighted(ds_can, ds_coh):
    """The effective number of looks is that of emperical_co_pc(..., return_n_looks=True)."""
    ph, t_coh, t_coh_w, eff_n_pairs, n_components = emperical_co_emi_temp_coh_pc(
        ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000, weighted=True)
    coh, n_looks = mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], return_n_looks=True)
    np.testing.assert_array_equal(coh, ds_coh)
    assert ((1 <= n_looks) & (n_looks <= np.count_nonzero(ds_can['is_shp'], axis=(1, 2)))).all()
    ref = ds_temp_coh(ds_coh, emi(ds_coh), n_looks=n_looks)
    for a, b in zip((t_coh, t_coh_w, eff_n_pairs, n_components), ref):
        np.testing.assert_array_equal(a, b)


def _blas_threads():
    from threadpoolctl import threadpool_info
    return [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']


@pytest.mark.skipif(not os.path.isdir('/proc/self/task'), reason='counts threads in /proc')
def test_emi_cpu_single_thread_blas(not_pd):
    """The CPU kernel limits the BLAS threads while it runs (a multithreaded BLAS called from every numba
    thread starts threads in each call), also when called from several threads, and restores them."""
    import threading
    before = _blas_threads()
    coh = np.tile(not_pd[0], (20, 1))
    workers = [threading.Thread(target=_emi, args=(coh,)) for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert _blas_threads() == before
    assert len(os.listdir('/proc/self/task')) < 4 * os.cpu_count() + 64
