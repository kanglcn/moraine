import numpy as np
import pytest

import moraine as mr
from moraine.api.pl import emi, ds_temp_coh, emperical_co_emi_temp_coh_pc


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
    ph, quality = emi(ds_coh)
    assert ph.shape == (ds_coh.shape[0], 17)
    assert quality.shape == (ds_coh.shape[0],)
    np.testing.assert_allclose(np.abs(ph), 1, rtol=1e-5)
    np.testing.assert_allclose(ph[:, 0], 1, rtol=1e-5)          # reference image


def test_emi_ill_conditioned_closing_phases(rng):
    """Phases that close, coherence magnitude 1 - 1e-3 (condition number 2e4): quality 1."""
    nimages = 20
    theta = rng.uniform(-np.pi, np.pi, (20, nimages))
    coh_mag = np.full((nimages, nimages), 1 - 1e-3) + 1e-3 * np.eye(nimages)
    ph, quality = emi(_compress(_consistent_coh(theta, coh_mag)).astype(np.complex64))
    assert _phase_error(ph, theta).max() < 1e-3
    np.testing.assert_allclose(quality, 1, atol=1e-2)


def test_emi_regularize_not_positive_definite(not_pd):
    coh, theta = not_pd
    ph, quality = emi(coh)
    assert np.mean(quality < 0) > 0.5                       # EMI fails where |coh| is not positive definite
    ph_reg, quality_reg = emi(coh, regularize=True)
    assert (quality_reg > 0).all()
    np.testing.assert_allclose(np.abs(ph_reg), 1, rtol=1e-5)
    error, error_reg = _phase_error(ph, theta), _phase_error(ph_reg, theta)
    assert np.median(error_reg) < 0.5 < np.median(error)


def test_emi_regularize_keeps_positive_definite(ds_coh):
    pd = _well_conditioned_pd(ds_coh)
    assert pd.sum() > 100
    ph, quality = emi(ds_coh)
    ph_reg, quality_reg = emi(ds_coh, regularize=True)
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
    ph, quality = emi((coh * (1 - beta)).astype(np.complex64))
    assert _phase_error(ph, theta).max() < 1e-4
    np.testing.assert_allclose(quality, 1, atol=1e-4)


def test_emi_regularize_numerically_singular(rng):
    """Phases that close, coherence magnitude 1 - 1e-6 (condition number ~1e7)."""
    nimages = 20
    theta = rng.uniform(-np.pi, np.pi, (5, nimages))
    coh_mag = np.full((nimages, nimages), 1 - 1e-6) + 1e-6 * np.eye(nimages)
    ph, quality = emi(_compress(_consistent_coh(theta, coh_mag)).astype(np.complex64), regularize=True)
    assert _phase_error(ph, theta).max() < 1e-3
    np.testing.assert_allclose(quality, 1, atol=1e-2)


@pytest.mark.gpu
def test_emi_regularize_gpu(not_pd, ds_coh):
    import cupy as cp
    for c in (not_pd[0], ds_coh):
        ph, quality = emi(c, regularize=True)
        ph_gpu, quality_gpu = (a.get() for a in emi(cp.asarray(c), regularize=True))
        # CPU and GPU eigen decompositions round differently in float32
        np.testing.assert_allclose(ph_gpu, ph, atol=1e-2)
        np.testing.assert_allclose(quality_gpu, quality, rtol=1e-2, atol=1e-2)
    pd = _well_conditioned_pd(ds_coh)
    ph, quality = (a.get() for a in emi(cp.asarray(ds_coh)))
    ph_reg, quality_reg = (a.get() for a in emi(cp.asarray(ds_coh), regularize=True))
    # batched GPU eigen solvers may round differently when other matrices of the batch change
    np.testing.assert_allclose(ph_reg[pd], ph[pd], atol=1e-5)
    np.testing.assert_allclose(quality_reg[pd], quality[pd], rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_ds_temp_coh_gpu(ds_coh):
    import cupy as cp
    ph = emi(ds_coh)[0]
    np.testing.assert_array_almost_equal(ds_temp_coh(ds_coh, ph), ds_temp_coh(cp.asarray(ds_coh), cp.asarray(ph)).get())


@pytest.mark.parametrize('regularize', [False, True])
def test_emperical_co_emi_temp_coh_pc(ds_can, ds_coh, regularize):
    ph, quality, t_coh = emperical_co_emi_temp_coh_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000,
                                                      regularize=regularize)
    ph_, quality_ = emi(ds_coh, regularize=regularize)
    np.testing.assert_array_equal(ph, ph_)
    np.testing.assert_array_equal(quality, quality_)
    np.testing.assert_array_equal(t_coh, ds_temp_coh(ds_coh, ph_))
