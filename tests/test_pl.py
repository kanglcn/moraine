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


def _phase_error(ph, theta, ref=0):
    """Largest absolute phase error in radians of each point, relative to image `ref`."""
    return np.abs(np.angle(ph * np.exp(-1j * (theta - theta[..., [ref]])))).max(axis=-1)


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


@pytest.mark.gpu
def test_ds_temp_coh_gpu(ds_coh):
    import cupy as cp
    ph = emi(ds_coh)[0]
    np.testing.assert_array_almost_equal(ds_temp_coh(ds_coh, ph), ds_temp_coh(cp.asarray(ds_coh), cp.asarray(ph)).get())


def test_emperical_co_emi_temp_coh_pc(ds_can, ds_coh):
    ph, quality, t_coh = emperical_co_emi_temp_coh_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000)
    ph_, quality_ = emi(ds_coh)
    np.testing.assert_array_equal(ph, ph_)
    np.testing.assert_array_equal(quality, quality_)
    np.testing.assert_array_equal(t_coh, ds_temp_coh(ds_coh, ph_))
