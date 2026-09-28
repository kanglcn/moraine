import numpy as np
import pytest
import zarr

import moraine as mr
from moraine.pu import mcf_pc, gamma_mcf_pt
from conftest import data_path


def wrap(unw):
    return np.mod(unw + np.pi, 2 * np.pi) - np.pi


@pytest.fixture(scope='module')
def ds_ph(ds_can):
    ph, quality, t_coh = mr.emperical_co_emi_temp_coh_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'], batch_size=1000)
    keep = (quality >= 1.0) & (quality < 1.2) & (t_coh > 0.7) & (t_coh <= 1.0)
    gix, ph = ds_can['gix'][keep], ph[keep]
    key = mr.pc_sort(mr.pc_hix(gix, shape=ds_can['rslc'].shape[:2]))
    gix, ph = gix[key], ph[key]
    ph = ph * ph[:, 0:1].conj()
    return gix, ph[:, 14] * ph[:, 9].conj()


def test_mcf_pc(ds_ph):
    gix, ph = ds_ph
    unw = mcf_pc(gix[:, 1], gix[:, 0], ph)
    np.testing.assert_array_almost_equal(wrap(unw), np.angle(ph), decimal=3)


def test_mcf_pc_matches_gamma(ds_ph):
    pytest.importorskip('py_gamma')
    gix, ph = ds_ph
    np.testing.assert_array_almost_equal(mcf_pc(gix[:, 1], gix[:, 0], ph), gamma_mcf_pt(gix[:, 1], gix[:, 0], ph), decimal=3)
