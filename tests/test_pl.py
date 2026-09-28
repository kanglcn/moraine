import numpy as np
import pytest

import moraine as mr
from moraine.pl import emi, ds_temp_coh, emperical_co_emi_temp_coh_pc


@pytest.fixture(scope='module')
def ds_coh(ds_can):
    return mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'])


def test_emi(ds_coh):
    ph, quality = emi(ds_coh)
    assert ph.shape == (ds_coh.shape[0], 17)
    assert quality.shape == (ds_coh.shape[0],)
    np.testing.assert_allclose(np.abs(ph), 1, rtol=1e-5)
    np.testing.assert_allclose(ph[:, 0], 1, rtol=1e-5)          # reference image


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
