import numpy as np
import pytest

import moraine as mr
from moraine.api.pqm import temp_coh


@pytest.mark.gpu
def test_temp_coh_ds_gpu(ds_can):
    import cupy as cp
    coh = mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'])
    ph = mr.emi(coh)[0]
    np.testing.assert_array_almost_equal(temp_coh(coh, ph), temp_coh(cp.asarray(coh), cp.asarray(ph)).get())


@pytest.mark.gpu
def test_temp_coh_ps_gpu(rslc):
    import cupy as cp
    s = rslc[:300, :300]
    pairs = mr.TempNet.from_bandwidth(s.shape[-1], 1).image_pairs
    intf = s[..., pairs[:, 0]] * s[..., pairs[:, 1]].conj()
    t = temp_coh(intf, s, pairs)
    assert t.shape == s.shape[:2]
    np.testing.assert_array_almost_equal(t, temp_coh(cp.asarray(intf), cp.asarray(s), pairs).get())
