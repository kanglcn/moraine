import numpy as np
import pytest

import moraine as mr
from moraine.api.pqm import temp_coh


@pytest.mark.gpu
def test_temp_coh_ds_gpu(ds_can):
    import cupy as cp
    coh = mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'])
    ph = mr.emi(coh)
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


def test_temp_coh_keeps_inputs(rng):
    # contiguous raster and point cloud stacks of non unit amplitude; the inputs must not change
    pairs = mr.TempNet.from_bandwidth(6, 2).image_pairs
    s = (rng.standard_normal((4, 5, 6)) + 1j * rng.standard_normal((4, 5, 6))).astype(np.complex64)
    noise = np.exp(1j * rng.normal(0, 0.5, (4, 5, pairs.shape[0])))
    intf = (s[..., pairs[:, 0]] * s[..., pairs[:, 1]].conj() * noise).astype(np.complex64)
    u = lambda z: z / np.abs(z)
    expected = np.abs((u(intf) * u(s[..., pairs[:, 0]]).conj() * u(s[..., pairs[:, 1]])).mean(axis=-1))
    for shape in (s.shape[:2], (s.shape[0] * s.shape[1],)):
        intf_in = intf.reshape(*shape, -1).copy(); s_in = s.reshape(*shape, -1).copy()
        t = temp_coh(intf_in, s_in, pairs)
        np.testing.assert_allclose(t, expected.reshape(shape), rtol=1e-5)
        np.testing.assert_array_equal(intf_in, intf.reshape(*shape, -1))
        np.testing.assert_array_equal(s_in, s.reshape(*shape, -1))
