import numpy as np
import pytest

from moraine.api.calamp import rslc2amp, calamp


def test_rslc2amp(rslc):
    s = rslc[:, :, 1]
    np.testing.assert_array_almost_equal(rslc2amp(s), np.abs(s), decimal=4)


@pytest.mark.gpu
def test_rslc2amp_gpu(rslc):
    import cupy as cp
    s = rslc[:, :, 1]
    np.testing.assert_array_almost_equal(rslc2amp(s), rslc2amp(cp.asarray(s)).get(), decimal=4)


@pytest.mark.gpu
def test_calamp_gpu(rslc):
    import cupy as cp
    amp = np.abs(rslc[:, :, 0])
    np.testing.assert_array_almost_equal(calamp(amp), calamp(cp.asarray(amp)).get(), decimal=3)
