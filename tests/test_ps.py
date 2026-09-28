import numpy as np
import pytest

from moraine.ps import amp_disp


@pytest.mark.gpu
def test_amp_disp_gpu(rslc):
    import cupy as cp
    s = rslc[:500, :500]
    np.testing.assert_array_almost_equal(amp_disp(s), amp_disp(cp.asarray(s)).get())
