import numpy as np

from moraine.api.coord_ import Coord


def test_coords2gixs_and_rasterize():
    coord = Coord(1.8, 0.2, 10, -1.2, 0.2, 10)
    x = np.array([1.91, 1.88, 1.87, 3.43, 2.8])
    y = np.array([-1.11, -1.09, -0.81, -0.4, 0.11])
    gix = coord.coords2gixs(np.stack((y, x), axis=-1))
    np.testing.assert_array_equal(gix, np.stack(([0, 1, 2, 4, 7], [1, 0, 0, 8, 5]), axis=-1))

    pc = np.random.default_rng(0).random((5, 3, 2))
    ras = coord.rasterize(pc, gix)
    iidx = coord.rasterize_iidx(gix)
    assert ras.shape[2:] == pc.shape[1:]
    assert iidx.shape == (10, 10)
    expected = pc[iidx]
    expected[iidx == -1] = np.nan
    np.testing.assert_array_equal(expected, ras)
