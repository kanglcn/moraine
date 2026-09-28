import numpy as np

from moraine.tnet import TempNet, nimage_from_npair


def test_from_bandwidth():
    tnet = TempNet.from_bandwidth(5, 2)
    np.testing.assert_array_equal(tnet.image_pairs, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])


def test_image_pairs_idx():
    tnet = TempNet(np.stack(([0, 1, 2, 3], [1, 2, 3, 4]), axis=-1))
    assert tnet.image_pairs_idx(ref=1, sec=2) == 1
    np.testing.assert_array_equal(tnet.image_pairs_idx(ref=[1, 2], sec=[2, 3]), [1, 2])


def test_nimage_from_npair():
    assert nimage_from_npair(6) == 4
