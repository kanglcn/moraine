import numpy as np
import pytest

from moraine.api.tnet import TempNet, check_image_pairs, nimage_from_npair

# image pairs against the data conventions: sorted without duplicates, every reference index smaller
AGAINST_CONVENTIONS = [
    ([[0, 1], [0, 1], [1, 2]], 'without duplicates'),
    ([[1, 2], [0, 1]], 'sorted'),
    ([[0, 2], [0, 1], [1, 2]], 'sorted'),
    ([[0, 1], [2, 1]], 'smaller'),
    ([[0, 0]], 'smaller'),
    ([[-1, 0]], '0 or larger'),
    (np.zeros((0, 2), int), 'no image pairs'),
    ([0, 1], 'shape'),
]


def test_from_bandwidth():
    tnet = TempNet.from_bandwidth(5, 2)
    np.testing.assert_array_equal(tnet.image_pairs, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])


def test_image_pairs_idx():
    tnet = TempNet(np.stack(([0, 1, 2, 3], [1, 2, 3, 4]), axis=-1))
    assert tnet.image_pairs_idx(ref=1, sec=2) == 1
    np.testing.assert_array_equal(tnet.image_pairs_idx(ref=[1, 2], sec=[2, 3]), [1, 2])


def test_image_pairs_idx_out_of_range():
    """A pair with an image index outside 0 .. n_images - 1 is not in the network, in every input form; such pairs
    gave the row of the pair of their code ref * n_images + sec, e.g. (0, 5) that of (1, 2) with 3 images."""
    tnet = TempNet([[0, 1], [0, 2], [1, 2]])
    outside = [(0, 5), (1, -1), (-1, 4)]                # the codes of (1, 2), (0, 2), (0, 1)
    for ref, sec in outside:
        assert tnet.image_pairs_idx(ref=ref, sec=sec) == -1
    pairs = np.array(outside + [(1, 2), (0, 1)])
    expected = [-1, -1, -1, 2, 0]
    np.testing.assert_array_equal(tnet.image_pairs_idx(image_pairs=pairs), expected)
    np.testing.assert_array_equal(tnet.image_pairs_idx(ref=pairs[:, 0], sec=pairs[:, 1]), expected)
    np.testing.assert_array_equal(tnet.image_pairs_idx(ref=pairs[:, 0].tolist(), sec=pairs[:, 1].tolist()), expected)
    big = np.array([[1431655766, 0]], np.int32)         # its int32 code 1431655766 * 3 overflows to 2, that of (0, 2)
    np.testing.assert_array_equal(tnet.image_pairs_idx(image_pairs=big), [-1])


def test_connected_networks_are_valid():
    """A network is valid when its pairs link all the images, whatever pairs it has: a reference image need not
    have a pair with the next image, and the last pair need not hold the last image."""
    TempNet(np.array([[0, 1], [0, 2], [1, 3], [2, 3]]))
    tnet = TempNet(np.array([[0, 1], [0, 3], [0, 4], [0, 5], [1, 2]]))
    np.testing.assert_array_equal(tnet.image_pairs_idx(ref=[0, 0, 0, 0, 1], sec=[1, 3, 4, 5, 2]), [0, 1, 2, 3, 4])
    assert tnet.image_pairs_idx(ref=1, sec=5) == -1


@pytest.mark.parametrize('nimages', [2, 3, 7])
def test_bandwidth_networks_are_valid(nimages):
    for bandwidth in range(1, nimages):
        TempNet(TempNet.from_bandwidth(nimages, bandwidth).image_pairs)


@pytest.mark.parametrize('pairs, match', [
    ([[0, 1], [2, 3]], 'not linked'),                       # two parts
    ([[0, 2]], 'not linked'),                               # image 1 has no pair
] + AGAINST_CONVENTIONS)
def test_invalid_networks(pairs, match):
    with pytest.raises(ValueError, match=match):
        TempNet(np.asarray(pairs))


@pytest.mark.parametrize('pairs, match', AGAINST_CONVENTIONS)
def test_check_image_pairs(pairs, match):
    with pytest.raises(ValueError, match=match):
        check_image_pairs(pairs)


def test_conventions_do_not_need_linked_images():
    """A network may leave out images or have parts: the conventions hold, only `TempNet` needs all images linked."""
    for pairs in ([[0, 1], [2, 3]], [[1, 2], [1, 3], [2, 3]]):
        check_image_pairs(pairs)
        with pytest.raises(ValueError, match='not linked'):
            TempNet(pairs)


def test_load_checks_the_conventions(tmp_path):
    TempNet([[1, 2], [2, 3]], check_if_valid=False).save(str(tmp_path / 'leaves_out_0.zarr'))
    np.testing.assert_array_equal(TempNet.load(str(tmp_path / 'leaves_out_0.zarr')).image_pairs, [[1, 2], [2, 3]])
    TempNet([[0, 2], [0, 1], [1, 2]], check_if_valid=False).save(str(tmp_path / 'unsorted.zarr'))
    with pytest.raises(ValueError, match='sorted'):
        TempNet.load(str(tmp_path / 'unsorted.zarr'))


def test_nimage_from_npair():
    assert nimage_from_npair(6) == 4
