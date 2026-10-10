"""Clear errors for wrong arguments (the checks that were asserts)."""
import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.cli.zarr_ import ZarrDir, check_ndim
from moraine.cli.pc import _path_lists
from moraine.api.chunk_ import fill_slice
from moraine.api.tnet import _imagepair_from_bandwidth


def test_check_ndim(tmp_path):
    z = zarr.open(str(tmp_path / 'a.zarr'), mode='w', shape=(4, 5), dtype='f4')
    check_ndim('rslc', z, 2, '(a, b)')
    with pytest.raises(ValueError, match=r'rslc must have 3 dimensions \(nlines, width, nimages\), got shape \(4, 5\)'):
        check_ndim('rslc', z, 3, '(nlines, width, nimages)')


def test_command_rejects_wrong_dimensions(tmp_path):
    zarr.open(str(tmp_path / 'rslc.zarr'), mode='w', shape=(4, 5), dtype='c8', chunks=(4, 5))
    with pytest.raises(ValueError, match='rslc must have 3 dimensions'):
        mc.shp_test(str(tmp_path / 'rslc.zarr'), str(tmp_path / 'is_shp.zarr'), str(tmp_path / 'shp_num.zarr'), 1, 1)
    assert not (tmp_path / 'is_shp.zarr').exists()


def test_path_lists():
    assert _path_lists(ras='a', pc='b') == [['a'], ['b']]
    assert _path_lists(ras=['a', 'c'], pc=('b', 'd')) == [['a', 'c'], ['b', 'd']]
    with pytest.raises(ValueError, match='ras is a list, pc is a path'):
        _path_lists(ras=['a'], pc='b')
    with pytest.raises(ValueError, match='same number of paths'):
        _path_lists(ras=['a', 'c'], pc=['b'])


def test_zarr_dir_chunks(tmp_path):
    zarr.open(str(tmp_path / '0.zarr'), mode='w', shape=(10, 3), dtype='f4', chunks=(5, 1))
    with pytest.raises(ValueError, match='points of a chunk array must be in one chunk'):
        ZarrDir([str(tmp_path / '0.zarr')])
    zarr.open(str(tmp_path / '1.zarr'), mode='w', shape=(10, 3), dtype='f4', chunks=(10, 2))
    with pytest.raises(ValueError, match='one image or the whole window per chunk'):
        ZarrDir([str(tmp_path / '1.zarr')])


def test_api_argument_errors():
    gix = np.array([[0, 1], [0, 2]]); hix = np.array([3, 5])
    with pytest.raises(ValueError, match='both be gix'):
        mr.pc_union(gix, hix)
    with pytest.raises(ValueError, match='sorted and unique'):
        mr.pc_union(np.array([5, 3]), hix)
    with pytest.raises(ValueError, match='bandwidth 10 is larger than the number of images 5'):
        _imagepair_from_bandwidth(5, 10)
    with pytest.raises(ValueError, match='slice stop 12 must be after the start 0 and within the shape 10'):
        fill_slice((10,), (slice(0, 12),))
    with pytest.raises(ValueError, match='slices with a step'):
        fill_slice((10,), (slice(0, 5, 2),))
