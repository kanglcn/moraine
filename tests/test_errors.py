"""Clear errors for wrong arguments, ArgumentErrors (decision 0051), and the errors of the data that are not."""
import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.cli.zarr_ import ZarrDir, check_ndim
from moraine.cli.pc import _path_lists
from moraine.api.utils_.chunk import fill_slice
from moraine.api.tnet import _imagepair_from_bandwidth
from moraine.api.utils_ import ArgumentError
from conftest import write_zarr, random_gix


def test_check_ndim(tmp_path):
    z = zarr.open(str(tmp_path / 'a.zarr'), mode='w', shape=(4, 5), dtype='f4')
    check_ndim('rslc', z, 2, '(a, b)')
    with pytest.raises(ArgumentError, match=r'rslc must have 3 dimensions \(nlines, width, nimages\), got shape \(4, 5\)'):
        check_ndim('rslc', z, 3, '(nlines, width, nimages)')


def test_command_rejects_wrong_dimensions(tmp_path):
    zarr.open(str(tmp_path / 'rslc.zarr'), mode='w', shape=(4, 5), dtype='c8', chunks=(4, 5))
    with pytest.raises(ArgumentError, match='rslc must have 3 dimensions'):
        mc.shp_test(str(tmp_path / 'rslc.zarr'), str(tmp_path / 'is_shp.zarr'), str(tmp_path / 'shp_num.zarr'), 1, 1)
    assert not (tmp_path / 'is_shp.zarr').exists()


def test_path_lists():
    assert _path_lists(ras='a', pc='b') == [['a'], ['b']]
    assert _path_lists(ras=['a', 'c'], pc=('b', 'd')) == [['a', 'c'], ['b', 'd']]
    with pytest.raises(ArgumentError, match='ras is a list, pc is a path'):
        _path_lists(ras=['a'], pc='b')
    with pytest.raises(ArgumentError, match='same number of paths'):
        _path_lists(ras=['a', 'c'], pc=['b'])


def test_zarr_dir_chunks(tmp_path):
    zarr.open(str(tmp_path / '0.zarr'), mode='w', shape=(10, 3), dtype='f4', chunks=(5, 1))
    with pytest.raises(ArgumentError, match='points of a chunk array must be in one chunk'):
        ZarrDir([str(tmp_path / '0.zarr')])
    zarr.open(str(tmp_path / '1.zarr'), mode='w', shape=(10, 3), dtype='f4', chunks=(10, 2))
    with pytest.raises(ArgumentError, match='one image or the whole window per chunk'):
        ZarrDir([str(tmp_path / '1.zarr')])


def test_api_argument_errors():
    gix = np.array([[0, 1], [0, 2]]); hix = np.array([3, 5])
    with pytest.raises(ArgumentError, match='both be gix'):
        mr.pc_union(gix, hix)
    for fn in (mr.pc_union, mr.pc_intersect, mr.pc_diff):
        with pytest.raises(ArgumentError, match='both be gix'):
            fn(hix.reshape(2, 1, 1), hix.reshape(2, 1, 1))
    with pytest.raises(ValueError, match='sorted and unique'):
        mr.pc_union(np.array([5, 3]), hix)
    with pytest.raises(ArgumentError, match='bandwidth 10 is larger than the number of images 5'):
        _imagepair_from_bandwidth(5, 10)
    with pytest.raises(ValueError, match='slice stop 12 must be after the start 0 and within the shape 10'):
        fill_slice((10,), (slice(0, 12),))
    with pytest.raises(ValueError, match='slices with a step'):
        fill_slice((10,), (slice(0, 5, 2),))


# ---------------------------------------------------------------- argument errors are ArgumentErrors (decision 0051)

def test_argument_error_class():
    """ArgumentError: a ValueError, the same class from moraine.api.utils_ and moraine.cli.utils_, not a public name,
    and the same class after a worker process sent it"""
    import pickle
    from moraine.api.utils_ import errors
    from moraine.cli import utils_
    assert issubclass(ArgumentError, ValueError)
    assert ArgumentError is errors.ArgumentError is utils_.ArgumentError
    assert 'ArgumentError' not in mr.__all__ and not hasattr(mc, 'ArgumentError')
    e = pickle.loads(pickle.dumps(ArgumentError('x')))
    assert type(e) is ArgumentError and str(e) == 'x'


_PAIRS3 = np.array([[0, 1], [0, 2], [1, 2]])


@pytest.mark.parametrize('call, message', [
    (lambda: mr.pc_reference(np.zeros((2, 2, 2), np.float32), 0),
     r'data must have shape \(n_points,\) or \(n_points, n\), got shape \(2, 2, 2\)'),
    (lambda: mr.temp_coh(np.ones((4, 3), np.complex64), np.ones((2, 2, 3), np.complex64)),
     'intf and rslc must both be rasters or both point clouds, got 2 and 3 dimensions'),
    (lambda: mr.TempNet.from_bandwidth(3).image_pairs_idx(ref=[0, 1], sec=[1]),
     'ref and sec must have the same length, got 2 and 1'),
    (lambda: mr.nimage_from_npair(7), 'input is not correct value of number of all pairs'),
    (lambda: mr.HilbertRtree.build(np.arange(4.), np.arange(4.)).bbox_query([0, 0, 1], np.arange(4.), np.arange(4.)),
     r'bounds must be \[x0, y0, xm, ym\], got 3 values'),
    (lambda: mr.HilbertRtree.build(np.zeros(3), np.zeros(2)), r'x and y must have the same shape, got \(3,\) and \(2,\)'),
    (lambda: mr.mcf_pc(np.array([0., 1.]), np.array([0., 1.]), np.ones(2, np.complex64)), 'need at least 3 points'),
    (lambda: mr.emcf_pc(np.zeros(3), np.zeros(3), np.ones(3, np.complex64)),
     r'ph must have shape \(n_points, nimages\), got \(3,\)'),
    (lambda: mr.emcf_pc(np.zeros(3), np.zeros(3), np.ones((3, 1), np.complex64)), 'at least 2 images are needed'),
    (lambda: mr.emcf_pc(np.zeros(3), np.zeros(3), np.ones((3, 3), np.complex64), image_pairs=np.zeros((0, 2), np.int32)),
     r'image_pairs must be integers of shape \(n_pairs, 2\), got int32 \(0, 2\)'),
    (lambda: mr.unwrap_correct_closure_pc(np.zeros(3), np.zeros(3), np.ones((3, 4), np.complex64),
                                          np.zeros((2, 3), np.float32), _PAIRS3),
     r'ph \(n_points, nimages\) and unw \(n_points, n_pairs\) do not match: \(3, 4\), \(2, 3\)'),
], ids=['pc_reference ndim', 'temp_coh ndim', 'image_pairs_idx', 'nimage_from_npair', 'bbox_query', 'rtree build',
        'mcf_pc 2 points', 'emcf_pc ph ndim', 'emcf_pc 1 image', 'emcf_pc no pairs', 'closure shapes'])
def test_api_argument_errors_are_argument_errors(call, message):
    with pytest.raises(ArgumentError, match=message):
        call()


@pytest.mark.gpu
def test_ds_temp_coh_gpu_block_size_is_an_argument_error():
    import cupy as cp
    with pytest.raises(ArgumentError, match='block_size x nimages must be at most 393216 on a GPU, not 394240'):
        mr.ds_temp_coh(cp.ones((1, 1), np.complex64), cp.ones((1, 385), np.complex64), block_size=1024, n_looks=10.0)


def test_executor_more_gpu_workers_than_gpus_is_an_argument_error(monkeypatch):
    from moraine.cli.executor import Executor
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ArgumentError, match='n_workers 2 GPU workers for 1 GPUs in CUDA_VISIBLE_DEVICES'):
        with Executor(cuda=True, n_workers=2):
            pass


def test_data_errors_are_not_argument_errors():
    """problems found in the values of the data stay ValueErrors (decision 0051)"""
    from moraine.api.utils_.orbit import bperp
    for call, message in [
        (lambda: mr.pc_union(np.array([5, 3]), np.array([3, 5])), 'sorted and unique'),
        (lambda: mr.mcf_pc(np.arange(10.), np.arange(10.), np.ones(10, np.complex64)), 'collinear or duplicated'),
        (lambda: mr.pc_reference(np.full((3, 2), np.nan, np.float32), np.ones(3, bool)), 'all nan'),
        (lambda: bperp(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3, 3)),
                       np.zeros((1, 3, 3))), 'at least 4 state vectors'),
    ]:
        with pytest.raises(ValueError, match=message) as e:
            call()
        assert not isinstance(e.value, ArgumentError), message


def test_cli_argument_errors_are_argument_errors(tmp_path):
    """argument checks of the commands not tested with their command: an ArgumentError before any output is written"""
    rng = np.random.default_rng(0)
    w = lambda name, data, chunks=None: write_zarr(tmp_path / name, data, chunks)
    out = str(tmp_path / 'out.zarr')
    gix = w('gix.zarr', random_gix(rng, (20, 20), 6)); short_gix = w('gix5.zarr', random_gix(rng, (20, 20), 5))
    ph = w('ph.zarr', np.exp(1j * rng.random((6, 3))).astype(np.complex64), (6, 1))
    unw = w('unw.zarr', np.zeros((6, 3), np.float32), (6, 1)); unw5 = w('unw5.zarr', np.zeros((5, 3), np.float32), (5, 1))
    pc = w('pc.zarr', rng.random(6).astype(np.float32)); pc2 = w('pc2.zarr', rng.random((6, 2)).astype(np.float32))
    polygons = str(tmp_path / 'p.geojson'); mr.write_polygons(polygons, [[(0, 0), (5, 0), (0, 5)]], 'radar_grid')
    spacing = dict(range_pixel_spacing=1.0, azimuth_pixel_spacing=1.0)
    for call, message in [
        (lambda: mc.image_pairs(str(tmp_path / 'pairs.txt')), 'give either `rslc` or `nimages`'),
        (lambda: mc.mcf_pc(short_gix, ph, out, _PAIRS3, **spacing), r'gix must have shape \(6, 2\), got \(5, 2\)'),
        (lambda: mc.emcf_pc(short_gix, ph, _PAIRS3, out, **spacing), r'gix must have shape \(6, 2\), got \(5, 2\)'),
        (lambda: mc.unwrap_correct_closure_pc(short_gix, ph, unw, _PAIRS3, out, **spacing),
         r'gix must have shape \(6, 2\), got \(5, 2\)'),
        (lambda: mc.unwrap_correct_closure_pc(gix, ph, unw5, _PAIRS3, out, **spacing),
         r'ph \(6, 3\) and unw_ph \(5, 3\) have different points'),
        (lambda: mc.polygon_mask(polygons, out, gix=gix, keep='in'), "keep must be 'inside' or 'outside', not 'in'"),
        (lambda: mc.polygon_mask(polygons, out, x=pc), 'give `y` with `x`'),
        (lambda: mc.polygon_mask(polygons, out, x=pc, y=pc2), r'x \(6,\) and y \(6, 2\) have different shapes'),
        (lambda: mc.polygon_mask(polygons, out, gix=pc), r'gix must have shape \(n_points, 2\), not \(6,\)'),
        (lambda: mc.gix2bool(pc, out, (20, 20)), r'gix must have shape \(n_points, 2\), got \(6,\)'),
        (lambda: mc.pc_concat(str(tmp_path), [out]), 'pcs must be a list of paths when pc is a list of paths'),
        (lambda: mc.pc_concat([pc], None), 'wrong input'),
        (lambda: mc.pc2ras(w('hix.zarr', np.arange(6)), pc, out, None),
         r'shape is needed with a hilbert index input \(idx with one column\)'),
        (lambda: mc.temp_coh(ph, unw5, str(tmp_path / 't_coh.zarr')),
         r'temp-coh: intf \(6, 3\) and rslc \(5, 3\) do not have the same points or pixels'),
        (lambda: mc.pc_reference(pc2, out, ref_point=0, method='mode'), "method must be 'median' or 'mean', not 'mode'"),
        (lambda: mc.pc_sc_dem_error(pc, pc2, pc, pc, str(tmp_path / 'dh.zarr'), out, ['20200101']),
         r'disp must have shape \(n_points, nimages\), got shape \(6,\)'),
        (lambda: mc.pc_sc_dem_error(gix, pc2, pc, pc, str(tmp_path / 'dh.zarr'), out, ['20200101', '20200113']),
         'disp must be floating point, got int32'),
        (lambda: mc.math(out, 'a + b', a=pc, b=pc2), r'b: shape \(6, 2\) differs from a: \(6,\)'),
    ]:
        with pytest.raises(ArgumentError, match=message):
            call()
        assert not any((tmp_path / name).exists() for name in ('out.zarr', 'pairs.txt', 't_coh.zarr', 'dh.zarr')), message
