"""The command `emperical-co-pc` on a small synthetic stack with missing data: the mean coherence magnitude of the
image pairs (`coh_mean`)."""
import numpy as np
import pytest

import moraine as mr
import moraine.cli as mc
from conftest import write_zarr, read_zarr as r

NIMAGES = 6
CHUNKS = (20, 25)


def _per_chunk(path):
    """the arrays of a directory of per chunk zarrs, concatenated in the order of the file names"""
    return np.concatenate([r(p) for p in sorted(path.glob('*.zarr'))])


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    """rslc (60, 50, 6) with missing data (nan): image 2 in a corner, image 4 everywhere; the SHPs of every pixel in a
    window of 3 x 5 pixels (the centre always one of them) and 40 % of the pixels as points"""
    d = tmp_path_factory.mktemp('co')
    rng = np.random.default_rng(1)
    shape = (60, 50)
    rslc = (rng.standard_normal((*shape, NIMAGES)) + 1j * rng.standard_normal((*shape, NIMAGES))).astype(np.complex64)
    rslc[:15, :20, 2] = np.nan
    rslc[:, :, 4] = np.nan
    is_shp = rng.random((*shape, 3, 5)) < 0.6
    is_shp[:, :, 1, 2] = True
    gix = np.stack(np.where(rng.random(shape) < 0.4), axis=-1).astype(np.int32)
    for name, data, chunks in (('rslc', rslc, (*CHUNKS, 1)), ('is_shp', is_shp, (*CHUNKS, 3, 5)), ('gix', gix, (500, 2))):
        write_zarr(d / f'{name}.zarr', data, chunks)
    return d


def _coh_mean(d, name, chunks=CHUNKS, **kwargs):
    """`emperical-co-pc` with `coh_mean`, the SHPs of the points split by `chunks` first; the mean and the coherence"""
    mc.ras2pc_ras_chunk(str(d / 'gix.zarr'), str(d / 'is_shp.zarr'), str(d / f'{name}_is_shp'),
                        str(d / f'{name}_key.zarr'), chunks=chunks)
    mc.emperical_co_pc(str(d / 'rslc.zarr'), str(d / f'{name}_is_shp'), str(d / 'gix.zarr'), str(d / f'{name}_coh'),
                       coh_mean=str(d / f'{name}_coh_mean.zarr'), chunks=chunks, **kwargs)
    return r(d / f'{name}_coh_mean.zarr'), _per_chunk(d / f'{name}_coh')


@pytest.fixture(scope='module')
def cpu(stack):
    return _coh_mean(stack, 'cpu', n_looks_dir=str(stack / 'cpu_n_looks'))


@pytest.mark.filterwarnings('ignore:Mean of empty slice')
def test_coh_mean(stack, cpu):
    """the mean of |coherence| of every image pair over the points where it is finite; the other outputs unchanged"""
    mean, coh = cpu
    assert mean.dtype == np.float32 and mean.shape == (NIMAGES, NIMAGES)
    np.testing.assert_allclose(mean, mr.uncompress_coh(np.nanmean(np.abs(coh), axis=0, dtype=np.float64)), rtol=1e-6)
    np.testing.assert_array_equal(mean, mean.T)
    np.testing.assert_array_equal(np.diag(mean), 1)
    # the missing data: the pairs of image 2 have fewer points with a finite coherence, those of image 4 none
    pairs = mr.TempNet.from_bandwidth(NIMAGES).image_pairs.tolist()
    finite = np.isfinite(coh).sum(axis=0)
    assert 0 < finite[pairs.index([1, 2])] < finite[pairs.index([1, 3])] == len(coh)
    assert np.isnan(mean[4, [0, 1, 2, 3, 5]]).all() and np.isfinite(np.delete(mean, 4, 0)[:, [0, 1, 2, 3, 5]]).all()
    # the coherence and the number of looks are those of the command without `coh_mean`
    mc.emperical_co_pc(str(stack / 'rslc.zarr'), str(stack / 'cpu_is_shp'), str(stack / 'gix.zarr'),
                       str(stack / 'plain_coh'), n_looks_dir=str(stack / 'plain_n_looks'), chunks=CHUNKS)
    np.testing.assert_array_equal(coh, _per_chunk(stack / 'plain_coh'))
    np.testing.assert_array_equal(_per_chunk(stack / 'cpu_n_looks'), _per_chunk(stack / 'plain_n_looks'))


def test_coh_mean_image_pairs(stack, cpu):
    """NaN for the image pairs that are not computed, the same mean for the others"""
    pairs = mr.TempNet.from_bandwidth(NIMAGES, bandwidth=2).image_pairs
    mean, _ = _coh_mean(stack, 'pairs', image_pairs=pairs)
    computed = np.eye(NIMAGES, dtype=bool)
    computed[pairs[:, 0], pairs[:, 1]] = computed[pairs[:, 1], pairs[:, 0]] = True
    assert np.isnan(mean[~computed]).all()
    np.testing.assert_allclose(mean[computed], cpu[0][computed], rtol=1e-6)


def test_coh_mean_chunks_and_workers(stack, cpu):
    """the same mean with other processing chunks (the last ones smaller) in worker processes"""
    mean, coh = _coh_mean(stack, 'chunks', chunks=(32, 40), processes=True, n_workers=2)
    assert len(coh) == len(cpu[1])
    np.testing.assert_allclose(mean, cpu[0], rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:Mean of empty slice')
def test_coh_mean_gpu(stack, cpu):
    """the mean of the coherence computed on the GPU, the same as on the CPU within float32 rounding"""
    mean, coh = _coh_mean(stack, 'gpu', cuda=True)
    np.testing.assert_allclose(mean, mr.uncompress_coh(np.nanmean(np.abs(coh), axis=0, dtype=np.float64)), rtol=1e-6)
    np.testing.assert_allclose(mean, cpu[0], rtol=1e-5)
