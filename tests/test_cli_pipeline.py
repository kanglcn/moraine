"""The CLI processing chain on a 400x400 crop of the sample data, checked against the API.

rslc -> amp_disp -> shp_test -> select_shp -> DS candidates -> emperical_co_pc -> emi / ds_temp_coh
     -> emperical_co_emi_temp_coh_pc -> temp_coh -> mcf_pc, plus pyramids and views.
"""
import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc

pytestmark = pytest.mark.slow
GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]
CHUNKS = (200, 200)


def r(p):
    return zarr.open(str(p), mode='r')[:]


@pytest.fixture(scope='module')
def work(rslc, tmp_path_factory):
    d = tmp_path_factory.mktemp('cli')
    crop = rslc[:400, :400]
    z = zarr.open(str(d / 'rslc.zarr'), mode='w', shape=crop.shape, dtype=crop.dtype, chunks=(*CHUNKS, 1))
    z[:] = crop
    return d, crop


@pytest.mark.parametrize('cuda', GPU)
def test_amp_disp(work, cuda):
    d, crop = work
    mc.amp_disp(str(d / 'rslc.zarr'), str(d / f'adi_{cuda}.zarr'), cuda=cuda)
    np.testing.assert_array_almost_equal(r(d / f'adi_{cuda}.zarr'), mr.amp_disp(crop))


@pytest.fixture(scope='module')
def shp(work):
    d, crop = work
    mc.shp_test(str(d / 'rslc.zarr'), str(d / 'pvalue.zarr'), az_half_win=5, r_half_win=5)
    mc.select_shp(str(d / 'pvalue.zarr'), str(d / 'is_shp.zarr'), str(d / 'shp_num.zarr'), p_max=0.05)
    mc.pc_logic_ras(str(d / 'shp_num.zarr'), str(d / 'ds_can_gix.zarr'), 'ras>=50')
    return d


def test_shp(work, shp):
    d, crop = work
    p = mr.ks_test(np.abs(crop) ** 2, az_half_win=5, r_half_win=5)
    # the KS distance is discrete (k/nimages); intensities equal up to float32 rounding may be ordered
    # differently by the CLI and the API, moving a handful of p values by one step
    mismatch = ~np.isclose(r(d / 'pvalue.zarr'), p, equal_nan=True)
    assert mismatch.mean() < 1e-5
    is_shp, shp_num = mr.select_shp(r(d / 'pvalue.zarr'), 0.05)
    np.testing.assert_array_equal(r(d / 'is_shp.zarr'), is_shp)
    np.testing.assert_array_equal(r(d / 'shp_num.zarr'), shp_num)
    gix = r(d / 'ds_can_gix.zarr')
    np.testing.assert_array_equal(gix, np.stack(np.where(shp_num >= 50), axis=-1))
    mc.gix2bool(str(d / 'ds_can_gix.zarr'), str(d / 'is_ds_can.zarr'), shape=shp_num.shape)
    np.testing.assert_array_equal(r(d / 'is_ds_can.zarr'), shp_num >= 50)


@pytest.mark.gpu
def test_shp_test_gpu(work, shp):
    d, _ = work
    mc.shp_test(str(d / 'rslc.zarr'), str(d / 'pvalue_gpu.zarr'), az_half_win=5, r_half_win=5, cuda=True)
    assert (~np.isclose(r(d / 'pvalue_gpu.zarr'), r(d / 'pvalue.zarr'), atol=1e-4, equal_nan=True)).mean() < 1e-5


@pytest.fixture(scope='module')
def co(work, shp):
    d, _ = work
    mc.ras2pc_ras_chunk(str(d / 'ds_can_gix.zarr'), str(d / 'is_shp.zarr'), str(d / 'ds_can_is_shp'),
                        str(d / 'ds_can_key.zarr'), chunks=CHUNKS)
    mc.emperical_co_pc(str(d / 'rslc.zarr'), str(d / 'ds_can_is_shp'), str(d / 'ds_can_gix.zarr'),
                       str(d / 'ds_can_coh'), chunks=CHUNKS)
    chunks = zarr.open(str(d / 'ds_can_gix.zarr'), mode='r').chunks[0]
    mc.pc_concat(str(d / 'ds_can_coh'), str(d / 'ds_can_coh.zarr'), key=str(d / 'ds_can_key.zarr'), chunks=chunks)
    return d


def test_emperical_co_pc(work, co):
    d, crop = work
    gix = r(d / 'ds_can_gix.zarr')
    is_shp = r(d / 'is_shp.zarr')[gix[:, 0], gix[:, 1]]
    np.testing.assert_array_almost_equal(r(d / 'ds_can_coh.zarr'), mr.emperical_co_pc(crop, gix, is_shp))
    n_point = gix.shape[0]
    mc.data_reduce(str(d / 'ds_can_coh.zarr'), str(d / 'coh_ave.zarr'), map_func=np.abs, reduce_func=np.sum,
                   post_map_func=lambda x: x / n_point)
    assert mr.uncompress_coh(r(d / 'coh_ave.zarr')).shape == (17, 17)


@pytest.fixture(scope='module')
def pl(co):
    d = co
    mc.emi(str(d / 'ds_can_coh.zarr'), str(d / 'ds_can_ph.zarr'), ref=0)
    mc.ds_temp_coh(str(d / 'ds_can_coh.zarr'), str(d / 'ds_can_ph.zarr'), str(d / 'ds_can_t_coh.zarr'))
    return d


def test_emi_ds_temp_coh(pl):
    d = pl
    coh = r(d / 'ds_can_coh.zarr')
    ph = mr.emi(coh)
    np.testing.assert_array_almost_equal(r(d / 'ds_can_ph.zarr'), ph)
    np.testing.assert_array_almost_equal(r(d / 'ds_can_t_coh.zarr'), mr.ds_temp_coh(coh, ph))


def test_emi_ref(pl):
    d = pl
    mc.emi(str(d / 'ds_can_coh.zarr'), str(d / 'ph_ref3.zarr'), ref=3)
    np.testing.assert_array_almost_equal(r(d / 'ph_ref3.zarr'), mr.emi(r(d / 'ds_can_coh.zarr'), ref=3))


def test_emi_regularize(pl):
    """`regularize=False` of the commands gives the result of the API (`pl` uses the default, True)."""
    d = pl
    coh = r(d / 'ds_can_coh.zarr')
    ph = mr.emi(coh, regularize=False)
    mc.emi(str(d / 'ds_can_coh.zarr'), str(d / 'ph_noreg.zarr'), regularize=False)
    np.testing.assert_array_equal(r(d / 'ph_noreg.zarr'), ph)
    names = [f'{n}_fused_noreg' for n in ('ph', 't_coh')]
    mc.emperical_co_emi_temp_coh_pc(str(d / 'rslc.zarr'), str(d / 'ds_can_is_shp'), str(d / 'ds_can_gix.zarr'),
                                    *[str(d / n) for n in names], chunks=CHUNKS, regularize=False)
    chunks = zarr.open(str(d / 'ds_can_gix.zarr'), mode='r').chunks[0]
    mc.pc_concat([str(d / n) for n in names], [str(d / f'{n}.zarr') for n in names],
                 key=str(d / 'ds_can_key.zarr'), chunks=chunks)
    for n, ref in zip(names, (ph, mr.ds_temp_coh(coh, ph))):
        np.testing.assert_array_equal(r(d / f'{n}.zarr'), ref)


def test_weighted_temp_coh(pl):
    """The weighted temporal coherence of the fused command equals emperical-co-pc with `n_looks_dir` followed
    by ds-temp-coh with `n_looks`."""
    d = pl
    key, chunks = str(d / 'ds_can_key.zarr'), zarr.open(str(d / 'ds_can_gix.zarr'), mode='r').chunks[0]
    names = [f'{n}_w' for n in ('ph', 't_coh', 't_coh_w', 'eff_n_pairs')]
    mc.emperical_co_emi_temp_coh_pc(str(d / 'rslc.zarr'), str(d / 'ds_can_is_shp'), str(d / 'ds_can_gix.zarr'),
                                    *[str(d / n) for n in names[:2]], t_coh_w_dir=str(d / names[2]),
                                    eff_n_pairs_dir=str(d / names[3]), chunks=CHUNKS)
    mc.pc_concat([str(d / n) for n in names], [str(d / f'{n}.zarr') for n in names], key=key, chunks=chunks)
    mc.emperical_co_pc(str(d / 'rslc.zarr'), str(d / 'ds_can_is_shp'), str(d / 'ds_can_gix.zarr'),
                       str(d / 'coh_n'), n_looks_dir=str(d / 'n_looks'), chunks=CHUNKS)
    mc.pc_concat([str(d / 'coh_n'), str(d / 'n_looks')], [str(d / 'coh_n.zarr'), str(d / 'n_looks.zarr')],
                 key=key, chunks=chunks)
    np.testing.assert_array_equal(r(d / 'coh_n.zarr'), r(d / 'ds_can_coh.zarr'))
    mc.ds_temp_coh(str(d / 'ds_can_coh.zarr'), str(d / 'ds_can_ph.zarr'), str(d / 't_coh_s.zarr'),
                   n_looks=str(d / 'n_looks.zarr'), t_coh_w=str(d / 't_coh_w_s.zarr'),
                   eff_n_pairs=str(d / 'eff_n_pairs_s.zarr'))
    for n in ('ph', 't_coh', 't_coh_w', 'eff_n_pairs'):
        a = r(d / f'{n}_w.zarr')
        b = r(d / 'ds_can_ph.zarr') if n == 'ph' else r(d / f'{n}_s.zarr')
        np.testing.assert_array_equal(a, b)
    coh, gix, n_looks = r(d / 'ds_can_coh.zarr'), r(d / 'ds_can_gix.zarr'), r(d / 'n_looks.zarr')
    n_shp = np.count_nonzero(r(d / 'is_shp.zarr')[gix[:, 0], gix[:, 1]], axis=(1, 2))
    assert n_looks.dtype == np.float32 and ((1 <= n_looks) & (n_looks <= n_shp)).all()
    for a, b in zip(mr.ds_temp_coh(coh, r(d / 'ds_can_ph.zarr'), n_looks=n_looks),
                    (r(d / 't_coh_s.zarr'), r(d / 't_coh_w_s.zarr'), r(d / 'eff_n_pairs_s.zarr'))):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='need n_looks'):
        mc.ds_temp_coh(str(d / 'ds_can_coh.zarr'), str(d / 'ds_can_ph.zarr'), t_coh_w=str(d / 'x.zarr'))


@pytest.mark.parametrize('cuda', GPU)
def test_emperical_co_emi_temp_coh_pc(pl, cuda):
    """The fused chunkwise version equals emperical_co_pc -> emi -> ds_temp_coh on the same device.

    CPU and GPU eigen decompositions differ in float32, so each device is compared with itself;
    on CPU the results are identical.
    """
    d = pl
    if cuda:
        mc.emi(str(d / 'ds_can_coh.zarr'), str(d / 'ref_ph_True.zarr'), cuda=True)
        mc.ds_temp_coh(str(d / 'ds_can_coh.zarr'), str(d / 'ref_ph_True.zarr'), str(d / 'ref_t_coh_True.zarr'), cuda=True)
        refs = ('ref_ph_True', 'ref_t_coh_True')
    else:
        refs = ('ds_can_ph', 'ds_can_t_coh')
    names = [f'{n}_{cuda}' for n in ('ph', 't_coh')]
    mc.emperical_co_emi_temp_coh_pc(str(d / 'rslc.zarr'), str(d / 'ds_can_is_shp'), str(d / 'ds_can_gix.zarr'),
                                    *[str(d / n) for n in names], chunks=CHUNKS, cuda=cuda)
    chunks = zarr.open(str(d / 'ds_can_gix.zarr'), mode='r').chunks[0]
    mc.pc_concat([str(d / n) for n in names], [str(d / f'{n}.zarr') for n in names],
                 key=str(d / 'ds_can_key.zarr'), chunks=chunks)
    for n, ref in zip(names, refs):
        a, b = r(d / f'{n}.zarr'), r(d / f'{ref}.zarr')
        if cuda:  # batched GPU eigen solvers round differently for different batch sizes
            assert np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-6)) < 1e-5
            np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('cuda', GPU)
def test_temp_coh(pl, cuda):
    d = pl
    mc.temp_coh(str(d / 'ds_can_coh.zarr'), str(d / 'ds_can_ph.zarr'), str(d / f'pqm_t_coh_{cuda}.zarr'), cuda=cuda)
    np.testing.assert_array_almost_equal(r(d / f'pqm_t_coh_{cuda}.zarr'),
                                         mr.temp_coh(r(d / 'ds_can_coh.zarr'), r(d / 'ds_can_ph.zarr')))


def test_mcf_pc(pl):
    d = pl
    gix, ph = r(d / 'ds_can_gix.zarr'), r(d / 'ds_can_ph.zarr')
    keep = r(d / 'ds_can_t_coh.zarr') > 0.7
    key = mr.pc_sort(mr.pc_hix(gix[keep], shape=(400, 400)))
    for name, data in [('ds_gix.zarr', gix[keep][key]), ('ds_ph.zarr', ph[keep][key])]:
        z = zarr.open(str(d / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=(data.shape[0], 1))
        z[:] = data
    pairs = mr.TempNet.from_bandwidth(17, bandwidth=1).image_pairs[:3]
    mc.mcf_pc(str(d / 'ds_gix.zarr'), str(d / 'ds_ph.zarr'), str(d / 'ds_unw.zarr'), pairs, range_pixel_spacing=1.0,
              azimuth_pixel_spacing=1.0)
    unw = r(d / 'ds_unw.zarr')
    intf = ph[keep][key][:, pairs[:, 0]] * ph[keep][key][:, pairs[:, 1]].conj()
    np.testing.assert_array_almost_equal(np.mod(unw + np.pi, 2 * np.pi) - np.pi, np.angle(intf), decimal=3)


def test_pyramids_and_views(work, shp):
    d, crop = work
    mc.amp_disp(str(d / 'rslc.zarr'), str(d / 'adi.zarr'))
    mc.ras_pyramid(str(d / 'adi.zarr'), str(d / 'adi_pyramid'))
    mc.ras_pyramid(str(d / 'rslc.zarr'), str(d / 'rslc_pyramid'))
    # point cloud pyramid of the DS candidates
    gix = r(d / 'ds_can_gix.zarr')
    for name, data in [('x.zarr', gix[:, 1].astype(np.float64)), ('y.zarr', gix[:, 0].astype(np.float64)),
                       ('pc.zarr', r(d / 'adi.zarr')[gix[:, 0], gix[:, 1]])]:
        z = zarr.open(str(d / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=data.shape)
        z[:] = data
    mc.pc_pyramid(str(d / 'pc.zarr'), str(d / 'pc_pyramid'), x=str(d / 'x.zarr'), y=str(d / 'y.zarr'), ras_resolution=20)
    # views of real data: adi with the DS candidates on it, next to the interferograms
    v = mc.view(str(d / 'adi_pyramid'), cmap='gray') * mc.view(str(d / 'pc_pyramid')) + \
        mc.view(str(d / 'rslc_pyramid'), show='intf_seq')
    assert [len(p) for p in v.widget.layers] == [2, 1] and v.widget.kdims == [{'name': 'image', 'max': 15}]
    assert v.png(str(d / 'views.png'), index={'image': 3}) and (d / 'views.png').stat().st_size > 10_000


def test_transform(tmp_path):
    pyproj = pytest.importorskip('pyproj')
    lon, lat = np.meshgrid(np.linspace(-150, -140, 50), np.linspace(60, 61, 40))
    for name, data in [('lon.zarr', lon), ('lat.zarr', lat)]:
        z = zarr.open(str(tmp_path / name), mode='w', shape=data.shape, dtype=data.dtype, chunks=(20, 50))
        z[:] = data
    mc.transform(str(tmp_path / 'lon.zarr'), str(tmp_path / 'lat.zarr'), str(tmp_path / 'e.zarr'), str(tmp_path / 'n.zarr'))
    e, n = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    np.testing.assert_allclose(r(tmp_path / 'e.zarr'), e)
    np.testing.assert_allclose(r(tmp_path / 'n.zarr'), n)
