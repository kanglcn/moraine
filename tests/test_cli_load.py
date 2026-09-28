"""Loading GAMMA results. Needs the GAMMA software on PATH and the sample GAMMA output."""
import shutil

import numpy as np
import pytest
import toml
import zarr

import moraine.cli as mc
from conftest import data_path

NLINES, WIDTH, NIMAGES, REF = 2500, 1834, 17, '20220620'


@pytest.fixture(scope='module')
def gamma():
    for cmd in ('base_calc', 'geocode', 'phase_sim_orb', 'create_offset'):
        if shutil.which(cmd) is None:
            pytest.skip(f'GAMMA command {cmd} not found')
    return data_path('gamma')


def test_load_gamma_range(gamma, tmp_path):
    mc.load_gamma_range(str(gamma / 'rslc' / f'{REF}.rslc.par'), str(tmp_path / 'range.zarr'))
    r = zarr.open(str(tmp_path / 'range.zarr'), mode='r')[:]
    assert r.shape == (NLINES, WIDTH)
    assert np.all(np.diff(r, axis=1) > 0)          # slant range increases with the range index


def test_load_gamma_metadata(gamma, tmp_path):
    mc.load_gamma_metadata(str(gamma / 'rslc'), str(gamma / 'DEM' / 'dem_seg_par'), REF, str(tmp_path / 'meta.toml'))
    meta = toml.load(tmp_path / 'meta.toml')
    assert len(meta['dates']) == NIMAGES
    assert REF in meta['dates']


def test_load_gamma_lat_lon_hgt(gamma, tmp_path):
    g = gamma / 'geocoding'
    out = [str(tmp_path / f) for f in ('lat.zarr', 'lon.zarr', 'hgt.zarr')]
    mc.load_gamma_lat_lon_hgt(str(g / '20210802.diff_par'), str(gamma / 'rslc' / f'{REF}.rslc.par'),
                              str(gamma / 'DEM' / 'dem_seg_par'), str(g / '20210802.hgt'), str(tmp_path / 'scratch'), *out)
    lat, lon = zarr.open(out[0], mode='r')[:], zarr.open(out[1], mode='r')[:]
    assert lat.shape == lon.shape == (NLINES, WIDTH)
    assert np.nanmin(lat) >= -90 and np.nanmax(lat) <= 90 and np.nanmin(lon) >= -180 and np.nanmax(lon) <= 360


def test_load_gamma_look_vector(gamma, tmp_path):
    g = gamma / 'geocoding'
    out = [str(tmp_path / f) for f in ('theta.zarr', 'phi.zarr')]
    mc.load_gamma_look_vector(str(g / '20210802.lv_theta'), str(g / '20210802.lv_phi'), str(g / '20210802.lt_fine'),
                              str(gamma / 'rslc' / f'{REF}.rslc.par'), str(gamma / 'DEM' / 'dem_seg_par'),
                              str(tmp_path / 'scratch'), *out)
    assert zarr.open(out[0], mode='r').shape == zarr.open(out[1], mode='r').shape == (NLINES, WIDTH)


@pytest.mark.slow
def test_load_gamma_flatten_rslc(gamma, tmp_path):
    mc.load_gamma_flatten_rslc(str(gamma / 'rslc'), REF, str(gamma / 'geocoding' / '20210802.hgt'),
                               str(tmp_path / 'scratch'), str(tmp_path / 'rslc.zarr'), chunks=(1000, 1000))
    z = zarr.open(str(tmp_path / 'rslc.zarr'), mode='r')
    assert z.shape == (NLINES, WIDTH, NIMAGES)
    assert np.iscomplexobj(z[:10, :10, 0])
