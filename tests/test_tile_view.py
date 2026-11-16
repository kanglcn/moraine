"""Views (moraine.cli.view): layers (tile geometry, colours, points, coordinates, show functions, time series),
composition, the widget (sliders, messages with the browser, polygons) and PNG images."""
import io

import numpy as np
import pytest
import zarr

import moraine.cli as mc
from moraine.cli import view, TileView
from moraine.cli.tiles import grid_geom, mercator_geom, mercator_pixel, WORLD, Overlay, Layout
from conftest import write_zarr

RAS = (-0.5, -0.5)       # view origin of rasters: pixel i centred at coordinate i


@pytest.fixture
def ras(tmp_path):
    """300 x 500 raster with value = 1000 * line + column, and its pyramid."""
    a = (1000 * np.arange(300)[:, None] + np.arange(500)[None, :]).astype(np.float32)
    write_zarr(tmp_path / 'ras.zarr', a, (100, 100))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'ras_pyr'))
    return a, tmp_path / 'ras_pyr'


@pytest.fixture
def stack(tmp_path, rng):
    """60 x 40 x 4 complex stack of unit phasors and its pyramid."""
    s = np.exp(1j * rng.uniform(-np.pi, np.pi, (60, 40, 4))).astype(np.complex64)
    write_zarr(tmp_path / 'stack.zarr', s, (30, 40, 1))
    mc.ras_pyramid(str(tmp_path / 'stack.zarr'), str(tmp_path / 'stack_pyr'))
    return s, tmp_path / 'stack_pyr'


@pytest.fixture
def grid_pc(tmp_path):
    """Points on the radar grid (lines 3 - 62, columns 2 - 41, one cell in three) with value
    1000 * line + column, and their pyramid."""
    pts = np.array([(y, x) for y in range(3, 63) for x in range(2, 42) if (x + y) % 3 == 0], dtype=float)
    write_zarr(tmp_path / 'gy.zarr', pts[:, 0])
    write_zarr(tmp_path / 'gx.zarr', pts[:, 1])
    write_zarr(tmp_path / 'val.zarr', (1000 * pts[:, 0] + pts[:, 1]).astype(np.float32))
    mc.pc_pyramid(str(tmp_path / 'val.zarr'), str(tmp_path / 'pc_pyr'), x=str(tmp_path / 'gx.zarr'),
                  y=str(tmp_path / 'gy.zarr'), ras_resolution=1)
    return [tuple(int(c) for c in p) for p in pts], tmp_path / 'pc_pyr'


TX, TY = 26000, 13000        # web mercator tile at zoom 15 (east Asia) holding the test scene


@pytest.fixture
def mercator_pc(tmp_path):
    """Points in web mercator coordinates with cells of one screen pixel at zoom 15: point (a, b) (line a from
    the north, column b; one cell in three) at the centre of pixel (a, b) of tile (TX, TY), value 1000 a + b."""
    res = mercator_pixel(15)
    west, top = -WORLD / 2 + TX * 256 * res, WORLD / 2 - TY * 256 * res
    ab = np.array([(a, b) for a in range(60) for b in range(40) if (a + b) % 3 == 0])
    write_zarr(tmp_path / 'mx.zarr', west + (ab[:, 1] + 0.5) * res)
    write_zarr(tmp_path / 'my.zarr', top - (ab[:, 0] + 0.5) * res)
    write_zarr(tmp_path / 'val.zarr', (1000 * ab[:, 0] + ab[:, 1]).astype(np.float32))
    mc.pc_pyramid(str(tmp_path / 'val.zarr'), str(tmp_path / 'merc_pyr'), x=str(tmp_path / 'mx.zarr'),
                  y=str(tmp_path / 'my.zarr'), ras_resolution=res)
    return [tuple(int(c) for c in q) for q in ab], res, west, top, tmp_path / 'merc_pyr'


# ---------------------------------------------------------------- rasters

def test_raster_tile_geometry(ras):
    a, pyr = ras
    layer = view(str(pyr))
    assert layer.kind == 'raster' and layer.crs == 'grid' and layer.shape == (300, 500) and layer.kdims == []
    assert layer.edge_origin == RAS and layer.extent == (-0.5, -0.5, 499.5, 299.5)
    # zoom 0: one screen pixel per pixel, tile (1, 0) is columns 256 - 511, lines 0 - 255
    t = layer.raster_values(grid_geom(0, 1, 0, RAS))
    np.testing.assert_array_equal(t[:, :244], a[:256, 256:])
    assert np.isnan(t[:, 244:]).all()                       # outside the data
    # zoom -1: level 1, every 2nd pixel
    t = layer.raster_values(grid_geom(-1, 0, 0, RAS))
    np.testing.assert_array_equal(t[:150, :250], a[::2, ::2])
    assert np.isnan(t[150:]).all() and np.isnan(t[:, 250:]).all()
    # zoomed in: 2**2 screen pixels per pixel, pixels enlarged
    t = layer.raster_values(grid_geom(2, 3, 1, RAS))
    np.testing.assert_array_equal(t, np.repeat(np.repeat(a[64:128, 192:256], 4, 0), 4, 1))
    # beyond the coarsest level (8, 2 x 2 pixels): the pixel nearest to each screen pixel centre
    t = layer.raster_values(grid_geom(-9, 0, 0, RAS))
    assert t[0, 0] == a[256, 256] and np.isnan(t[0, 1]) and np.isnan(t[1, 0])
    assert np.isnan(layer.raster_values(grid_geom(0, 5, 0, RAS))).all()
    assert np.isnan(layer.raster_values(grid_geom(0, -1, 0, RAS))).all()


def test_raster_colours_and_value(ras):
    a, pyr = ras
    layer = view(str(pyr))
    lo, hi = layer.clim
    assert 0 < lo < hi < a.max()                             # 1 - 99 % range of the values
    assert layer.colors[0] == '#440154'                      # viridis
    rgba = layer.colorize(np.array([[lo - 1, hi + 1, np.nan]]))
    assert rgba.shape == (1, 3, 4) and rgba.dtype == np.uint8
    assert tuple(rgba[0, 0, :3]) == (0x44, 0x01, 0x54) and rgba[0, 0, 3] == 255
    assert tuple(rgba[0, 1, :3]) == (0xfd, 0xe7, 0x25)       # last viridis colour
    assert rgba[0, 2, 3] == 0                                # nan is transparent
    # palette indices: 0 .. 254 over the colour limits, 255 transparent (nan), also when the limits are equal
    idx = layer.indices(np.array([[lo - 1, hi + 1, np.nan]]))
    assert idx.dtype == np.uint8 and idx.tolist() == [[0, 254, 255]]
    assert layer.palette.shape == (256, 4) and layer.palette[255, 3] == 0 and (layer.palette[:255, 3] == 255).all()
    assert view(str(pyr), clim=(1, 1)).indices(np.array([3.0, np.nan])).tolist() == [0, 255]
    assert layer.value(7.2, 3.4, 1) == {'x': 7.0, 'y': 3.0, 'key': [3, 7], 'value': float(a[3, 7])}
    # zoomed out: the value of the cell drawn (level 2, every 4th pixel), the key of the pixel under the cursor
    assert layer.value(101.6, 100.0, 4) == {'x': 102.0, 'y': 100.0, 'key': [100, 102], 'value': float(a[100, 100])}
    assert layer.value(499.6, 0, 1) is None and layer.value(-0.6, 0, 1) is None
    custom = view(str(pyr), cmap='magma', clim=(0, 50))
    assert custom.clim == (0, 50) and custom.colors[0] == '#000004'


def test_raster_in_memory(tmp_path, ras, monkeypatch):
    a, pyr = ras
    layer = view(a, label='array')
    ref = view(str(pyr))
    assert layer.max_level == 8 and layer.label == 'array'
    for z, tx, ty in [(0, 1, 0), (-1, 0, 0), (2, 3, 1), (-9, 0, 0)]:
        np.testing.assert_array_equal(layer.raster_values(grid_geom(z, tx, ty, RAS)),
                                      ref.raster_values(grid_geom(z, tx, ty, RAS)))
    assert layer.clim == pytest.approx(ref.clim, rel=0.05)
    # a zarr array that is not a pyramid is read into memory, if small
    layer = view(str(tmp_path / 'ras.zarr'))
    assert layer.label == 'ras.zarr' and layer.shape == (300, 500)
    monkeypatch.setattr('moraine.cli.tiles.MEMORY_BYTES', 1000)
    with pytest.raises(ValueError, match='ras-pyramid'):
        view(str(tmp_path / 'ras.zarr'))
    # bounds put the pixels at other coordinates, e.g. 10 x 10 units per pixel
    layer = view(a, bounds=(100, 200, 100 + 499 * 10, 200 + 299 * 10))
    assert (layer.cx0, layer.cy0, layer.rx, layer.ry) == (100, 200, 10, 10)
    assert layer.value(100 + 7 * 10 + 3, 200 + 3 * 10 - 4, 1)['key'] == [3, 7]


def test_phase_stack(stack):
    import colorcet
    s, pyr = stack
    layer = view(str(pyr))
    assert layer.clim == pytest.approx((-np.pi, np.pi)) and layer.bar_label == 'phase (rad)'
    assert layer.colors[0] == colorcet.colorwheel[0].lower()
    assert layer.kdims == [{'name': 'image', 'max': 3}] and layer.default_index == {'image': 0}
    geom = grid_geom(0, 0, 0, RAS)
    np.testing.assert_allclose(layer.raster_values(geom, {'image': 2})[:60, :40], np.angle(s[..., 2]), rtol=1e-6)
    layer = view(str(pyr), show='intf_seq')
    assert layer.kdims == [{'name': 'image', 'max': 2}]
    np.testing.assert_allclose(layer.raster_values(geom, [1])[:60, :40],
                               np.angle(s[..., 1] * s[..., 2].conj()), rtol=1e-5, atol=1e-6)
    layer = view(str(pyr), show='intf_all')
    assert [k['name'] for k in layer.kdims] == ['ref', 'sec']
    assert layer.default_index == {'ref': 0, 'sec': 1}        # an interferogram, not image 0 with itself
    assert layer.widget.panels[0]['layers'][0]['sliders'] == ['ref', 'sec']
    with pytest.raises(ValueError, match='show must be'):
        view(str(pyr), show='interferogram')


def test_show_function(stack, monkeypatch):
    s, pyr = stack
    # sliders named like the arguments, over the images of the stack; only the images used are read
    layer = view(str(pyr), show=lambda v, ref, sec: np.angle(v[..., ref] * np.conj(v[..., sec])))
    assert [(k['name'], k['max']) for k in layer.kdims] == [('ref', 3), ('sec', 3)]
    assert layer.clim == pytest.approx((-np.pi, np.pi))        # phases of complex data: cyclic colours
    geom = grid_geom(0, 0, 0, RAS)
    np.testing.assert_allclose(layer.raster_values(geom, {'ref': 1, 'sec': 3})[:60, :40],
                               np.angle(s[..., 1] * s[..., 3].conj()), rtol=1e-5, atol=1e-6)
    read = []
    zarr_getitem = zarr.Array.__getitem__
    monkeypatch.setattr(zarr.Array, '__getitem__', lambda self, key: read.append(key) or zarr_getitem(self, key))
    layer.raster_values(geom, {'ref': 1, 'sec': 3})
    assert sorted(k[-1] for k in read) == [1, 3]              # two images read, not the whole stack
    monkeypatch.undo()
    # a real result: colours from its values; the whole visible stack with v[...]
    layer = view(str(pyr), show=lambda v: np.abs(v[...]).mean(axis=-1), cmap='gray')
    assert layer.kdims == [] and layer.clim[1] == pytest.approx(1, abs=1e-5)
    # one image of a 1D stack and no way to guess the slider range
    with pytest.raises(ValueError, match='sliders'):
        view(np.ones((20, 30), np.float32), show=lambda v, i: v[...] * i)
    layer = view(np.ones((20, 30), np.float32), show=lambda v, i: v[...] * i, sliders={'i': 5})
    assert layer.kdims == [{'name': 'i', 'max': 4}]
    with pytest.raises(ValueError, match='at most 2 sliders'):
        view(str(pyr), show=lambda v, a, b, c: v[..., a])


# ---------------------------------------------------------------- point clouds

def test_point_cloud_raster_zoom(grid_pc):
    pts, pyr = grid_pc
    layer = view(str(pyr))
    assert layer.kind == 'point cloud' and layer.n_points == len(pts)
    # cell centres at the grid coordinates, the first at (x, y) = (2, 3)
    assert (layer.cx0, layer.cy0, layer.rx) == (2, 3, 1) and layer.edge_origin == (1.5, 2.5)
    assert layer.shape == (60, 40) and layer.crs == 'grid'
    t = layer.raster_values(grid_geom(0, 0, 0, layer.edge_origin))
    for y, x in pts:
        assert t[y - 3, x - 2] == 1000 * y + x
    assert np.isnan(t[0, 0])                                 # (y, x) = (3, 2): no point
    layer.render(grid_geom(0, 0, 0, layer.edge_origin))
    assert layer._rtree is None                              # overviews do not read the coordinates
    # probing where the points are rasterized: the point of the cell drawn, without the coordinate tree
    k = pts.index((30, 21))
    assert layer.value(21.1, 29.9, 1) == {'point': k, 'key': k, 'x': 21.0, 'y': 30.0, 'value': 30021.0}
    assert layer.value(22, 30, 1) is None                    # an empty cell
    assert layer.value(21.1, 29.9, 2) == {'point': k, 'key': k, 'x': 21.0, 'y': 30.0, 'value': 30021.0}   # level 1
    assert layer._rtree is None


def test_point_cloud_points_zoom(grid_pc):
    pts, pyr = grid_pc
    layer = view(str(pyr))
    # zoom 2: cells of 4 screen pixels, points as disks of radius 1.6 pixels at their coordinates; a tile is the
    # palette indices of its pixels
    img = layer.render(grid_geom(2, 0, 0, layer.edge_origin))
    assert img.shape == (256, 256) and img.dtype == np.uint8
    row, col = int((30 - 2.5) * 4), int((21 - 1.5) * 4)      # point (y, x) = (30, 21)
    assert img[row, col] == layer.indices(np.array([30021.0]))[0]
    assert img[row, col + 4] == 255                          # (30, 22): no point, 4 pixels from the others
    k = pts.index((30, 21))
    assert layer.value(21.1, 29.9, 0.25) == {'point': k, 'key': k, 'x': 21.0, 'y': 30.0, 'value': 30021.0}
    assert layer.value(22, 30, 1 / 16) is None               # centre of an empty cell, zoomed in
    assert layer.value(0, 0, 1 / 16) is None


def test_point_data_in_memory(grid_pc):
    pts, pyr = grid_pc
    d = pyr.parent
    gy, gx, val = (zarr.open(str(d / f'{n}.zarr'), mode='r')[:] for n in ('gy', 'gx', 'val'))
    layer = view(val, x=gx, y=gy, label='pts')
    ref = view(str(pyr))
    assert layer.kind == 'point cloud'
    assert (layer.cx0, layer.cy0, layer.rx, layer.shape) == (ref.cx0, ref.cy0, ref.rx, ref.shape)
    for z in (0, -1, 2):
        geom = grid_geom(z, 0, 0, ref.edge_origin)
        np.testing.assert_array_equal(layer.render(geom), ref.render(geom))
    assert layer.value(21.1, 29.9, 0.25)['value'] == 30021.0
    assert layer.value(21.1, 29.9, 2)['value'] == 30021.0 and layer.value(22, 30, 1) is None
    # zarr paths work too; coordinates off the integer grid need a resolution
    assert view(str(d / 'val.zarr'), x=str(d / 'gx.zarr'), y=str(d / 'gy.zarr')).shape == (60, 40)
    assert view(val, x=gx + 0.5, y=gy, resolution=1).shape == (60, 40)
    with pytest.raises(ValueError, match='resolution'):
        view(val, x=gx + 0.5, y=gy)
    with pytest.raises(ValueError, match='x.*and.*y'):
        view(val)
    with pytest.raises(ValueError, match='both'):
        view(val, x=gx)


def test_point_cloud_rtree_from_pyramid(grid_pc, monkeypatch):
    """The points of a pyramid are found with the bounding box tree saved by `pc_pyramid`, without reading all
    coordinates (a pyramid without the tree is rejected: tests/test_contracts.py); points in memory get one built."""
    from moraine.api.pc.rtree import HilbertRtree
    pts, pyr = grid_pc
    layer = view(str(pyr))
    monkeypatch.setattr(HilbertRtree, 'build', lambda *a, **k: pytest.fail('the tree was built instead of read'))
    box = (19.5, 28.5, 23.5, 32.5)                            # columns 20 - 23, lines 29 - 32
    idx = layer.points_in(box)
    assert sorted(pts[i] for i in idx) == [(y, x) for y in range(29, 33) for x in range(20, 24) if (x + y) % 3 == 0]
    monkeypatch.undo()
    gy, gx, val = (zarr.open(str(pyr.parent / f'{n}.zarr'), mode='r')[:] for n in ('gy', 'gx', 'val'))
    np.testing.assert_array_equal(np.sort(view(val, x=gx, y=gy).points_in(box)), np.sort(idx))


def test_web_mercator(mercator_pc):
    ab, res, west, top, pyr = mercator_pc
    layer = view(str(pyr))
    assert layer.crs == 'web_mercator' and layer.shape == (60, 40)
    assert layer.extent == pytest.approx((west, top - 60 * res, west + 40 * res, top), abs=1e-6)
    # zoom 15: one cell per screen pixel, north up
    t = layer.raster_values(mercator_geom(15, TX, TY))
    for a, b in ab:
        assert t[a, b] == 1000 * a + b
    assert np.isnan(t[1, 0]) and np.isnan(t[100, 100])       # no point, outside the data
    assert np.isnan(layer.raster_values(mercator_geom(15, TX + 1, TY))).all()
    # zoom 14: level 1 (cells of 2 pixels at zoom 15, one pixel here), the data in the top left quarter
    t = layer.raster_values(mercator_geom(14, TX // 2, TY // 2))
    assert np.isfinite(t[:30, :20]).mean() > 0.5 and np.isnan(t[31:, 21:]).all()
    assert layer._rtree is None
    # zoom 17: cells of 4 pixels, tile (4 TX, 4 TY) covers pixels 0 - 63 of the zoom 15 tile
    img = layer.render(mercator_geom(17, 4 * TX, 4 * TY))
    row, col = 4 * 30 + 2, 4 * 21 + 2                        # point (a, b) = (30, 21)
    assert img[row, col] == layer.indices(np.array([30021.0]))[0]
    assert img[row, col + 4] == 255
    x, y = west + 21.5 * res, top - 30.5 * res
    found = layer.value(x + 0.3 * res, y - 0.2 * res, mercator_pixel(17))
    assert found['point'] == ab.index((30, 21)) and found['value'] == 30021.0
    assert found['x'] == pytest.approx(x) and found['y'] == pytest.approx(y)
    assert layer.value(west - res, y, mercator_pixel(17)) is None
    assert layer.value(x + 0.3 * res, y - 0.2 * res, mercator_pixel(15)) == found    # the cell drawn at zoom 15
    # the map: zoom showing all data, at most 16 screen pixels per cell
    w = layer.widget
    assert w.crs == 'web_mercator' and w.axis_labels == ['longitude', 'latitude']
    assert w.max_zoom == 19 and w.zoom == 18


def test_rejects(tmp_path, ras):
    a, pyr = ras
    with pytest.raises(FileNotFoundError):
        view(str(tmp_path / 'missing.zarr'))
    write_zarr(tmp_path / 'line.zarr', np.arange(10.0))
    with pytest.raises(ValueError, match='x.*and.*y'):
        view(str(tmp_path / 'line.zarr'))
    with pytest.raises(ValueError, match='resolution'):
        view(a, resolution=2)
    lon = 120 + 1e-4 * np.arange(20)                          # longitude / latitude: not supported
    with pytest.raises(ValueError, match='longitude'):
        view(lon.astype(np.float32), x=lon, y=lon - 90, resolution=1e-4)


# ---------------------------------------------------------------- coherence

def _coh_pc(tmp_path, pairs):
    """Point cloud of 12 points on a grid with compressed coherence: pair k of point p has magnitude
    (k + 1) / 10 and phase 0.1 * (p + 1) * (k + 1)."""
    gy, gx = np.divmod(np.arange(12), 4)
    k = np.arange(len(pairs))[None, :]
    pt = np.arange(12)[:, None]
    coh = ((k + 1) / 10 * np.exp(1j * 0.1 * (pt + 1) * (k + 1))).astype(np.complex64)
    write_zarr(tmp_path / 'gy.zarr', gy.astype(float)); write_zarr(tmp_path / 'gx.zarr', gx.astype(float))
    write_zarr(tmp_path / 'coh.zarr', coh, (12, 1))
    mc.pc_pyramid(str(tmp_path / 'coh.zarr'), str(tmp_path / 'coh_pyr'), x=str(tmp_path / 'gx.zarr'),
                  y=str(tmp_path / 'gy.zarr'), ras_resolution=1)
    return coh, tmp_path / 'coh_pyr'


def test_coherence_sliders(tmp_path):
    pairs = np.array([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])          # (0, 3) not in the network
    np.savetxt(tmp_path / 'pairs.txt', pairs, fmt='%d')
    coh, pyr = _coh_pc(tmp_path, pairs)
    layer = view(str(pyr), show='coh', image_pairs=str(tmp_path / 'pairs.txt'))
    assert [(k['name'], k['max']) for k in layer.kdims] == [('ref', 3), ('sec', 3)]
    assert layer.default_index == {'ref': 0, 'sec': 1} and layer.bar_label == 'phase (rad)'
    assert layer.ts is None                                  # no time series of pairs
    assert layer.widget.panels[0]['layers'][0]['sliders'] == ['ref', 'sec']
    geom = grid_geom(0, 0, 0, layer.edge_origin)
    t = layer.raster_values(geom, {'ref': 1, 'sec': 3})[:3, :4].ravel()      # pair (1, 3) is k = 3
    np.testing.assert_allclose(t, np.angle(coh[:, 3]), rtol=1e-5)
    np.testing.assert_allclose(layer.raster_values(geom, [3, 1])[:3, :4].ravel(), -np.angle(coh[:, 3]), rtol=1e-5)
    assert (layer.raster_values(geom, [2, 2])[:3, :4] == 0).all()        # diagonal: phase 0
    assert np.isnan(layer.raster_values(geom, [0, 3])[:3, :4]).all()     # not in the network
    assert layer.value(1, 0, 0.25, [1, 3])['value'] == pytest.approx(np.angle(coh[1, 3]), rel=1e-5)
    # magnitude, all pairs by default
    pairs_all = np.array([(i, j) for i in range(4) for j in range(i + 1, 4)])
    coh, pyr = _coh_pc(tmp_path / 'all', pairs_all)
    layer = view(str(pyr), show='coh_abs')
    assert layer.clim == (0, 1) and layer.bar_label == 'coherence' and layer.colors[0] == '#440154'
    np.testing.assert_allclose(layer.raster_values(geom, [3, 2])[:3, :4].ravel(), np.abs(coh[:, 5]), rtol=1e-5)
    assert (layer.raster_values(geom, [1, 1])[:3, :4] == 1).all()
    with pytest.raises(ValueError, match='image pairs'):
        view(str(pyr), show='coh', image_pairs=str(tmp_path / 'pairs.txt'))


def test_coherence_of_data_that_are_not_compressed_coherence(tmp_path):
    """'coh' and 'coh_abs' of data that are not compressed coherence say what compressed coherence is, with the shape
    of the data (of the points for point data): one column per image pair, n (n - 1) / 2 for all pairs of n images or
    as many as the image pairs given. They said "input is not correct value of number of all pairs" (`nimage_from_npair`)
    and "3 image pairs for data with 6 pairs", data without columns failed with a numpy error, and the shape of point
    data was that of their raster."""
    import re
    x, y = np.arange(12.0) % 4, np.arange(12.0) // 4
    rule = ('one column per image pair, n_pairs = n (n - 1) / 2 for all pairs of n images (1, 3, 6, 10, ...) or the '
            'number of the image pairs given')
    for data, kw, error in [
            (np.ones((10, 10, 4), np.complex64), {}, f'not shape (10, 10, 4): {rule} (none are given)'),
            (np.ones((10, 10, 0), np.complex64), {}, f'not shape (10, 10, 0): {rule} (none are given)'),
            (np.ones((12, 4), np.complex64), dict(x=x, y=y), f'not shape (12, 4): {rule} (none are given)'),
            (np.ones((10, 10, 6), np.complex64), dict(image_pairs=[[0, 1], [0, 2], [1, 2]]),
             f'not shape (10, 10, 6): {rule} (3)'),
            (np.ones(12, np.complex64), dict(x=x, y=y), 'not shape (12,)')]:
        for show in ('coh', 'coh_abs'):
            with pytest.raises(ValueError, match='^' + re.escape(f'show={show!r} needs compressed coherence (..., '
                                                                 f'n_pairs), {error}') + '$'):
                view(data, show=show, **kw)
    assert [k['max'] for k in view(np.ones((10, 10, 6), np.complex64), show='coh').kdims] == [3, 3]    # 4 images
    assert [k['max'] for k in view(np.ones((10, 10, 1), np.complex64), show='coh_abs').kdims] == [1, 1]


def test_coherence_sliders_reach_the_last_image(tmp_path):
    pairs = np.array([(0, 1), (0, 3), (1, 2)])          # the last pair does not hold the last image
    np.savetxt(tmp_path / 'pairs.txt', pairs, fmt='%d')
    coh, pyr = _coh_pc(tmp_path, pairs)
    layer = view(str(pyr), show='coh', image_pairs=str(tmp_path / 'pairs.txt'))
    assert [(k['name'], k['max']) for k in layer.kdims] == [('ref', 3), ('sec', 3)]
    geom = grid_geom(0, 0, 0, layer.edge_origin)
    np.testing.assert_allclose(layer.raster_values(geom, {'ref': 0, 'sec': 3})[:3, :4].ravel(), np.angle(coh[:, 1]),
                               rtol=1e-5)


def test_coherence_image_pairs_of_the_data_conventions(tmp_path):
    """Image pairs against the data conventions are rejected (with these unsorted pairs the sliders (0, 1) showed
    the coherence of (0, 2)); a network may leave out images."""
    pairs = np.array([(0, 2), (0, 1), (1, 2)])
    np.savetxt(tmp_path / 'pairs.txt', pairs, fmt='%d')
    _, pyr = _coh_pc(tmp_path, pairs)
    with pytest.raises(ValueError, match='sorted'):
        view(str(pyr), show='coh', image_pairs=str(tmp_path / 'pairs.txt'))
    with pytest.raises(ValueError, match='smaller'):
        view(str(pyr), show='coh_abs', image_pairs=np.array([(0, 1), (2, 0), (1, 2)]))
    layer = view(str(pyr), show='coh_abs', image_pairs=np.array([(1, 2), (1, 3), (2, 3)]))     # no image 0
    geom = grid_geom(0, 0, 0, layer.edge_origin)
    assert layer.raster_values(geom, {'ref': 1, 'sec': 3})[0, 0] == pytest.approx(0.2)     # column 1: 2 / 10


# ---------------------------------------------------------------- named shows of real stacks and of matrices

def _grid_points(tmp_path, data, name):
    """Point data `data` (12, ...) at the points of a 3 x 4 grid (point p at line p // 4, column p % 4): the
    pyramid, and the coordinates x, y (12,)."""
    y, x = (c.astype(float) for c in np.divmod(np.arange(12), 4))
    write_zarr(tmp_path / f'{name}_x.zarr', x); write_zarr(tmp_path / f'{name}_y.zarr', y)
    write_zarr(tmp_path / f'{name}.zarr', data, (12,) + (1,) * (data.ndim - 1))
    mc.pc_pyramid(str(tmp_path / f'{name}.zarr'), str(tmp_path / f'{name}_pyr'), x=str(tmp_path / f'{name}_x.zarr'),
                  y=str(tmp_path / f'{name}_y.zarr'), ras_resolution=1)
    return tmp_path / f'{name}_pyr', x, y


def _check_points(layers, index, expected):
    """The values (12,) `expected` of the points of `_grid_points` in each of `layers` for the slider values
    `index`, to float32 rounding (numpy may compute the phases of contiguous arrays with other SIMD code):
    rasterized (one cell per point) and drawn one by one (the value of the point itself)."""
    for layer in layers:
        geom = grid_geom(0, 0, 0, layer.edge_origin)
        np.testing.assert_allclose(layer.raster_values(geom, index)[:3, :4].ravel(), expected, rtol=1e-6, atol=1e-6)
        for k in (0, 6, 11):
            line, col = divmod(k, 4)
            found = layer.value(col + 0.1, line - 0.1, 0.25, index)
            assert found['point'] == k and found['value'] == pytest.approx(float(expected[k]), rel=1e-6, abs=1e-6)


# (show, sliders [(name, max)], slider values, the two images (i, j) of the slider values)
NAMED_SHOWS = [('intf_0', [('image', 3)], {'image': 2}, (0, 2)),
               ('intf_seq', [('image', 2)], {'image': 1}, (1, 2)),
               ('intf_all', [('ref', 3), ('sec', 3)], {'ref': 3, 'sec': 1}, (3, 1))]


def test_named_shows_of_real_stacks(tmp_path, rng):
    """'intf_0', 'intf_seq' and 'intf_all' of real stacks, e.g. unwrapped phase or a time series: the difference
    of the values of images i and j (not wrapped), of rasters and points; 'phase' needs complex data."""
    ts = rng.uniform(-20, 20, (30, 20, 4)).astype(np.float32)        # differences of up to 40
    write_zarr(tmp_path / 'ts.zarr', ts, (15, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'ts_pyr'))
    pts = rng.uniform(-20, 20, (12, 4)).astype(np.float32)
    pyr, x, y = _grid_points(tmp_path, pts, 'pts')
    geom = grid_geom(0, 0, 0, RAS)
    for show, sliders, index, (i, j) in NAMED_SHOWS:
        for layer in (view(str(tmp_path / 'ts_pyr'), show=show), view(ts, show=show)):     # pyramid, in memory
            assert [(k['name'], k['max']) for k in layer.kdims] == sliders
            np.testing.assert_array_equal(layer.raster_values(geom, index)[:30, :20], ts[..., i] - ts[..., j])
        _check_points([view(str(pyr), show=show), view(pts, x=x, y=y, show=show)], index, pts[:, i] - pts[:, j])
    assert np.abs(ts[..., 3] - ts[..., 1]).max() > 2 * np.pi
    with pytest.raises(ValueError, match='phase needs complex data with one index dimension'):
        view(ts, show='phase').raster_values(geom, {'image': 0})


def test_named_shows_of_stacks_of_matrices(tmp_path, rng):
    """'intf_0', 'intf_seq' and 'intf_all' of stacks of matrices (..., n, n), e.g. coherence matrices: element
    (i, j) of the images of the sliders, its phase for complex data, of rasters and points; 'phase' needs one
    stack dimension, and more than two are rejected."""
    c = (rng.uniform(0.1, 1, (30, 20, 4, 4)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (30, 20, 4, 4)))
         ).astype(np.complex64)
    r = rng.normal(size=(30, 20, 4, 4)).astype(np.float32)
    geom = grid_geom(0, 0, 0, RAS)
    for name, m in (('c', c), ('r', r)):
        write_zarr(tmp_path / f'{name}.zarr', m, (15, 20, 1, 1))
        mc.ras_pyramid(str(tmp_path / f'{name}.zarr'), str(tmp_path / f'{name}_pyr'))
        p = m[:3, :4].reshape(12, 4, 4)
        pyr, x, y = _grid_points(tmp_path, p, f'{name}_pts')
        element = np.angle if np.iscomplexobj(m) else (lambda a: a)
        for show, sliders, index, (i, j) in NAMED_SHOWS:
            for layer in (view(str(tmp_path / f'{name}_pyr'), show=show), view(m, show=show)):
                assert [(k['name'], k['max']) for k in layer.kdims] == sliders
                np.testing.assert_allclose(layer.raster_values(geom, index)[:30, :20], element(m[..., i, j]),
                                           rtol=1e-6, atol=1e-6)
            _check_points([view(str(pyr), show=show), view(p, x=x, y=y, show=show)], index, element(p[:, i, j]))
        with pytest.raises(ValueError, match='phase needs complex data with one index dimension'):
            view(m, show='phase').raster_values(geom, {'image': 0})
    with pytest.raises(ValueError, match='one or two index dimensions'):
        view(np.ones((10, 10, 2, 2, 2), np.complex64), show='intf_all').raster_values(geom, {'ref': 0, 'sec': 1})


def test_named_shows_that_do_not_fit_the_data(tmp_path, rng):
    """A named show that does not fit the data is an error of `view`, before anything is drawn (`moraine quicklook`
    makes it a usage error): 'phase' needs complex data with one stack dimension, 'intf_0', 'intf_seq' and 'intf_all'
    one or two, 'intf_seq' two images. The views were made and failed in every tile, 'intf_all' of data without a
    stack with an IndexError when it was made."""
    ts = rng.normal(size=(30, 20, 3)).astype(np.float32)
    write_zarr(tmp_path / 'ts.zarr', ts, (15, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'ts_pyr'))
    pyr, x, y = _grid_points(tmp_path, ts[:3, :4].reshape(12, 3), 'pts')
    for data, show, error in [
            (ts, 'phase', 'phase needs complex data with one index dimension, got float32 with 1'),
            (str(tmp_path / 'ts_pyr'), 'phase', 'phase needs complex data with one index dimension, got float32 with 1'),
            (str(pyr), 'phase', 'phase needs complex data with one index dimension, got float32 with 1'),
            (np.ones((10, 10, 3, 3), np.complex64), 'phase', 'phase needs complex data with one index dimension, got '
                                                             'complex64 with 2'),
            (np.ones((10, 10), np.complex64), 'phase', 'phase needs complex data with one index dimension, got '
                                                       'complex64 with 0'),
            (np.ones((10, 10), np.float32), 'intf_0', 'the data must have one or two index dimensions, got 0'),
            (np.ones((10, 10), np.complex64), 'intf_all', 'the data must have one or two index dimensions, got 0'),
            (np.ones((10, 10, 2, 2, 2), np.complex64), 'intf_seq', 'the data must have one or two index dimensions, '
                                                                   'got 3'),
            (np.ones((10, 10, 1), np.complex64), 'intf_seq', 'intf_seq needs at least two images, got 1')]:
        with pytest.raises(ValueError, match=f'^{error}$'):
            view(data, show=show)
    with pytest.raises(ValueError, match='^phase needs complex data'):
        view(ts[:3, :4].reshape(12, 3), x=x, y=y, show='phase')
    assert view(np.ones((10, 10, 2), np.complex64), show='intf_seq').kdims == [{'name': 'image', 'max': 0}]


def test_default_show_of_more_than_two_stack_dimensions(tmp_path, rng):
    """The default show of data with three or more stack dimensions, rasters and points, real and complex, is an error
    of `view` that names the shape and the shows (the view was made and failed when drawn: "2 indices given for data
    with 3 index dimensions"); a show function, e.g. that of the message, draws them."""
    import re
    a = rng.random((10, 8, 2, 3, 4)).astype(np.float32)
    write_zarr(tmp_path / 'a.zarr', a, (10, 8, 1, 1, 1))
    mc.ras_pyramid(str(tmp_path / 'a.zarr'), str(tmp_path / 'a_pyr'))
    p = a[:3, :4].reshape(12, 2, 3, 4)
    pyr, x, y = _grid_points(tmp_path, p, 'pts')
    shows = ("more than the shows of a view take (the default and 'intf_0', 'intf_seq', 'intf_all': one or two; "
             "'phase', 'coh', 'coh_abs': one); show them with a function of the visible stack and its sliders, e.g. ")
    for data, kw, error in [
            (a, {}, "(10, 8, 2, 3, 4): 3 stack dimensions, {}show=lambda v, i, j: v[..., i, j, 0], "
                    "sliders={{'i': 2, 'j': 3}}"),
            (str(tmp_path / 'a_pyr'), {}, "(10, 8, 2, 3, 4): 3 stack dimensions, {}show=lambda v, i, j: v[..., i, j, "
                                          "0], sliders={{'i': 2, 'j': 3}}"),
            (a.astype(np.complex64), {}, "(10, 8, 2, 3, 4): 3 stack dimensions, {}show=lambda v, i, j: v[..., i, j, "
                                         "0], sliders={{'i': 2, 'j': 3}}"),
            (p, dict(x=x, y=y), "(12, 2, 3, 4): 3 stack dimensions, {}show=lambda v, i, j: v[..., i, j, 0], "
                                "sliders={{'i': 2, 'j': 3}}"),
            (str(pyr), {}, "(12, 2, 3, 4): 3 stack dimensions, {}show=lambda v, i, j: v[..., i, j, 0], "
                           "sliders={{'i': 2, 'j': 3}}"),
            (np.ones((5, 5, 2, 2, 2, 2), np.float32), {}, "(5, 5, 2, 2, 2, 2): 4 stack dimensions, {}show=lambda v, i, "
                                                          "j: v[..., i, j, 0, 0], sliders={{'i': 2, 'j': 2}}")]:
        with pytest.raises(ValueError, match='^' + re.escape('data of shape ' + error.format(shows)) + '$'):
            view(data, **kw)
    geom = grid_geom(0, 0, 0, RAS)
    for layer in (view(a, show=lambda v, i, j: v[..., i, j, 0], sliders={'i': 2, 'j': 3}),
                  view(str(tmp_path / 'a_pyr'), show=lambda v, i, j: v[..., i, j, 0], sliders={'i': 2, 'j': 3})):
        assert [(k['name'], k['max']) for k in layer.kdims] == [('i', 1), ('j', 2)]
        np.testing.assert_array_equal(layer.raster_values(geom, {'i': 1, 'j': 2})[:10, :8], a[..., 1, 2, 0])
    _check_points([view(str(pyr), show=lambda v, i, j: v[..., i, j, 3], sliders={'i': 2, 'j': 3})], {'i': 0, 'j': 1},
                  p[:, 0, 1, 3])


def test_named_shows_of_real_data_have_linear_colours(tmp_path, rng):
    """'intf_0', 'intf_seq' and 'intf_all' of real data are coloured like other real values, viridis with the colour
    bar of the layer: the differences of two images of a stack over +-(p99 - p01) of the statistics of the data (the
    differences of two values within the 1 % - 99 % range), the elements of matrices (..., n, n) over p01 .. p99.
    They had the cyclic colours of phases over (-pi, pi] and the colour bar 'phase (rad)', so that differences beyond
    pi saturated; complex data keep the colours of phases."""
    import colorcet
    from moraine.cli.plot import _stats
    ts = np.cumsum(rng.normal(-5, 2, (30, 20, 4)), axis=-1).astype(np.float32)      # a time series, e.g. in mm
    ts[..., 0] = 0
    m = rng.normal(5, 2, (30, 20, 3, 3)).astype(np.float32)                           # matrices
    for name, a in (('ts', ts), ('m', m)):
        write_zarr(tmp_path / f'{name}.zarr', a, (15, 20) + (1,) * (a.ndim - 2))
        mc.ras_pyramid(str(tmp_path / f'{name}.zarr'), str(tmp_path / f'{name}_pyr'))
    pts = ts[:3, :4].reshape(12, 4)
    pyr, x, y = _grid_points(tmp_path, pts, 'pts')
    for show, _, _, (i, j) in NAMED_SHOWS:
        for layer, data in [(view(str(tmp_path / 'ts_pyr'), show=show), ts), (view(ts, show=show), ts),
                            (view(str(pyr), show=show), pts), (view(pts, x=x, y=y, show=show), pts),
                            (view(str(tmp_path / 'm_pyr'), show=show), m), (view(m, show=show), m)]:
            st = _stats(data.ravel())
            assert layer.colors[0] == '#440154' and layer.colors[-1] == '#fde725'          # viridis
            assert layer.bar_label == layer.label and 'phase (rad)' not in repr(layer)
            if data is m:
                assert layer.clim == pytest.approx((st['p01'], st['p99']), rel=1e-5)
            else:
                width = st['p99'] - st['p01']
                assert layer.clim == pytest.approx((-width, width), rel=1e-5)
                diff = data[..., i] - data[..., j]
                assert np.abs(diff).max() > np.pi            # beyond the colours of phases, within these
                inside = np.abs(diff) < width
                assert inside.mean() > 0.9 and np.all(np.isin(layer.indices(diff[inside]), [0, 254], invert=True))
    s = np.exp(1j * ts).astype(np.complex64)                  # complex data: the colours of phases
    for show, *_ in NAMED_SHOWS:
        layer = view(s, show=show)
        assert layer.clim == pytest.approx((-np.pi, np.pi)) and layer.bar_label == 'phase (rad)'
        assert layer.colors[0] == colorcet.colorwheel[0].lower()


def test_default_show_of_complex_stacks_of_matrices(tmp_path, rng):
    """By default complex data with two stack dimensions, e.g. coherence matrices (..., n, n), show the phase of
    element (i, j) of the sliders i and j in the colours of phases, as real ones show its value (it failed when
    drawn); complex data without a stack show their phase."""
    c = (rng.uniform(0.1, 1, (30, 20, 3, 3)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (30, 20, 3, 3)))
         ).astype(np.complex64)
    write_zarr(tmp_path / 'c.zarr', c, (15, 20, 1, 1))
    mc.ras_pyramid(str(tmp_path / 'c.zarr'), str(tmp_path / 'c_pyr'))
    p = c[:3, :4].reshape(12, 3, 3)
    pyr, x, y = _grid_points(tmp_path, p, 'pts')
    rasters = [view(str(tmp_path / 'c_pyr')), view(c)]
    points = [view(str(pyr)), view(p, x=x, y=y)]
    for layer in rasters + points:
        assert [(k['name'], k['max']) for k in layer.kdims] == [('i', 2), ('j', 2)]
        assert layer.default_index == {'i': 0, 'j': 0} and layer.show_name is None
        assert layer.clim == pytest.approx((-np.pi, np.pi)) and layer.bar_label == 'phase (rad)'
    geom = grid_geom(0, 0, 0, RAS)
    for layer in rasters:
        np.testing.assert_allclose(layer.raster_values(geom, {'i': 1, 'j': 2})[:30, :20], np.angle(c[..., 1, 2]),
                                   rtol=1e-6, atol=1e-6)
        assert layer.value(7, 3, 1, {'i': 2, 'j': 0})['value'] == pytest.approx(float(np.angle(c[3, 7, 2, 0])),
                                                                                rel=1e-6, abs=1e-6)
    _check_points(points, {'i': 2, 'j': 0}, np.angle(p[:, 2, 0]))
    view(str(tmp_path / 'c_pyr')).png(str(tmp_path / 'c.png'), index=(1, 2))
    assert (tmp_path / 'c.png').stat().st_size > 1000
    real = view(c.real)
    assert [(k['name'], k['max']) for k in real.kdims] == [('i', 2), ('j', 2)] and real.bar_label == 'raster'
    np.testing.assert_array_equal(real.raster_values(geom, {'i': 1, 'j': 2})[:30, :20], c.real[..., 1, 2])
    flat = view(c[..., 0, 1])
    assert flat.kdims == [] and flat.clim == pytest.approx((-np.pi, np.pi))
    np.testing.assert_allclose(flat.raster_values(geom)[:30, :20], np.angle(c[..., 0, 1]), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- time series

def test_raster_time_series(tmp_path, stack, rng):
    s, pyr = stack
    layer = view(str(pyr))
    t = layer.series(7, 3, 1)                                 # pixel (3, 7): the phase of the stack
    assert t['key'] == [3, 7] and t['ref'] is None
    np.testing.assert_allclose(t['values'], np.angle(s[3, 7]), rtol=1e-6)
    t = layer.series(7, 3, 1, ref=[10, 2])                    # relative to pixel (10, 2)
    np.testing.assert_allclose(t['values'], np.angle(s[3, 7] * s[10, 2].conj()), rtol=1e-5, atol=1e-6)
    assert layer.series(-1, 3, 1) is None
    # another time series, e.g. unwrapped phase, with nan; an array works as well as a zarr path
    ts = rng.random((60, 40, 7)).astype(np.float32); ts[3, 7, 2] = np.nan
    write_zarr(tmp_path / 'ts.zarr', ts)
    for series in (str(tmp_path / 'ts.zarr'), ts):
        t = view(str(pyr), series=series).series(7, 3, 1, ref=[0, 0])
        assert t['values'][2] is None
        np.testing.assert_allclose([t['values'][k] for k in (0, 1, 3)], (ts[3, 7] - ts[0, 0])[[0, 1, 3]], rtol=1e-6)
    with pytest.raises(ValueError, match='series'):            # another grid
        view(str(pyr), series=str(pyr / '1.zarr'))


def test_point_time_series(grid_pc):
    pts, pyr = grid_pc
    assert view(str(pyr)).ts is None                         # one value per point: no time series
    n = len(pts)
    ts = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    layer = view(str(pyr), series=ts)
    k, k_ref = pts.index((30, 21)), pts.index((3, 3))
    t = layer.series(21.1, 29.9, 0.25, ref=k_ref)
    assert t['point'] == k and t['values'] == [3.0 * (k - k_ref)] * 3
    with pytest.raises(ValueError, match='n_points'):
        view(str(pyr), series=ts[:5])


# ---------------------------------------------------------------- composition and the widget

def test_composition(ras, grid_pc, mercator_pc, rng):
    pts, pc_pyr = grid_pc
    amp = view(np.abs(rng.normal(size=(80, 60))).astype(np.float32), label='amp', cmap='gray',
               dates=['d0', 'd1', 'd2', 'd3', 'd4'])
    stack = view(rng.normal(size=(80, 60, 5)).astype(np.float32), label='stack', opacity=0.5)
    points = view(str(pc_pyr))
    over = amp * stack * points
    assert isinstance(over, Overlay) and [layer.label for layer in over.layers] == ['amp', 'stack', 'pc_pyr']
    lay = over + amp + (stack * points)
    assert isinstance(lay, Layout) and [len(p) for p in lay.panels] == [3, 1, 2]
    with pytest.raises(TypeError):
        lay * amp
    w = over.widget
    assert isinstance(w, TileView) and w is over.widget and len(w.panels) == 1
    assert w.kdims == [{'name': 'i', 'max': 4}] and w.index == {'i': 0}
    assert w.dates == ['d0', 'd1', 'd2', 'd3', 'd4']           # from the layer that has them
    assert w.view_origin == [-0.5, -0.5]                     # of the first layer
    assert w.extent == [-0.5, -0.5, 59.5, 79.5]              # all layers
    assert [p['opacity'] for p in w.panels[0]['layers']] == [1.0, 0.5, 1.0]
    assert w.panels[0]['series'] and w.panels[0]['layers'][0]['colors'][0] == '#000000'    # gray
    data, _ = over._repr_mimebundle_()                       # displayed as a widget in a notebook
    assert 'application/vnd.jupyter.widget-view+json' in data
    assert len(lay.widget.panels) == 3 and max(lay.widget.frame) <= 560      # smaller maps side by side
    assert [info['sliders'] for info in w.panels[0]['layers']] == [[], ['i'], []]   # the sliders of each layer
    with pytest.raises(ValueError, match='coordinates'):
        (points * view(str(mercator_pc[-1]))).widget


def test_description_for_readers(stack, grid_pc):
    s, pyr = stack
    text = repr(view(str(pyr), show='intf_all', dates=['a', 'b', 'c', 'd']) * view(str(grid_pc[1])))
    assert text.startswith('moraine view: 1 map(s), radar grid')
    assert "stack_pyr: raster (60, 40, 4) complex64, show='intf_all'" in text
    assert 'pc_pyr: point cloud' in text and 'slider ref: 0 .. 3 (a .. d), now 0' in text
    assert 'slider sec: 0 .. 3 (a .. d), now 1' in text and 'time series of stack_pyr' in text
    assert repr(view(str(pyr)).widget) == repr(view(str(pyr)))


def test_state_in_python(grid_pc):
    pts, pyr = grid_pc
    layer = view(str(pyr), series=np.zeros((len(pts), 3), np.float32))
    assert layer.selected == {} and layer.reference == {}
    layer.widget.selected = {'panel': 0, 'layer': 0, 'label': 'pc_pyr', 'key': 5, 'x': 3.0, 'y': 4.0, 'point': 5}
    assert layer.selected['point'] == 5                       # what the map clicked
    stack = view(np.zeros((10, 10, 4), np.float32))
    stack.index = {'i': 2}                                   # moves the slider on the map
    assert stack.widget.index == {'i': 2} and stack.index == {'i': 2}
    layer.polygons = [[[5, 5], [20, 5], [20, 30]]]
    assert layer.widget.polygons == [[[5, 5], [20, 5], [20, 30]]]


def test_index_set_in_python_within_the_sliders(tmp_path, stack):
    """`.index = {...}` takes the sliders of the view within their ranges, as `png(index=...)` does: other values are a
    ValueError and the sliders of the map do not move, so nothing is drawn with them (an image beyond the stack
    reached the tiles: an IndexError for stacks, nan for coherence)."""
    s, pyr = stack                                                      # 4 images
    _, coh_pyr = _coh_pc(tmp_path, np.array([(i, j) for i in range(4) for j in range(i + 1, 4)]))
    for v in (view(str(pyr), show='intf_all'), view(str(coh_pyr), show='coh')):
        moved = []
        v.widget.observe(moved.append, names='index')
        for value, match in [
                ({'sec': 6}, r"^slider 'sec' = 6 is outside its range 0 \.\. 3; the sliders are ref 0 \.\. 3, "
                             r"sec 0 \.\. 3$"),
                ({'ref': 2, 'sec': -1}, r"^slider 'sec' = -1 is outside its range 0 \.\. 3;"),
                ({'image': 1}, r"^no slider 'image'; the view has the sliders ref 0 \.\. 3, sec 0 \.\. 3$")]:
            with pytest.raises(ValueError, match=match):
                v.index = value
            assert v.index == {'ref': 0, 'sec': 1} and moved == []
        v.index = {'ref': 3, 'sec': 0}                                  # the last image too
        assert v.index == {'ref': 3, 'sec': 0} and len(moved) == 1
        v.index = {'sec': 2}                                            # some of the sliders
        assert v.index == {'ref': 3, 'sec': 2}
    layout = view(str(pyr)) + view(np.ones((60, 40, 6), np.complex64))   # the merged slider: image 0 .. 3
    with pytest.raises(ValueError, match=r"^slider 'image' = 5 is outside its range 0 \.\. 3$"):
        layout.index = {'image': 5}
    layout.index = {'image': 3}
    assert layout.index == {'image': 3}


def test_view_messages(grid_pc, monkeypatch):
    from PIL import Image
    pts, pyr = grid_pc
    n = len(pts)
    ts = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    v = (view(np.ones((70, 50), np.float32), label='amp') * view(str(pyr), series=ts)).widget
    assert v.view_origin == [-0.5, -0.5]
    sent = []
    monkeypatch.setattr(v, 'send', lambda content, buffers=None: sent.append((content, buffers)))
    # tile of the second layer: map position = data - view origin
    v._on_msg(v, {'type': 'tile', 'id': 1, 'panel': 0, 'layer': 1, 'z': 0, 'x': 0, 'y': 0, 'index': {}}, [])
    content, buffers = sent.pop()
    img = Image.open(io.BytesIO(buffers[0]))                 # a palette PNG, index 255 transparent
    assert content == {'type': 'tile', 'id': 1} and img.mode == 'P' and img.size == (256, 256)
    layer = v.layers[0][1]
    np.testing.assert_array_equal(np.asarray(img.convert('RGBA')),
                                  layer.palette[layer.render(grid_geom(0, 0, 0, (-0.5, -0.5)))])
    # values of all layers under the cursor
    k, k_ref = pts.index((30, 21)), pts.index((3, 3))
    v._on_msg(v, {'type': 'value', 'id': 2, 'x': 21.5, 'y': 30.4, 'z': 2}, [])
    msg = sent.pop()[0]
    assert (msg['x'], msg['y']) == (21.0, pytest.approx(29.9))
    assert msg['values'] == [{'label': 'amp', 'value': 1.0}, {'label': 'pc_pyr', 'value': 30021.0, 'point': k}]
    # continuous zoom: at zoom 2.5 the probe reaches 4 * 2**-2.5 = 0.7 units, point (30, 21) is 0.9 from (21.9, 30)
    v._on_msg(v, {'type': 'value', 'id': 8, 'x': 22.4, 'y': 30.5, 'z': 2.5}, [])
    assert sent.pop()[0]['values'] == [{'label': 'amp', 'value': 1.0}]
    # time series of the top layer with one, relative to a reference of the same layer
    v._on_msg(v, {'type': 'locate', 'id': 3, 'x': 3.5, 'y': 3.5, 'z': 2}, [])
    loc = sent.pop()[0]
    assert loc['point'] == k_ref and loc['layer'] == 1 and loc['label'] == 'pc_pyr'
    ref = {'panel': 0, 'layer': 1, 'key': k_ref}
    v._on_msg(v, {'type': 'series', 'id': 4, 'x': 21.5, 'y': 30.4, 'z': 2, 'ref': ref}, [])
    msg = sent.pop()[0]
    assert msg['point'] == k and msg['ref'] == k_ref and msg['values'] == [3.0 * (k - k_ref)] * 3
    v._on_msg(v, {'type': 'series', 'id': 5, 'x': 21.5, 'y': 30.4, 'z': 2, 'ref': {**ref, 'layer': 0}}, [])
    assert sent.pop()[0]['ref'] is None                       # a reference of another layer is ignored
    v._on_msg(v, {'type': 'series', 'id': 6, 'x': -5, 'y': 0, 'z': 2}, [])
    assert sent.pop()[0] == {'type': 'series', 'id': 6}       # nothing there
    v._on_msg(v, {'type': 'tile', 'id': 7, 'panel': 0, 'layer': 5, 'z': 0, 'x': 0, 'y': 0}, [])
    assert 'error' in sent.pop()[0]


def test_map_size(ras, grid_pc):
    a, pyr = ras
    w = view(str(pyr)).widget
    assert w.frame == [900, 540] and w.zoom == 0 and w.max_zoom == 4
    assert w.panels[0]['layers'][0]['label'] == 'ras_pyr' and not w.panels[0]['series']
    assert w.panels[0]['layers'][0]['sliders'] == []
    assert view(str(grid_pc[1])).widget.view_origin == [1.5, 2.5]
    # the size of the maps: the width of the notebook by default, or given in pixels (by any layer)
    assert w.size == []
    assert view(str(pyr), size=(600, 400)).widget.size == [600, 400]
    assert (view(str(pyr)) * view(str(grid_pc[1]), size=(500, 300))).widget.size == [500, 300]
    with pytest.raises(ValueError, match='size'):
        view(str(pyr), size=(0, 400))
    assert w.pixel_size(2.5) == 2 ** -2.5                    # the zoom is continuous


def test_terrain(mercator_pc, grid_pc):
    """3D views: the `terrain` of the layers reaches the widget; web mercator layers only."""
    from moraine.cli.tiles import TERRAIN_TILES
    pyr = mercator_pc[4]
    layer = view(str(pyr), terrain=True)
    assert layer.terrain == TERRAIN_TILES
    assert all(k in TERRAIN_TILES['url'] for k in ('{z}', '{x}', '{y}')) and TERRAIN_TILES['max_zoom'] == 15
    w = layer.widget
    assert w.terrain == TERRAIN_TILES and w.crs == 'web_mercator'
    assert '3D view over the terrain' in repr(layer)
    # the maps of a view share one terrain, from any of their layers
    assert (view(str(pyr)) * layer).widget.terrain == TERRAIN_TILES
    assert (view(str(pyr)) + layer).widget.terrain == TERRAIN_TILES
    other = view(str(pyr), terrain='https://tiles.example.org/{z}/{x}/{y}.png')    # another Terrarium service
    assert other.terrain['url'] == 'https://tiles.example.org/{z}/{x}/{y}.png' and other.terrain['max_zoom'] == 15
    with pytest.raises(ValueError, match='one terrain'):
        (layer * other).widget
    # 2D maps by default
    flat = view(str(pyr))
    assert flat.terrain is None and flat.widget.terrain == {}
    assert view(str(pyr), terrain=False).terrain is None and '3D' not in repr(flat)
    with pytest.raises(ValueError, match='web mercator'):
        view(str(grid_pc[1]), terrain=True)
    with pytest.raises(ValueError, match='terrain must be'):
        view(str(pyr), terrain='https://tiles.example.org/{z}/{x}.png')


def test_polygon_file(tmp_path, grid_pc, mercator_pc):
    from moraine.api.polygon import read_polygons
    pts, pyr = grid_pc
    path = tmp_path / 'areas.geojson'
    layer = view(str(pyr), polygons=str(path))
    assert not path.exists() and layer.widget.polygon_coordinates == 'radar_grid'
    layer.polygons = [[[5, 5], [20, 5], [20, 30]], [[30, 40], [35, 40], [35, 45], [30, 45]]]
    polys, coords = read_polygons(path)                       # written on every change
    assert coords == 'radar_grid' and [len(q) for q in polys] == [3, 4]
    assert view(str(pyr), polygons=str(path)).polygons == layer.polygons      # read when the view opens
    assert 'polygons saved to' in repr(layer)
    merc = view(str(mercator_pc[-1]))
    assert merc.widget.polygon_coordinates == 'lonlat'
    with pytest.raises(ValueError, match='radar_grid'):
        merc.widget.load_polygons(path)
    with pytest.raises(ValueError, match='one polygon file'):
        (view(str(pyr), polygons='a.geojson') * view(str(pyr), polygons='b.geojson')).widget


# ---------------------------------------------------------------- PNG images

def test_png(tmp_path, ras, stack, grid_pc, mercator_pc):
    from PIL import Image
    a, pyr = ras
    s, spyr = stack
    out = view(str(pyr)).png(str(tmp_path / 'r.png'), width=400)
    img = Image.open(out)
    assert img.size[0] >= 400 and img.size[1] > 200
    assert (view(str(spyr), show='intf_all') + view(str(spyr))).png(str(tmp_path / 'l.png'), index={'ref': 1})
    pts, ppyr = grid_pc
    # few points and a wide image: drawn one by one; many pixels per cell otherwise rasterized
    for width in (1000, 50):
        view(str(ppyr)).png(str(tmp_path / f'p{width}.png'), width=width)
        assert (tmp_path / f'p{width}.png').stat().st_size > 1000
    assert view(str(mercator_pc[-1])).png(str(tmp_path / 'm.png'))


def test_png_extent(tmp_path, ras, mercator_pc):
    """A part of the scene is drawn from the finest level that fits, down to the data themselves; extents are
    pixels on the radar grid and degrees on web mercator; the view describes the extent and the finest cell."""
    from moraine.cli.tiles import _data_extent, _level_of, _panel_geom, _user_extent
    a, pyr = ras
    layer = view(str(pyr))
    full = _panel_geom(layer.extent, 'grid', 100)[0]
    part = (100, 50, 200, 150)                                         # range, azimuth
    geom, size = _panel_geom(_data_extent(part, 'grid'), 'grid', 100)
    assert _level_of(layer, abs(full.sx)) > 0 and _level_of(layer, abs(geom.sx)) == 0
    values = layer.raster_values(geom, size=size)
    cols = np.floor(geom.x0 + np.arange(size[0]) * geom.sx + 0.5).astype(int)
    rows = np.floor(geom.y0 + np.arange(size[1]) * geom.sy + 0.5).astype(int)
    np.testing.assert_array_equal(values, a[np.ix_(rows, cols)])      # the data, not a coarser level
    assert layer.png(str(tmp_path / 'part.png'), width=300, extent=part)
    assert _user_extent(_data_extent(part, 'grid'), 'grid') == 'range 100 .. 200, azimuth 50 .. 150'
    with pytest.raises(ValueError, match='extent'):
        layer.png(str(tmp_path / 'bad.png'), extent=(200, 50, 100, 150))
    # web mercator: degrees
    pts, res, west, top, mpyr = mercator_pc
    mlayer = view(str(mpyr))
    x0, y0, x1, y1 = mlayer.extent
    text = _user_extent((x0, y0, x1, y1), 'web_mercator')
    lon0, lon1 = (float(v) for v in text.split(',')[0].split()[1::2])
    lat0, lat1 = (float(v) for v in text.split(',')[1].split()[1::2])
    np.testing.assert_allclose(_data_extent((lon0, lat0, lon1, lat1), 'web_mercator'), (x0, y0, x1, y1), atol=10)
    assert mlayer.png(str(tmp_path / 'mpart.png'), extent=(lon0, lat0, (lon0 + lon1) / 2, (lat0 + lat1) / 2))
    assert 'extent lon ' in repr(mlayer) and 'finest cell' in repr(mlayer) and 'levels 0..' in repr(mlayer)
    assert 'extent range -0.5 .. 499.5, azimuth -0.5 .. 299.5' in repr(layer)          # cell edges


def test_png_extent_of_words(tmp_path, ras, mercator_pc):
    """An extent of words (e.g. 'west' typed for its number) is an error that says that the extent is four numbers and
    what they are, on the radar grid and on web mercator; it was "could not convert string to float: 'west'"."""
    a, pyr = ras
    out = tmp_path / 'e.png'
    with pytest.raises(ValueError, match=r"^extent must be four numbers \(range_min, azimuth_min, range_max, "
                                         r"azimuth_max\), pixels on the radar grid, got \('west', 'south', 'east', "
                                         r"'north'\)$"):
        view(str(pyr)).png(str(out), extent=('west', 'south', 'east', 'north'))
    with pytest.raises(ValueError, match=r"^extent must be four numbers \(west, south, east, north\), longitudes and "
                                         r"latitudes in degrees, got \(14.1, 'south', 14.2, None\)$"):
        view(str(mercator_pc[-1])).png(str(out), extent=(14.1, 'south', 14.2, None))
    assert not out.exists()


def test_png_index_within_the_sliders(tmp_path, stack):
    """`png(index=...)` draws the slider values given, by name or in the order of the sliders, within the ranges of
    the sliders only: values beyond the images (they drew nan for coherence or failed with an IndexError) and
    negative ones (they failed or drew an image counted from the end) are errors that name the ranges."""
    from moraine.cli.tiles import TILE
    s, pyr = stack                                                      # 4 images
    out = tmp_path / 'i.png'
    for show, index, match in [
            ('intf_all', (0, 6), r"slider 'sec' = 6 is outside its range 0 \.\. 3; the sliders are ref 0 \.\. 3, "
                                 r"sec 0 \.\. 3$"),
            ('intf_all', {'ref': -1}, r"slider 'ref' = -1 is outside its range 0 \.\. 3;"),
            ('phase', {'image': 4}, r"slider 'image' = 4 is outside its range 0 \.\. 3$"),
            ('intf_seq', (3,), r"slider 'image' = 3 is outside its range 0 \.\. 2$")]:
        with pytest.raises(ValueError, match=match):
            view(str(pyr), show=show).png(str(out), index=index)
    _, coh_pyr = _coh_pc(tmp_path, np.array([(i, j) for i in range(4) for j in range(i + 1, 4)]))
    with pytest.raises(ValueError, match=r"slider 'sec' = 6 is outside its range 0 \.\. 3"):
        view(str(coh_pyr), show='coh').png(str(out), index=(0, 6))
    with pytest.raises(ValueError, match=r"slider 'image' = 5 is outside its range 0 \.\. 3$"):   # the merged slider
        (view(str(pyr)) + view(np.ones((60, 40, 6), np.complex64))).png(str(out), index={'image': 5})
    assert not out.exists()
    # the values given are drawn, the last ones of the sliders too
    layer = view(str(pyr), show='intf_all')
    drawn = []
    raster_values = layer.raster_values
    layer.raster_values = lambda geom, index=(), size=(TILE, TILE): \
        drawn.append((geom, size, raster_values(geom, index, size))) or drawn[-1][2]
    for index, (i, j) in [((3, 0), (3, 0)), ({'sec': 3}, (0, 3))]:
        layer.png(str(out), width=200, index=index)
        geom, size, values = drawn[-1]
        cols = np.floor(geom.x0 + np.arange(size[0]) * geom.sx + 0.5).astype(int)
        rows = np.floor(geom.y0 + np.arange(size[1]) * geom.sy + 0.5).astype(int)
        np.testing.assert_allclose(values, np.angle(s[..., i] * s[..., j].conj())[np.ix_(rows, cols)], rtol=1e-5,
                                   atol=1e-6)


def test_png_index_of_the_sliders_of_the_view(tmp_path, stack):
    """`png(index=...)` takes the sliders of the view only: a name that is not a slider and more values than sliders
    are errors that name the sliders of the view (they were ignored and the default images drawn)."""
    s, pyr = stack                                                      # 4 images
    out = tmp_path / 'i.png'
    for show, index, match in [
            (None, {'images': 2}, r"^no slider 'images'; the view has the slider image 0 \.\. 3$"),
            ('intf_all', {'ref': 1, 'sek': 2, 'i': 0}, r"^no sliders 'sek', 'i'; the view has the sliders ref 0 \.\. 3, "
                                                      r"sec 0 \.\. 3$"),
            (None, (0, 2), r"^2 values for 1 slider; the view has the slider image 0 \.\. 3$"),
            ('intf_all', [1, 2, 3], r"^3 values for 2 sliders; the view has the sliders ref 0 \.\. 3, sec 0 \.\. 3$")]:
        with pytest.raises(ValueError, match=match):
            view(str(pyr), show=show).png(str(out), index=index)
    with pytest.raises(ValueError, match=r'^1 value for 0 sliders; the view has no sliders$'):
        view(np.ones((10, 10), np.float32)).png(str(out), index=(0,))
    with pytest.raises(ValueError, match=r"^no slider 'image'; the view has no sliders$"):
        view(np.ones((10, 10), np.float32)).png(str(out), index={'image': 0})
    layout = view(str(pyr)) + view(np.ones((60, 40, 2), np.float32))     # sliders image and i, of different layers
    with pytest.raises(ValueError, match=r"^3 values for 2 sliders; the view has the sliders image 0 \.\. 3, i 0 \.\. 1$"):
        layout.png(str(out), index=(0, 1, 0))
    assert not out.exists()
    layout.png(str(out), width=200, index={'i': 1})                     # the sliders of any layer of the view
    assert out.stat().st_size > 1000
