"""Views (moraine.cli.view): layers (tile geometry, colours, points, coordinates, show functions, time series),
composition, the widget (sliders, messages with the browser, polygons) and PNG images."""
import io

import numpy as np
import pytest
import zarr

import moraine.cli as mc
from moraine.cli import view, TileView
from moraine.cli.tiles import grid_geom, mercator_geom, mercator_pixel, WORLD, Overlay, Layout

RAS = (-0.5, -0.5)       # view origin of rasters: pixel i centred at coordinate i


def _zarr(path, a, chunks=None):
    z = zarr.open(str(path), mode='w', shape=a.shape, dtype=a.dtype, chunks=chunks or a.shape)
    z[...] = a
    return z


@pytest.fixture
def ras(tmp_path):
    """300 x 500 raster with value = 1000 * line + column, and its pyramid."""
    a = (1000 * np.arange(300)[:, None] + np.arange(500)[None, :]).astype(np.float32)
    _zarr(tmp_path / 'ras.zarr', a, (100, 100))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'ras_pyr'))
    return a, tmp_path / 'ras_pyr'


@pytest.fixture
def stack(tmp_path, rng):
    """60 x 40 x 4 complex stack of unit phasors and its pyramid."""
    s = np.exp(1j * rng.uniform(-np.pi, np.pi, (60, 40, 4))).astype(np.complex64)
    _zarr(tmp_path / 'stack.zarr', s, (30, 40, 1))
    mc.ras_pyramid(str(tmp_path / 'stack.zarr'), str(tmp_path / 'stack_pyr'))
    return s, tmp_path / 'stack_pyr'


@pytest.fixture
def grid_pc(tmp_path):
    """Points on the radar grid (lines 3 - 62, columns 2 - 41, one cell in three) with value
    1000 * line + column, and their pyramid."""
    pts = np.array([(y, x) for y in range(3, 63) for x in range(2, 42) if (x + y) % 3 == 0], dtype=float)
    _zarr(tmp_path / 'gy.zarr', pts[:, 0])
    _zarr(tmp_path / 'gx.zarr', pts[:, 1])
    _zarr(tmp_path / 'val.zarr', (1000 * pts[:, 0] + pts[:, 1]).astype(np.float32))
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
    _zarr(tmp_path / 'mx.zarr', west + (ab[:, 1] + 0.5) * res)
    _zarr(tmp_path / 'my.zarr', top - (ab[:, 0] + 0.5) * res)
    _zarr(tmp_path / 'val.zarr', (1000 * ab[:, 0] + ab[:, 1]).astype(np.float32))
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
    assert layer.value(7.2, 3.4, 1) == {'x': 7.0, 'y': 3.0, 'key': [3, 7], 'value': float(a[3, 7])}
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


def test_point_cloud_points_zoom(grid_pc):
    pts, pyr = grid_pc
    layer = view(str(pyr))
    # zoom 2: cells of 4 screen pixels, points as disks of radius 1.6 pixels at their coordinates
    rgba = layer.render(grid_geom(2, 0, 0, layer.edge_origin))
    assert rgba.shape == (256, 256, 4)
    row, col = int((30 - 2.5) * 4), int((21 - 1.5) * 4)      # point (y, x) = (30, 21)
    np.testing.assert_array_equal(rgba[row, col], layer.colorize(np.array([30021.0]))[0])
    assert rgba[row, col + 4, 3] == 0                        # (30, 22): no point, 4 pixels from the others
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
    # zarr paths work too; coordinates off the integer grid need a resolution
    assert view(str(d / 'val.zarr'), x=str(d / 'gx.zarr'), y=str(d / 'gy.zarr')).shape == (60, 40)
    assert view(val, x=gx + 0.5, y=gy, resolution=1).shape == (60, 40)
    with pytest.raises(ValueError, match='resolution'):
        view(val, x=gx + 0.5, y=gy)
    with pytest.raises(ValueError, match='x.*and.*y'):
        view(val)
    with pytest.raises(ValueError, match='both'):
        view(val, x=gx)


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
    rgba = layer.render(mercator_geom(17, 4 * TX, 4 * TY))
    row, col = 4 * 30 + 2, 4 * 21 + 2                        # point (a, b) = (30, 21)
    np.testing.assert_array_equal(rgba[row, col], layer.colorize(np.array([30021.0]))[0])
    assert rgba[row, col + 4, 3] == 0
    x, y = west + 21.5 * res, top - 30.5 * res
    found = layer.value(x + 0.3 * res, y - 0.2 * res, mercator_pixel(17))
    assert found['point'] == ab.index((30, 21)) and found['value'] == 30021.0
    assert found['x'] == pytest.approx(x) and found['y'] == pytest.approx(y)
    assert layer.value(west - res, y, mercator_pixel(17)) is None
    # the map: zoom showing all data, at most 16 screen pixels per cell
    w = layer.widget
    assert w.crs == 'web_mercator' and w.axis_labels == ['longitude', 'latitude']
    assert w.max_zoom == 19 and w.zoom == 18


def test_rejects(tmp_path, ras):
    a, pyr = ras
    with pytest.raises(FileNotFoundError):
        view(str(tmp_path / 'missing.zarr'))
    _zarr(tmp_path / 'line.zarr', np.arange(10.0))
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
    _zarr(tmp_path / 'gy.zarr', gy.astype(float)); _zarr(tmp_path / 'gx.zarr', gx.astype(float))
    _zarr(tmp_path / 'coh.zarr', coh, (12, 1))
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
    _zarr(tmp_path / 'ts.zarr', ts)
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
    img = np.asarray(Image.open(io.BytesIO(buffers[0])))
    assert content == {'type': 'tile', 'id': 1} and img.shape == (256, 256, 4)
    np.testing.assert_array_equal(img, v.layers[0][1].render(grid_geom(0, 0, 0, (-0.5, -0.5))))
    # values of all layers under the cursor
    k, k_ref = pts.index((30, 21)), pts.index((3, 3))
    v._on_msg(v, {'type': 'value', 'id': 2, 'x': 21.5, 'y': 30.4, 'z': 2}, [])
    msg = sent.pop()[0]
    assert (msg['x'], msg['y']) == (21.0, pytest.approx(29.9))
    assert msg['values'] == [{'label': 'amp', 'value': 1.0}, {'label': 'pc_pyr', 'value': 30021.0, 'point': k}]
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
    assert view(str(grid_pc[1])).widget.view_origin == [1.5, 2.5]


def test_polygon_file(tmp_path, grid_pc, mercator_pc):
    from moraine.polygon import read_polygons
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
