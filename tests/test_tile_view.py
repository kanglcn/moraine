"""Tile viewer of pyramids: tile geometry, colours, points and the messages with the browser."""
import io

import numpy as np
import pytest
import zarr

pytest.importorskip('anywidget')

import moraine.cli as mc
from moraine.command.tile_view import tile_view, TileView


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


def test_tile_geometry(ras):
    a, pyr = ras
    v = tile_view(str(pyr))
    assert v.shape == [300, 500] and v.index == [] and v.kdims == []
    # the initial zoom shows the whole scene in the frame: 500 x 300 pixels in 900 x 540 screen pixels
    assert v.frame == [900, 540] and v.zoom == 0
    assert v.origin == [-0.5, -0.5] and v.res == 1          # pixel i centred at coordinate i
    # zoom 0: one screen pixel per pixel, tile (1, 0) is columns 256 - 511, lines 0 - 255
    t = v.tile_values(0, 1, 0)
    assert t.shape == (256, 256)
    np.testing.assert_array_equal(t[:, :244], a[:256, 256:])
    assert np.isnan(t[:, 244:]).all()                       # outside the scene
    # zoom -1: level 1, every 2nd pixel
    t = v.tile_values(-1, 0, 0)
    np.testing.assert_array_equal(t[:150, :250], a[::2, ::2])
    assert np.isnan(t[150:]).all() and np.isnan(t[:, 250:]).all()
    # zoomed in: 2**2 screen pixels per pixel, tiles of 64 pixels, drawn enlarged by the browser
    t = v.tile_values(2, 3, 1)
    np.testing.assert_array_equal(t, a[64:128, 192:256])
    # beyond the coarsest level (8, 2 x 2 pixels): every 2nd pixel of it
    t = v.tile_values(-9, 0, 0)
    assert t.shape == (256, 256) and t[0, 0] == a[0, 0] and np.isnan(t[0, 1]) and np.isnan(t[1, 0])
    assert np.isnan(v.tile_values(0, 5, 0)).all() and np.isnan(v.tile_values(0, -1, 0)).all()


def test_colours_and_value(ras):
    a, pyr = ras
    v = tile_view(str(pyr))
    lo, hi = v.clim
    assert 0 < lo < hi < a.max()                             # 1 - 99 % range of the values
    assert v.colors[0] == '#440154'                          # viridis
    rgba = v.colorize(np.array([[lo - 1, hi + 1, np.nan]]))
    assert rgba.shape == (1, 3, 4) and rgba.dtype == np.uint8
    assert tuple(rgba[0, 0, :3]) == (0x44, 0x01, 0x54) and rgba[0, 0, 3] == 255
    assert tuple(rgba[0, 1, :3]) == (0xfd, 0xe7, 0x25)       # last viridis colour
    assert rgba[0, 2, 3] == 0                                # nan is transparent
    assert v.value(7.2, 3.9) == {'x': 7, 'y': 3, 'value': float(a[3, 7])}
    assert v.value(500, 0) is None and v.value(-0.1, 0) is None


def test_phase_stack(tmp_path, rng):
    import colorcet
    stack = np.exp(1j * rng.uniform(-np.pi, np.pi, (60, 40, 4))).astype(np.complex64)
    _zarr(tmp_path / 'stack.zarr', stack, (30, 40, 1))
    mc.ras_pyramid(str(tmp_path / 'stack.zarr'), str(tmp_path / 'stack_pyr'))
    v = tile_view(str(tmp_path / 'stack_pyr'))
    assert v.clim == pytest.approx([-np.pi, np.pi]) and v.label == 'phase (rad)'
    assert v.colors[0] == colorcet.colorwheel[0].lower()
    assert v.kdims == [{'name': 'i', 'max': 3}] and v.index == [0]
    assert v.zoom == 3                                       # 40 x 60 pixels, 2**3 screen pixels each
    np.testing.assert_allclose(v.tile_values(0, 0, 0, [2])[:60, :40], np.angle(stack[..., 2]), rtol=1e-6)
    v = tile_view(str(tmp_path / 'stack_pyr'), post_proc='intf_seq')
    assert v.kdims == [{'name': 'i', 'max': 2}]
    np.testing.assert_allclose(v.tile_values(0, 0, 0, [1])[:60, :40],
                               np.angle(stack[..., 1] * stack[..., 2].conj()), rtol=1e-5, atol=1e-6)
    v = tile_view(str(tmp_path / 'stack_pyr'), post_proc='intf_all')
    assert [k['name'] for k in v.kdims] == ['i', 'j'] and v.index == [0, 0]


def test_messages(ras, monkeypatch):
    from PIL import Image
    a, pyr = ras
    v = tile_view(str(pyr))
    sent = []
    monkeypatch.setattr(v, 'send', lambda content, buffers=None: sent.append((content, buffers)))
    v._on_msg(v, {'type': 'tile', 'id': 7, 'z': 1, 'x': 0, 'y': 0, 'index': []}, [])
    content, buffers = sent.pop()
    assert content == {'type': 'tile', 'id': 7}
    img = Image.open(io.BytesIO(buffers[0]))
    assert img.format == 'PNG' and img.mode == 'RGBA' and img.size == (128, 128)
    np.testing.assert_array_equal(np.asarray(img), v.colorize(a[:128, :128].astype(float)))
    v._on_msg(v, {'type': 'value', 'id': 8, 'x': 4.5, 'y': 2.5, 'z': 0}, [])
    assert sent.pop() == ({'type': 'value', 'id': 8, 'x': 4, 'y': 2, 'value': float(a[2, 4])}, None)
    v._on_msg(v, {'type': 'value', 'id': 10, 'x': -3, 'y': 2, 'z': 0}, [])
    assert sent.pop() == ({'type': 'value', 'id': 10, 'value': None}, None)   # outside the scene
    v._on_msg(v, {'type': 'tile', 'id': 9, 'z': 0, 'x': 0, 'y': 0, 'index': [5]}, [])   # wrong index
    content, _ = sent.pop()
    assert content['id'] == 9 and 'error' in content


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


def test_point_cloud_raster_zoom(grid_pc):
    pts, pyr = grid_pc
    v = tile_view(str(pyr))
    # map in cells of level 0: cell centres at the grid coordinates, the first at (x, y) = (2, 3)
    assert v.origin == [1.5, 2.5] and v.res == 1 and v.axis_labels == ['range', 'azimuth']
    t = v.tile_values(0, 0, 0)
    for y, x in pts:
        if y < 62 and x < 41:        # the last line and column share cells with the previous (pc_pyramid)
            assert t[y - 3, x - 2] == 1000 * y + x
    assert np.isnan(t[0, 0])                                 # (y, x) = (3, 2): no point
    v.tile_rgba(0, 0, 0)
    assert v._rtree is None                                  # overviews do not read the coordinates


def test_point_cloud_points_zoom(grid_pc):
    pts, pyr = grid_pc
    v = tile_view(str(pyr))
    # zoom 2: cells of 4 screen pixels, points as disks of radius 1.6 pixels at their coordinates
    rgba = v.tile_rgba(2, 0, 0)
    assert rgba.shape == (256, 256, 4)
    row, col = int((30 - 2.5) * 4), int((21 - 1.5) * 4)      # point (y, x) = (30, 21)
    np.testing.assert_array_equal(rgba[row, col], v.colorize(np.array([30021.0]))[0])
    assert rgba[row, col + 4, 3] == 0                        # (30, 22): no point, 4 pixels from the others
    # value of the nearest point under the cursor
    k = pts.index((30, 21))
    assert v.value(19.6, 27.4, z=2) == {'point': k, 'x': 21.0, 'y': 30.0, 'value': 30021.0}
    assert v.value(20.5, 27.5, z=4) is None                  # centre of an empty cell, zoomed in
    assert v.value(-1, 0, z=4) is None


def test_rejects_other_inputs(tmp_path, ras):
    a, _ = ras
    with pytest.raises(ValueError, match='not a pyramid'):
        tile_view(str(tmp_path / 'ras.zarr'))
    x = np.arange(20, dtype=float) * 7.5 - 1.6e7              # map coordinates (web mercator)
    for name, arr in [('x.zarr', x), ('y.zarr', x + 2.2e7), ('v.zarr', x.astype(np.float32))]:
        _zarr(tmp_path / name, arr)
    mc.pc_pyramid(str(tmp_path / 'v.zarr'), str(tmp_path / 'map_pyr'), x=str(tmp_path / 'x.zarr'),
                  y=str(tmp_path / 'y.zarr'), ras_resolution=7.5)
    with pytest.raises(ValueError, match='map coordinates'):
        TileView(str(tmp_path / 'map_pyr'))
