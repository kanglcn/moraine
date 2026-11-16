"""The code keeps the promises of docs/contracts/: JSON output, pipeline files, pyramids and data conventions."""
import re

import numpy as np
import pytest
import zarr

import moraine.cli as mc
from moraine.command import main, JSON_VERSION, UsageError
from moraine.command.pipeline import load_pipeline, PIPELINE_VERSION
from moraine.command.summary import pyramid_levels, summarize
from moraine.cli.plot import PYRAMID_VERSION
from conftest import REPO, write_zarr, json_out, random_gix

CONTRACTS = REPO / 'docs' / 'contracts'


def _tables(doc):
    """{section heading: {field: optional}} of the field tables of a contract."""
    out, section = {}, None
    for line in (CONTRACTS / doc).read_text().splitlines():
        if line.startswith('#'):
            section = line.lstrip('#').strip()
        m = re.match(r'\| `([^`]+)` \|[^|]*\|(.*)\|$', line)
        if m and section:
            out.setdefault(section, {})[m[1]] = m[2].strip().startswith('*optional*')
    return out


JSON = _tables('json-output.md')


def _check(obj, *sections):
    """`obj` has every required field of `sections` and no field they do not document."""
    fields = {}
    for s in sections:
        fields.update(JSON[s])
    missing = [f for f, optional in fields.items() if not optional and f not in obj]
    extra = [k for k in obj if k not in fields]
    assert not missing and not extra, f'{sections}: missing {missing}, undocumented {extra}'


def _summary(s):
    if 'error' in s:
        return _check(s, 'failed summary')
    _check(s, s['kind'].replace('raster ', '').replace('point cloud ', '') + ' summary')


def test_versions_in_the_index():
    index = (CONTRACTS / 'README.md').read_text()
    for name, version in [('JSON output', JSON_VERSION), ('Pipeline files', PIPELINE_VERSION),
                          ('Pyramids', PYRAMID_VERSION)]:
        assert re.search(rf'\[{name}\].*\| {version} \|', index), name


# ---------------------------------------------------------------- JSON output

def test_json_processing_command_list_info_error(tmp_path, capsys, rng):
    write_zarr(tmp_path / 'ras.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
    assert main(['pc-logic-ras', '--ras', str(tmp_path / 'ras.zarr'), '--gix', str(tmp_path / 'gix.zarr'),
                 '--operation', 'ras>0.5', '--json', '-q']) == 0
    out = json_out(capsys)
    assert out['version'] == JSON_VERSION and out['ok'] is True
    _check(out, 'Every output', 'processing command')
    for s in out['summaries']:
        _summary(s)

    assert main(['list', '--json']) == 0
    out = json_out(capsys)
    _check(out, 'Every output', 'list')
    assert set(out['commands'][0]) == {'name', 'module', 'summary'}

    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    (tmp_path / 'dir').mkdir(); write_zarr(tmp_path / 'dir' / '0.zarr', np.zeros(3))
    zarr.open_group(str(tmp_path / 'grp.zarr'), mode='w')
    (tmp_path / 'f.txt').write_text('x')
    capsys.readouterr()
    paths = ['ras.zarr', 'pyr', 'dir', 'grp.zarr', 'f.txt']
    assert main(['info', *(str(tmp_path / p) for p in paths), '--json']) == 0
    out = json_out(capsys)
    _check(out, 'Every output', 'info')
    assert [s['kind'] for s in out['summaries']] == ['array', 'raster pyramid', 'directory', 'group', 'file']
    for s in out['summaries']:
        _summary(s)

    assert main(['info', str(tmp_path / 'missing.zarr'), '--json']) == 1
    out = json_out(capsys)
    assert out['ok'] is False
    _check(out, 'Every output', 'error')


@pytest.mark.parametrize('data', [
    np.full((60, 40), np.nan, np.float32),
    np.ones((60, 40), np.complex64),
    np.arange(2400).reshape(60, 40) % 2 == 0,
], ids=['nan', 'complex', 'bool'])
def test_json_pyramid_summaries(tmp_path, data):
    write_zarr(tmp_path / 'a.zarr', data, (20, 20))
    mc.ras_pyramid(str(tmp_path / 'a.zarr'), str(tmp_path / 'pyr'))
    _summary(summarize(str(tmp_path / 'pyr')))


def test_json_quicklook(tmp_path, capsys, rng):
    write_zarr(tmp_path / 'ras.zarr', rng.random((64, 48)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    capsys.readouterr()
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'q.png'), '--json']) == 0
    _check(json_out(capsys), 'Every output', 'quicklook')
    assert main(['view', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'v.ipynb'), '--json']) == 0
    _check(json_out(capsys), 'Every output', 'view')


PIPELINE = '''
[[step]]
name = "select"
run = "pc-logic-ras"
ras = "ras.zarr"
gix = "gix.zarr"
operation = "{operation}"

[[step]]
name = "pyramid"
run = "ras-pyramid"
ras = "ras.zarr"
out_dir = "pyr"
'''


def test_json_run_and_status(tmp_path, capsys, rng):
    write_zarr(tmp_path / 'ras.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
    path = tmp_path / 'p.toml'
    path.write_text(PIPELINE.format(operation='ras>0.5'))
    assert main(['run', str(path), '--json']) == 0
    out = json_out(capsys)
    _check(out, 'Every output', 'run')
    assert set(out['plan'][0]) == {'name', 'command', 'action'}
    for step in out['steps']:
        _check(step, 'step record')
        for s in step['summaries']:
            _summary(s)
    assert out['steps'][1]['quicklooks']

    assert main(['status', str(path), '--json']) == 0
    out = json_out(capsys)
    _check(out, 'Every output', 'status')
    assert {'name', 'command', 'status'} <= set(out['steps'][0])

    path.write_text(PIPELINE.format(operation='nonsense>'))
    assert main(['run', str(path), '--json']) == 1
    out = json_out(capsys)
    assert out['ok'] is False and 'error' not in out
    _check(out, 'Every output', 'run')
    assert out['steps'][-1]['status'] == 'failed'
    _check(out['steps'][-1], 'step record')


# ---------------------------------------------------------------- pipeline files

STEP = '[[step]]\nname = "pairs"\nrun = "image-pairs"\nout = "p.txt"\nnimages = 3\n'


def test_pipeline_keys_are_the_documented_ones(tmp_path):
    keys = [k.split()[1] for k in _tables('pipeline-file.md')['Tables'] if k.startswith('[pipeline] ')]
    assert sorted(keys) == ['quicklook', 'version', 'workdir']
    values = {'version': PIPELINE_VERSION, 'workdir': '"."', 'quicklook': 'false'}
    for k in keys:
        (tmp_path / 'p.toml').write_text(f'[pipeline]\n{k} = {values[k]}\n' + STEP)
        load_pipeline(tmp_path / 'p.toml')
    (tmp_path / 'p.toml').write_text('[pipeline]\nname = "x"\n' + STEP)
    with pytest.raises(UsageError):
        load_pipeline(tmp_path / 'p.toml')


def test_pipeline_version(tmp_path):
    (tmp_path / 'p.toml').write_text(f'[pipeline]\nversion = {PIPELINE_VERSION + 1}\n' + STEP)
    with pytest.raises(UsageError, match='update moraine'):
        load_pipeline(tmp_path / 'p.toml')


def test_pipeline_variables_from_files(tmp_path):
    rows = [k for k in _tables('pipeline-file.md')['Tables'] if k.startswith('[vars] ')]
    assert rows == ['[vars] NAME = { file = "PATH", key = "KEY" }']
    (tmp_path / 'm.toml').write_text('n = 3\n')
    var = rows[0].replace('[vars] NAME', 'n').replace('PATH', 'm.toml').replace('KEY', 'n')
    (tmp_path / 'p.toml').write_text(f'[vars]\n{var}\n' + STEP.replace('nimages = 3', 'nimages = "${n}"'))
    assert load_pipeline(tmp_path / 'p.toml')['steps'][0]['values']['nimages'] == 3     # a whole value keeps its type
    (tmp_path / 'p.toml').write_text('[vars]\nn = { file = "m.toml", key = "n", default = 1 }\n' + STEP)
    with pytest.raises(UsageError, match='a variable read from a file'):          # no other keys
        load_pipeline(tmp_path / 'p.toml')


# ---------------------------------------------------------------- pyramids

@pytest.mark.parametrize('rows', [1024, 16])     # one band, or bands of 16 lines with levels beyond the band
def test_raster_pyramid_layout(tmp_path, rng, rows):
    ras = rng.random((50, 37, 2)).astype(np.float32)
    write_zarr(tmp_path / 'ras.zarr', ras, (20, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'), chunks=(16, 16), rows=rows)
    pyr = tmp_path / 'pyr'
    meta = zarr.open(str(pyr / '0.zarr'), mode='r').attrs['moraine_pyramid']
    assert meta['version'] == PYRAMID_VERSION and meta['kind'] == 'raster'
    # the statistics of the whole raster, in the marker and per channel
    assert meta['stats']['min'] == pytest.approx(ras.min(), rel=1e-5) and meta['stats']['nan_fraction'] == 0
    assert meta['stats']['mean'] == pytest.approx(ras.mean(), rel=1e-5) and meta['stats']['std'] == pytest.approx(ras.std(), rel=1e-4)
    assert meta['stats']['p50'] == pytest.approx(np.median(ras), rel=1e-4)
    table = zarr.open(str(pyr / 'stats.zarr'), mode='r')
    assert table.shape == (2, 8) and table.attrs['columns'][3] == 'mean'
    np.testing.assert_allclose(table[:, 3], ras.mean(axis=(0, 1)), rtol=1e-6)
    maxlevel = int(np.floor(np.log2(37)))
    assert sorted(p.name for p in pyr.iterdir()) == sorted([*(f'{l}.zarr' for l in range(maxlevel + 1)), 'stats.zarr'])
    for level in range(maxlevel + 1):
        z = zarr.open(str(pyr / f'{level}.zarr'), mode='r')
        np.testing.assert_array_equal(z[:], ras[::2**level, ::2**level])
        assert z.chunks == (16, 16, 1)
    assert pyramid_levels(pyr) == list(range(maxlevel + 1))
    with pytest.raises(ValueError, match='power of 2'):
        mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr2'), rows=100)


def test_point_cloud_pyramid_layout(tmp_path, rng):
    n = 200
    g = random_gix(rng, (40, 30), n)
    write_zarr(tmp_path / 'x.zarr', g[:, 1].astype(float)); write_zarr(tmp_path / 'y.zarr', g[:, 0].astype(float))
    write_zarr(tmp_path / 'pc.zarr', rng.random(n).astype(np.float32))
    mc.pc_pyramid(str(tmp_path / 'pc.zarr'), str(tmp_path / 'pyr'), x=str(tmp_path / 'x.zarr'),
                  y=str(tmp_path / 'y.zarr'), ras_resolution=1)
    pyr = tmp_path / 'pyr'
    meta = zarr.open(str(pyr / '0.zarr'), mode='r').attrs['moraine_pyramid']
    assert meta['version'] == PYRAMID_VERSION and meta['kind'] == 'point cloud'
    pc_values = zarr.open(str(tmp_path / 'pc.zarr'), mode='r')[:]
    assert meta['stats']['max'] == pytest.approx(pc_values.max(), rel=1e-5)      # of the points, not the cells
    assert zarr.open(str(pyr / 'stats.zarr'), mode='r').shape == (8,)
    levels = pyramid_levels(pyr)
    expected = {'bounds.toml', 'x.zarr', 'y.zarr', 'pc.zarr', 'rtree.zarr', 'stats.zarr', *(f'{l}.zarr' for l in levels),
                *(f'idx_{l}.zarr' for l in levels)}
    assert {p.name for p in pyr.iterdir()} == expected
    tree = zarr.open(str(pyr / 'rtree.zarr'), mode='r')       # bounding box tree of the points
    assert tree.ndim == 2 and tree.shape[1] == 4 and tree.dtype == np.float64
    assert tree.attrs['n_points'] == n and tree.attrs['page_size'] >= 1
    pc = zarr.open(str(pyr / 'pc.zarr'), mode='r')[:]
    for level in levels:
        ras = zarr.open(str(pyr / f'{level}.zarr'), mode='r')[:]
        idx = zarr.open(str(pyr / f'idx_{level}.zarr'), mode='r')[:]
        assert np.isnan(ras[idx == -1]).all()
        np.testing.assert_array_equal(ras[idx != -1], pc[idx[idx != -1]])


def _block_means(a, f):
    """Expected means of the f x f blocks of `a` (h, w) over its finite values, nan where none."""
    h, w = a.shape
    H, W = -(-h // f), -(-w // f)
    padded = np.full((H * f, W * f), np.nan, np.result_type(a, np.float64))
    padded[:h, :w] = a
    blocks = padded.reshape(H, f, W, f).transpose(0, 2, 1, 3).reshape(H, W, f * f)
    with np.errstate(invalid='ignore'):
        return np.nanmean(blocks, axis=2)


def test_raster_pyramid_mean(tmp_path, rng):
    """method mean: every level is the mean of the blocks of the data over the finite pixels, in bands too."""
    ras = rng.random((50, 37)).astype(np.float32)
    ras[5:20, 3:9] = np.nan; ras[:2] = np.nan
    write_zarr(tmp_path / 'ras.zarr', ras, (20, 20))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'), method='mean', rows=16)
    pyr = tmp_path / 'pyr'
    assert zarr.open(str(pyr / '0.zarr'), mode='r').attrs['moraine_pyramid']['method'] == 'mean'
    np.testing.assert_array_equal(zarr.open(str(pyr / '0.zarr'), mode='r')[:], ras)
    for level in range(1, 6):
        z = zarr.open(str(pyr / f'{level}.zarr'), mode='r')
        assert z.dtype == np.float32
        np.testing.assert_allclose(z[:], _block_means(ras.astype(np.float64), 2 ** level), rtol=1e-6, atol=1e-7)
    # complex data: complex means; integers: float32 levels
    c = np.exp(1j * rng.uniform(-np.pi, np.pi, (20, 24, 2))).astype(np.complex64)
    write_zarr(tmp_path / 'c.zarr', c, (20, 24, 1))
    mc.ras_pyramid(str(tmp_path / 'c.zarr'), str(tmp_path / 'cpyr'), method='mean')
    z = zarr.open(str(tmp_path / 'cpyr' / '2.zarr'), mode='r')
    assert z.dtype == np.complex64
    for k in range(2):
        np.testing.assert_allclose(z[..., k], _block_means(c[..., k].astype(np.complex128), 4), rtol=1e-5, atol=1e-6)
    n = (rng.random((20, 24)) * 10).astype(np.int32)
    write_zarr(tmp_path / 'n.zarr', n)
    mc.ras_pyramid(str(tmp_path / 'n.zarr'), str(tmp_path / 'npyr'), method='mean')
    assert zarr.open(str(tmp_path / 'npyr' / '0.zarr'), mode='r').dtype == np.int32
    z = zarr.open(str(tmp_path / 'npyr' / '1.zarr'), mode='r')
    assert z.dtype == np.float32
    np.testing.assert_allclose(z[:], _block_means(n.astype(np.float64), 2), rtol=1e-6)
    assert summarize(str(tmp_path / 'npyr'))['method'] == 'mean'
    with pytest.raises(ValueError, match='method'):
        mc.ras_pyramid(str(tmp_path / 'n.zarr'), str(tmp_path / 'bad'), method='average')


def test_point_cloud_pyramid_mean(tmp_path, rng):
    """method mean of point clouds: the mean of the cells with points; integer points give float32 cells with nan
    where a cell has no point (first too)."""
    pts = np.array([(y, x) for y in range(12) for x in range(10) if (x + y) % 3 == 0])
    val = rng.random(len(pts)).astype(np.float32)
    write_zarr(tmp_path / 'x.zarr', pts[:, 1].astype(float)); write_zarr(tmp_path / 'y.zarr', pts[:, 0].astype(float))
    write_zarr(tmp_path / 'v.zarr', val)
    mc.pc_pyramid(str(tmp_path / 'v.zarr'), str(tmp_path / 'pyr'), x=str(tmp_path / 'x.zarr'), y=str(tmp_path / 'y.zarr'),
                  ras_resolution=1, method='mean')
    pyr = tmp_path / 'pyr'
    assert zarr.open(str(pyr / '0.zarr'), mode='r').attrs['moraine_pyramid']['method'] == 'mean'
    grid = np.full((12, 10), np.nan, np.float64); grid[pts[:, 0], pts[:, 1]] = val
    np.testing.assert_allclose(zarr.open(str(pyr / '0.zarr'), mode='r')[:], grid.astype(np.float32), equal_nan=True)
    for level in (1, 2, 3):
        np.testing.assert_allclose(zarr.open(str(pyr / f'{level}.zarr'), mode='r')[:], _block_means(grid, 2 ** level),
                                   rtol=1e-6, atol=1e-7)
        idx = zarr.open(str(pyr / f'idx_{level}.zarr'), mode='r')[:]            # still the first point of the block
        assert ((idx >= 0) == np.isfinite(_block_means(grid, 2 ** level))).all()
    counts = (rng.random(len(pts)) * 5 + 1).astype(np.int16)
    write_zarr(tmp_path / 'n.zarr', counts)
    for method in ('first', 'mean'):
        mc.pc_pyramid(str(tmp_path / 'n.zarr'), str(tmp_path / f'n_{method}'), x=str(tmp_path / 'x.zarr'),
                      y=str(tmp_path / 'y.zarr'), ras_resolution=1, method=method)
        z = zarr.open(str(tmp_path / f'n_{method}' / '0.zarr'), mode='r')
        assert z.dtype == np.float32 and np.isnan(z[0, 1]) and z[0, 0] == counts[0]      # (0, 1): no point
        s = summarize(str(tmp_path / f'n_{method}'))
        assert s['dtype'] == 'int16' and s['method'] == method and s['min'] == 1
    with pytest.raises(ValueError, match='method'):
        mc.pc_pyramid(str(tmp_path / 'n.zarr'), str(tmp_path / 'bad'), x=str(tmp_path / 'x.zarr'), y=str(tmp_path / 'y.zarr'),
                      ras_resolution=1, method='decimate')


@pytest.mark.parametrize('res, x0, y0', [(1, 2, 3), (4.777314267823516, -16498435.873784425, 8649592.229351616)])
def test_point_cloud_pyramid_every_point_in_its_cell(tmp_path, res, x0, y0):
    # one point in three cells of a 60 x 40 grid, including the last line and column
    yi, xi = np.nonzero((np.arange(60)[:, None] + np.arange(40)[None, :]) % 3 == 0)
    x, y = x0 + xi * res, y0 + yi * res
    write_zarr(tmp_path / 'x.zarr', x); write_zarr(tmp_path / 'y.zarr', y)
    write_zarr(tmp_path / 'pc.zarr', np.arange(len(x), dtype=np.float32))
    mc.pc_pyramid(str(tmp_path / 'pc.zarr'), str(tmp_path / 'pyr'), x=str(tmp_path / 'x.zarr'),
                  y=str(tmp_path / 'y.zarr'), ras_resolution=res)
    import tomllib
    bx0, by0, bxm, bym = tomllib.loads((tmp_path / 'pyr' / 'bounds.toml').read_text())['bounds']
    idx = zarr.open(str(tmp_path / 'pyr' / 'idx_0.zarr'), mode='r')[:]
    assert idx.shape == (60, 40)
    assert (bx0, by0) == (pytest.approx(x0), pytest.approx(y0))
    assert (bxm, bym) == (pytest.approx(x0 + 39 * res), pytest.approx(y0 + 59 * res))
    # cell (i, j) centred at (x0 + j res, y0 + i res) holds the point there: no two points share a cell
    np.testing.assert_array_equal(idx[yi, xi], np.arange(len(x)))
    assert (idx != -1).sum() == len(x)


def test_pyramid_statistics_keep_their_digits(tmp_path, rng):
    """The standard deviation keeps its digits where the mean is large against it, in the marker and in stats.zarr:
    values of 1e5 +- 0.01 in bands and channels of a raster, the amplitude of complex points of modulus 1."""
    ras = 1e5 + 0.01 * rng.standard_normal((64, 37, 3))
    write_zarr(tmp_path / 'ras.zarr', ras, (16, 37, 1))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'), rows=16)
    pyr = tmp_path / 'pyr'
    assert zarr.open(str(pyr / '0.zarr'), mode='r').attrs['moraine_pyramid']['stats']['std'] == \
        pytest.approx(ras.std(), rel=1e-5)
    table = zarr.open(str(pyr / 'stats.zarr'), mode='r')
    np.testing.assert_allclose(table[:, table.attrs['columns'].index('std')], ras.std(axis=(0, 1)), rtol=1e-9)
    n = 2000
    g = random_gix(rng, (60, 50), n)
    write_zarr(tmp_path / 'x.zarr', g[:, 1].astype(float)); write_zarr(tmp_path / 'y.zarr', g[:, 0].astype(float))
    ph = np.exp(1j * rng.uniform(-np.pi, np.pi, (n, 2))).astype(np.complex64)
    write_zarr(tmp_path / 'ph.zarr', ph, (n, 1))
    mc.pc_pyramid(str(tmp_path / 'ph.zarr'), str(tmp_path / 'pc_pyr'), x=str(tmp_path / 'x.zarr'),
                  y=str(tmp_path / 'y.zarr'), ras_resolution=1)
    stats = zarr.open(str(tmp_path / 'pc_pyr' / '0.zarr'), mode='r').attrs['moraine_pyramid']['stats']
    assert stats['amplitude_std'] == pytest.approx(np.abs(ph).astype(np.float64).std(), rel=1e-5)


def test_newer_pyramid_is_rejected(tmp_path, rng):
    write_zarr(tmp_path / 'ras.zarr', rng.random((16, 16)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    zarr.open(str(tmp_path / 'pyr' / '0.zarr'), mode='r+').attrs['moraine_pyramid'] = {
        'version': PYRAMID_VERSION + 1, 'kind': 'raster'}
    with pytest.raises(ValueError):
        pyramid_levels(tmp_path / 'pyr')


def test_marker_is_written_last(tmp_path, rng, monkeypatch):
    """A pyramid is complete when it has the marker: one whose statistics table could not be written has none."""
    import moraine.cli.plot as plot

    def full_disk(part):
        raise OSError('no space left on device')
    write_zarr(tmp_path / 'ras.zarr', rng.random((16, 16)).astype(np.float32))
    monkeypatch.setattr(plot, '_channel_row', full_disk)
    with pytest.raises(OSError, match='no space left'):
        mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    assert 'moraine_pyramid' not in zarr.open(str(tmp_path / 'pyr' / '0.zarr'), mode='r').attrs


@pytest.mark.parametrize('case', ['unfinished', 'no method', 'no stats', 'no rtree'])
def test_pyramid_to_rebuild_is_rejected(tmp_path, rng, monkeypatch, capsys, case):
    """Every reader rejects a pyramid of moraine before 0.10.0 (without the marker, a marker without `method` or
    `stats`, a point cloud pyramid without rtree.zarr) and one whose command did not finish, and asks to rebuild it;
    for `moraine info`, `quicklook` and `view` it is a usage error (exit code 2) with the error of the JSON output."""
    import shutil
    import moraine.cli.plot as plot
    from moraine.command.summary import quicklook, view as notebook
    pyr = tmp_path / 'pyr'
    if case == 'no rtree':
        g = random_gix(rng, (20, 10), 50)
        write_zarr(tmp_path / 'x.zarr', g[:, 1].astype(float)); write_zarr(tmp_path / 'y.zarr', g[:, 0].astype(float))
        write_zarr(tmp_path / 'pc.zarr', rng.random(50).astype(np.float32))
        mc.pc_pyramid(str(tmp_path / 'pc.zarr'), str(pyr), x=str(tmp_path / 'x.zarr'), y=str(tmp_path / 'y.zarr'),
                      ras_resolution=1)
        shutil.rmtree(pyr / 'rtree.zarr')
    else:
        write_zarr(tmp_path / 'ras.zarr', rng.random((30, 20)).astype(np.float32))
        if case == 'unfinished':     # the levels are there, the marker is not
            monkeypatch.setattr(plot, '_channel_row', lambda part: 1 / 0)
            with pytest.raises(ZeroDivisionError):
                mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(pyr))
        else:
            mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(pyr))
            z0 = zarr.open(str(pyr / '0.zarr'), mode='r+')
            z0.attrs['moraine_pyramid'] = {k: v for k, v in z0.attrs['moraine_pyramid'].items() if k != case[3:]}
    command = 'pc-pyramid' if case == 'no rtree' else 'ras-pyramid'
    for read in (pyramid_levels, summarize, mc.view):
        with pytest.raises(ValueError, match=f'before 0.10.0.*rebuild it with `moraine {command}`'):
            read(str(pyr))
    for read in (lambda p: quicklook(p, str(tmp_path / 'q.png')), lambda p: notebook([p], str(tmp_path / 'v.ipynb'))):
        with pytest.raises(UsageError, match=f'before 0.10.0.*rebuild it with `moraine {command}`'):  # a wrong input
            read(str(pyr))
    capsys.readouterr()
    for args in (['info'], ['quicklook', '-o', str(tmp_path / 'q.png')], ['view', '-o', str(tmp_path / 'v.ipynb')]):
        assert main([*args, str(pyr), '--json']) == 2, args
        out = json_out(capsys)
        _check(out, 'Every output', 'error')
        assert out['ok'] is False and re.search(f'before 0.10.0.*rebuild it with `moraine {command}`', out['error'])


# ---------------------------------------------------------------- data conventions

@pytest.mark.parametrize('pairs, match', [
    ([[0, 2], [0, 1], [1, 2]], 'sorted'),
    ([[0, 1], [0, 1], [1, 2]], 'without duplicates'),
    ([[0, 1], [2, 1]], 'smaller'),
    ([[0, 1], [1, 1]], 'smaller'),
])
def test_commands_take_image_pairs_of_the_data_conventions(tmp_path, pairs, match):
    """The image pairs of the commands and pipeline files (text files, .npy files, lists) are sorted without
    duplicates, every reference index smaller than its secondary index (data.md)."""
    from moraine.command import image_pairs as read_image_pairs
    np.savetxt(tmp_path / 'pairs.txt', pairs, fmt='%d')
    np.save(tmp_path / 'pairs.npy', np.array(pairs))
    for value in (str(tmp_path / 'pairs.txt'), str(tmp_path / 'pairs.npy'), str(pairs), pairs):
        with pytest.raises(UsageError, match=match):
            read_image_pairs(value)
    with pytest.raises(UsageError, match='pairs.txt'):              # the error names the file
        read_image_pairs(str(tmp_path / 'pairs.txt'))
    mc.image_pairs(str(tmp_path / 'made.txt'), nimages=6, bandwidth=2)
    np.testing.assert_array_equal(read_image_pairs(str(tmp_path / 'made.txt'))[:3], [[0, 1], [0, 2], [1, 2]])
