"""The `moraine` command line and pipeline runner, on small synthetic data (no GPU / sample data needed)."""
import json

import numpy as np
import pytest
import zarr

from moraine.command import (main, commands, get_command, bind_args, parse_docstring, literal, image_pairs,
                             UsageError)
from moraine.command.pipeline import run_pipeline, pipeline_status
from moraine.command.summary import summarize, quicklook


def _zarr(path, data, chunks=None):
    z = zarr.open(str(path), mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks or data.shape)
    z[:] = data


def _json_out(capsys):
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------- building blocks

def test_parse_docstring():
    summary, params = parse_docstring('''Do things.

    Parameters
    ----------
    a : int, default: 1
        first
        second line
    **kw
        extra

    Returns
    -------
    int
        result
    ''')
    assert summary == 'Do things.'
    assert params == {'a': ('int, default: 1', 'first second line'), 'kw': ('', 'extra')}


def test_parse_docstring_badly_indented_summary():
    doc = 'Summary.\nsecond summary line at column 0\n\n    Parameters\n    ----------\n    a : str\n        first\n    b : int\n        second\n'
    summary, params = parse_docstring(doc)
    assert summary == 'Summary. second summary line at column 0'
    assert params == {'a': ('str', 'first'), 'b': ('int', 'second')}


def test_every_command_documents_its_parameters():
    missing = [f'{c.name}.{p.name}' for c in commands().values() for p in c.params if not p.help]
    assert not missing, missing


def test_literal_and_image_pairs(tmp_path):
    assert literal('1000,1000') == (1000, 1000)
    assert literal('None') is None
    assert literal('raw/rslc.zarr') == 'raw/rslc.zarr'
    np.testing.assert_array_equal(image_pairs('[[0,1],[1,2]]'), [[0, 1], [1, 2]])
    np.savetxt(tmp_path / 'p.txt', [[0, 1], [0, 2]], fmt='%d')
    np.testing.assert_array_equal(image_pairs(str(tmp_path / 'p.txt')), [[0, 1], [0, 2]])
    with pytest.raises(UsageError):
        image_pairs('not/a/file.txt')


def test_commands_registry():
    cmds = commands()
    assert {'amp-disp', 'shp-test', 'emi', 'n2f', 'mcf-pc', 'pc-concat', 'ras-pyramid'} <= set(cmds)
    assert 'data-reduce' not in cmds          # needs python callables
    shp = get_command('shp_test')
    by_name = {p.name: p for p in shp.params}
    assert by_name['rslc'].required and by_name['rslc'].help
    assert by_name['cuda'].kind == 'bool' and by_name['cuda'].default is False
    with pytest.raises(UsageError, match='Did you mean: amp-disp'):
        get_command('amp-dsp')


def test_bind_args():
    cmd = get_command('amp-disp')
    kwargs = bind_args(cmd, {'rslc': 'r.zarr', 'adi': 'a.zarr', 'chunks': [100, 100], 'cuda': True})
    assert kwargs == {'rslc': 'r.zarr', 'adi': 'a.zarr', 'chunks': (100, 100), 'cuda': True}
    with pytest.raises(UsageError, match='adi_out -> adi'):     # typos are not swallowed by **kwargs
        bind_args(cmd, {'rslc': 'r.zarr', 'adi_out': 'a.zarr'})
    with pytest.raises(UsageError, match='missing required argument'):
        bind_args(cmd, {'rslc': 'r.zarr'})
    with pytest.raises(UsageError, match='takes no extra keyword arguments'):   # no **kwargs since decision 0034
        bind_args(cmd, {'rslc': 'r', 'adi': 'a'}, kw={'memory_limit': '2GB'})
    math = get_command('math')          # **data: the inputs of the expression are extra keyword arguments
    assert bind_args(math, {'output': 'o.zarr', 'operation': 'a*2'}, kw={'a': 'a.zarr'})['a'] == 'a.zarr'


# ---------------------------------------------------------------- summary / quicklook

def test_summarize_is_metadata_only(tmp_path, rng):
    _zarr(tmp_path / 'f.zarr', rng.random((300, 200)).astype(np.float32), (100, 100))
    assert summarize(str(tmp_path / 'f.zarr')) == {'path': str(tmp_path / 'f.zarr'), 'kind': 'array',
                                                   'shape': [300, 200], 'dtype': 'float32', 'chunks': [100, 100]}
    _zarr(tmp_path / 'd' / '0.zarr', np.ones((6, 8), np.float32)); _zarr(tmp_path / 'd' / '1.zarr', np.ones((3, 4), np.float32))
    assert summarize(str(tmp_path / 'd'))['kind'] == 'raster pyramid'   # levels halving in size
    with pytest.raises(FileNotFoundError):
        summarize(str(tmp_path / 'missing.zarr'))


def test_quicklook_needs_a_pyramid(tmp_path, rng):
    _zarr(tmp_path / 'f.zarr', rng.random((30, 20)).astype(np.float32))
    with pytest.raises(ValueError, match='moraine ras-pyramid'):
        quicklook(str(tmp_path / 'f.zarr'), str(tmp_path / 'q.png'))


# ---------------------------------------------------------------- command line

def test_main_list_info(tmp_path, capsys, rng):
    assert main(['list', '--json']) == 0
    assert any(c['name'] == 'amp-disp' for c in _json_out(capsys)['commands'])
    _zarr(tmp_path / 'a.zarr', rng.random((20, 20)).astype(np.float32))
    assert main(['info', str(tmp_path / 'a.zarr'), '--json']) == 0
    assert _json_out(capsys)['summaries'][0]['shape'] == [20, 20]


def test_main_runs_a_command(tmp_path, capsys, rng):
    ras = rng.random((30, 40)).astype(np.float32)
    _zarr(tmp_path / 'ras.zarr', ras, (10, 40))
    code = main(['pc-logic-ras', '--ras', str(tmp_path / 'ras.zarr'), '--gix', str(tmp_path / 'gix.zarr'),
                 '--operation', 'ras>0.5', '--json', '-q'])
    assert code == 0
    out = _json_out(capsys)
    assert out['ok'] and out['outputs'] == [str(tmp_path / 'gix.zarr')]
    assert out['summaries'][0]['shape'] == [int((ras > 0.5).sum()), 2]


def test_main_error(tmp_path, capsys):
    assert main(['info', str(tmp_path / 'missing.zarr'), '--json']) == 1
    assert _json_out(capsys)['ok'] is False


# ---------------------------------------------------------------- pipeline

PIPELINE = '''
[pipeline]
quicklook = true

[[step]]
name = "scale"
run = "math"
output = "b.zarr"
operation = "a*{factor}"
[step.kw]
a = "a.zarr"

[[step]]
name = "select"
run = "pc-logic-ras"
ras = "b.zarr"
gix = "gix.zarr"
operation = "ras>1"

[[step]]
name = "values"
run = "ras2pc"
idx = "gix.zarr"
ras = "b.zarr"
pc = "pc.zarr"
'''


@pytest.fixture
def pipe(tmp_path, rng):
    _zarr(tmp_path / 'a.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
    def write(factor):
        (tmp_path / 'pipeline.toml').write_text(PIPELINE.format(factor=factor))
        return str(tmp_path / 'pipeline.toml')
    return tmp_path, write


def _actions(result):
    return [p['action'].split(' (')[0] for p in result['plan']]


def test_pipeline_run_resume_and_invalidate(pipe):
    d, write = pipe
    path = write(2)
    res = run_pipeline(path, echo=lambda *a: None)
    assert res['ok'] and _actions(res) == ['run'] * 3
    a = zarr.open(str(d / 'a.zarr'), mode='r')[:]
    np.testing.assert_array_equal(zarr.open(str(d / 'pc.zarr'), mode='r')[:], (2 * a)[2 * a > 1])
    assert not (d / '.moraine' / 'pipeline' / 'quicklook').exists()   # quicklooks are only made of pyramids
    # nothing changed: everything is skipped
    assert _actions(run_pipeline(path, echo=lambda *a: None)) == ['skip'] * 3
    # a changed argument reruns that step and every step reading its outputs
    path = write(3)
    plan = run_pipeline(path, dry_run=True, echo=lambda *a: None)['plan']
    assert plan[0]['action'] == 'run (arguments changed)'
    assert all(p['action'].startswith('run (input will change: ') for p in plan[1:])
    run_pipeline(path, echo=lambda *a: None)
    np.testing.assert_array_equal(zarr.open(str(d / 'pc.zarr'), mode='r')[:], (3 * a)[3 * a > 1])
    assert [s['status'] for s in pipeline_status(path)['steps']] == ['done'] * 3
    # a deleted output reruns its step
    import shutil; shutil.rmtree(d / 'pc.zarr')
    assert _actions(run_pipeline(path, dry_run=True, echo=lambda *a: None)) == ['skip', 'skip', 'run']


def test_pipeline_only_from_force(pipe):
    d, write = pipe
    path = write(2)
    run_pipeline(path, echo=lambda *a: None)
    assert _actions(run_pipeline(path, dry_run=True, from_step='select', echo=lambda *a: None)) == ['skip', 'run', 'run']
    assert _actions(run_pipeline(path, dry_run=True, only=['values'], force=True, echo=lambda *a: None)) == ['skip', 'skip', 'run']
    with pytest.raises(UsageError, match='unknown step'):
        run_pipeline(path, only=['nope'], echo=lambda *a: None)


def test_pipeline_failure_and_resume(pipe):
    d, write = pipe
    path = write(2)
    (d / 'a.zarr').rename(d / 'a_hidden.zarr')
    res = run_pipeline(path, echo=lambda *a: None)
    assert not res['ok'] and res['steps'][0]['status'] == 'failed' and 'a.zarr' in res['steps'][0]['error']
    assert (d / '.moraine' / 'pipeline' / 'logs' / 'scale.log').exists()
    assert pipeline_status(path)['steps'][0]['status'] == 'failed'
    (d / 'a_hidden.zarr').rename(d / 'a.zarr')
    assert run_pipeline(path, echo=lambda *a: None)['ok']


@pytest.mark.parametrize('text,match', [
    ('[[step]]\nname="x"\nrun="amp-dsp"\n', 'Did you mean: amp-disp'),
    ('[[step]]\nname="x"\nrun="amp-disp"\nrslc="r"\nadi_out="a"\n', 'adi_out -> adi'),
    ('[[step]]\nrun="amp-disp"\n', 'has no name'),
    ('[steps]\n', 'unknown table'),
])
def test_pipeline_bad_files(tmp_path, text, match):
    (tmp_path / 'p.toml').write_text(text)
    with pytest.raises(UsageError, match=match):
        run_pipeline(str(tmp_path / 'p.toml'), dry_run=True, echo=lambda *a: None)


def test_dashed_option_spelling():
    from moraine.command import _normalize_options
    assert _normalize_options(['shp-test', '--az-half-win', '5', '--no-cuda', '--r-half-win=5']) == \
        ['shp-test', '--az_half_win', '5', '--no-cuda', '--r_half_win=5']
    assert _normalize_options(['run', 'p.toml', '--dry-run']) == ['run', 'p.toml', '--dry-run']


@pytest.mark.parametrize('tokens', [['40', '30'], ['40,30'], ['(40,30)'], ['40,', '30']])
def test_tuple_option_spellings(tmp_path, capsys, rng, tokens):
    gix = np.stack(np.unravel_index(np.sort(rng.choice(1200, 50, replace=False)), (40, 30)), -1).astype(np.int32)
    _zarr(tmp_path / 'gix.zarr', gix)
    _zarr(tmp_path / 'pc.zarr', rng.random(50).astype(np.float32))
    assert main(['pc2ras', '--idx', str(tmp_path / 'gix.zarr'), '--pc', str(tmp_path / 'pc.zarr'),
                 '--ras', str(tmp_path / 'ras.zarr'), '--shape', *tokens, '--chunks', '20', '30', '--json', '-q']) == 0
    assert _json_out(capsys)['summaries'][0]['shape'] == [40, 30]


@pytest.mark.parametrize('tokens,msg', [(['2500'], 'needs 2 integers'), (['1', '2', '3'], 'needs 2 integers'),
                                        (['abc'], 'needs 2 integers'), (['2500.5', '1834'], 'needs 2 integers')])
def test_tuple_option_errors(tmp_path, capsys, tokens, msg):
    code = main(['pc2ras', '--idx', 'g.zarr', '--pc', 'p.zarr', '--ras', 'r.zarr', '--shape', *tokens, '--json'])
    assert code == 2
    out = _json_out(capsys)
    assert not out['ok'] and msg in out['error'] and '--shape 1000 1000' in out['error']


def test_int_or_tuple_option():
    cmd = get_command('temp-coh')
    assert cmd.convert('chunks', ['200']) == 200
    assert cmd.convert('chunks', ['200', '100']) == (200, 100)
    with pytest.raises(UsageError, match='an integer or 2 integers'):
        cmd.convert('chunks', ['1', '2', '3'])


def test_tuple_in_pipeline_file(tmp_path):
    (tmp_path / 'p.toml').write_text('[[step]]\nname="a"\nrun="amp-disp"\nrslc="r"\nadi="a"\nchunks=[100, 100, 1]\n')
    with pytest.raises(UsageError, match="step 'a'.*--chunks needs 2 integers"):
        run_pipeline(str(tmp_path / 'p.toml'), dry_run=True, echo=lambda *a: None)


def test_pyramids(tmp_path, rng):
    import moraine.cli as mc
    from moraine.command.summary import pyramid_levels
    ras = (rng.random((300, 200, 3)) * np.exp(1j * rng.random((300, 200, 3)))).astype(np.complex64)
    _zarr(tmp_path / 'ras.zarr', ras, (100, 100, 1))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'ras_pyramid'))
    assert pyramid_levels(tmp_path / 'ras_pyramid')[:2] == [0, 1] and pyramid_levels(tmp_path / 'ras.zarr') == []
    s = summarize(str(tmp_path / 'ras_pyramid'))
    assert s['kind'] == 'raster pyramid' and s['shape'] == [300, 200, 3] and s['levels'] > 1
    assert s['stats_level'] == 0 and abs(s['amplitude_mean'] - np.abs(ras).mean()) < 1e-3 and 'warnings' not in s
    assert summarize(str(tmp_path / 'ras_pyramid'), max_bytes=100_000) == s         # the statistics of the pyramid
    z0 = zarr.open(str(tmp_path / 'ras_pyramid' / '0.zarr'), mode='r+')               # a pyramid made without them
    z0.attrs['moraine_pyramid'] = {k: v for k, v in z0.attrs['moraine_pyramid'].items() if k != 'stats'}
    assert summarize(str(tmp_path / 'ras_pyramid'), max_bytes=100_000)['stats_level'] > 0     # coarser level
    # the images of a stack that are all nan or constant are named
    bad = ras.copy(); bad[..., 1] = np.nan; bad[..., 2] = 1 + 0j
    _zarr(tmp_path / 'bad.zarr', bad, (100, 100, 1))
    mc.ras_pyramid(str(tmp_path / 'bad.zarr'), str(tmp_path / 'bad_pyramid'))
    assert summarize(str(tmp_path / 'bad_pyramid'))['warnings'] == ['all values nan: image 1', 'constant values: image 2']
    for kw in [{}, {'index': (2,)}, {'show': 'intf_seq', 'index': (1,)}, {'show': 'intf_all', 'index': (0, 2)}]:
        out = tmp_path / 'r.png'
        quicklook(str(tmp_path / 'ras_pyramid'), str(out), width=200, **kw)
        assert out.stat().st_size > 1000
        out.unlink()

    n = 500
    g = np.stack(np.unravel_index(np.sort(rng.choice(300 * 200, n, replace=False)), (300, 200)), -1)
    for name, a in [('x.zarr', g[:, 1].astype(float)), ('y.zarr', g[:, 0].astype(float)),
                    ('pc.zarr', rng.random(n).astype(np.float32))]:
        _zarr(tmp_path / name, a)
    mc.pc_pyramid(str(tmp_path / 'pc.zarr'), str(tmp_path / 'pc_pyramid'), x=str(tmp_path / 'x.zarr'),
                  y=str(tmp_path / 'y.zarr'), ras_resolution=1)
    s = summarize(str(tmp_path / 'pc_pyramid'))
    assert s['kind'] == 'point cloud pyramid'
    assert s['nan_fraction'] == 0.0            # cells without points are not counted as nan
    for width in (1000, 50):       # points drawn one by one (fine) and rasterized (coarse)
        quicklook(str(tmp_path / 'pc_pyramid'), str(tmp_path / f'p{width}.png'), width=width)
        assert (tmp_path / f'p{width}.png').stat().st_size > 1000


def test_quicklook_command(tmp_path, capsys, rng):
    import moraine.cli as mc
    _zarr(tmp_path / 'ras.zarr', rng.random((64, 48)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    capsys.readouterr()                  # drop what ras_pyramid printed
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'q.png'), '--json']) == 0
    assert _json_out(capsys)['png'] == str(tmp_path / 'q.png')
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'part.png'), '--extent', '10,5,30,20',
                 '--json']) == 0
    assert _json_out(capsys)['png'] == str(tmp_path / 'part.png') and (tmp_path / 'part.png').stat().st_size > 1000
    assert main(['quicklook', str(tmp_path / 'ras.zarr'), '--json']) == 1
    assert 'not a pyramid' in _json_out(capsys)['error']


@pytest.mark.parametrize('data,warning', [
    (np.full((60, 40), np.nan, np.float32), 'all values are nan'),
    (np.where(np.arange(2400).reshape(60, 40) % 97 == 0, np.inf, 1.5).astype(np.float32), 'infinite values'),
    (np.zeros((60, 40), np.float32), 'all values are 0.0'),
])
def test_pyramid_warnings(tmp_path, data, warning):
    import moraine.cli as mc
    _zarr(tmp_path / 'a.zarr', data, (20, 20))
    mc.ras_pyramid(str(tmp_path / 'a.zarr'), str(tmp_path / 'pyr'))
    assert any(warning in w for w in summarize(str(tmp_path / 'pyr'))['warnings'])


def test_pipeline_vars_and_workdir(tmp_path, rng, capsys):
    data = tmp_path / 'data'; work = tmp_path / 'work'
    data.mkdir()
    _zarr(data / 'a.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
    (tmp_path / 'p.toml').write_text('''
[vars]
data = "/nowhere"
thr = "0.5"

[[step]]
name = "select"
run = "pc-logic-ras"
ras = "${data}/a.zarr"
gix = "out/gix.zarr"
operation = "ras>${thr}"
''')
    res = run_pipeline(str(tmp_path / 'p.toml'), workdir=str(work), variables={'data': str(data)}, echo=lambda *a: None)
    assert res['ok'] and (work / 'out' / 'gix.zarr').exists()
    assert (work / '.moraine' / 'p' / 'state.json').exists()
    # a changed variable changes the arguments
    plan = run_pipeline(str(tmp_path / 'p.toml'), workdir=str(work), dry_run=True, echo=lambda *a: None,
                        variables={'data': str(data), 'thr': '0.6'})['plan']
    assert plan[0]['action'] == 'run (arguments changed)'
    # command line: --workdir / --var, and a clear error for an unknown variable
    assert main(['status', str(tmp_path / 'p.toml'), '--workdir', str(work), '--var', f'data={data}', '--json']) == 0
    assert _json_out(capsys)['steps'][0]['status'] == 'done'
    (tmp_path / 'q.toml').write_text('[[step]]\nname="a"\nrun="amp-disp"\nrslc="${gamma}/r"\nadi="a"\n')
    with pytest.raises(UsageError, match=r'unknown variable \$\{gamma\}'):
        run_pipeline(str(tmp_path / 'q.toml'), dry_run=True, echo=lambda *a: None)


def test_image_pairs_command(tmp_path, capsys):
    _zarr(tmp_path / 'rslc.zarr', np.zeros((4, 5, 6), np.complex64))
    assert main(['image-pairs', '--rslc', str(tmp_path / 'rslc.zarr'), '--bandwidth', '2', '--out',
                 str(tmp_path / 'p.txt'), '-q', '--json']) == 0
    capsys.readouterr()
    np.testing.assert_array_equal(image_pairs(str(tmp_path / 'p.txt')), [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5]])


def test_chunk_directories_are_not_pyramids(tmp_path, rng):
    from moraine.command.summary import pyramid_levels
    for i, n in enumerate([500, 320, 410]):      # per-chunk arrays of ras2pc_ras_chunk: 0.zarr, 1.zarr, ...
        _zarr(tmp_path / 'chunks' / f'{i}.zarr', rng.random((n, 11, 11)).astype(np.float32))
    assert pyramid_levels(tmp_path / 'chunks') == []
    assert summarize(str(tmp_path / 'chunks'))['kind'] == 'directory'


def test_pipeline_makes_missing_parent_directories(tmp_path, capsys):
    # the image pair file of a later step is made by an earlier step, in a directory that does not exist yet
    _zarr(tmp_path / 'rslc.zarr', np.ones((20, 30, 4), np.complex64), (10, 30, 1))
    (tmp_path / 'p.toml').write_text('''
[[step]]
name = "pairs"
run = "image-pairs"
rslc = "rslc.zarr"
bandwidth = 1
out = "new/dir/pairs.txt"

[[step]]
name = "t_coh"
run = "temp-coh"
intf = "intf.zarr"
rslc = "rslc.zarr"
t_coh = "t_coh.zarr"
image_pairs = "new/dir/pairs.txt"
''')
    _zarr(tmp_path / 'intf.zarr', np.ones((20, 30, 3), np.complex64), (10, 30, 1))
    assert run_pipeline(str(tmp_path / 'p.toml'), echo=lambda *a: None)['ok']
    np.testing.assert_allclose(zarr.open(str(tmp_path / 't_coh.zarr'), mode='r')[:], 1, rtol=1e-5)


def test_outputs_without_slash_or_dot(tmp_path, capsys, rng, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _zarr(tmp_path / 'ras.zarr', rng.random((16, 16)).astype(np.float32))
    assert main(['ras-pyramid', '--ras', 'ras.zarr', '--out_dir', 'pyr', '--json', '-q']) == 0
    out = _json_out(capsys)
    assert out['outputs'] == ['pyr'] and out['inputs'].keys() == {'ras.zarr'}


# ---------------------------------------------------------------- interactive views

@pytest.fixture
def pyramids(tmp_path, rng):
    import moraine.cli as mc
    d = tmp_path
    stack = np.exp(1j * rng.uniform(-np.pi, np.pi, (60, 40, 4))).astype(np.complex64)
    _zarr(d / 'stack.zarr', stack, (30, 40, 1))
    mc.ras_pyramid(str(d / 'stack.zarr'), str(d / 'stack_pyr'))
    _zarr(d / 'long.zarr', rng.random((400, 20)).astype(np.float32), (100, 20))     # elongated scene
    mc.ras_pyramid(str(d / 'long.zarr'), str(d / 'long_pyr'))
    n = 300
    g = np.stack(np.unravel_index(np.sort(rng.choice(60 * 40, n, replace=False)), (60, 40)), -1)
    for name, a in [('gy.zarr', g[:, 0].astype(float)), ('gx.zarr', g[:, 1].astype(float)),
                    ('my.zarr', 6e6 + 7.5 * g[:, 0]), ('mx.zarr', -1.6e7 + 7.5 * g[:, 1]),     # map coordinates
                    ('val.zarr', np.linspace(0, 10, n * 3).reshape(n, 3).astype(np.float32))]:
        _zarr(d / name, a)
    mc.pc_pyramid(str(d / 'val.zarr'), str(d / 'grid_pyr'), x=str(d / 'gx.zarr'), y=str(d / 'gy.zarr'), ras_resolution=1)
    mc.pc_pyramid(str(d / 'val.zarr'), str(d / 'map_pyr'), x=str(d / 'mx.zarr'), y=str(d / 'my.zarr'), ras_resolution=7.5)
    return d


def test_view_colours_axes_sliders(pyramids):
    import colorcet
    import moraine.cli as mc
    d = pyramids
    # complex stack: phase with the cyclic colorwheel over (-pi, pi], radar axes with azimuth down
    w = mc.view(str(d / 'stack_pyr')).widget
    bar = w.panels[0]['layers'][0]
    assert bar['colors'][0] == colorcet.colorwheel[0].lower() and bar['clim'] == pytest.approx([-np.pi, np.pi])
    assert w.crs == 'grid' and w.axis_labels == ['range', 'azimuth']
    assert w.kdims == [{'name': 'image', 'max': 3}]
    assert w.frame == [round(700 * 40 / 60), 700]
    assert mc.view(str(d / 'stack_pyr'), show='intf_seq').widget.kdims == [{'name': 'image', 'max': 2}]
    assert mc.view(str(d / 'stack_pyr'), show='intf_all').widget.kdims == [{'name': 'ref', 'max': 3},
                                                                          {'name': 'sec', 'max': 3}]
    # real point cloud stack on the radar grid: viridis over the 1 - 99 % range, sliders over the stack
    w = mc.view(str(d / 'grid_pyr')).widget
    lo, hi = w.panels[0]['layers'][0]['clim']
    assert w.panels[0]['layers'][0]['colors'][0] == '#440154' and 0 < lo < hi < 10
    assert w.kdims == [{'name': 'i', 'max': 2}]
    # map coordinates: north up, longitude / latitude axes
    assert mc.view(str(d / 'map_pyr')).widget.axis_labels == ['longitude', 'latitude']
    # a 1:20 scene is drawn at most 1:4
    assert mc.view(str(d / 'long_pyr')).widget.frame == [175, 700]


def test_view_command_writes_a_runnable_notebook(pyramids, capsys, tmp_path):
    from moraine.cli.tiles import _Shown
    d = pyramids
    (d / 'meta.toml').write_text('dates = ["20210101", "20210113", "20210125", "20210206"]\n')
    nb_path = d / 'v.ipynb'
    assert main(['view', str(d / 'stack_pyr'), str(d / 'map_pyr'), '-o', str(nb_path), '--show', 'intf_seq',
                 '--dates', str(d / 'meta.toml'), '--json']) == 0
    out = _json_out(capsys)
    assert out['notebook'] == str(nb_path) and len(out['pyramids']) == 2
    nb = json.loads(nb_path.read_text())
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert all(not c['outputs'] for c in nb['cells'] if c['cell_type'] == 'code')
    assert code[0] == 'import moraine.cli as mc'
    assert str((d / 'map_pyr').resolve()) in code[-1] and "show='intf_seq'" in code[-1]   # absolute paths
    ns = {}
    for src in code:                                                   # the cells run and give views
        result = eval(compile(src, 'cell', 'exec' if src is code[0] else 'eval'), ns)
    assert isinstance(result, _Shown) and result.widget.dates[1] == '20210113'
    assert 'moraine view' in repr(result)                              # a text description for readers
    assert main(['view', str(d / 'stack_pyr'), '-o', str(nb_path), '--json']) == 1   # no silent overwrite
    assert 'exists' in _json_out(capsys)['error']
    assert main(['view', str(d / 'stack_pyr'), '-o', str(nb_path), '--overwrite', '--post_proc', 'phase',
                 '--json']) == 0                                       # the old option name still works
