"""The `moraine` command line and pipeline runner, on small synthetic data (no GPU / sample data needed)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from moraine.command import (main, commands, get_command, bind_args, parse_docstring, literal, image_pairs,
                             UsageError, Command, Param)
from moraine.command.pipeline import load_pipeline, run_pipeline, pipeline_status
from moraine.command.summary import summarize, quicklook
from conftest import write_zarr, json_out, random_gix


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


def test_bool_arguments_from_text(tmp_path):
    # text from `--var` or a pipeline variable: 'false' gave True (bool of a non-empty string)
    cmd = get_command('amp-disp')
    assert [cmd.convert('cuda', v) for v in ('false', 'False', ' TRUE ', 'true', False, 1)] == \
        [False, False, True, True, False, True]
    with pytest.raises(UsageError, match="cuda = 'no' is not a boolean"):
        cmd.convert('cuda', 'no')
    (tmp_path / 'p.toml').write_text('[vars]\ngpu = false\n\n[[step]]\nname = "adi"\nrun = "amp-disp"\n'
                                     'rslc = "r.zarr"\nadi = "a.zarr"\ncuda = "${gpu}"\n')

    def cuda(**variables):
        return load_pipeline(str(tmp_path / 'p.toml'), variables=variables)['steps'][0]['kwargs']['cuda']
    assert cuda() is False and cuda(gpu='false') is False and cuda(gpu='true') is True


# ---------------------------------------------------------------- summary / quicklook

def test_summarize_is_metadata_only(tmp_path, rng):
    write_zarr(tmp_path / 'f.zarr', rng.random((300, 200)).astype(np.float32), (100, 100))
    assert summarize(str(tmp_path / 'f.zarr')) == {'path': str(tmp_path / 'f.zarr'), 'kind': 'array',
                                                   'shape': [300, 200], 'dtype': 'float32', 'chunks': [100, 100]}
    # levels halving in size without the marker: a pyramid of moraine before 0.10.0 (or one that did not finish)
    write_zarr(tmp_path / 'd' / '0.zarr', np.ones((6, 8), np.float32)); write_zarr(tmp_path / 'd' / '1.zarr', np.ones((3, 4), np.float32))
    with pytest.raises(ValueError, match='before 0.10.0.*rebuild it with `moraine ras-pyramid`'):
        summarize(str(tmp_path / 'd'))
    with pytest.raises(FileNotFoundError):
        summarize(str(tmp_path / 'missing.zarr'))


def test_quicklook_needs_a_pyramid(tmp_path, rng):
    write_zarr(tmp_path / 'f.zarr', rng.random((30, 20)).astype(np.float32))
    with pytest.raises(UsageError, match='moraine ras-pyramid'):
        quicklook(str(tmp_path / 'f.zarr'), str(tmp_path / 'q.png'))


# ---------------------------------------------------------------- command line

def test_main_list_info(tmp_path, capsys, rng):
    assert main(['list', '--json']) == 0
    assert any(c['name'] == 'amp-disp' for c in json_out(capsys)['commands'])
    write_zarr(tmp_path / 'a.zarr', rng.random((20, 20)).astype(np.float32))
    assert main(['info', str(tmp_path / 'a.zarr'), '--json']) == 0
    assert json_out(capsys)['summaries'][0]['shape'] == [20, 20]


def test_main_runs_a_command(tmp_path, capsys, rng):
    ras = rng.random((30, 40)).astype(np.float32)
    write_zarr(tmp_path / 'ras.zarr', ras, (10, 40))
    code = main(['pc-logic-ras', '--ras', str(tmp_path / 'ras.zarr'), '--gix', str(tmp_path / 'gix.zarr'),
                 '--operation', 'ras>0.5', '--json', '-q'])
    assert code == 0
    out = json_out(capsys)
    assert out['ok'] and out['outputs'] == [str(tmp_path / 'gix.zarr')]
    assert out['summaries'][0]['shape'] == [int((ras > 0.5).sum()), 2]


def test_main_error(tmp_path, capsys):
    assert main(['info', str(tmp_path / 'missing.zarr'), '--json']) == 1
    assert json_out(capsys)['ok'] is False


def test_arguments_a_command_rejects_are_usage_errors(tmp_path, capsys, rng):
    """Arguments that a command rejects as not fitting together (an ArgumentError) are a usage error, exit code 2,
    with the message of the command: pc-select-data of a grid index by a hilbert index without --shape (a processing
    error before, exit code 1: "TypeError: cannot unpack non-iterable NoneType object"). With --shape it runs."""
    import moraine as mr
    gix = random_gix(rng, (40, 30), 50)
    write_zarr(tmp_path / 'gix.zarr', gix); write_zarr(tmp_path / 'hix.zarr', np.sort(mr.pc_hix(gix, (40, 30))))
    write_zarr(tmp_path / 'pc.zarr', rng.random(50).astype(np.float32))
    args = ['pc-select-data', '--idx_in', str(tmp_path / 'gix.zarr'), '--idx', str(tmp_path / 'hix.zarr'),
            '--pc_in', str(tmp_path / 'pc.zarr'), '--pc', str(tmp_path / 'out.zarr'), '--json', '-q']
    capsys.readouterr()
    assert main(args) == 2
    captured = capsys.readouterr()
    error = ('shape (nlines, width) is needed when a grid index and a hilbert index are mixed: idx_in is a grid index, '
             'idx a hilbert index')
    assert json.loads(captured.out) == {'version': 1, 'ok': False, 'error': error}
    assert f'moraine: error: {error}' in captured.err and '--traceback' not in captured.err
    assert not (tmp_path / 'out.zarr').exists()
    assert main(args + ['--shape', '40', '30']) == 0 and json_out(capsys)['ok']


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
    write_zarr(tmp_path / 'a.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
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
    gix = random_gix(rng, (40, 30), 50)
    write_zarr(tmp_path / 'gix.zarr', gix)
    write_zarr(tmp_path / 'pc.zarr', rng.random(50).astype(np.float32))
    assert main(['pc2ras', '--idx', str(tmp_path / 'gix.zarr'), '--pc', str(tmp_path / 'pc.zarr'),
                 '--ras', str(tmp_path / 'ras.zarr'), '--shape', *tokens, '--chunks', '20', '30', '--json', '-q']) == 0
    assert json_out(capsys)['summaries'][0]['shape'] == [40, 30]


@pytest.mark.parametrize('tokens,msg', [(['2500'], 'needs 2 integers'), (['1', '2', '3'], 'needs 2 integers'),
                                        (['abc'], 'needs 2 integers'), (['2500.5', '1834'], 'needs 2 integers')])
def test_tuple_option_errors(tmp_path, capsys, tokens, msg):
    code = main(['pc2ras', '--idx', 'g.zarr', '--pc', 'p.zarr', '--ras', 'r.zarr', '--shape', *tokens, '--json'])
    assert code == 2
    out = json_out(capsys)
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
    write_zarr(tmp_path / 'ras.zarr', ras, (100, 100, 1))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'ras_pyramid'))
    assert pyramid_levels(tmp_path / 'ras_pyramid')[:2] == [0, 1] and pyramid_levels(tmp_path / 'ras.zarr') == []
    s = summarize(str(tmp_path / 'ras_pyramid'))
    assert s['kind'] == 'raster pyramid' and s['shape'] == [300, 200, 3] and s['levels'] > 1 and s['method'] == 'decimate'
    assert s['stats_level'] == 0 and abs(s['amplitude_mean'] - np.abs(ras).mean()) < 1e-3 and 'warnings' not in s
    # the images of a stack that are all nan or constant are named
    bad = ras.copy(); bad[..., 1] = np.nan; bad[..., 2] = 1 + 0j
    write_zarr(tmp_path / 'bad.zarr', bad, (100, 100, 1))
    mc.ras_pyramid(str(tmp_path / 'bad.zarr'), str(tmp_path / 'bad_pyramid'))
    assert summarize(str(tmp_path / 'bad_pyramid'))['warnings'] == ['all values nan: image 1', 'constant values: image 2']
    for kw in [{}, {'index': (2,)}, {'show': 'intf_seq', 'index': (1,)}, {'show': 'intf_all', 'index': (0, 2)}]:
        out = tmp_path / 'r.png'
        quicklook(str(tmp_path / 'ras_pyramid'), str(out), width=200, **kw)
        assert out.stat().st_size > 1000
        out.unlink()

    n = 500
    g = random_gix(rng, (300, 200), n)
    for name, a in [('x.zarr', g[:, 1].astype(float)), ('y.zarr', g[:, 0].astype(float)),
                    ('pc.zarr', rng.random(n).astype(np.float32))]:
        write_zarr(tmp_path / name, a)
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
    write_zarr(tmp_path / 'ras.zarr', rng.random((64, 48)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'pyr'))
    capsys.readouterr()                  # drop what ras_pyramid printed
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'q.png'), '--json']) == 0
    assert json_out(capsys)['png'] == str(tmp_path / 'q.png')
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(tmp_path / 'part.png'), '--extent', '10,5,30,20',
                 '--json']) == 0
    assert json_out(capsys)['png'] == str(tmp_path / 'part.png') and (tmp_path / 'part.png').stat().st_size > 1000
    assert main(['quicklook', str(tmp_path / 'ras.zarr'), '--json']) == 2          # a usage error
    assert 'not a pyramid' in json_out(capsys)['error']


def test_quicklook_wrong_inputs_are_usage_errors(tmp_path, capsys, rng, monkeypatch):
    """Wrong inputs of `moraine quicklook` are usage errors (exit code 2) with the messages of their checks, before
    anything is drawn: a path that is not a pyramid or one to rebuild, an extent that is not (min, min, max, max), a
    show that does not fit the data (they were processing errors, exit code 1; a show of the wrong data failed when it
    was drawn). An error while drawing stays a processing error."""
    import moraine.cli as mc
    import moraine.cli.tiles as tiles
    write_zarr(tmp_path / 'ts.zarr', rng.random((30, 20, 4)).astype(np.float32), (30, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'pyr'))
    write_zarr(tmp_path / 'r.zarr', rng.random((30, 20)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'r.zarr'), str(tmp_path / 'r_pyr'))
    write_zarr(tmp_path / 'm5.zarr', rng.random((30, 20, 2, 2, 2)).astype(np.float32), (30, 20, 1, 1, 1))
    mc.ras_pyramid(str(tmp_path / 'm5.zarr'), str(tmp_path / 'm5_pyr'))
    write_zarr(tmp_path / 'old' / '0.zarr', np.ones((6, 8), np.float32))     # levels without the marker
    write_zarr(tmp_path / 'old' / '1.zarr', np.ones((3, 4), np.float32))
    pyr, out = str(tmp_path / 'pyr'), tmp_path / 'q.png'
    for args, error in [
            ([str(tmp_path / 'ts.zarr')], f"{tmp_path / 'ts.zarr'} is not a pyramid; build one first with `moraine "
                                          f"ras-pyramid --ras {tmp_path / 'ts.zarr'} --out_dir <dir>` (rasters) or "
                                          f"`moraine pc-pyramid` (point clouds)"),
            ([str(tmp_path / 'missing')], f"{tmp_path / 'missing'} is not a pyramid; build one first with"),
            ([str(tmp_path / 'old')], f"{tmp_path / 'old'}: a pyramid without its marker (the attribute moraine_pyramid "
                                      f"of 0.zarr), made by moraine before 0.10.0"),
            ([pyr, '--extent', '30,5,10,20'], 'extent must be (range_min, azimuth_min, range_max, azimuth_max) with the '
                                              'minimum before the maximum, got (30.0, 5.0, 10.0, 20.0)'),
            ([pyr, '--extent', '10,5,30'], 'extent must be (range_min, azimuth_min, range_max, azimuth_max) with the '
                                           'minimum before the maximum, got (10.0, 5.0, 30.0)'),
            ([pyr, '--extent', 'west,5,30,20'], "extent must be four numbers (range_min, azimuth_min, range_max, "
                                                "azimuth_max), pixels on the radar grid, got ('west', 5.0, 30.0, 20.0)"),
            ([pyr, '--extent', 'west,south,east,north'], 'extent must be four numbers (range_min, azimuth_min, '
                                                         'range_max, azimuth_max), pixels on the radar grid, got '
                                                         "('west', 'south', 'east', 'north')"),
            ([pyr, '--show', 'phase'], 'phase needs complex data with one index dimension, got float32 with 1'),
            ([pyr, '--show', 'coh'], "show='coh' needs compressed coherence (..., n_pairs), not shape (30, 20, 4): one "
                                     'column per image pair, n_pairs = n (n - 1) / 2 for all pairs of n images (1, 3, '
                                     '6, 10, ...) or the number of the image pairs given (none are given)'),
            ([str(tmp_path / 'r_pyr'), '--show', 'intf_seq'], 'the data must have one or two index dimensions, got 0'),
            ([str(tmp_path / 'r_pyr'), '--show', 'coh'], "show='coh' needs compressed coherence (..., n_pairs), not "
                                                         "shape (30, 20)"),
            ([str(tmp_path / 'm5_pyr')], 'data of shape (30, 20, 2, 2, 2): 3 stack dimensions, more than the shows of '
                                         "a view take (the default and 'intf_0', 'intf_seq', 'intf_all': one or two; ")]:
        capsys.readouterr()
        assert main(['quicklook', *args, '-o', str(out), '--json']) == 2, args
        result = json_out(capsys)
        assert result['ok'] is False and result['error'].startswith(error), (args, result)
        assert not out.exists()
    assert main(['quicklook', pyr, '-o', str(out)]) == 0 and out.stat().st_size > 1000
    out.unlink()
    # an error while drawing is a processing error

    def render(*args, **kwargs):
        raise ValueError('a chunk of the level cannot be read')
    monkeypatch.setattr(tiles.RasterLayer, 'render', render)
    capsys.readouterr()
    assert main(['quicklook', pyr, '-o', str(out), '--json']) == 1
    assert json_out(capsys)['error'] == 'ValueError: a chunk of the level cannot be read' and not out.exists()


def test_view_and_info_reject_what_quicklook_rejects(tmp_path, capsys, rng):
    """`moraine view` rejects a path that is not a pyramid, and `moraine view` and `moraine info` a pyramid to rebuild
    or of a newer moraine, as `moraine quicklook` does: a usage error (exit code 2) with the same message (they
    exited with 1, after "ValueError: "). info still summarizes what is not a pyramid, and a missing path stays a
    processing error of info."""
    import moraine.cli as mc
    write_zarr(tmp_path / 'ts.zarr', rng.random((30, 20, 4)).astype(np.float32), (30, 20, 1))
    write_zarr(tmp_path / 'old' / '0.zarr', np.ones((6, 8), np.float32))     # levels without the marker
    write_zarr(tmp_path / 'old' / '1.zarr', np.ones((3, 4), np.float32))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'new'))
    z0 = zarr.open(str(tmp_path / 'new' / '0.zarr'), mode='r+')
    z0.attrs['moraine_pyramid'] = {**z0.attrs['moraine_pyramid'], 'version': 99}       # of a newer moraine
    for i, n in enumerate([50, 25]):         # per-chunk arrays (ras2pc_ras_chunk), not levels of a pyramid
        write_zarr(tmp_path / 'chunks' / f'{i}.zarr', rng.random((n, 11, 11)).astype(np.float32))
    capsys.readouterr()
    png, nb = tmp_path / 'q.png', tmp_path / 'v.ipynb'
    options = {'quicklook': ['-o', str(png)], 'view': ['-o', str(nb)], 'info': []}
    for path, error, commands in [
            ('ts.zarr', f"{tmp_path / 'ts.zarr'} is not a pyramid; build one first with `moraine ras-pyramid --ras "
                        f"{tmp_path / 'ts.zarr'} --out_dir <dir>` (rasters) or `moraine pc-pyramid` (point clouds)",
             ('quicklook', 'view')),
            ('missing', f"{tmp_path / 'missing'} is not a pyramid; build one first with", ('quicklook', 'view')),
            ('chunks', f"{tmp_path / 'chunks'} is not a pyramid; build one first with", ('quicklook', 'view')),
            ('old', f"{tmp_path / 'old'}: a pyramid without its marker (the attribute moraine_pyramid of 0.zarr), made "
                    "by moraine before 0.10.0 or by a ras-pyramid that did not finish; rebuild it with `moraine "
                    "ras-pyramid`", ('quicklook', 'view', 'info')),
            ('new', f"{tmp_path / 'new'}: pyramid layout version 99 is newer than this moraine supports (1); update "
                    "moraine", ('quicklook', 'view', 'info'))]:
        errors = []
        for command in commands:
            assert main([command, str(tmp_path / path), *options[command], '--json']) == 2, (command, path)
            result = json_out(capsys)
            assert set(result) == {'version', 'ok', 'error'} and result['ok'] is False, (command, path)
            errors.append(result['error'])
        assert errors[0].startswith(error) and errors == errors[:1] * len(commands), (path, errors)
    assert not png.exists() and not nb.exists()
    assert main(['info', str(tmp_path / 'ts.zarr'), str(tmp_path / 'chunks'), '--json']) == 0
    assert [s['kind'] for s in json_out(capsys)['summaries']] == ['array', 'directory']
    assert main(['info', str(tmp_path / 'missing'), '--json']) == 1
    assert json_out(capsys)['error'] == f"FileNotFoundError: {tmp_path / 'missing'} does not exist"


def test_quicklook_index_within_the_sliders(tmp_path, capsys, rng):
    """`moraine quicklook --index` outside the ranges of the sliders is a usage error that names them (beyond the
    last image it drew nan for coherence)."""
    import moraine.cli as mc
    gy, gx = np.divmod(np.arange(12), 4)
    write_zarr(tmp_path / 'gy.zarr', gy.astype(float)); write_zarr(tmp_path / 'gx.zarr', gx.astype(float))
    coh = (0.5 * np.exp(1j * rng.uniform(-np.pi, np.pi, (12, 6)))).astype(np.complex64)     # all pairs of 4 images
    write_zarr(tmp_path / 'coh.zarr', coh, (12, 1))
    mc.pc_pyramid(str(tmp_path / 'coh.zarr'), str(tmp_path / 'pyr'), x=str(tmp_path / 'gx.zarr'),
                  y=str(tmp_path / 'gy.zarr'), ras_resolution=1)
    capsys.readouterr()
    out = tmp_path / 'q.png'
    args = ['quicklook', str(tmp_path / 'pyr'), '-o', str(out), '--show', 'coh', '--json', '--index']
    assert main(args + ['0', '6']) == 2
    result = json_out(capsys)
    assert result['ok'] is False and result['error'] == ("--index 0 6: slider 'sec' = 6 is outside its range 0 .. 3; "
                                                         "the sliders are ref 0 .. 3, sec 0 .. 3")
    assert not out.exists()
    assert main(args + ['3', '0']) == 0 and out.stat().st_size > 1000
    with pytest.raises(UsageError, match=r"--index 0 -1: slider 'sec' = -1 is outside its range 0 \.\. 3"):
        quicklook(str(tmp_path / 'pyr'), str(tmp_path / 'p.png'), index=(0, -1), show='coh')


def test_quicklook_index_of_the_sliders(tmp_path, capsys, rng):
    """`moraine quicklook --index` with more values than sliders is a usage error that names the sliders (the values
    beyond the sliders were ignored: `--index 0 6` drew image 0 of a stack)."""
    import moraine.cli as mc
    write_zarr(tmp_path / 'ts.zarr', rng.random((30, 20, 4)).astype(np.float32), (30, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'pyr'))
    write_zarr(tmp_path / 'r.zarr', rng.random((30, 20)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'r.zarr'), str(tmp_path / 'r_pyr'))
    capsys.readouterr()
    out = tmp_path / 'q.png'
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(out), '--json', '--index', '0', '6']) == 2
    assert json_out(capsys)['error'] == '--index 0 6: 2 values for 1 slider; the view has the slider i 0 .. 3'
    assert main(['quicklook', str(tmp_path / 'r_pyr'), '-o', str(out), '--json', '--index', '0']) == 2
    assert json_out(capsys)['error'] == '--index 0: 1 value for 0 sliders; the view has no sliders'
    assert not out.exists()
    assert main(['quicklook', str(tmp_path / 'pyr'), '-o', str(out), '--json', '--index', '3']) == 0
    assert out.stat().st_size > 1000


def test_quicklook_clim(tmp_path, capsys, rng):
    """`moraine quicklook --clim LOW,HIGH` draws with the colour range of `clim` of `moraine.cli.view` (there was no
    way to set it: the differences of a real stack were drawn over +-(p99 - p01), too wide for sequential pairs); a
    LOW with a minus sign is the value of the option. Anything but two finite numbers with LOW < HIGH is a usage
    error, before anything is drawn."""
    import moraine.cli as mc
    ts = np.cumsum(rng.normal(-5, 2, (30, 20, 4)), axis=-1).astype(np.float32)      # a time series in mm
    write_zarr(tmp_path / 'ts.zarr', ts, (30, 20, 1))
    mc.ras_pyramid(str(tmp_path / 'ts.zarr'), str(tmp_path / 'pyr'))
    capsys.readouterr()
    pyr, out = str(tmp_path / 'pyr'), tmp_path / 'q.png'
    expected = Path(mc.view(pyr, show='intf_seq', clim=(-11, 6)).png(str(tmp_path / 'v.png'), width=300)).read_bytes()
    default = Path(mc.view(pyr, show='intf_seq').png(str(tmp_path / 'd.png'), width=300)).read_bytes()
    assert expected != default
    args = ['quicklook', pyr, '-o', str(out), '--show', 'intf_seq', '--width', '300', '--json']
    for clim in (['--clim', '-11,6'], ['--clim=-11,6'], ['--clim', '-11 6'], ['--clim', '(-11, 6)'],
                 ['--clim', '-1.1e1,6.0']):
        assert main(args + clim) == 0, clim
        assert json_out(capsys)['png'] == str(out) and out.read_bytes() == expected, clim
        out.unlink()
    assert quicklook(pyr, str(out), show='intf_seq', width=300, clim=(-11, 6)) == str(out)     # two numbers in Python
    assert out.read_bytes() == expected
    out.unlink()
    for clim in ('6,-11', '1,1', '1', '1,2,3', 'a,b', 'nan,1', '1,inf', ''):
        assert main(args + ['--clim', clim]) == 2, clim
        assert json_out(capsys)['error'] == ('clim must be two numbers LOW,HIGH with LOW < HIGH, the values of the '
                                             f'first and the last colour (e.g. -11,6), got {clim!r}')
    with pytest.raises(UsageError, match=r'^clim must be two numbers LOW,HIGH with LOW < HIGH.*got \(6, -11\)$'):
        quicklook(pyr, str(out), clim=(6, -11))
    assert not out.exists()
    with pytest.raises(SystemExit):
        main(['quicklook', '--help'])
    assert '--clim LOW,HIGH' in ' '.join(capsys.readouterr().out.split())


def test_quicklook_extent_starting_with_a_minus_sign(tmp_path, capsys, rng):
    """`moraine quicklook --extent` takes a first number with a minus sign, a longitude west of Greenwich or the edge
    -0.5 of the first pixel that the views give, as `--clim` does; argparse took the value for an unknown option
    ("argument --extent: expected one argument")."""
    import moraine.cli as mc
    from moraine.cli.tiles import _lonlat_to_merc
    write_zarr(tmp_path / 'r.zarr', rng.random((30, 20)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'r.zarr'), str(tmp_path / 'r_pyr'))
    gy, gx = np.divmod(np.arange(100), 10)
    x0, y0 = _lonlat_to_merc(-122.42, 37.77)                                       # points in San Francisco
    write_zarr(tmp_path / 'mx.zarr', x0 + 30.0 * gx); write_zarr(tmp_path / 'my.zarr', y0 + 30.0 * gy)
    write_zarr(tmp_path / 'v.zarr', rng.random(100).astype(np.float32))
    mc.pc_pyramid(str(tmp_path / 'v.zarr'), str(tmp_path / 'm_pyr'), x=str(tmp_path / 'mx.zarr'),
                  y=str(tmp_path / 'my.zarr'), ras_resolution=30)
    capsys.readouterr()
    for pyr, extent in [('r_pyr', '-0.5,-0.5,10,15'), ('m_pyr', '-122.42,37.77,-122.418,37.772')]:
        out = tmp_path / f'{pyr}.png'
        for args in (['--extent', extent], [f'--extent={extent}']):
            assert main(['quicklook', str(tmp_path / pyr), '-o', str(out), *args, '--json']) == 0, args
            assert json_out(capsys)['png'] == str(out) and out.stat().st_size > 1000
            out.unlink()


def test_quicklook_output_name(tmp_path, capsys, rng, monkeypatch):
    """`moraine quicklook -o NAME` without a suffix writes and reports NAME.png (matplotlib wrote NAME.png while the
    command reported NAME, also as `png` of the JSON output); another suffix than .png is a usage error before
    anything is drawn (matplotlib failed after drawing, exit code 1, or wrote another format, e.g. a PDF)."""
    import moraine.cli as mc
    import moraine.cli.tiles as tiles
    write_zarr(tmp_path / 'r.zarr', rng.random((30, 20)).astype(np.float32))
    mc.ras_pyramid(str(tmp_path / 'r.zarr'), str(tmp_path / 'pyr'))
    capsys.readouterr()
    pyr = str(tmp_path / 'pyr')
    assert main(['quicklook', pyr, '-o', str(tmp_path / 'q'), '--json']) == 0
    assert json_out(capsys)['png'] == str(tmp_path / 'q.png')
    assert (tmp_path / 'q.png').stat().st_size > 1000 and not (tmp_path / 'q').exists()
    assert main(['quicklook', pyr, '-o', str(tmp_path / 'a.b' / 'text')]) == 0             # a dot in a directory
    assert capsys.readouterr().out == f"saved {tmp_path / 'a.b' / 'text.png'}\n"
    assert (tmp_path / 'a.b' / 'text.png').exists()
    assert quicklook(pyr, str(tmp_path / 'p')) == str(tmp_path / 'p.png') and (tmp_path / 'p.png').exists()
    for name in ('q2.png', 'Q.PNG'):                                                        # .png in any case
        assert main(['quicklook', pyr, '-o', str(tmp_path / name), '--json']) == 0
        assert json_out(capsys)['png'] == str(tmp_path / name) and (tmp_path / name).stat().st_size > 1000
    drawn = []
    monkeypatch.setattr(tiles, 'render_png', lambda *args, **kwargs: drawn.append(args))
    for name in ('x.txt', 'x.pdf', 'x.jpg', 'v1.2'):
        assert main(['quicklook', pyr, '-o', str(tmp_path / name), '--json']) == 2, name
        assert json_out(capsys)['error'] == (f'{tmp_path / name}: the image name must end with .png, or have no '
                                             'suffix (then .png is appended)')
        assert not (tmp_path / name).exists()
    with pytest.raises(UsageError, match='must end with .png'):
        quicklook(pyr, str(tmp_path / 'x.svg'))
    assert drawn == []


def test_quicklook_and_view_help_agree_with_the_docstrings(capsys):
    """The help of `moraine quicklook` and `moraine view`, which the manual shows, says what the docstrings of the
    functions say: the default of --show, and the sliders of --index, also i j of 4D data (it did not say that)."""
    from moraine.command import summary
    helps = {}
    for name in ('quicklook', 'view'):
        with pytest.raises(SystemExit):
            main([name, '--help'])
        helps[name] = ' '.join(capsys.readouterr().out.split())            # without the wrapping of argparse
        doc = parse_docstring(getattr(summary, name).__doc__)[1]['show'][1]
        m = re.fullmatch(r'(.*), see `moraine.cli.view`; (.*) by default', doc)
        assert m, doc
        assert f'{m[1]}, see moraine.cli.view (default: {m[2]})' in helps[name]
    index = parse_docstring(summary.quicklook.__doc__)[1]['index'][1]
    assert "(i, j) for 4D data (..., n, n), (ref, sec) for 'intf_all' / 'coh' / 'coh_abs'" in index
    assert ('image of a stack, i j of 4D data (..., n, n), or two images (ref sec) with --show intf_all / coh / '
            'coh_abs') in helps['quicklook']


@pytest.mark.parametrize('data,warning', [
    (np.full((60, 40), np.nan, np.float32), 'all values are nan'),
    (np.where(np.arange(2400).reshape(60, 40) % 97 == 0, np.inf, 1.5).astype(np.float32), 'infinite values'),
    (np.zeros((60, 40), np.float32), 'all values are 0.0'),
])
def test_pyramid_warnings(tmp_path, data, warning):
    import moraine.cli as mc
    write_zarr(tmp_path / 'a.zarr', data, (20, 20))
    mc.ras_pyramid(str(tmp_path / 'a.zarr'), str(tmp_path / 'pyr'))
    assert any(warning in w for w in summarize(str(tmp_path / 'pyr'))['warnings'])


def test_pipeline_vars_and_workdir(tmp_path, rng, capsys):
    data = tmp_path / 'data'; work = tmp_path / 'work'
    data.mkdir()
    write_zarr(data / 'a.zarr', rng.random((30, 40)).astype(np.float32), (10, 40))
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
    assert json_out(capsys)['steps'][0]['status'] == 'done'
    (tmp_path / 'q.toml').write_text('[[step]]\nname="a"\nrun="amp-disp"\nrslc="${gamma}/r"\nadi="a"\n')
    with pytest.raises(UsageError, match=r'unknown variable \$\{gamma\}'):
        run_pipeline(str(tmp_path / 'q.toml'), dry_run=True, echo=lambda *a: None)


PAIRS = '''
[vars]
n = { file = "meta.toml", key = "n" }
pairs = { file = "meta.toml", key = "pairs" }
bw = { file = "meta.toml", key = "bw" }

[defaults]
bandwidth = "${bw}"

[[step]]
name = "pairs"
run = "image-pairs"
nimages = "${n}"
out = "pairs_${n}.txt"

[[step]]
name = "t_coh"
run = "temp-coh"
intf = "intf.zarr"
rslc = "rslc.zarr"
image_pairs = "${pairs}"
'''


def test_pipeline_variables_from_a_toml_file(tmp_path):
    meta, p, quiet = tmp_path / 'meta.toml', str(tmp_path / 'p.toml'), dict(echo=lambda *a: None)
    meta.write_text('n = 4\npairs = [[0, 1], [1, 2], [2, 3]]\nbw = 1\nname = "x"\n')
    (tmp_path / 'p.toml').write_text(PAIRS)
    pairs, t_coh = load_pipeline(p)['steps']
    # a whole value keeps the type of the variable (also in [defaults]), inside a longer string it is text
    assert pairs['values'] == {'bandwidth': 1, 'nimages': 4, 'out': 'pairs_4.txt'}
    np.testing.assert_array_equal(t_coh['kwargs']['image_pairs'], [[0, 1], [1, 2], [2, 3]])
    assert run_pipeline(p, only=['pairs'], **quiet)['ok']
    np.testing.assert_array_equal(image_pairs(str(tmp_path / 'pairs_4.txt')), [[0, 1], [1, 2], [2, 3]])
    # the hash holds the values read: another key of the file changes nothing, a value reruns the step
    meta.write_text('n = 4\npairs = [[0, 1], [1, 2], [2, 3]]\nbw = 1\nname = "y"\n')
    assert run_pipeline(p, only=['pairs'], dry_run=True, **quiet)['plan'][0]['action'] == 'skip (done)'
    meta.write_text('n = 4\npairs = [[0, 1], [1, 2], [2, 3]]\nbw = 2\nname = "y"\n')
    assert run_pipeline(p, only=['pairs'], dry_run=True, **quiet)['plan'][0]['action'] == 'run (arguments changed)'
    # --var overrides it (text), and the file is not read then
    meta.unlink()
    pairs = load_pipeline(p, variables={'n': '3', 'pairs': '[[0, 1]]', 'bw': '2'})['steps'][0]
    assert pairs['kwargs'] == {'bandwidth': 2, 'nimages': 3, 'out': 'pairs_3.txt'} and pairs['var_files'] == []
    # errors
    meta.write_text('n = 4\npairs = [[0, 1]]\nbw = 1\n')
    (tmp_path / 'bad.toml').write_text('n = \n')
    for old, new, match in [('key = "n"', 'key = "nn"', "no key 'nn' in meta.toml. Did you mean 'n'? Keys: n, pairs, bw"),
                            ('key = "n"', 'key = "n", default = 1', 'is written { file = "PATH", key = "KEY" }'),
                            ('key = "n"', 'key = 1', 'is written { file = "PATH", key = "KEY" }'),
                            ('file = "meta.toml", key = "n"', 'file = "bad.toml", key = "n"',
                             'bad.toml is not a readable TOML file'),
                            ('"pairs_${n}.txt"', '"pairs_${pairs}.txt"', 'is a list and can only be a whole value')]:
        (tmp_path / 'q.toml').write_text(PAIRS.replace(old, new, 1))
        with pytest.raises(UsageError, match=re.escape(match)):
            load_pipeline(str(tmp_path / 'q.toml'))


def test_pipeline_variable_from_a_file_made_later(tmp_path, monkeypatch):
    # the file of a variable may be made by an earlier step of the pipeline (or by another pipeline)
    def write_meta(out:str, n:int):
        Path(out).write_text(f'n = {n}\npairs = [[0, 1]]\nbw = 1\n')
    monkeypatch.setitem(commands(), 'write-meta', Command(
        'write-meta', write_meta, 'test', 'Write a TOML file.',
        [Param('out', 'str', True, None, 'str', 'output'), Param('n', 'value', True, None, 'int', 'a number')]))
    text = PAIRS.replace('[[step]]', '[[step]]\nname = "meta"\nrun = "write-meta"\nout = "meta.toml"\nn = NUMBER\n\n[[step]]', 1)
    p, quiet = tmp_path / 'p.toml', dict(echo=lambda *a: None)
    p.write_text(text.replace('NUMBER', '3'))
    plan = run_pipeline(str(p), dry_run=True, **quiet)['plan']       # the names of the arguments are checked
    assert [a['action'] for a in plan] == ['run (not run yet)', 'run (variable bw: meta.toml does not exist yet)',
                                           'run (variable pairs: meta.toml does not exist yet)']
    assert pipeline_status(str(p))['steps'][1]['status'] == 'pending'
    assert run_pipeline(str(p), only=['meta', 'pairs'], **quiet)['ok']
    assert (tmp_path / 'pairs_3.txt').exists()
    # a step rewriting the file reruns the steps using it, with the new values
    p.write_text(text.replace('NUMBER', '5'))
    plan = run_pipeline(str(p), only=['meta', 'pairs'], dry_run=True, **quiet)['plan']
    assert [a['action'] for a in plan[:2]] == ['run (arguments changed)', 'run (input will change: meta.toml)']
    assert run_pipeline(str(p), only=['meta', 'pairs'], **quiet)['ok']
    assert len(image_pairs(str(tmp_path / 'pairs_5.txt'))) == 4
    # still missing when the step runs: the step fails, and the next run resumes there once the file exists
    (tmp_path / 'meta.toml').unlink()
    res = run_pipeline(str(p), only=['pairs'], **quiet)
    assert not res['ok'] and res['steps'][0]['error'] == 'UsageError: variable bw: meta.toml does not exist yet'
    write_meta(str(tmp_path / 'meta.toml'), 6)
    res = run_pipeline(str(p), only=['pairs'], **quiet)
    assert res['ok'] and res['steps'][0]['args']['nimages'] == 6


def test_image_pairs_command(tmp_path, capsys):
    write_zarr(tmp_path / 'rslc.zarr', np.zeros((4, 5, 6), np.complex64))
    assert main(['image-pairs', '--rslc', str(tmp_path / 'rslc.zarr'), '--bandwidth', '2', '--out',
                 str(tmp_path / 'p.txt'), '-q', '--json']) == 0
    capsys.readouterr()
    np.testing.assert_array_equal(image_pairs(str(tmp_path / 'p.txt')), [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5]])


def test_chunk_directories_are_not_pyramids(tmp_path, rng):
    from moraine.command.summary import pyramid_levels
    for i, n in enumerate([500, 320, 410]):      # per-chunk arrays of ras2pc_ras_chunk: 0.zarr, 1.zarr, ...
        write_zarr(tmp_path / 'chunks' / f'{i}.zarr', rng.random((n, 11, 11)).astype(np.float32))
    assert pyramid_levels(tmp_path / 'chunks') == []
    assert summarize(str(tmp_path / 'chunks'))['kind'] == 'directory'


def test_pipeline_makes_missing_parent_directories(tmp_path, capsys):
    # the image pair file of a later step is made by an earlier step, in a directory that does not exist yet
    write_zarr(tmp_path / 'rslc.zarr', np.ones((20, 30, 4), np.complex64), (10, 30, 1))
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
    write_zarr(tmp_path / 'intf.zarr', np.ones((20, 30, 3), np.complex64), (10, 30, 1))
    assert run_pipeline(str(tmp_path / 'p.toml'), echo=lambda *a: None)['ok']
    np.testing.assert_allclose(zarr.open(str(tmp_path / 't_coh.zarr'), mode='r')[:], 1, rtol=1e-5)


def test_outputs_without_slash_or_dot(tmp_path, capsys, rng, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_zarr(tmp_path / 'ras.zarr', rng.random((16, 16)).astype(np.float32))
    assert main(['ras-pyramid', '--ras', 'ras.zarr', '--out_dir', 'pyr', '--json', '-q']) == 0
    out = json_out(capsys)
    assert out['outputs'] == ['pyr'] and out['inputs'].keys() == {'ras.zarr'}


# ---------------------------------------------------------------- interactive views

@pytest.fixture
def pyramids(tmp_path, rng):
    import moraine.cli as mc
    d = tmp_path
    stack = np.exp(1j * rng.uniform(-np.pi, np.pi, (60, 40, 4))).astype(np.complex64)
    write_zarr(d / 'stack.zarr', stack, (30, 40, 1))
    mc.ras_pyramid(str(d / 'stack.zarr'), str(d / 'stack_pyr'))
    write_zarr(d / 'long.zarr', rng.random((400, 20)).astype(np.float32), (100, 20))     # elongated scene
    mc.ras_pyramid(str(d / 'long.zarr'), str(d / 'long_pyr'))
    n = 300
    g = random_gix(rng, (60, 40), n)
    for name, a in [('gy.zarr', g[:, 0].astype(float)), ('gx.zarr', g[:, 1].astype(float)),
                    ('my.zarr', 6e6 + 7.5 * g[:, 0]), ('mx.zarr', -1.6e7 + 7.5 * g[:, 1]),     # map coordinates
                    ('val.zarr', np.linspace(0, 10, n * 3).reshape(n, 3).astype(np.float32))]:
        write_zarr(d / name, a)
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
    out = json_out(capsys)
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
    assert main(['view', str(d / 'stack_pyr'), '-o', str(nb_path), '--json']) == 2   # no silent overwrite
    assert 'exists' in json_out(capsys)['error']
    assert main(['view', str(d / 'stack_pyr'), '-o', str(nb_path), '--overwrite', '--show', 'phase',
                 '--json']) == 0
    with pytest.raises(SystemExit):                                    # --post_proc, its name of 2026-09, is gone
        main(['view', str(d / 'stack_pyr'), '-o', str(nb_path), '--overwrite', '--post_proc', 'phase'])
    # 3D maps over the terrain, for web mercator pyramids
    nb3d = d / 'v3d.ipynb'
    assert main(['view', str(d / 'map_pyr'), '-o', str(nb3d), '--terrain', '--json']) == 0
    code = [''.join(c['source']) for c in json.loads(nb3d.read_text())['cells'] if c['cell_type'] == 'code']
    assert code[-1].endswith(', terrain=True)')
    ns = {}
    exec(code[0], ns)
    assert eval(code[-1], ns).widget.terrain['url'].endswith('{z}/{x}/{y}.png')


def test_view_output_name(pyramids, capsys, tmp_path):
    """`moraine view -o NAME` without a suffix writes and reports NAME.ipynb (it exited with 1: "the notebook name must
    end with .ipynb"); another suffix and an existing notebook without --overwrite are usage errors (exit code 2; they
    were processing errors, exit code 1) raised before anything is read: before the pyramid paths are checked."""
    from moraine.command.summary import view as view_notebook
    pyr = str(pyramids / 'stack_pyr')
    assert main(['view', pyr, '-o', str(tmp_path / 'v'), '--json']) == 0
    assert json_out(capsys)['notebook'] == str(tmp_path / 'v.ipynb')
    assert (tmp_path / 'v.ipynb').exists() and not (tmp_path / 'v').exists()
    assert main(['view', pyr, '-o', str(tmp_path / 'a.b' / 'nb')]) == 0                   # a dot in a directory
    assert capsys.readouterr().out.startswith(f"saved {tmp_path / 'a.b' / 'nb.ipynb'}: open it in Jupyter")
    assert view_notebook([pyr], str(tmp_path / 'p')) == str(tmp_path / 'p.ipynb') and (tmp_path / 'p.ipynb').exists()
    (tmp_path / 'old.ipynb').write_text('{}')
    for name, error in [
            ('x.txt', f"{tmp_path / 'x.txt'}: the notebook name must end with .ipynb, or have no suffix (then .ipynb is "
                      'appended)'),
            ('X.IPYNB', f"{tmp_path / 'X.IPYNB'}: the notebook name must end with .ipynb, or have no suffix (then "
                        '.ipynb is appended)'),
            ('v1.2', f"{tmp_path / 'v1.2'}: the notebook name must end with .ipynb, or have no suffix (then .ipynb is "
                     'appended)'),
            ('old.ipynb', f"{tmp_path / 'old.ipynb'} exists; choose another name or overwrite it"),
            ('old', f"{tmp_path / 'old.ipynb'} exists; choose another name or overwrite it")]:
        for path in (pyr, str(tmp_path / 'missing')):          # the name first, then the pyramids
            assert main(['view', path, '-o', str(tmp_path / name), '--json']) == 2, (name, path)
            result = json_out(capsys)
            assert set(result) == {'version', 'ok', 'error'} and result['error'] == error, (name, path)
        with pytest.raises(UsageError, match='^' + re.escape(error) + '$'):
            view_notebook([str(tmp_path / 'missing')], str(tmp_path / name))
    assert not (tmp_path / 'x.txt').exists() and (tmp_path / 'old.ipynb').read_text() == '{}'
    assert main(['view', pyr, '-o', str(tmp_path / 'old'), '--overwrite', '--json']) == 0
    assert json_out(capsys)['notebook'] == str(tmp_path / 'old.ipynb')
    assert json.loads((tmp_path / 'old.ipynb').read_text())['nbformat'] == 4
