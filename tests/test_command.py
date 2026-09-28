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
    assert bind_args(cmd, {'rslc': 'r', 'adi': 'a'}, kw={'memory_limit': '2GB'})['memory_limit'] == '2GB'


# ---------------------------------------------------------------- summary / quicklook

def test_summarize(tmp_path, rng):
    _zarr(tmp_path / 'f.zarr', np.where(rng.random((300, 200)) < 0.1, np.nan, 1.0).astype(np.float32))
    s = summarize(str(tmp_path / 'f.zarr'))
    assert s['kind'] == 'array' and s['shape'] == [300, 200] and 0.05 < s['nan_fraction'] < 0.15
    _zarr(tmp_path / 'c.zarr', (rng.random((50, 3)) * np.exp(1j * rng.random((50, 3)))).astype(np.complex64))
    assert 'amplitude_mean' in summarize(str(tmp_path / 'c.zarr'))
    _zarr(tmp_path / 'b.zarr', np.ones((10, 10), bool))
    assert summarize(str(tmp_path / 'b.zarr'))['true_fraction'] == 1.0
    big = summarize(str(tmp_path / 'f.zarr'), max_elements=1000)
    assert 'sampled_step' in big
    with pytest.raises(FileNotFoundError):
        summarize(str(tmp_path / 'missing.zarr'))


def test_quicklook(tmp_path, rng):
    _zarr(tmp_path / 'ras.zarr', rng.random((100, 80, 3)).astype(np.float32))
    _zarr(tmp_path / 'pc.zarr', np.exp(1j * rng.random(50)).astype(np.complex64))
    _zarr(tmp_path / 'gix.zarr', np.stack(np.unravel_index(rng.choice(400, 50, replace=False), (20, 20)), -1).astype(np.int32))
    for args in [('ras.zarr', {}), ('pc.zarr', {'gix': str(tmp_path / 'gix.zarr')}), ('pc.zarr', {})]:
        quicklook(str(tmp_path / args[0]), str(tmp_path / 'q.png'), **args[1])
        assert (tmp_path / 'q.png').stat().st_size > 1000
        (tmp_path / 'q.png').unlink()


# ---------------------------------------------------------------- command line

def test_main_list_info_tnet(tmp_path, capsys, rng):
    assert main(['list', '--json']) == 0
    assert any(c['name'] == 'amp-disp' for c in _json_out(capsys)['commands'])
    _zarr(tmp_path / 'a.zarr', rng.random((20, 20)).astype(np.float32))
    assert main(['info', str(tmp_path / 'a.zarr'), '--json']) == 0
    assert _json_out(capsys)['summaries'][0]['shape'] == [20, 20]
    assert main(['tnet', '--nimages', '5', '--bandwidth', '1', '-o', str(tmp_path / 'p.txt'), '--json']) == 0
    assert _json_out(capsys)['n_pairs'] == 4
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'p.txt', dtype=int), [[0, 1], [1, 2], [2, 3], [3, 4]])


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
    assert (d / '.moraine' / 'quicklook' / 'scale__b.zarr.png').exists()
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
    assert (d / '.moraine' / 'logs' / 'scale.log').exists()
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
