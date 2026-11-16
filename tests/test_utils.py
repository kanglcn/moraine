from pathlib import Path

import numba
import numpy as np
import pytest

import moraine as mr
from moraine.api.utils_ import jit
from moraine.api.utils_.jit import _source_hash, mjit
from moraine.api.utils_.device import _default_cuda_home


def test_source_hash(tmp_path):
    (tmp_path / 'a.py').write_text('x = 1\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.py').write_text('y = 2\n')
    h = _source_hash(tmp_path)
    (tmp_path / '.ipynb_checkpoints').mkdir()
    (tmp_path / '.ipynb_checkpoints' / 'c.py').write_text('z = 3\n')
    assert _source_hash(tmp_path) == h
    (tmp_path / 'sub' / 'b.py').write_text('y = 4\n')  # a change in any file gives another cache directory
    assert _source_hash(tmp_path) != h


def test_source_hash_of_the_whole_package():
    """the cache directory changes with every source file of moraine, not only those of moraine.api (decision 0029)"""
    assert _source_hash() == _source_hash(Path(mr.__file__).parent)


def test_mjit_keeps_numba_cache_dir():
    before = numba.config.CACHE_DIR

    @mjit(nopython=True)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert numba.config.CACHE_DIR == before
    if jit._CACHE_DIR is not None:
        assert str(f._cache._cache_path).startswith(jit._CACHE_DIR)


def test_emi_kernel_cached(rng):
    # the EMI kernel (with its intrinsics) is saved to the cache
    from moraine.api.ds.pl import _emi_numba
    x = rng.standard_normal((4, 30, 6)) + 1j * rng.standard_normal((4, 30, 6))
    c = np.conj(x).transpose(0, 2, 1) @ x
    d = 1 / np.sqrt(np.diagonal(c, axis1=1, axis2=2).real)
    pairs = mr.TempNet.from_bandwidth(6).image_pairs
    coh = (c[:, pairs[:, 0], pairs[:, 1]].conj() * d[:, pairs[:, 0]] * d[:, pairs[:, 1]]).astype(np.complex64)
    assert mr.emi(coh).shape == (4, 6)
    if jit._CACHE_DIR is not None:
        assert list(Path(_emi_numba._cache._cache_path).glob('pl._emi_numba-*.nbi'))


def test_default_cuda_home(tmp_path, monkeypatch):
    for v in ('CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr('sys.prefix', str(tmp_path))
    _default_cuda_home()
    assert 'CUDA_HOME' not in __import__('os').environ          # no CUDA libraries in this prefix
    (tmp_path / 'nvvm').mkdir()
    _default_cuda_home()
    assert __import__('os').environ['CUDA_HOME'] == str(tmp_path)
    monkeypatch.setenv('CUDA_HOME', '/somewhere/else')
    _default_cuda_home()
    assert __import__('os').environ['CUDA_HOME'] == '/somewhere/else'   # set by the user: kept


@pytest.mark.gpu
def test_cuda_kernels_cached():
    """GPU kernels are saved to moraine's numba cache directory (decision 0029), like the CPU functions."""
    import cupy as cp
    from moraine.api import ps
    before = numba.config.CACHE_DIR
    rslc = cp.asarray((np.random.default_rng(0).standard_normal((8, 8, 5, 2)) @ [1, 1j]).astype(np.complex64))
    mr.amp_disp(rslc)
    assert numba.config.CACHE_DIR == before
    if jit._CACHE_DIR is not None:
        kernel = ps._amp_disp_cuda
        assert str(kernel._cache._cache_path).startswith(jit._CACHE_DIR)
        assert list(Path(kernel._cache._cache_path).glob('ps._amp_disp_cuda-*.nbi'))


def test_write_toml(tmp_path):
    """The files of `write_toml` (meta.toml, bounds.toml of the pyramids) are read back by tomllib with their values
    and types; numpy numbers as the Python numbers they convert to, aware date-times with their offset."""
    import datetime
    import tomllib
    from moraine.cli.utils_ import write_toml
    utc = datetime.timezone.utc
    base = np.array([-97.6136, 0, 1e-5, 3e38, np.nan, np.inf, -np.inf], dtype=np.float32)
    table = {'dates': ['20210802', '20210816'],
             'times': [datetime.datetime(2021, 8, 2, 9, 24, 42, 695163, tzinfo=utc),
                       datetime.datetime(2021, 8, 16, 9, 24, 42, tzinfo=utc)],
             'radar_wavelength': 0.05546576, 'n': 3, 'flags': [True, np.bool_(False)], 'base': base,
             'bounds': [np.float64(-16498435.873784425), np.float32(40.8), np.int32(2), np.int64(-3), np.uint16(7)],
             'day': datetime.date(2021, 8, 2), 'local': datetime.datetime(2021, 8, 2, 9, 24),
             'text': 'a "quote", a back\\slash, \t\n\x00\x1f\x7f, é 地', 'a key': [], 'nested': [[1, 2.5], ['x']]}
    text = write_toml(tmp_path / 'm.toml', table)
    assert (tmp_path / 'm.toml').read_text(encoding='utf-8') == text
    back = tomllib.loads(text)
    assert list(back) == list(table)
    assert back['bounds'] == [-16498435.873784425, float(np.float32(40.8)), 2, -3, 7]
    assert [type(v) for v in back['bounds']] == [float, float, int, int, int]
    assert all(type(v) is float for v in back['base'])
    np.testing.assert_array_equal(back['base'], base.astype(np.float64))      # the float32 values exactly, nan too
    assert back['times'] == table['times'] and all(t.utcoffset() == datetime.timedelta(0) for t in back['times'])
    for key in ('dates', 'radar_wavelength', 'day', 'local', 'text', 'a key', 'nested'):
        assert back[key] == table[key] and type(back[key]) is type(table[key])
    assert back['n'] == 3 and type(back['n']) is int and back['flags'] == [True, False]
    assert all(type(v) is bool for v in back['flags'])
    with pytest.raises(TypeError, match='no TOML value'):
        write_toml(tmp_path / 'bad.toml', {'a': None})


def test_download_dl_model(tmp_path, monkeypatch, capsys):
    """The models are written as served; a missing file (HTTP 404), a file cut short and a refused connection print
    an error, raise nothing and write no file; a missing output directory raises. A local HTTP server stands in for
    GitHub, and connections elsewhere fail the test: no network."""
    import http.server
    import socket
    import threading
    import urllib.request
    body = bytes(range(256)) * 40
    served = {'/kanglcn/n2f/refs/heads/main/n2f.pth': (body, len(body)),
              '/kanglcn/n2ft/refs/heads/main/n2ft.pth': (body[:100], len(body))}       # cut short
    requested = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            if self.path not in served:
                return self.send_error(404)
            content, length = served[self.path]
            self.send_response(200)
            self.send_header('Content-Length', str(length))
            self.end_headers()
            self.wfile.write(content)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        no_server = s.getsockname()[1]                  # a port nobody listens on
    port = [server.server_address[1]]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda url, *a, **k: opener.open(
        url.replace('https://raw.githubusercontent.com', f'http://127.0.0.1:{port[0]}'), *a, **k))
    getaddrinfo = socket.getaddrinfo

    def local_only(host, *args, **kwargs):
        assert host == '127.0.0.1', f'the test must not connect to {host}'
        return getaddrinfo(host, *args, **kwargs)
    monkeypatch.setattr(socket, 'getaddrinfo', local_only)
    try:
        mr.download_dl_model(path=tmp_path)
        assert requested == ['/kanglcn/n2f/refs/heads/main/n2f.pth', '/kanglcn/n2f/refs/heads/main/n2fs3d.pth',
                             '/kanglcn/n2ft/refs/heads/main/n2ft.pth']
        assert (tmp_path / 'n2f.pth').read_bytes() == body
        assert not (tmp_path / 'n2fs3d.pth').exists() and not (tmp_path / 'n2ft.pth').exists()
        out = capsys.readouterr().out
        assert "n2f.pth' downloaded successfully" in out and out.count('Error downloading file') == 2
        assert 'HTTP Error 404' in out and 'IncompleteRead' in out
        port[0] = no_server
        mr.download_dl_model('n2fs3d', path=tmp_path)
        assert 'Error downloading file' in capsys.readouterr().out and not (tmp_path / 'n2fs3d.pth').exists()
        port[0] = server.server_address[1]
        with pytest.raises(FileNotFoundError):
            mr.download_dl_model('n2f', path=tmp_path / 'no_such_directory')
    finally:
        server.shutdown()
        server.server_close()
