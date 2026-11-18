"""ARCHITECTURE.md lists every module, and the layer dependency rules hold."""
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = sorted(p.relative_to(REPO).as_posix() for p in (REPO / 'moraine').rglob('*.py')
                 if '.ipynb_checkpoints' not in p.parts and '__pycache__' not in p.parts)
ENTRY_POINTS = {'moraine/__main__.py'}


def _layer(module):
    if module.startswith('moraine/command/') or module in ('moraine.command',) or module.startswith('moraine.command'):
        return 'command'
    if module.startswith('moraine/cli/') or module.startswith('moraine.cli'):
        return 'cli'
    return 'api'


def _imports(path):
    """Absolute module names imported by `path` (relative imports resolved)."""
    tree = ast.parse((REPO / path).read_text())
    package = path[:-3].split('/')[:-1] if not path.endswith('__init__.py') else path[:-3].split('/')[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                name = '.'.join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ''
            yield node.lineno, name


def test_every_module_is_in_the_map():
    listed = set(re.findall(r'`(moraine/[\w/]+\.py)`', (REPO / 'ARCHITECTURE.md').read_text()))
    assert not set(MODULES) - listed, f'add to ARCHITECTURE.md: {sorted(set(MODULES) - listed)}'
    assert not listed - set(MODULES), f'no such module: {sorted(listed - set(MODULES))}'


@pytest.mark.parametrize('module', MODULES)
def test_layer_dependencies(module):
    if module in ENTRY_POINTS:
        return
    src = _layer(module)
    for lineno, name in _imports(module):
        if not name.startswith('moraine'):
            continue
        dst = _layer(name)
        assert not (src == 'api' and dst in ('cli', 'command')), f'{module}:{lineno} API imports {name}'
        assert not (src == 'cli' and dst == 'command'), f'{module}:{lineno} CLI imports {name}'


def test_import_moraine_does_not_import_torch():
    code = 'import sys, moraine, moraine.cli, moraine.command; print("torch" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip().endswith('False'), out.stdout + out.stderr


def test_import_moraine_does_not_import_dask():
    """dask is not a dependency any more (decision 0034)."""
    code = 'import sys, moraine, moraine.cli, moraine.command; print("dask" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip().endswith('False'), out.stdout + out.stderr


def test_import_moraine_does_not_import_numexpr_or_pandas():
    """numexpr starts its threads when it is imported, pandas imports numexpr where it is installed (decision 0045)."""
    code = ('import sys, moraine, moraine.cli, moraine.command; '
            'print([m for m in ("numexpr", "pandas") if m in sys.modules])')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip().endswith('[]'), out.stdout + out.stderr


def test_import_moraine_does_not_import_cupy():
    """With a GPU visible, cupy (which imports pytest) is imported by the GPU code paths only: not by `import moraine`
    and not by the dispatch of numpy arrays. zarr imports cupy for its GPU buffers wherever cupy is installed, so
    moraine imports zarr with cupy hidden (decision 0050) and restores sys.modules: cupy imports afterwards."""
    code = ('import sys, importlib.util, numpy, moraine, moraine.cli, moraine.command\n'
            'from moraine.api.utils_ import get_array_module\n'
            'assert get_array_module(numpy.zeros(1)) is numpy\n'
            'print([m for m in ("cupy", "pytest") if m in sys.modules])\n'
            'if importlib.util.find_spec("cupy"):\n'
            '    import cupy\n'
            '    assert sys.modules["cupy"] is cupy and hasattr(cupy, "ndarray")\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    assert out.returncode == 0 and out.stdout.strip().endswith('[]'), out.stdout + out.stderr


def test_import_moraine_does_not_import_requests():
    """requests is not a dependency any more: the deep learning models are downloaded with urllib"""
    code = 'import sys, moraine, moraine.cli, moraine.command; print("requests" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip().endswith('False'), out.stdout + out.stderr


# ---------------------------------------------------------------- inside the API (decision 0041)

API_FOUNDATIONS = {'pc', 'tnet', 'polygon', 'slc'}
API_STAGES = {'ps', 'ds', 'dl', 'pqm', 'unwrap', 'tropo', 'tide', 'ts'}


def _api_part(name):
    """the part of the API (the first name below moraine.api, e.g. 'ds') of a module path or a dotted module name;
    None outside the API and for moraine/api/__init__.py"""
    parts = name[:-3].split('/') if name.endswith('.py') else name.split('.')
    if parts[:2] != ['moraine', 'api'] or len(parts) < 3 or parts[2] == '__init__':
        return None
    return parts[2]


@pytest.mark.parametrize('module', [m for m in MODULES if _api_part(m)])
def test_api_dependency_direction(module):
    """utils_ <- foundations <- stages: a stage does not import another stage"""
    src = _api_part(module)
    assert src in {'utils_'} | API_FOUNDATIONS | API_STAGES, f'{module}: add {src} to API_FOUNDATIONS or API_STAGES'
    allowed = {'utils_'} if src == 'utils_' else {src, 'utils_'} | API_FOUNDATIONS
    for lineno, name in _imports(module):
        dst = _api_part(name)
        assert dst is None or dst in allowed, f'{module}:{lineno} ({src}) imports {name} ({dst})'


@pytest.mark.parametrize('module', [m for m in MODULES if m.startswith('moraine/api/')])
def test_api_imports_are_relative(module):
    """the API modules import each other relatively, never through `import moraine`"""
    for node in ast.walk(ast.parse((REPO / module).read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or '']
        else:
            continue
        bad = [n for n in names if n == 'moraine' or n.startswith('moraine.')]
        assert not bad, f'{module}:{node.lineno} imports {bad[0]}'


def test_moraine_names_are_the_public_api():
    """moraine.* holds the public functions and classes of the foundations and stages, and no modules"""
    import moraine
    parts = [importlib.import_module(f'moraine.api.{p}') for p in sorted(API_FOUNDATIONS | API_STAGES)]
    names = [n for m in parts for n in m.__all__]
    assert len(names) == len(set(names)), 'a name is exported by two parts of the API'
    assert sorted(moraine.__all__) == sorted(names)
    assert not [n for n in names if inspect.ismodule(getattr(moraine, n))]


@pytest.mark.parametrize('package', sorted({m.split('/')[2] for m in MODULES
                                            if m.startswith('moraine/api/') and m.count('/') == 3}))
def test_submodules_are_not_hidden(package):
    """a submodule is not named like a name its package exports (the name would hide the module)"""
    pkg = importlib.import_module(f'moraine.api.{package}')
    stems = {p.stem for p in (REPO / 'moraine' / 'api' / package).glob('*.py')}
    assert not stems & set(pkg.__all__)
