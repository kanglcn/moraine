"""ARCHITECTURE.md lists every module, and the layer dependency rules hold."""
import ast
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
