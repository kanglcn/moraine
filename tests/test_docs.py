"""Keep AGENTS.md, docs/ (the manual, decision 0032) and examples/ in sync with the code."""
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from moraine.command import commands
from moraine.command.pipeline import load_pipeline

REPO = Path(__file__).resolve().parents[1]
# every handwritten markdown file of the manual (docs/) and the agent / contributor files
DOCS = [REPO / 'AGENTS.md', REPO / 'README.md', REPO / 'ARCHITECTURE.md',
        *sorted(p for p in (REPO / 'docs').rglob('*.md') if 'overrides' not in p.parts
                and not (p.parent.name in ('cli', 'api') and p.name.split('.')[0] not in ('index', 'cli'))
                and 'CampiFlegrei' not in p.parts),      # not the pages generated at build time
        REPO / '.claude' / 'skills' / 'moraine-processing' / 'SKILL.md',
        REPO / '.claude' / 'skills' / 'moraine-development' / 'SKILL.md']
BUILTIN = {'list', 'info', 'quicklook', 'view', 'run', 'status', 'COMMAND', 'FILE'}
EXAMPLES = sorted((REPO / 'examples').glob('*.toml'))


@pytest.mark.parametrize('doc', DOCS, ids=lambda p: p.name)
def test_documented_commands_exist(doc):
    names = set(commands()) | BUILTIN
    text = doc.read_text()
    if doc.parent.name == 'decisions':
        pytest.skip('decision records may name planned commands')
    code = '\n'.join(re.findall(r'```.*?\n(.*?)```', text, re.S))       # fenced code blocks
    used = set(re.findall(r'`moraine ([a-z][a-z0-9-]*)', text))
    used |= set(re.findall(r'^\s*moraine ([a-z][a-z0-9-]*)', code, re.M))
    used |= set(re.findall(r'`([a-z0-9]+(?:-[a-z0-9]+)+)`', text))      # `amp-disp` style names
    unknown = sorted(u for u in used if u not in names and not u.startswith(('load-gamma', 'moraine-')))
    assert not unknown, f'{doc.name} mentions unknown commands: {unknown}'


@pytest.mark.parametrize('doc', DOCS, ids=lambda p: p.name)
def test_documented_repo_files_exist(doc):
    refs = set(re.findall(r'`((?:examples|docs|nbs/Tutorials/CLI|moraine|tests)/[\w./-]+\.(?:toml|md|ipynb|py))`',
                          doc.read_text()))
    missing = sorted(r for r in refs if not (REPO / r).exists())
    assert not missing, f'{doc.name} refers to missing files: {missing}'


def test_every_example_has_a_guide():
    guides = {p.stem for p in (REPO / 'docs' / 'workflows').glob('*.md')}
    assert {p.stem for p in EXAMPLES} <= guides


@pytest.mark.parametrize('example', EXAMPLES, ids=lambda p: p.name)
def test_examples_parse(example, tmp_path):
    pipe = load_pipeline(str(example), workdir=str(tmp_path), variables={'gamma': str(tmp_path)})
    assert pipe['steps']


# ---------------------------------------------------------------- the manual (decision 0032)

def _refgen():
    spec = importlib.util.spec_from_file_location('refgen', REPO / 'docs' / '_scripts' / 'refgen.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_covers_every_command():
    pages = _refgen().cli_pages()
    documented = set()
    for text, _ in pages.values():
        documented |= set(re.findall(r'^## ([a-z0-9-]+)$', text, re.M))
    assert documented == set(commands()), documented ^ set(commands())
    for mod in {c.module for c in commands().values()}:
        assert f'cli/{mod}.md' in pages


def test_cli_reference_shows_examples_and_io():
    pages = _refgen().cli_pages()
    text = pages['cli/ps.md'][0]
    assert 'examples/02_ps.toml' in text and 'run = "amp-disp"' in text
    assert 'mo-io--in' in text and 'mo-io--out' in text


def test_api_reference_covers_every_public_name():
    refgen = _refgen()
    members = refgen.api_members()
    for item in refgen.API_MODULES:
        name, subs = item if isinstance(item, tuple) else (item, None)
        mods = [f'moraine.api.{name}.{s}' for s in subs] if subs else [f'moraine.api.{name}']
        public = set()
        for m in mods:
            public |= set(__import__(m, fromlist=['__all__']).__all__)
        assert set(members[f'api/{name.rstrip("_")}.md']) == public, name


def test_tutorial_pages_are_the_notebooks():
    pages = _refgen().tutorial_pages()
    assert len(pages) == len(_refgen().TUTORIALS), 'a tutorial notebook is missing'
    text = pages['tutorials/CampiFlegrei/02_ps.md'][0]
    assert text.startswith('# PS Processing') and '```python' in text and '.ipynb)' not in text.split('!!! info')[1][300:]


@pytest.mark.slow
def test_manual_builds(tmp_path):
    if shutil.which('mkdocs') is None and importlib.util.find_spec('mkdocs') is None:
        pytest.skip('mkdocs is not installed (pip install -e ".[docs]")')
    out = subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '--strict', '-d', str(tmp_path / 'site')],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-3000:]
    assert (tmp_path / 'site' / 'cli' / 'ps' / 'index.html').exists()
