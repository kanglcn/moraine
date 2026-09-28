"""Keep AGENTS.md, docs/workflows and examples/ in sync with the code."""
import re
from pathlib import Path

import pytest

from moraine.command import commands
from moraine.command.pipeline import load_pipeline

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / 'AGENTS.md', REPO / 'README.md', REPO / 'ARCHITECTURE.md',
        *sorted((REPO / 'docs' / 'workflows').glob('*.md')),
        *sorted((REPO / 'docs' / 'decisions').glob('*.md')),
        *sorted((REPO / 'docs' / 'contracts').glob('*.md')), REPO / 'docs' / 'development.md',
        REPO / '.claude' / 'skills' / 'moraine-processing' / 'SKILL.md']
BUILTIN = {'list', 'info', 'quicklook', 'run', 'status', 'COMMAND', 'FILE'}
EXAMPLES = sorted((REPO / 'examples').glob('*.toml'))


@pytest.mark.parametrize('doc', DOCS, ids=lambda p: p.name)
def test_documented_commands_exist(doc):
    names = set(commands()) | BUILTIN
    text = doc.read_text()
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
