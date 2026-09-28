"""The design decision records in docs/decisions follow the format of docs/decisions/README.md."""
import re
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / 'docs' / 'decisions'
RECORDS = sorted(p for p in DIR.glob('[0-9][0-9][0-9][0-9]-*.md'))
SECTIONS = ['Status', 'Date', 'Context', 'Decision', 'Consequences', 'Do not']


def test_numbers_are_unique_and_consecutive():
    numbers = [int(p.name[:4]) for p in RECORDS]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def test_index_lists_every_record():
    index = (DIR / 'README.md').read_text()
    linked = set(re.findall(r'\]\((\d{4}-[\w-]+\.md)\)', index))
    assert linked == {p.name for p in RECORDS}


@pytest.mark.parametrize('record', RECORDS, ids=lambda p: p.name)
def test_record_format(record):
    text = record.read_text()
    assert text.startswith(f'# {record.name[:4]} '), 'first line must be "# NNNN Title"'
    headings = re.findall(r'^## (.+)$', text, re.M)
    assert headings == SECTIONS, headings
    body = dict(zip(headings, re.split(r'^## .+$', text, flags=re.M)[1:]))
    status = body['Status'].strip()
    assert re.fullmatch(r'Proposed|Accepted|Deprecated|Superseded by \d{4}', status), status
    if status.startswith('Superseded'):
        assert any(p.name.startswith(status[-4:]) for p in RECORDS), f'{status}: no such record'
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', body['Date'].strip())
    index = (DIR / 'README.md').read_text()
    row = next(l for l in index.splitlines() if f']({record.name})' in l)
    assert row.rstrip(' |').endswith(status), f'index status differs from the record: {row}'


def test_agents_md_points_to_the_decisions():
    agents = (DIR.parents[1] / 'AGENTS.md').read_text()
    assert 'docs/decisions/README.md' in agents
