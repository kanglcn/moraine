# Design decisions

Architecture decision records (ADR): one file per decision that shapes how moraine is built or used,
with its reasons and what must not be done because of it. Agents and contributors read this index
before changing the design; a decision is changed by a new record, not by editing code against it.

## Index

| # | decision | status |
|---|---|---|
| [0001](0001-record-decisions.md) | Record design decisions as ADRs in docs/decisions | Accepted |
| [0002](0002-pytorch-for-deep-learning.md) | Deep learning inference uses PyTorch only | Accepted |
| [0003](0003-plain-python-no-nbdev.md) | Develop moraine as a plain Python package, not with nbdev | Accepted |
| [0004](0004-no-docs-website.md) | No documentation website for now | Accepted |
| [0005](0005-cli-generated-from-docstrings.md) | The command line is generated from moraine.cli signatures and docstrings | Accepted |
| [0006](0006-toml-pipelines.md) | Processing chains are TOML pipelines that resume | Accepted |
| [0007](0007-visualization-through-pyramids.md) | Visualization and result statistics go through pyramids | Accepted |
| [0008](0008-tool-independent-agent-docs.md) | Agent documentation is tool independent and tested | Accepted |
| [0009](0009-verified-examples.md) | Example pipelines are verified on real data | Accepted |
| [0010](0010-versioned-contracts.md) | Formats others depend on are versioned contracts | Accepted |
| [0011](0011-architecture-map.md) | The module map and layer rules are tested | Accepted |

## Writing a record

- File name `NNNN-short-title.md`, the next free number; add it to the index in the same change.
- Sections, in this order: the title line `# NNNN Title`, then `Status`, `Date`, `Context`, `Decision`,
  `Consequences`, `Do not`.
- Status is one of `Proposed`, `Accepted`, `Superseded by NNNN`, `Deprecated`.
- To change a decision, write a new record and mark the old one `Superseded by NNNN`; keep the old file.
- Keep records short: the reason and the rule matter, the history is in git.
- `tests/test_decisions.py` checks the index, the sections and the status of every record.
