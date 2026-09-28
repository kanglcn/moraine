# 0003 Develop moraine as a plain Python package, not with nbdev

## Status

Accepted

## Date

2026-09-28

## Context

moraine was written with nbdev: the source was in notebooks, exported to `.py`. Tests only ran as
notebooks (several passed only because of notebook globals and hid real bugs), parameter docs were
inline comments that tools could not read, and packaging was split over settings.ini and setup.py.

## Decision

The `.py` files in `moraine/` are the source. Packaging is `pyproject.toml` (PEP 621). Functions have
numpy style docstrings. Tests are pytest in `tests/`, skipping what needs the sample data, a GPU, GAMMA
or the models. The API / CLI notebooks were removed; tutorial notebooks stay in `nbs/Tutorials`.

## Consequences

- Porting the notebook tests found and fixed bugs (CPU `ad_intf_pc`, `isPD` / `nearestPD` on numpy,
  `HilbertRtree.save` / `load`, `is_cuda_available`).
- Docstrings are also the input of the command line help (see 0005).

## Do not

- Do not reintroduce nbdev (`nbdev_export` would overwrite the source) or code generated from notebooks.
- Do not put tests in notebooks; tutorials are examples, not tests.
