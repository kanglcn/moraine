# 0033 The sample data set of the examples, guides and manual is Campi Flegrei

## Status

Accepted

## Date

2026-10-08

## Context

Decision 0009 verified the example pipelines on an anonymous sample data set (`data/gamma`, 2500 x 1834 x 17,
2021-2022), whose numbers filled the workflow guides. The manual (decision 0032) shows one data set to users:
the maintainer chose Campi Flegrei (Sentinel-1 descending track 22, 92 dates 2019-2021, 981 x 4160 pixels,
`data/CampiFlegrei/gamma`, described in `data/CampiFlegrei/README.md`): a known uplift with many PS and DS, which the
tutorials already use. Two data sets with different numbers would confuse users and agents.

## Decision

- The sample data set of the examples and guides is Campi Flegrei: the `[vars]` defaults of `examples/*.toml`
  (reference 20200707, geocoding prefix 20200707, shape 981 x 4160, pixel spacings 2.329562 m and 13.9516 m) and
  the expected results, run times and figures of `docs/workflows/` come from a run of the examples on it
  (first run: 2026-10-08).
- The manual uses no other data set: its figures, the recorded agent session and the tutorial pages are Campi
  Flegrei.
- `data/gamma` and `data/rslc.zarr` stay the data of the tests (`tests/conftest.py`); they document nothing.

## Consequences

- The rule of decision 0009 stands with the new data set: an example changes only after a run on Campi Flegrei,
  and the guide numbers are updated from that run.
- The Xinpu tutorials stay in `nbs/Tutorials/CLI/Xinpu/` for the maintainer but are not part of the manual.

## Do not

- Do not put numbers or figures of another data set into the guides or the manual.
- Do not change the `[vars]` defaults of the examples to another data set.
