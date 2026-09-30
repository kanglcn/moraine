# 0015 The API layer is the package moraine.api, grouped by topic

## Status

Accepted

## Date

2026-09-29

## Context

The API modules were files directly in `moraine/`, next to the `cli` and `command` packages. The layer of
a module was not visible from its path, and a topic with several modules (phase unwrapping: Delaunay
triangulation, MCF, EMCF, GAMMA wrapper) either grew into one large file (`pu.py`, now split into `mcf.py`, `emcf.py` and `gamma.py`) or spread over
unrelated names in `moraine/`. More unwrapping and inversion code is planned (decision 0012).

## Decision

- The API layer is the package `moraine/api/`; `moraine/cli/` and `moraine/command/` stay where they are.
- A topic with several modules is a subpackage of `moraine/api/`, starting with `moraine/api/unwrap/`
  (`delaunay_.py`, `mcf.py`, `emcf.py`, `gamma.py`); single module topics stay files in `moraine/api/`.
- `moraine/api/__init__.py` and each subpackage `__init__.py` re-export their public names; `import moraine`
  re-exports `moraine.api`, so `moraine.mcf_pc` etc. are unchanged.
- `moraine/dl_model/` (where the models are downloaded) stays in place: it holds data, not code.
- There are no compatibility aliases for the old module paths (`moraine.pu`, `moraine.pc`, ...).
- File paths in earlier records are updated to the new locations (the decisions themselves are unchanged).

## Consequences

- Code importing modules directly changes (`from moraine.pu import ...` -> `from moraine.api.unwrap import ...`);
  code using `moraine.*` or `moraine.cli.*` does not.
- New API code goes into the topic's module or subpackage under `moraine/api/`, listed in `ARCHITECTURE.md`.

## Do not

- Do not add API modules directly in `moraine/`.
- Do not grow one module per topic past a few hundred lines when it has separable parts; make a subpackage.
