# 0041 The API is grouped by processing stage

## Status

Accepted

## Date

2026-11-10

## Context

Decision 0015 made `moraine/api/` a package grouped by topic, with a subpackage for a topic of several modules, but
only phase unwrapping had become one: 19 modules sat side by side, internal helpers next to the algorithms, and some
modules held several separable parts (`dl.py` 744 lines: model download, n2f / n2fs3d, n2ft; `pc.py` 611 lines:
indices, set operations, raster <-> point cloud). Dependencies were hidden: modules called each other through
`import moraine as mr` (`mr.TempNet`, `mr.emperical_co_pc`), `co.py` and `pl.py` imported each other, and `moraine.*`
also held module objects (`mr.co`, `mr.gamma` was `unwrap/gamma.py`). The examples, guides, manual and commands follow
the processing chain (02_ps ... 06_products), and the stages pass arrays to each other.

## Decision

- The API is grouped by the stages of the processing chain and the foundations they share:
  - internal utilities, not re-exported as `moraine.*`: `moraine/api/utils_/` (numba decorators, CPU and GPU of the
    process, chunks, raster coordinates, GAMMA files);
  - foundations: `pc/` (point cloud indices, sorting, set operations, raster <-> point cloud, R-tree), `tnet.py`
    (image pairs), `polygon.py`, `slc.py` (amplitude, amplitude calibration, interferograms, multilooking);
  - stages: `ps.py` (amplitude dispersion), `ds/` (SHP, coherence matrices, phase linking, DS temporal coherence),
    `dl/` (deep learning filters), `pqm.py` (temporal coherence), `unwrap/`, `ts.py` (deformation products).
- A module is one method or one data structure; new code goes to the stage or foundation of its method. A topic of
  one module stays a file and becomes a subpackage with its second module (decision 0015).
- Dependencies go one way: `utils_` imports no other part of the API; a foundation imports `utils_` and other
  foundations; a stage imports `utils_` and the foundations, never another stage. Stages pass arrays to each other and
  are chained by the CLI and the pipelines; what two stages need is a foundation.
- The API modules import each other relatively, never through `import moraine`.
- Every `__init__.py` lists its public names in `__all__`, so `moraine.*` holds exactly the public functions and
  classes of the API. A submodule is not named like a name its package exports (it would be hidden by it).
- The public names (`moraine.*`), `moraine.cli`, the commands and the contracts do not change; module paths do, without
  aliases (as in decision 0015). `tests/test_architecture.py` checks the rules.

## Consequences

- `moraine.api.<module>` paths changed, e.g. `moraine.api.co` -> `moraine.api.ds.co`, `moraine.api.chunk_` ->
  `moraine.api.utils_.chunk`, the functions of `moraine.api.dl` in `moraine.api.dl.model`, `.raster` and `.pc`; code
  using `moraine.*`, `moraine.cli.*` or the public names of `moraine.api.utils_` does not change.
- The manual has one API page per stage or foundation with a section per module.
- The CLI layer (`moraine/cli/`) is not regrouped; the speckle correlation and effective number of looks of
  `ds/co.py` stay in that module.

## Do not

- Do not import one stage from another; move what both need into a foundation (or `utils_`).
- Do not `import moraine` inside `moraine/api/`.
- Do not make a subpackage for a topic of one module.
- Do not name a submodule like a function or class of its package (`dl/n2f.py` would be hidden by the function `n2f`).
