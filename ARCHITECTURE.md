# Architecture

A map of the code: layers, what each module does and which way dependencies go. Read it before
changing code; `tests/test_architecture.py` checks that every module is listed here and that the
dependency rules hold. Design decisions are in `docs/decisions/`, interface promises in `docs/contracts/`.

## Layers

```
moraine/command/   `moraine` command line and pipelines      (command layer)
      │ uses
moraine/cli/       zarr in -> zarr out, chunk by chunk          (CLI layer)
      │ uses
moraine/api/       numpy / cupy arrays in memory              (API layer)
```

- **API** (`moraine/api/`): algorithms on arrays in memory, grouped by the stages of the processing chain
  (`ps`, `ds`, `dl`, `pqm`, `unwrap`, `ts`) and the foundations they share (`pc`, `tnet`, `polygon`, `slc`), the
  internal utilities in `moraine/api/utils_/` (decisions 0015, 0041). A module is one method or one data structure,
  a topic of several modules is a subpackage. `import moraine` re-exports the public names as `moraine.*`. Most
  functions accept numpy (CPU, numba) or cupy (GPU) arrays and return the same kind
  (`moraine.api.utils_.get_array_module`).
- **CLI** (`moraine/cli/`): the same operations on zarr datasets larger than memory, processed chunk by
  chunk by the workers of `moraine/cli/executor.py` (threads, processes or one process per GPU). Every function decorated with
  `@mc_logger` logs its arguments and is exposed as a command.
- **Command** (`moraine/command/`): the `moraine` executable. Commands and their help are generated from
  the CLI functions (decision 0005); pipelines run and resume chains of commands (decision 0006).

### Dependency rules

- The API layer does not import `moraine.cli` or `moraine.command`.
- The CLI layer does not import `moraine.command`.
- Inside the API (decision 0041): `moraine/api/utils_/` imports no other part of the API; a foundation (`pc`,
  `tnet`, `polygon`, `slc`) imports `utils_` and other foundations; a stage (`ps`, `ds`, `dl`, `pqm`, `unwrap`,
  `ts`) imports `utils_` and the foundations, never another stage. The API modules import each other relatively,
  never through `import moraine`.
- Exception: `moraine/__main__.py` is the `python -m moraine` entry point and imports `moraine.command`.
- GPU packages (`cupy`, `numba.cuda`) are imported only behind `moraine.api.utils_.is_cuda_available()`;
  `torch` only inside the functions that run a model (decision 0002).

## API layer (moraine/api/)

| module | responsibility | main names |
|---|---|---|
| `moraine/__init__.py` | version; re-exports the public API as `moraine.*` | |
| `moraine/__main__.py` | `python -m moraine` entry point | |
| `moraine/api/__init__.py` | re-exports the public names of the foundations and stages below (`__all__`) | |
| *internal utilities* | | |
| `moraine/api/utils_/__init__.py` | internal utilities, not re-exported as `moraine.*`; re-exports the names of `jit.py` and `device.py` as `moraine.api.utils_.*` | |
| `moraine/api/utils_/jit.py` | numba decorators with the compiled code cached by source hash (decision 0029) | `mjit`, `ngjit`, `ngpjit`, `mcuda_jit` |
| `moraine/api/utils_/device.py` | GPU detection, numpy / cupy dispatch, available cores and memory | `is_cuda_available`, `get_array_module`, `get_mem_avail` |
| `moraine/api/utils_/chunk.py` | slices and halos for chunkwise processing of rasters and point clouds | `chunkwise_slicing_mapping`, `chunkwise_knn_mapping` |
| `moraine/api/utils_/coord.py` | regular coordinate grid, coordinate <-> grid index, rasterization | `Coord` |
| `moraine/api/utils_/gamma_io.py` | read / write GAMMA binary files | `read_gamma_image` |
| `moraine/api/utils_/orbit.py` | perpendicular baselines at points on the ground from the orbit state vectors, independent of the SAR processor | `bperp` |
| `moraine/api/utils_/expr.py` | elementwise expressions of named arrays, the `operation` of `math`, `pc-logic-ras`, `pc-logic-pc` (decision 0045) | `evaluate` |
| *foundations* | | |
| `moraine/api/pc/__init__.py` | point clouds; re-exports the names of the modules below | |
| `moraine/api/pc/index.py` | grid and hilbert indices of point clouds, sorting | `pc_hix`, `pc_gix`, `pc_sort` |
| `moraine/api/pc/sets.py` | set operations of point clouds | `pc_union`, `pc_intersect`, `pc_diff` |
| `moraine/api/pc/raster.py` | raster <-> point cloud, points in the order of raster chunks | `ras2pc`, `pc2ras` |
| `moraine/api/pc/rtree.py` | hilbert R-tree for bounding box queries on point clouds | `HilbertRtree` |
| `moraine/api/tnet.py` | temporal network: image pairs | `TempNet` |
| `moraine/api/polygon.py` | polygon GeoJSON files (longitude / latitude or radar grid), point in polygon | `read_polygons`, `write_polygons`, `polygons_contain` |
| `moraine/api/slc.py` | basic operations on SLC stacks: amplitude, amplitude calibration, interferograms, multilooking | `rslc2amp`, `calamp`, `intf` |
| *stages* | | |
| `moraine/api/ps.py` | amplitude dispersion index for PS selection | `amp_disp` |
| `moraine/api/ds/__init__.py` | distributed scatterers; re-exports the names of `shp.py`, `co.py` and `pl.py` | |
| `moraine/api/ds/shp.py` | statistically homogeneous pixels (KS test) | `ks_test`, `select_shp` |
| `moraine/api/ds/co.py` | covariance / coherence matrices and the effective number of looks of the SHPs, positive definiteness, regularization | `emperical_co_pc`, `uncompress_coh` |
| `moraine/api/ds/pl.py` | phase linking (EMI) and DS temporal coherence | `emi`, `ds_temp_coh` |
| `moraine/api/ds/emi_cuda_.py` | GPU kernel of the regularized EMI (imported by `pl.py` only with a GPU) | |
| `moraine/api/ds/linalg_.py` | small dense linear algebra of the phase linking on the CPU without LAPACK: Householder tridiagonalization, Sturm multisection, inverse iteration, Gauss-Jordan inverse | |
| `moraine/api/dl/__init__.py` | deep learning filters; re-exports the names of `model.py`, `raster.py` and `pc.py` | |
| `moraine/api/dl/model.py` | model download, cached loading | `download_dl_model` |
| `moraine/api/dl/raster.py` | n2f / n2fs3d filtering of raster interferograms | `n2f`, `n2fs3d` |
| `moraine/api/dl/pc.py` | n2ft filtering of point cloud interferograms | `n2ft` |
| `moraine/api/dl/unet_torch_.py` | UNet of n2f / n2fs3d (must match the published weights) | `UNet` |
| `moraine/api/dl/n2ft_torch_.py` | point transformer of n2ft (must match the published weights) | `N2FT` |
| `moraine/api/pqm.py` | pixel quality: temporal coherence | `temp_coh` |
| `moraine/api/unwrap/__init__.py` | phase unwrapping; re-exports `mcf_pc`, `emcf_pc`, `unwrap_correct_closure_pc`, `gamma_mcf_pt` | |
| `moraine/api/unwrap/delaunay_.py` | 2D Delaunay triangulation (sweep-hull, numba, exact for integer coordinates) as half-edges | `delaunay_halfedges`, `delaunay` |
| `moraine/api/unwrap/mcf.py` | minimum cost flow (successive shortest paths) on the Delaunay half-edges, unwrapping of one interferogram | `mcf_pc` |
| `moraine/api/unwrap/emcf.py` | extended minimum cost flow (EMCF) of any network of interferograms | `emcf_pc` |
| `moraine/api/unwrap/closure.py` | correction of unwrapping errors by phase closure per region (MintPy-like) to the unwrapped image phases; L1 fit on a network of image pairs | `unwrap_correct_closure_pc` |
| `moraine/api/unwrap/gamma.py` | wrapper of GAMMA `mcf_pt` | `gamma_mcf_pt` |
| `moraine/api/ts.py` | deformation products of point clouds (decision 0040): spatial reference, displacement | `pc_reference`, `pc_disp` |
| `moraine/dl_model/__init__.py` | package directory where the `.pth` models are downloaded (data, not API: it stays in place so downloaded models remain valid) | |

## CLI layer (moraine/cli/)

| module | responsibility | commands |
|---|---|---|
| `moraine/cli/__init__.py` | re-exports the CLI functions as `moraine.cli.*` | |
| `moraine/cli/logging.py` | `@mc_logger` (argument logging, marks a function as a command), zarr log helper | |
| `moraine/cli/zarr_.py` | parallel zarr read / write, directories of per chunk zarrs | |
| `moraine/cli/executor.py` | `Executor`: the workers of a command (threads of the process, spawned processes, one per GPU) running tasks on zarr chunks (`Chunk`, `chunk_task`) | |
| `moraine/cli/utils_.py` | small helpers (clean output directories) | |
| `moraine/cli/load.py` | load GAMMA results (runs GAMMA programs; the perpendicular baselines from the orbits of the par files) | `load-gamma-*` |
| `moraine/cli/transform.py` | coordinate transformation (pyproj) | `transform` |
| `moraine/cli/tnet.py` | image pair files | `image-pairs` |
| `moraine/cli/math.py` | elementwise expressions of zarr arrays | `math` |
| `moraine/cli/polygon.py` | masks of rasters / point clouds from polygon files | `polygon-mask` |
| `moraine/cli/pc.py` | point cloud operations on zarr | `ras2pc`, `pc-union`, `pc-sort`, ... |
| `moraine/cli/ps.py` | amplitude dispersion index | `amp-disp` |
| `moraine/cli/shp.py` | SHP test and selection | `shp-test`, `select-shp` |
| `moraine/cli/co.py` | coherence of point clouds per raster chunk | `emperical-co-pc` |
| `moraine/cli/pl.py` | phase linking, DS temporal coherence, fused coherence + phase linking | `emi`, `ds-temp-coh`, `emperical-co-emi-temp-coh-pc` |
| `moraine/cli/pqm.py` | temporal coherence | `temp-coh` |
| `moraine/cli/pu.py` | point cloud unwrapping | `mcf-pc`, `emcf-pc`, `unwrap-correct-closure-pc`, `gamma-mcf-pt` |
| `moraine/cli/ts.py` | deformation products of point clouds (decision 0040) | `pc-reference`, `pc-disp` |
| `moraine/cli/dl.py` | n2f / n2ft filtering of stacks | `n2f`, `n2ft` |
| `moraine/cli/plot.py` | pyramids of rasters / point clouds, pyramid reading and the rules to show them | `ras-pyramid`, `pc-pyramid` |
| `moraine/cli/tiles.py` | views (`view`): rasters and point clouds from pyramids or arrays, tiles and PNG images, value / point / time series at a position, `*` overlay and `+` layout, text description | (notebook, no command) |
| `moraine/cli/viewer.py` | the notebook widget of views (`TileView`; anywidget + Leaflet, or deck.gl over the terrain in 3D; tiles rendered by the kernel): maps linked in zoom and pan, merged sliders, time series, reference, polygons; `viewer.js` / `.css` beside it | (notebook, no command) |

## Command layer (moraine/command/)

| module | responsibility |
|---|---|
| `moraine/command/__init__.py` | command registry generated from `moraine.cli`, argument parsing and validation, `execute`, `--json` output, `main` |
| `moraine/command/pipeline.py` | TOML pipeline loading (`[vars]`, `[defaults]`, `[[step]]`), planning, resume, state in `.moraine/<file name>/` |
| `moraine/command/summary.py` | `info` metadata and pyramid statistics, `quicklook` (PNG) and `view` (notebook of interactive maps) through `moraine.cli.view` |

## Other parts of the repository

| path | content |
|---|---|
| `tests/` | pytest; `conftest.py` has the sample data fixtures and the `gpu` / `slow` markers |
| `tests/browser/` | the viewer's JavaScript in jsdom and in headless Chromium, run by hand (not pytest) |
| `examples/` | verified pipelines of the whole processing chain (decision 0009) |
| `docs/workflows/` | one guide per example pipeline |
| `docs/decisions/` | design decision records |
| `docs/contracts/` | promises on formats others depend on (JSON output, pipeline files, pyramids, data) |
| `docs/development.md` | how to change moraine |
| `docs/viewer.md` | how the viewer works (tiles, widget, 2D and 3D front end), what was learnt, its tests and open items |
| `docs/roadmap.md` | planned features not started yet |
| `nbs/Tutorials/` | tutorial notebooks (examples, not tests) |
| `data/` | sample data (not in git) |
