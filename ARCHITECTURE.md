# Architecture

A map of the code: layers, what each module does and which way dependencies go. Read it before
changing code; `tests/test_architecture.py` checks that every module is listed here and that the
dependency rules hold. Design decisions are in `docs/decisions/`, interface promises in `docs/contracts/`.

## Layers

```
moraine/command/   `moraine` command line and pipelines      (command layer)
      │ uses
moraine/cli/       zarr in -> zarr out, chunked with dask     (CLI layer)
      │ uses
moraine/*.py       numpy / cupy arrays in memory              (API layer)
```

- **API** (`moraine/*.py`): algorithms on arrays in memory. Most functions accept numpy (CPU, numba)
  or cupy (GPU) arrays and return the same kind (`moraine.utils_.get_array_module`).
- **CLI** (`moraine/cli/`): the same operations on zarr datasets larger than memory, processed chunk by
  chunk with dask (CPU `LocalCluster` or `LocalCUDACluster`). Every function decorated with
  `@mc_logger` logs its arguments and is exposed as a command.
- **Command** (`moraine/command/`): the `moraine` executable. Commands and their help are generated from
  the CLI functions (decision 0005); pipelines run and resume chains of commands (decision 0006).

### Dependency rules

- The API layer does not import `moraine.cli` or `moraine.command`.
- The CLI layer does not import `moraine.command`.
- Exception: `moraine/__main__.py` is the `python -m moraine` entry point and imports `moraine.command`.
- GPU packages (`cupy`, `dask_cuda`, `rmm`) are imported only behind `moraine.utils_.is_cuda_available()`;
  `torch` only inside the functions that run a model (decision 0002).

## API layer (moraine/)

| module | responsibility | main names |
|---|---|---|
| `moraine/__init__.py` | re-exports the public API as `moraine.*` | |
| `moraine/__main__.py` | `python -m moraine` entry point | |
| `moraine/utils_.py` | numba decorators, GPU detection, numpy / cupy dispatch | `ngjit`, `is_cuda_available`, `get_array_module` |
| `moraine/chunk_.py` | slices and halos for chunkwise processing of rasters and point clouds | `chunkwise_slicing_mapping`, `chunkwise_knn_mapping` |
| `moraine/coord_.py` | regular coordinate grid, coordinate <-> grid index, rasterization | `Coord` |
| `moraine/gamma_.py` | read / write GAMMA binary files | `read_gamma_image` |
| `moraine/tnet.py` | temporal network: image pairs | `TempNet` |
| `moraine/polygon.py` | polygon GeoJSON files (longitude / latitude or radar grid), point in polygon | `read_polygons`, `write_polygons`, `polygons_contain` |
| `moraine/pc.py` | point cloud indices (grid / hilbert), sorting, set operations, raster <-> point cloud | `pc_hix`, `pc_union`, `ras2pc` |
| `moraine/rtree.py` | hilbert R-tree for bounding box queries on point clouds | `HilbertRtree` |
| `moraine/calamp.py` | SLC amplitude and amplitude calibration | `rslc2amp`, `calamp` |
| `moraine/ps.py` | amplitude dispersion index for PS selection | `amp_disp` |
| `moraine/shp.py` | statistically homogeneous pixels (KS test) | `ks_test`, `select_shp` |
| `moraine/co.py` | interferograms, covariance / coherence matrices, positive definiteness, regularization | `emperical_co_pc`, `uncompress_coh` |
| `moraine/pl.py` | phase linking (EMI) and DS temporal coherence | `emi`, `ds_temp_coh` |
| `moraine/pqm.py` | pixel quality: temporal coherence | `temp_coh` |
| `moraine/delaunay_.py` | 2D Delaunay triangulation (sweep-hull, numba, exact for integer coordinates) as half-edges | `delaunay_halfedges`, `delaunay` |
| `moraine/pu.py` | phase unwrapping: minimum cost flow (successive shortest paths) on the Delaunay half-edges, GAMMA wrapper | `mcf_pc`, `gamma_mcf_pt` |
| `moraine/dl.py` | deep learning filters, model download and cached loading | `n2f`, `n2fs3d`, `n2ft`, `download_dl_model` |
| `moraine/unet_torch_.py` | UNet of n2f / n2fs3d (must match the published weights) | `UNet` |
| `moraine/n2ft_torch_.py` | point transformer of n2ft (must match the published weights) | `N2FT` |
| `moraine/dl_model/__init__.py` | package directory where the `.pth` models are downloaded | |
| `moraine/plot.py` | interactive holoviews plots of in-memory rasters / point clouds / time series | `ras_plot`, `pc_plot`, `ts_plot` |

## CLI layer (moraine/cli/)

| module | responsibility | commands |
|---|---|---|
| `moraine/cli/__init__.py` | re-exports the CLI functions as `moraine.cli.*` | |
| `moraine/cli/logging.py` | `@mc_logger` (argument logging, marks a function as a command), zarr / dask log helpers | |
| `moraine/cli/dask_.py` | parallel zarr read / write, zarr <-> dask arrays | |
| `moraine/cli/utils_.py` | small helpers (clean output directories) | |
| `moraine/cli/load.py` | load GAMMA results (runs GAMMA programs) | `load-gamma-*` |
| `moraine/cli/transform.py` | coordinate transformation (pyproj) | `transform` |
| `moraine/cli/tnet.py` | image pair files | `image-pairs` |
| `moraine/cli/math.py` | elementwise numexpr expressions on zarr arrays | `math` |
| `moraine/cli/polygon.py` | masks of rasters / point clouds from polygon files | `polygon-mask` |
| `moraine/cli/pc.py` | point cloud operations on zarr | `ras2pc`, `pc-union`, `pc-sort`, ... |
| `moraine/cli/ps.py` | amplitude dispersion index | `amp-disp` |
| `moraine/cli/shp.py` | SHP test and selection | `shp-test`, `select-shp` |
| `moraine/cli/co.py` | coherence of point clouds per raster chunk | `emperical-co-pc` |
| `moraine/cli/pl.py` | phase linking, DS temporal coherence, fused coherence + phase linking | `emi`, `ds-temp-coh`, `emperical-co-emi-temp-coh-pc` |
| `moraine/cli/pqm.py` | temporal coherence | `temp-coh` |
| `moraine/cli/pu.py` | point cloud unwrapping | `mcf-pc`, `gamma-mcf-pt` |
| `moraine/cli/dl.py` | n2f / n2ft filtering of stacks | `n2f`, `n2ft` |
| `moraine/cli/plot.py` | pyramids of rasters / point clouds and their interactive plots | `ras-pyramid`, `pc-pyramid` |

## Command layer (moraine/command/)

| module | responsibility |
|---|---|
| `moraine/command/__init__.py` | command registry generated from `moraine.cli`, argument parsing and validation, `execute`, `--json` output, `main` |
| `moraine/command/pipeline.py` | TOML pipeline loading (`[vars]`, `[defaults]`, `[[step]]`), planning, resume, state in `.moraine/<file name>/` |
| `moraine/command/summary.py` | `info` metadata and pyramid statistics, `quicklook` (PNG) and `view` (notebook with interactive plots) through `ras_plot` / `pc_plot` |
| `moraine/command/tile_view.py` | tile viewer of raster and point cloud pyramids in notebooks (anywidget + Leaflet, tiles rendered by the kernel; `tile_view.js` / `.css` beside it) |

## Other parts of the repository

| path | content |
|---|---|
| `tests/` | pytest; `conftest.py` has the sample data fixtures and the `gpu` / `slow` markers |
| `examples/` | verified pipelines of the whole processing chain (decision 0009) |
| `docs/workflows/` | one guide per example pipeline |
| `docs/decisions/` | design decision records |
| `docs/contracts/` | promises on formats others depend on (JSON output, pipeline files, pyramids, data) |
| `docs/development.md` | how to change moraine |
| `nbs/Tutorials/` | tutorial notebooks (examples, not tests) |
| `data/` | sample data (not in git) |
