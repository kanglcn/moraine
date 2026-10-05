# Release notes

<!-- do not remove -->

## Unreleased

`load-gamma-flatten-rslc` has the argument `gamma_threads` (default: the number of CPU cores, at most 64): the number of threads of each `phase_sim_orb` run (GAMMA uses 8 without it); on 128 cores one image takes 1.5 s instead of 10 s with the same result

The CLI tutorials (`nbs/Tutorials/CLI/`) use two sample data sets, one folder each with the same five notebooks (`01_load` ... `05_unwrap`): `Xinpu/` (landslide, Three Gorges, Sentinel-1 ascending) and `CampiFlegrei/` (caldera uplift, Sentinel-1 descending); the data are in `data/` and read from the GAMMA results

`moraine quicklook --extent` and `view(...).png(..., extent=...)` draw a part of the scene, from the finest pyramid level that fits, down to the data (degrees on web mercator maps, pixels on the radar grid); the title gives the extent and the level, `repr` of a view the extent and the finest cell of every layer, so that agents can zoom into what they look at

Data conventions (`docs/contracts/data.md`): stacks are chunked by blocks in space and one image (or image pair) per chunk; commands expect this layout (decision 0019)

The API modules moved from `moraine/` to `moraine/api/`, phase unwrapping to `moraine/api/unwrap/` (`mcf.py`, `emcf.py`, `gamma.py`, `delaunay_.py`; `moraine/pu.py` is split). `import moraine` still exports the same names (`moraine.mcf_pc`, ...); code importing modules directly changes, e.g. `from moraine.pu import mcf_pc` becomes `from moraine.api.unwrap import mcf_pc`. The downloaded deep learning models stay in `moraine/dl_model/`

Deep learning models run on PyTorch instead of ONNX Runtime; torch is an optional dependency (`pip install moraine[dl]`) and the models are downloaded as `.pth` files by `download_dl_model()`

`n2ft` results are reproducible (fixed farthest point sampling start)

Development moved from nbdev notebooks to plain python: packaging in `pyproject.toml`, numpy style docstrings, pytest tests in `tests/`; the nbdev documentation site is removed

`moraine ... --json` output has `version` and `ok` fields; pyramids record their format version in `0.zarr`; pipeline files may declare `[pipeline] version`. The formats are specified in `docs/contracts/`

New `emcf_pc`: extended minimum cost flow (EMCF) unwrapping of the interferograms of any network of image pairs (`image_pairs`, by default every image with the next three); the loops of the network are used against unwrapping errors; `temporal_cost` (constant or time span) and `spatial_cost` (constant, edge length, point `weight`; both constant by default) choose where corrections go; by default as many interferograms are unwrapped at the same time as the available cores and half of the available memory allow. On a synthetic benchmark with known truth (`tests/unwrap_benchmark.py`, every image with the next three): median 0.83 % wrong (point, interferogram) against 1.54 % for `mcf_pc`. Decisions 0012 (stage 3) and 0020. The `emcf-pc` command takes the image pairs (e.g. from `image-pairs --bandwidth 3`), the pixel spacings and optionally the dates instead of `meta.toml`, needs `ph` chunked one image per chunk and processes the points in blocks and the interferograms one by one (10 million points, 100 images: 8.2 GB peak memory with 4 interferograms at the same time); `misclosure`, the `pairs` output, `t_scale`, `bperp_scale`, `repair` and `exclude` are removed

New `unwrap_correct_closure_pc`: correction of unwrapping errors by phase closure per region, as in MintPy (Yunjun et al. 2019): where the unwrapped interferograms of a loop of image pairs do not add up to zero, the interferograms of every region (points connected without long edges, e.g. an island) are corrected by whole cycles; points of small regions one by one. Works after any unwrapper; returns the unwrapped phase of every image `ts` (n_points, nimages) relative to the reference image `ref` (default 0), whose differences close every loop, the per point `misclosure_fraction` (before the correction), `change_fraction` (interferograms changed by the correction, a point quality for masking; errors that close every loop are not seen) and the regions. After `emcf_pc` on the benchmark: median 0.81 % wrong. Decisions 0021 and 0022 (no separate time series inversion). Command `unwrap-correct-closure-pc` (per block of points, per interferogram, per block of points; optional outputs `misclosure_fraction`, `change_fraction` and `region`; 10 million points, 100 images, 294 interferograms: 145 s, 3.3 GB)

One viewer, `moraine.cli.view(data, ...)`, replaces the holoviews plots: interactive maps in Jupyter / VS Code notebooks (no server or port forwarding) of pyramids, rasters in memory and point data; `show=` a name ('phase', 'intf_seq', 'intf_all', 'coh', ...) or a function `lambda v, ref, sec: ...` whose arguments are sliders; `a * b` overlays views, `a + b` shows them side by side with linked zoom and pan; click a pixel / point for its time series (`series=`), double click for its reference; web mercator data over a satellite / street base map; polygons drawn on the map saved to `polygons='file.geojson'`; `.selected`, `.reference`, `.index` in python; `repr` describes a view and `.png(path)` saves an image. `moraine view PYRAMID ... -o view.ipynb [--show] [--dates]` writes a notebook of such maps and `moraine quicklook` draws with it (`--show`, the old `--post_proc` still works). Removed: `ras_plot`, `pc_plot`, `ts_plot`, `bg_alpha`, `view_pyramid` and the dependencies holoviews, bokeh, jupyter_bokeh; anywidget is a dependency

New command `polygon-mask`: bool mask of a raster or point cloud inside or outside the polygons of a GeoJSON file (longitude / latitude or radar grid coordinates); `moraine.read_polygons`, `write_polygons`, `polygons_contain`

`mcf_pc` / `mcf-pc`: new option `spatial_cost` ('constant', the default and the former behaviour, or 'length': phase jumps first on long edges). The `mcf-pc` command takes the pixel spacings (`range_pixel_spacing`, `azimuth_pixel_spacing`; the grid index is converted to meters), needs `ph` chunked one image per chunk and unwraps the interferograms in threads instead of dask processes: one copy of the network, `n_workers` by default bounded by the cores and half of the available memory; `threads_per_worker` and the dask cluster arguments are removed (10 million points, 16 interferograms, 8 at the same time: 28 s and 10 GB against 36 s and 17.5 GB with processes)

`mcf_pc` / `mcf-pc` use an own Delaunay triangulation and a successive shortest path min cost flow on its half-edges: exactly optimal, independent of the point order, about 7 times faster than before (10 million points in 16 s, GAMMA mcf_pt 35 s) with about 5 times less memory; new option `earth_cost` (default 1). OR-Tools is no longer a dependency

`gamma_mcf_pt`: float64 weights are converted to the FLOAT type mcf_pt reads (they were misread); the `gamma-mcf-pt` command passes `ref_point` on (it was ignored and the first point always used) and its default is 0, the first point as documented

`pc_pyramid` / `pc-pyramid`: the grid reaches the cells of the largest coordinates; the points of the last line and column were merged into the previous cells (one point per cell kept), and whether they were depended on rounding for coordinates that are not multiples of `ras_resolution`. Rebuild point cloud pyramids to see those points

Commands recognize outputs named without `/` or `.` (e.g. `--out_dir pyr`); `main()` returns 1 for a failed pipeline instead of raising `SystemExit`

`emi` command: `ref` was ignored, the first image was always the reference

`emi` on the CPU: the EMI quality of points with an ill-conditioned coherence magnitude matrix was wrong (by 0.1 at a condition number of 2·10⁴, more above) because the float32 inverse was not symmetric; the phase histories change only by rounding

Bugs squashed: CPU `ad_intf_pc` (undefined name), `isPD`/`nearestPD` on numpy arrays, `HilbertRtree.save`/`load` with zarr 3

## 0.9.0

Add mcf_pc API and CLI

Add interative point plot tool

## 0.8.5


## 0.8.4

Add n2ft model

Add API for plot

## 0.8.3

Move two deep learning models out of the package to prevent too big pypi package

update cli.co use image_pairs in memory rather than tnet file;

fix plot bugs in functions for general read zarr file, and change seq_intf from (i, i-1) to (i,i+1);

add multilook and intf to api.co;

finish gamma phase wrapping API and CLI;

move read/write gamma file to a seperate module from cli.load

## 0.8.2

bug fixed for cli.emperical_co_pc

### Bugs Squashed

- Data format scomplex is not supported. ([#20](https://github.com/kanglcn/moraine/issues/20))
  - ### Description of the problem

The package doesn't support scomplex data, so if the preprocessed rslc data is saved as scomplex, you are supposed to convert it to fcomplex and then process it with this package. 

### Minimal Complete Verifiable Example

_No response_

### Full error message

_No response_

### System information

```bash
numpy version 1.22.4
```


### Are you willing to help fix this bug?

No





## 0.8.1

implement parallel zarr io

modify `emperical_co_pc` and necessary utils
for independent ras chunkwise processing 

add `emperical_co_emi_temp_coh_pc` to 
- only processing a small batch in one chunk to prevent
holding the coherence matrix for all chunk which may exceed
memory limit;
- prevent writing coherence matrix which may exceed disk space limit.

## 0.8.0

- add cpu version of all functions and make use cpu as default
- add default args to dask cluster and allow users to configure that
- only calculate low tri of coherence matrix and copy its conj to up tri
- more flexiable plot functions
- carefully deal with nan values for all functions
- Update `pl.temp_coh` with elementwise kernel


## 0.7.0
Make cuda depedency optional

## 0.6.6
Test

## 0.6.5
Test

## 0.6.4
Test

## 0.6.3
Test

## 0.6.2
Test

## 0.6.1

Bug fix

## 0.6.0

Rename to Moraine

Add CLI plot

Package Reconstruction

## 0.5.1

Fix the doc generation

## 0.5.0

New features:

Add CLI for temporal coherence estimation for DS

Add dispersion index calculation for PS

Add logic operation for pc index

Add `transform` for coordinate reprojection

Add holoviews plot


Maintain:

Using `with ... as:` to prevent dask cluster unclosed

Remove all plot options in CLI

Set all thread per worker for local cuda cluster to 1

Use progressbar

Update doc theme

Reorginaze Tutorial

## 0.4.2

Finish point cloud manipulation functions and commands

Add utils for automatically determine chunk_size

## 0.4.1

Fix a small deploy issue

## 0.4.0

Modify gamma load function for more precise look vector

Add point cloud manipulation

Add Introduction section and add more detail on readme

Add utils, logger.zarr_info logger.darr_info

Rename emperical_co_sp to emperical_co_pc

## 0.3.2

Modify the gamma load funtion to make it faster

## 0.3.1

Add DS processing from CLI tutorial and fix bugs

## 0.3.0

Add interface to load gamma result

## 0.2.0

Add ks_test, select_ds_can, emperical_co_sp, emi CLI

Add log and sparse utils

Add test script

## 0.1.0

Refine phase linking EMI

Add function to estimate temporal coherence

Add tutorial for dask processing

Update the test in API notebooks

## 0.0.4

Add covariance/coherence matrix estimation for sparse data

Add phase linking method: EMI

Add tutorial for DS processing (still under construction)

## 0.0.3

Add ks test

Add covariance/coherence matrix estimation

Add tutorial for adaptive multilooking

## 0.0.2

Remove cupy requirement for automatically publish to pypi

## 0.0.1




