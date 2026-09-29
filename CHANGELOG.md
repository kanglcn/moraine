# Release notes

<!-- do not remove -->

## Unreleased

Deep learning models run on PyTorch instead of ONNX Runtime; torch is an optional dependency (`pip install moraine[dl]`) and the models are downloaded as `.pth` files by `download_dl_model()`

`n2ft` results are reproducible (fixed farthest point sampling start)

Development moved from nbdev notebooks to plain python: packaging in `pyproject.toml`, numpy style docstrings, pytest tests in `tests/`; the nbdev documentation site is removed

`moraine ... --json` output has `version` and `ok` fields; pyramids record their format version in `0.zarr`; pipeline files may declare `[pipeline] version`. The formats are specified in `docs/contracts/`

New command `moraine view PYRAMID ... -o view.ipynb`: a notebook with interactive plots of pyramids (cyclic colorwheel for phases, viridis over the 1-99 % range otherwise, radar or map axes, plot size from the scene, sliders for stacks); `quicklook` uses the colorwheel for phases too

`mcf_pc` / `mcf-pc` use an own Delaunay triangulation and a successive shortest path min cost flow on its half-edges: exactly optimal, independent of the point order, about 7 times faster than before (10 million points in 16 s, GAMMA mcf_pt 35 s) with about 5 times less memory; new option `earth_cost` (default 1). OR-Tools is no longer a dependency

`gamma_mcf_pt`: float64 weights are converted to the FLOAT type mcf_pt reads (they were misread); the `gamma-mcf-pt` command passes `ref_point` on (it was ignored and the first point always used) and its default is 0, the first point as documented

Commands recognize outputs named without `/` or `.` (e.g. `--out_dir pyr`); `main()` returns 1 for a failed pipeline instead of raising `SystemExit`

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




