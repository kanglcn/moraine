# moraine: guide for AI agents

moraine post-processes co-registered SAR images for InSAR time series: persistent and distributed
scatterer (PS / DS) selection, coherence estimation, phase linking, deep learning filtering, phase
unwrapping and visualization. Data are numpy / cupy arrays in memory and zarr arrays on disk.

This file is for coding agents (Claude Code, Codex, Cursor, ...). A user normally prepares the input
data and asks you to process them: run the `moraine` command line and TOML pipelines, check the
results, adjust parameters and rerun. Read the workflow guide in `docs/workflows/` that matches the
request before running anything.

## Environment

- Python >= 3.11. Install with `pip install -e '.[dev,dl]'` (`dl` = PyTorch, needed only by the deep
  learning filters `n2f`, `n2ft`). Download the trained models once:
  `python -c "import moraine; moraine.download_dl_model()"`.
- GPU processing (`--cuda`, `cuda = true`) needs cupy, numba-cuda, dask-cuda and rmm, installed with conda for the
  local CUDA version (see README). moraine treats a GPU as available only when `CUDA_VISIBLE_DEVICES`
  is set to a non-empty value; without it, run with `cuda = false`. GPU commands use all GPUs listed
  there (one dask worker per GPU) unless `n_workers` is given; list fewer GPUs to leave some free.
- Loading GAMMA results (`load-gamma-*`) runs GAMMA programs (`phase_sim_orb`, `create_offset`,
  `geocode`, `base_calc`); check with `which base_calc` first.
- Everything else runs on CPU with numba.

## Running processing

1. Check the tool: `moraine list` (all commands), `moraine COMMAND --help` (arguments with shapes,
   dtypes and defaults, generated from the function docstrings).
2. Prefer a pipeline file over single commands. Start from the verified examples in `examples/`:

   | file | tutorial | what it does |
   |---|---|---|
   | `examples/01_load.toml` | `docs/workflows/01_load.md` | GAMMA results -> zarr, web mercator coordinates |
   | `examples/02_ps.toml` | `docs/workflows/02_ps.md` | PS candidates (amplitude dispersion + n2f temporal coherence) |
   | `examples/03_ds.toml` | `docs/workflows/03_ds.md` | SHP, DS candidates, phase linking, DS refinement |
   | `examples/04_refine.toml` | `docs/workflows/04_refine.md` | merge PS and DS, refine with n2ft |
   | `examples/05_unwrap.toml` | `docs/workflows/05_unwrap.md` | minimum cost flow unwrapping |

   They share one working directory and read each other's outputs, so run them in this order:

   ```bash
   moraine run examples/01_load.toml --workdir WORK --var gamma=/path/to/gamma --var reference=YYYYMMDD
   moraine run examples/02_ps.toml --workdir WORK
   ...
   ```

   To change parameters, copy the example into the working directory (or anywhere) and edit the copy;
   do not edit `examples/` for one data set.
3. `moraine run FILE --dry-run` shows what would run and why. `moraine status FILE` shows the state.
   A rerun skips steps whose arguments and inputs did not change and reruns the steps downstream of a
   change, so after editing a parameter just run the file again.
4. Use `--json` when you parse the output: stdout is then one JSON object, logs go to stderr
   (fields in `docs/contracts/json-output.md`).
5. A failing step prints the error and its log (`WORK/.moraine/<file name>/logs/<step>.log`). Fix the
   cause and run the file again; it resumes at the failed step.

Pipeline file format (full description in `docs/contracts/pipeline-file.md`):

```toml
[vars]                        # ${name} is replaced in any value; --var name=value overrides
gamma = "/path/to/gamma"

[defaults]                    # applied to the steps whose command has these arguments
cuda = true

[[step]]
name = "adi"                  # unique step name
run = "amp-disp"              # a command of `moraine list`
rslc = "raw/rslc.zarr"        # the command arguments, relative to the working directory
adi = "ps/ras_adi.zarr"
[step.kw]                     # optional extra keyword arguments (e.g. dask cluster options)
memory_limit = "20GB"
```

Unknown argument names are errors (with a suggestion); tuples are written `[1000, 1000]` or
`"1000,1000"`; image pairs are a file made by the `image-pairs` command.

## Checking results

Never load large arrays to look at them. Use:

- `moraine info PATH`: shape, dtype and chunks of an array (no data read). For a pyramid it adds
  statistics from a coarse level (nan_fraction, min, max, mean, std, p01, p50, p99; amplitude for
  complex data) and `warnings` for all-nan, infinite or constant values.
- `moraine quicklook PYRAMID -o out.png` draws the whole scene of a pyramid (use
  `--show intf_seq --index I` for the I-th sequential interferogram of an rslc or phase stack).
  Look at the PNG: fringes should be continuous, noise should be where coherence is low. Zoom in with
  `--extent west,south,east,north` (degrees; `range_min,azimuth_min,range_max,azimuth_max` pixels on the
  radar grid): a smaller part is drawn from a finer pyramid level, down to single pixels / points; the title
  gives the extent and the level. Look at the whole scene first, then zoom into what looks wrong.
- In Python, `mc.view(data)` (`import moraine.cli as mc`) views a pyramid, a raster array or point data
  (`x=`, `y=`): `show=` what to show of a stack (`'intf_seq'`, ... or a function `lambda v, ref, sec: ...`),
  `a * b` overlays views, `a + b` puts them side by side. `repr(v)` describes a view in text and
  `v.png('out.png', index={...}, extent=(...))` saves an image you can look at (`repr` gives the extent of
  the data and the finest cell). In a notebook it is an interactive map:
  zoom and pan load details, sliders choose the image, a click plots the time series of a pixel / point and
  a double click makes it the reference, polygons drawn with `polygons='areas.geojson'` are saved for
  `moraine polygon-mask`; `v.selected`, `v.reference`, `v.index` follow the map.
- `moraine view PYRAMID [PYRAMID ...] -o view.ipynb [--show ...] [--dates meta.toml]` writes a notebook of
  such maps. Give it to the user to open in Jupyter / VS Code; it needs no server or port forwarding.
  Do not write plotting code.
- Pyramids are made by the `ras-pyramid` (rasters) and `pc-pyramid` (point clouds) commands; the
  examples build them for the results worth checking, and `moraine run` saves their PNGs to
  `WORK/.moraine/<file name>/quicklook/`.

Each workflow guide lists the expected ranges of its results and what to do when they are off.
Report anything outside those ranges to the user instead of silently continuing.

## Data conventions

Shapes and orders of the arrays (rasters azimuth first, point clouds in hilbert order with `gix` / `hix`,
image pairs, compressed coherence, per-chunk directories) are in `docs/contracts/data.md`. Read it before
combining results of different commands.

## Rules

- Keep outputs in a working directory outside the repository; never write into `data/` (sample data).
- Do not delete a user's results or working directories without asking.
- Long jobs: `moraine run` blocks until done. On a SLURM cluster submit it, e.g.
  `sbatch --gpus=1 --wrap "moraine run FILE --workdir WORK --json > WORK/run.json"`, and poll with
  `moraine status FILE --workdir WORK`.
- Only one GPU pipeline at a time: each GPU command reserves most of the GPU memory (rmm pool).

## Developing moraine

When the task is to change moraine itself rather than to process data:

- `docs/development.md`: how to make, validate and report a change, docstring rules and the pre-commit
  hook (read it first).
- `ARCHITECTURE.md`: layers, module map and dependency rules.
- `docs/decisions/README.md`: design decisions; do not change code against an accepted one, propose a
  new record and ask the user.
- `docs/roadmap.md`: planned features not started yet.
- `docs/contracts/README.md`: formats others depend on (`--json` output, pipeline files, pyramids, data
  conventions); changing them needs the contract, its tests and possibly a new version.
