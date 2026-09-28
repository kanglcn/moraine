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
- GPU processing (`--cuda`, `cuda = true`) needs cupy, dask-cuda and rmm, installed with conda for the
  local CUDA version (see README). moraine treats a GPU as available only when `CUDA_VISIBLE_DEVICES`
  is set to a non-empty value; without it, run with `cuda = false`.
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
4. Use `--json` when you parse the output: stdout is then one JSON object, logs go to stderr.
5. A failing step prints the error and its log (`WORK/.moraine/<file name>/logs/<step>.log`). Fix the
   cause and run the file again; it resumes at the failed step.

Pipeline file format (full description in `moraine/command/pipeline.py`):

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
  `--post_proc intf_seq --index I` for the I-th sequential interferogram of an rslc or phase stack).
  Look at the PNG: fringes should be continuous, noise should be where coherence is low.
- Pyramids are made by the `ras-pyramid` (rasters) and `pc-pyramid` (point clouds) commands; the
  examples build them for the results worth checking, and `moraine run` saves their PNGs to
  `WORK/.moraine/<file name>/quicklook/`.

Each workflow guide lists the expected ranges of its results and what to do when they are off.
Report anything outside those ranges to the user instead of silently continuing.

## Data conventions

- Raster: `(nlines, width[, n])`, azimuth first. rslc stack `(nlines, width, nimages)` complex64.
- Point cloud: arrays of shape `(n_points, ...)` indexed by
  - `gix`: grid index `(n_points, 2)` int32, (azimuth, range) of each point in the raster;
  - `hix`: hilbert index `(n_points,)` int64. After `pc-sort`, point clouds are in hilbert order, so
    points close in the array are close on the ground. Index arrays must be sorted for
    `pc-union` / `pc-intersect` / `pc-diff` / `pc-select-data`.
- Image pairs `(n_pairs, 2)`: reference and secondary image index; files have two integer columns.
- Interferograms and phase histories are complex; the phase is `np.angle(...)`. Filtered interferograms
  and phase histories have unit amplitude.
- Coherence of point clouds is stored compressed: the upper triangle of the coherence matrix,
  `(n_points, n_image_pairs)`; `moraine.uncompress_coh` restores full matrices.
- Commands that process raster chunks (`ras2pc-ras-chunk`, `emperical-co-pc`,
  `emperical-co-emi-temp-coh-pc`) write a directory with one zarr per raster chunk; merge it with
  `pc-concat` and the key written by `ras2pc-ras-chunk` (and the one of `pc-sort` for hilbert order).

## Rules

- Keep outputs in a working directory outside the repository; never write into `data/` (sample data).
- Do not delete a user's results or working directories without asking.
- Long jobs: `moraine run` blocks until done. On a SLURM cluster submit it, e.g.
  `sbatch --gpus=1 --wrap "moraine run FILE --workdir WORK --json > WORK/run.json"`, and poll with
  `moraine status FILE --workdir WORK`.
- Only one GPU pipeline at a time: each GPU command reserves most of the GPU memory (rmm pool).

## Design decisions

Read `docs/decisions/README.md` (the index of the design decision records) before changing how moraine
is built or used, e.g. dependencies, the command line, pipelines, visualization, documentation. Do not
change code against an accepted decision: propose a new record instead and ask the user. Record new
decisions there in the same change.

## Developing moraine

- The source is `moraine/` (API, numpy / cupy functions) and `moraine/cli/` (zarr in, zarr out,
  chunked with dask; every function decorated with `@mc_logger` becomes a `moraine` command).
- Command options and help are generated from the signatures and numpy style docstrings of
  `moraine/cli/*.py`, so keep type annotations and docstrings exact: shapes, dtypes, input or output,
  real defaults. `tests/test_command.py` checks that every argument is documented.
- Tests: `pytest -m "not slow"` (about 2 min). Tests needing the sample data (`MORAINE_TEST_DATA`,
  default `./data`), a GPU, GAMMA or the models are skipped when these are missing; `-m slow` runs the
  CLI chain tests. GPU tests are marked `@pytest.mark.gpu`.
- GPU code: import cupy / dask_cuda / rmm only behind `moraine.utils_.is_cuda_available()`.
- `.gitignore` ignores `*.toml`; new TOML files outside `examples/` need an exception.
