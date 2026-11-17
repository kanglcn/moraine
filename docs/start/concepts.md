# Concepts

What the pieces of moraine are and how they fit together. Ten minutes here save hours in the reference.

## A collection of functions, not a workflow

There is no workflow that always gives a satisfactory InSAR result: decorrelation, atmosphere and strong
deformation gradients defeat every fixed chain somewhere. moraine therefore is a collection of functions that
each implement one technique (an amplitude dispersion index, a statistical test for homogeneous pixels, EMI
phase linking, a deep learning filter, minimum cost flow unwrapping, ...) and the data infrastructure around
them. You compose the chain for your case; the [verified example pipelines](../workflows/index.md) are a
starting point, not the only way.

The data stay simple on purpose: **numpy or cupy arrays in memory, zarr arrays on disk**. No containers, no
project files; every result is an array you can open with any tool.

## Three layers

| layer | namespace | data | one call processes |
|---|---|---|---|
| API | `moraine.*` (`import moraine as mr`) | numpy (CPU, numba) or cupy (GPU) arrays in memory; the result is the same kind as the input | one unit: one image or image pair of the scene, or one block of pixels / points with its whole time series |
| CLI | `moraine.cli.*` (`import moraine.cli as mc`) | zarr datasets on disk, larger than memory; processed chunk by chunk by moraine's own workers, threads on the CPU or one process per GPU | a whole data set |
| command | `moraine COMMAND ...`, `moraine run FILE` | the same zarr datasets | a whole data set, from the shell or a pipeline file |

The CLI functions are not wrappers of the API functions: they cut the data into the units the API functions
take, run them as the tasks of an executor (threads on the CPU, one process per GPU; decisions 0033, 0034), and write zarr. Every `moraine.cli` function is a
command of the `moraine` executable, with its help generated from the docstring (decision 0005), so the three
layers always agree. `help(mr.emi)`, `help(mc.emi)` and `moraine emi --help` describe the same thing.

```python
import moraine as mr
import moraine.cli as mc

ph = mr.emi(coh)                                   # a block of points, in memory
mc.emi('ds/ds_can_coh.zarr', 'ds/ds_can_ph.zarr', cuda=True)   # a whole data set, on the GPUs
```

```bash
moraine emi --coh ds/ds_can_coh.zarr --ph ds/ds_can_ph.zarr --cuda
```

The modules of each layer and the rules on which layer may use which are on the developer page
[Architecture](../dev/architecture.md).

## Rasters and point clouds

Two kinds of data sets run through a processing chain (the full conventions are in the
[data contract](../contracts/data.md)):

- **Rasters** on the radar grid: shape `(nlines, width[, n])`, azimuth first, range second, stacks last. The
  rslc stack is `(nlines, width, nimages)` complex64; `nan` marks missing data.
- **Point clouds** of selected pixels (PS, DS): arrays `(n_points, ...)`. A point is addressed by its grid
  index `gix` `(n_points, 2)` (azimuth, range) or by its hilbert index `hix` `(n_points,)`, and point clouds are
  kept **in hilbert order**, so that points close in the array are close on the ground. `pc-union`,
  `pc-intersect`, `pc-diff` and `pc-select-data` are set operations on sorted indices; `ras2pc` and `pc2ras`
  move data between the two kinds.

Interferometric data are complex: the interferogram of the image pair `(ref, sec)` is `ref * conj(sec)`, phase
linked histories and filtered interferograms have unit amplitude, coherence of point clouds is stored
compressed (the upper triangle of the coherence matrix, `(n_points, n_pairs)`), unwrapped phase is float32
radians `(n_points, n_pairs)`.

## Data on disk

The data on disk are zarr arrays, and a zarr array is a directory: `zarr.json` with the shape, the dtype and the
chunks, and one file per chunk, compressed with zstd (decision 0039). Copy, move or delete the directory as a whole,
as if it were one file. The rslc stack of the sample data set:

```text
raw/rslc.zarr/       complex64 (981, 4160, 92), chunks (1000, 1000, 1)
├── zarr.json        shape, dtype, chunks, compression
└── c/               460 chunk files: c/<azimuth block>/<range block>/<image>
    ├── 0/0/0        azimuth 0-980, range 0-999, image 0
    ├── 0/0/1        azimuth 0-980, range 0-999, image 1
    ├── ...
    └── 0/4/91       azimuth 0-980, range 4000-4159, image 91
```

`moraine info PATH` prints the shape, the dtype and the chunks without reading any data. Any zarr reader opens the
array, and what it reads is a numpy array; read the part you need, not the whole stack:

```python
import zarr

rslc = zarr.open('raw/rslc.zarr', mode='r')   # reads the metadata only
block = rslc[:500, 1000:2000, 0]              # numpy array (500, 1000) complex64
```

## Chunks

zarr arrays are stored in chunks, the workers process them in parallel; the chunk layout decides the memory and the
speed. moraine's convention (decision 0019): **blocks in space, one image (or image pair) per chunk**,
e.g. `(lines_block, width_block, 1)` for a raster stack and `(points_block, 1)` for a point cloud stack. A step per
image then reads whole chunks, and a step per block reads one chunk of every image, both without rechunking.
Commands take `chunks` (processing) and `out_chunks` (storage) arguments; too small chunks spend the time in
scheduling, too large ones run out of memory. Divide rasters along azimuth rather than range.

## How a command runs

```text
moraine run FILE            pipeline: steps in order, unchanged ones skipped
└── moraine COMMAND ...     command: help and argument checks from the docstring
    └── moraine.cli.<name>  opens the inputs, creates the outputs, lists the tasks
        └── executor        runs the tasks: threads on the CPU, one process per GPU
            └── task        zarr ──► API function on numpy / cupy arrays ──► zarr
```

A command never holds a whole data set in memory. Its function in `moraine.cli` (`mc.<name>(...)` in Python)
cuts the work into tasks, one per unit the API function takes: a block of pixels or points (`chunks`), or an image
or image pair. A task reads its input chunks, calls the API function on them in memory and writes its output
chunks, which no other task writes. moraine's executor runs the tasks on its own workers (decisions 0033, 0034): on
the CPU in threads of the command's process, the numba kernels spreading each task over the cores; with `cuda` in
one process per GPU of `CUDA_VISIBLE_DEVICES`, which reads the chunks of its next task while it computes the current
one. `n_workers`, `threads_per_worker` and `processes` set how many tasks run at the same time and where
([command line](../cli/index.md)). A command therefore needs the memory of the tasks it runs at the same time, plus
the small arrays a few commands keep for the whole run (their help gives the bytes per point), not the memory of the
data set: when one runs out of memory, make `chunks` smaller or run fewer tasks at a time.

For data that fit in memory, a crop or a test, the API functions are quicker: no zarr to read and write, no workers
to start. Some commands have no API function of their own: the loading of GAMMA results, the pyramids and data
manipulation that is a line of numpy on arrays in memory (`math`, `pc-logic-ras`, `pc-select-data`, `pc-concat`,
...).

## The processing chain

The example pipelines implement one common chain; every step is a command and can be replaced:

1. **Load** GAMMA results into zarr: flattened rslc stack, coordinates, look vector, metadata ([01](../workflows/01_load.md)).
2. **PS candidates**: amplitude dispersion index and the temporal coherence of Noise2Fringe filtered
   interferograms ([02](../workflows/02_ps.md)).
3. **DS**: statistically homogeneous pixels (KS test), DS candidates, coherence matrices, EMI phase linking with
   an adaptive regularization, DS selection by the weighted temporal coherence and the connectivity of the
   coherent image pairs ([03](../workflows/03_ds.md)).
4. **Merge and refine**: PS and DS together, filtered with the Noise2Fringe Transformer, kept by temporal
   coherence with their DEM error relative to the neighbourhood fitted, which is removed from their phase
   ([04](../workflows/04_refine.md)).
5. **Unwrap**: minimum cost flow on a Delaunay network of the points, a redundant network of image pairs together
   (`emcf-pc`) or every interferogram alone (`mcf-pc`), unwrapping errors corrected by phase closure, which gives the
   unwrapped phase of every image ([05](../workflows/05_unwrap.md)).
6. **Products**: line of sight displacement in millimeters, corrected for the troposphere with ERA5 and relative to a
   reference area, the DEM error and the velocity of every point ([06](../workflows/06_products.md); decisions 0048,
   0051, 0054).

## Pipelines, resuming and JSON

A chain is a TOML file of steps run by `moraine run FILE` ([pipeline files](../contracts/pipeline-file.md)).
A step is skipped when it ran before with the same arguments and its inputs did not change, so after a failure
or a changed parameter you simply run the file again. `moraine run FILE --dry-run` shows what would run,
`moraine status FILE` the state. With `--json` every command prints one JSON object on stdout and logs on
stderr ([JSON output](../contracts/json-output.md)): this is what scripts and [AI agents](../agent/index.md)
read.

## GPU or CPU

API functions accept numpy or cupy arrays and dispatch on the type (`moraine.api.utils_.get_array_module`);
CLI functions and commands take `cuda` (one worker process per GPU, allocating from cupy's memory pool). A GPU is used only when
`CUDA_VISIBLE_DEVICES` names one. The phase linking of many images (more than about 100) can be faster on a
machine with many CPU cores than on one GPU; the guides give the numbers.
