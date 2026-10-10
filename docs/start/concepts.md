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

## Chunks

zarr arrays are stored in chunks, the workers process them in parallel; the chunk layout decides the memory and the
speed. moraine's convention (decision 0019): **blocks in space, one image (or image pair) per chunk**,
e.g. `(lines_block, width_block, 1)` for a raster stack and `(points_block, 1)` for a point cloud stack. A step per
image then reads whole chunks, and a step per block reads one chunk of every image, both without rechunking.
Commands take `chunks` (processing) and `out_chunks` (storage) arguments; too small chunks spend the time in
scheduling, too large ones run out of memory. Divide rasters along azimuth rather than range.

## The processing chain

The example pipelines implement one common chain; every step is a command and can be replaced:

1. **Load** GAMMA results into zarr: flattened rslc stack, coordinates, look vector, metadata ([01](../workflows/01_load.md)).
2. **PS candidates**: amplitude dispersion index and the temporal coherence of Noise2Fringe filtered
   interferograms ([02](../workflows/02_ps.md)).
3. **DS**: statistically homogeneous pixels (KS test), DS candidates, coherence matrices, EMI phase linking with
   an adaptive regularization, DS selection by the weighted temporal coherence and the connectivity of the
   coherent image pairs ([03](../workflows/03_ds.md)).
4. **Merge and refine**: PS and DS together, filtered with the Noise2Fringe Transformer, kept by temporal
   coherence ([04](../workflows/04_refine.md)).
5. **Unwrap**: minimum cost flow on a Delaunay network of the points, every interferogram alone (`mcf-pc`) or
   a redundant network together (`emcf-pc`), unwrapping errors corrected by phase closure ([05](../workflows/05_unwrap.md)).

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
