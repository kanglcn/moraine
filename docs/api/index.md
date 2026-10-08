# Python API

`import moraine as mr` gives the array functions: algorithms on numpy arrays (CPU, numba) or cupy arrays
(GPU) in memory, grouped by topic in the modules of `moraine.api` (decision 0015) and re-exported as `mr.<name>`.
Most functions accept both kinds of array and return the same kind.

```python
import numpy as np
import moraine as mr
import moraine.api.utils_

ph = mr.emi(coh)                          # numpy in, numpy out
if moraine.api.utils_.is_cuda_available():
    import cupy as cp
    ph_gpu = mr.emi(cp.asarray(coh))      # cupy in, cupy out
```

One call processes one unit: an image or image pair of the whole scene, or a block of pixels / points with
their whole time series (plus a halo where neighbours are needed). For data larger than memory the
[command line](../cli/index.md) maps these functions over zarr chunks with dask.

| module | topic | main names |
|---|---|---|
| [calamp](calamp.md) | SLC amplitude and amplitude calibration | `rslc2amp`, `calamp` |
| [pc](pc.md) | point cloud indices (grid / hilbert), sorting, set operations, raster <-> point cloud | `pc_hix`, `pc_sort`, `pc_union`, `ras2pc`, `pc2ras` |
| [rtree](rtree.md) | hilbert R-tree for bounding box queries on point clouds | `HilbertRtree` |
| [tnet](tnet.md) | temporal network: image pairs | `TempNet` |
| [polygon](polygon.md) | polygon GeoJSON files, point in polygon | `read_polygons`, `polygons_contain` |
| [ps](ps.md) | amplitude dispersion index for PS selection | `amp_disp` |
| [shp](shp.md) | statistically homogeneous pixels (KS test) | `ks_test`, `select_shp` |
| [co](co.md) | interferograms, coherence matrices, regularization | `emperical_co_pc`, `uncompress_coh`, `nearestPD` |
| [pl](pl.md) | phase linking (EMI) and DS temporal coherence | `emi`, `ds_temp_coh`, `emperical_co_emi_temp_coh_pc` |
| [dl](dl.md) | deep learning filters and model download | `n2f`, `n2fs3d`, `n2ft`, `download_dl_model` |
| [pqm](pqm.md) | pixel quality: temporal coherence | `temp_coh` |
| [unwrap](unwrap.md) | Delaunay triangulation, minimum cost flow, EMCF, phase closure correction, GAMMA `mcf_pt` | `delaunay`, `mcf_pc`, `emcf_pc`, `unwrap_correct_closure_pc` |
| [utils](utils.md) | numba decorators, GPU detection, numpy / cupy dispatch (`moraine.api.utils_`, not re-exported) | `is_cuda_available`, `get_array_module` |

Each page shows the signature and the docstring of every public function (shapes, dtypes, units, defaults) and
short examples on synthetic data that run when this manual is built, so they are known to work with the
version documented. The data conventions shared by all functions are in the [data contract](../contracts/data.md).
