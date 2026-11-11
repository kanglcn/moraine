"""Noise2Fringe Transformer (n2ft) filtering of point cloud interferograms"""

__all__ = ['n2ft']

import functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import prange
from scipy.spatial import KDTree
import fpsample
from ..utils_.jit import ngpjit
from ..utils_.chunk import chunkwise_knn_mapping
from .model import _import_torch, _get_model

@ngpjit
def _weights(dd):
    N, K = dd.shape #(N, 3)
    out = np.empty_like(dd)
    for i in prange(N):  # OpenMP-style parallel loop
        denom = 0.0
        for j in range(K):
            out[i, j] = 1.0 / (dd[i, j] + 1e-8)
            denom += out[i, j]
        for j in range(K):
            out[i, j] /= denom
    return out

def _sample_and_knn(pos, k=16, workers=8):  # (N, 2)
    # 8 query threads: for the 20 000 - 30 000 points of a chunk, one thread per core (128) costs more than it saves
    # fixed start_idx makes the farthest point sampling, hence the n2ft result, reproducible
    N0 = pos.shape[0]
    N1 = N0 // 4
    N2 = N1 // 4
    N3 = N2 // 4

    p0 = pos  # [N0, 2]
    idx01 = fpsample.bucket_fps_kdtree_sampling(p0, N1, start_idx=0)
    p1 = p0[idx01]
    idx12 = fpsample.bucket_fps_kdtree_sampling(p1, N2, start_idx=0)
    p2 = p1[idx12]
    idx23 = fpsample.bucket_fps_kdtree_sampling(p2, N3, start_idx=0)
    p3 = p2[idx23]

    tree0 = KDTree(p0)
    tree1 = KDTree(p1)
    tree2 = KDTree(p2)
    tree3 = KDTree(p3)

    # KNN queries
    _, ii_00 = tree0.query(p0, k=k, workers=workers) #[N0, k]
    _, ii_01 = tree0.query(p1, k=k, workers=workers) #[N1, k]
    _, ii_11 = tree1.query(p1, k=k, workers=workers)
    _, ii_12 = tree1.query(p2, k=k, workers=workers)
    _, ii_22 = tree2.query(p2, k=k, workers=workers)
    _, ii_23 = tree2.query(p3, k=k, workers=workers)
    _, ii_33 = tree3.query(p3, k=k, workers=workers)

    dd_10, ii_10 = tree1.query(p0, k=3, workers=workers)
    dd_21, ii_21 = tree2.query(p1, k=3, workers=workers)
    dd_32, ii_32 = tree3.query(p2, k=3, workers=workers)

    ww_10 = _weights(dd_10).astype(np.float32)
    ww_21 = _weights(dd_21).astype(np.float32)
    ww_32 = _weights(dd_32).astype(np.float32)

    return (
        ii_00, ii_01, ii_11, ii_12, ii_22, ii_23, ii_33,
        ii_10, ii_21, ii_32,
        ww_10, ww_21, ww_32
    )

@ngpjit
def _pos_norm(x, y): # (N,), (N,)
    '''return normalized pos, (N, 2)'''
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)
    y_norm = (y - y_min)/(y_max - y_min)
    x_norm = (x - x_min)/(x_max - x_min)

    return np.stack((x_norm, y_norm),axis=-1).astype(np.float32)

def _n2ft_structure(
    x,
    y,
    device,
):
    """normalized positions and the sampling and neighbour indices of the points, as torch tensors on `device`;
    they depend on the coordinates only and serve all interferograms of the points

    Parameters
    ----------
    x
        (n,)
    y
        (n,)
    device
        torch device
    """
    torch = _import_torch()
    pos = _pos_norm(x,y)
    keys = _sample_and_knn(pos)
    pos = torch.from_numpy(pos).to(device).unsqueeze(0)
    keys = tuple(torch.from_numpy(key).to(device).unsqueeze(0) for key in keys)
    return pos, keys

# The filtering pays the compilation of the model (15-40 s per process) from this many points times interferograms:
# compiled, the model runs about 3 times faster (A100: 1.9 instead of 5.5 ms per interferogram of 23 000 points).
_N2FT_COMPILE_MIN_POINT_INTFS = 100_000_000
_N2FT_MAX_POINT_INTFS = 200_000      # points times interferograms per model call on the CPU (about 1.6 GB)
_N2FT_POINT_INTF_BYTES = 8_000       # GPU memory of a model call per point and interferogram (A100: 6.5-7.7 kB)
_N2FT_MEMORY_FRACTION = 0.2          # of the GPU memory for one model call
_N2FT_PREFETCH = 2                   # processing chunks prepared (structure, phasors) ahead of the model

def _n2ft_compile_default(n_points, n_image_pairs):
    """whether to compile the n2ft model for this much work"""
    return n_points*n_image_pairs >= _N2FT_COMPILE_MIN_POINT_INTFS

@functools.lru_cache(maxsize=None)
def _n2ft_max_point_intfs(device):
    """points times interferograms of one model call on `device`: a fifth of the memory of a GPU, 200 000 on the CPU"""
    torch = _import_torch()
    device = torch.device(device)
    if device.type != 'cuda':
        return _N2FT_MAX_POINT_INTFS
    total = torch.cuda.get_device_properties(device).total_memory
    return max(50_000, int(total*_N2FT_MEMORY_FRACTION)//_N2FT_POINT_INTF_BYTES)

def _n2ft_phasors(intf, device):
    """interferograms (n, m) complex64 -> unit phasors (m, n, 2) float32 on `device`"""
    torch = _import_torch()
    x = torch.from_numpy(np.ascontiguousarray(intf)).to(device)
    x = torch.view_as_real(x.T.contiguous())                                      # (m, n, 2)
    return x/torch.linalg.vector_norm(x, dim=-1, keepdim=True)

def _n2ft_prepare(x, y, intf, device):
    """structure of the points and phasors of their interferograms on `device`: the input of `_infer_n2ft_prepared`"""
    return _n2ft_structure(x, y, device), _n2ft_phasors(intf, device)

def _n2ft_batches(m, batch):
    """(start, stop) of the batches of `m` interferograms of about `batch` each: as equal as possible, and none of a
    single interferogram when `batch` allows it (a compiled model would compile that shape separately)"""
    if m == 0:
        return []
    n_batches = -(-m//batch)
    if batch >= 2 and m >= 2:
        n_batches = min(n_batches, m//2)
    bounds = [round(i*m/n_batches) for i in range(n_batches+1)]
    return list(zip(bounds[:-1], bounds[1:]))

def _infer_n2ft_prepared(
    prepared,
    model,
    max_point_intfs:int=None,
):
    """Parameters
    ----------
    prepared
        (structure, phasors) of `_n2ft_prepare` on the device of `model`
    model
    max_point_intfs : int, optional
        points times interferograms of one model call (the batch of interferograms); a call holds about 8 kB per
        point and interferogram on a GPU. Default: a fifth of the GPU memory, 200 000 on the CPU

    Returns
    -------
    np.ndarray
        filtered interferograms, (n, m) complex64
    """
    torch = _import_torch()
    (pos, keys), x = prepared
    if max_point_intfs is None:
        max_point_intfs = _n2ft_max_point_intfs(next(model.parameters()).device)
    m, n = x.shape[:2]
    # the model runs several interferograms of the same points at once: the structure is the same for all of them
    batch = max(1, min(m, max_point_intfs//max(n, 1)))
    with torch.inference_mode():
        out = torch.empty_like(x)
        for start, stop in _n2ft_batches(m, batch):
            b = stop-start
            # expanded from the unbatched tensors, the strides are the same for every b: one shape for torch.compile
            pos_b = pos[0].expand(b, *pos.shape[1:])
            keys_b = tuple(key[0].expand(b, *key.shape[1:]) for key in keys)
            out[start:stop] = model(pos_b, x[start:stop], *keys_b)
        return torch.view_as_complex(out).T.contiguous().cpu().numpy()

def _infer_n2ft_structure(
    structure,
    intf,
    model,
    max_point_intfs:int=None,
):
    """`_infer_n2ft_prepared` of the structure of the points (`_n2ft_structure`) and their interferograms (n, m)"""
    device = next(model.parameters()).device
    return _infer_n2ft_prepared((structure, _n2ft_phasors(intf, device)), model, max_point_intfs)

def _infer_n2ft(
    x,
    y,
    intf,
    model,
):
    """Parameters
    ----------
    x
        (n,)
    y
        (n,)
    intf
        (n,m)
    model
    """
    return _infer_n2ft_prepared(_n2ft_prepare(x, y, intf, next(model.parameters()).device), model)

def _prefetched(items, prepare, n_prefetch=_N2FT_PREFETCH):
    """yields (item, prepare(item)) in order, with up to `n_prefetch` further items prepared in threads meanwhile:
    the sampling and neighbour search of the next processing chunks (CPU) run while the model filters this one"""
    items = list(items)
    with ThreadPoolExecutor(max_workers=max(1, n_prefetch)) as pool:
        futures = [pool.submit(prepare, item) for item in items[:n_prefetch]]
        for i, item in enumerate(items):
            if i+n_prefetch < len(items):
                futures.append(pool.submit(prepare, items[i+n_prefetch]))
            yield item, futures[i].result()
            futures[i] = None

def n2ft(
    x:np.ndarray,
    y:np.ndarray,
    intf:np.ndarray,
    chunks:int=None,
    k:int=128,
    model:str=None,
    cuda:bool=False,
    compile:bool=None,
):
    """Parameters
    ----------
    x : np.ndarray
        x coordinate, e.g., longitude, shape (n,) np.floating
    y : np.ndarray
        y coordinate, e.g., latitude, shape (n,) np.floating
    intf : np.ndarray
        interferogram, shape(n,) or shape(n,m) np.complex64
    chunks : int, optional
        chunksize, intf.shape[0] by default
    k : int, default: 128
        number of nearest neighbours of every point of a chunk that are filtered with it (halo)
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        use gpu for inference
    compile : bool, optional
        compile the model with torch.compile: the model then runs about 3 times faster on a GPU, but the compilation
        takes 15-40 s once per process (torch caches its result on disk). Default: when points times
        interferograms is at least 1e8
    """
    single_intf = False
    if len(intf.shape) == 1:
        single_intf = True
        intf = intf[:,None]
    n, m = intf.shape
    if compile is None:
        compile = _n2ft_compile_default(n, m)
    model = _get_model('n2ft', model, 'cuda' if cuda else 'cpu', compile)
    device = next(model.parameters()).device

    if (chunks is None) or (chunks >= n):
        out = _infer_n2ft(x, y, intf, model)
    else:
        out = np.empty_like(intf)
        in_indices, out_slices, map_indices = chunkwise_knn_mapping(x, y, chunks, k=k)
        prepare = lambda in_idx: _n2ft_prepare(x[in_idx], y[in_idx], intf[in_idx], device)
        for (in_idx, out_slice, map_idx), prepared in _prefetched(zip(in_indices, out_slices, map_indices), lambda item: prepare(item[0])):
            out[out_slice] = _infer_n2ft_prepared(prepared, model)[map_idx]

    if single_intf:
        out = out[:,0]
    return out
