"""internal utilities for chunkwise data processing"""


__all__ = ['fill_slice', 'all_chunk_slices', 'all_chunk_slices_with_overlap', 'chunkwise_slicing_mapping',
           'chunkwise_knn_mapping']

import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree

def fill_slice(
    shape,
    slices,
):
    """Parameters
    ----------
    shape
        numpy arr, cupy arr, zarr,
    slices
        tuple of slice object, len == data_arr.ndim
    """
    out_slices = []
    for i in range(len(slices)):
        slice_i = slices[i]
        if slice_i.start is None:
            start = 0
        else:
            start = slice_i.start
        if not 0 <= start < shape[i]:
            raise ValueError(f'slice start {start} is outside the shape {shape[i]} of dimension {i}')
        if slice_i.stop is None:
            stop = shape[i]
        else:
            stop = slice_i.stop
        if not start < stop <= shape[i]:
            raise ValueError(f'slice stop {stop} must be after the start {start} and within the shape {shape[i]} of dimension {i}')
        if slice_i.step not in (None, 1):
            raise ValueError('slices with a step are not supported')
        step = 1
        out_slices.append(slice(start,stop,step))
    return tuple(out_slices)

def all_chunk_slices(
    shape,
    chunks,
):
    """get the slices for every input chunks

    Parameters
    ----------
    shape
        np.array, cp.array,zarr
    chunks
    """
    out_slices = []
    for shape_, chunk_ in zip(shape,chunks):
        if chunk_ <0: chunk_ = shape_
        bound_1dim = np.arange(0,shape_+chunk_,chunk_)
        if bound_1dim[-1] > shape_: bound_1dim[-1] = shape_

        slice_1dim = []
        for j in range(bound_1dim.shape[0]-1):
            slice_1dim.append(slice(int(bound_1dim[j]),int(bound_1dim[j+1])))
        out_slices.append(slice_1dim)
    out_slices = list(itertools.product(*out_slices))
    return out_slices

def all_chunk_slices_with_overlap(
    shape, 
    chunks, 
    depths,
):
    '''get the slices for every input chunks with overlap'''
    out_slices = []
    for shape_, chunk_, depth_ in zip(shape,chunks,depths):
        if chunk_ <0: chunk_ = shape_
        starts_1dim = np.arange(-depth_,shape_-depth_,chunk_)
        starts_1dim[starts_1dim<0] = 0
        ends_1dim = np.arange(chunk_+depth_,shape_+chunk_+depth_,chunk_)
        ends_1dim[ends_1dim>shape_] = shape_

        slice_1dim = []
        for j in range(starts_1dim.shape[0]):
            slice_1dim.append(slice(int(starts_1dim[j]),int(ends_1dim[j])))
        out_slices.append(slice_1dim)
    out_slices = list(itertools.product(*out_slices))
    return out_slices

def chunkwise_slicing_mapping(
    shape,
    chunks,
    depths,
):
    '''get the slices for every input chunks with overlap
    output chunks without overlap and their mapping slices'''
    in_slices = []; out_slices = []; map_slices = []
    for shape_, chunk_, depth_ in zip(shape,chunks,depths):
        if chunk_ <0: chunk_ = shape_
        in_starts_1dim = np.arange(-depth_,shape_-depth_,chunk_)
        in_starts_1dim[in_starts_1dim<0] = 0
        in_ends_1dim = np.arange(chunk_+depth_,shape_+chunk_+depth_,chunk_)
        in_ends_1dim[in_ends_1dim>shape_] = shape_
        out_starts_1dim = np.arange(0,shape_,chunk_)
        out_ends_1dim = np.arange(chunk_,shape_+chunk_,chunk_)
        out_ends_1dim[out_ends_1dim>shape_] = shape_

        assert len(in_starts_1dim) == len(out_starts_1dim)
        assert len(in_ends_1dim) == len(out_ends_1dim)

        in_slice_1dim = []; out_slice_1dim = []; map_slice_1dim = []
        for in_start, in_end, out_start, out_end in zip(in_starts_1dim, in_ends_1dim, out_starts_1dim, out_ends_1dim):
            in_slice_1dim.append(slice(int(in_start),int(in_end)))
            out_slice_1dim.append(slice(int(out_start),int(out_end)))
            offset = out_start-in_start
            map_slice_1dim.append(slice(int(offset),int(offset+out_end-out_start)))

        in_slices.append(in_slice_1dim)
        out_slices.append(out_slice_1dim)
        map_slices.append(map_slice_1dim)

    in_slices = list(itertools.product(*in_slices))
    out_slices = list(itertools.product(*out_slices))
    map_slices = list(itertools.product(*map_slices))
    return in_slices, out_slices, map_slices

def chunkwise_knn_mapping(x, y, chunks, k=128, bounds=None, workers=32):
    """
    Input and output indices for processing a point cloud in chunks, every point with its k nearest neighbours.

    Parameters
    ----------
    x : np.ndarray
        x coordinate of the points, shape (n,)
    y : np.ndarray
        y coordinate of the points, shape (n,)
    chunks : int
        number of points per chunk
    k : int, default: 128
        number of nearest neighbours (the point itself counts) of every point of a chunk that are processed
        with the chunk
    bounds : np.ndarray, optional
        start of every chunk and the number of points, increasing, shape (n_chunks+1,); chunks of `chunks`
        points by default
    workers : int, default: 32
        threads of the neighbour search, each needs about 16 x k x chunks bytes of memory

    Returns
    -------
    in_indices : list of np.ndarray
        sorted indices of the points of every chunk and of their neighbours (halo)
    out_slices : list of slice
        the points of every chunk
    map_indices : list of np.ndarray
        positions of the points of every chunk in its `in_indices`
    """
    n = y.shape[0]
    if bounds is None:
        bound = np.arange(0, n + chunks, chunks)
        if bound[-1] > n:
            bound[-1] = n
    else:
        bound = np.asarray(bounds)
    pos = np.stack((x, y), axis=-1)
    # the sliding midpoint tree builds 3 times faster than the balanced one and is queried as fast
    tree = KDTree(pos, leafsize=32, balanced_tree=False, compact_nodes=False)
    k = min(k, n)

    def one_chunk(start, end):
        neighbours = tree.query(pos[start:end], k=k, workers=1)[1]
        # the chunk itself is added: a point is not among its own neighbours when more than k points share its position
        in_idx = np.unique(np.concatenate((np.arange(start, end), neighbours.ravel())))
        return in_idx, slice(start, end), np.flatnonzero((in_idx >= start) & (in_idx < end))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one_chunk, bound[:-1].tolist(), bound[1:].tolist()))
    in_indices, out_slices, map_indices = zip(*results)
    return list(in_indices), list(out_slices), list(map_indices)
