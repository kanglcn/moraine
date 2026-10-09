"""parallel reads and writes of zarr arrays, directories of per chunk zarrs"""


__all__ = ['parallel_read_zarr', 'parallel_write_zarr', 'ZarrDir']

import concurrent.futures
import itertools
import math
from pathlib import Path
import zarr
import numpy as np
from ..api.chunk_ import fill_slice as _fill_slice, all_chunk_slices


def _one_chunk_slices_and_out_shape(data_zarr,slices):
    '''divide input slices into slices of every zarr chunk
    and return the shape of subdata to be reade'''
    zarr_1chunk_slice = []
    out_1chunk_slice = []
    out_shape = []
    for i in range(len(slices)):
        chunk_size = data_zarr.chunks[i]
        shape = data_zarr.shape[i]
        zarr_slices = slices[i]
        start = zarr_slices.start
        stop = zarr_slices.stop
        out_shape.append(stop-start)

        start_ = math.floor(start/chunk_size)*chunk_size
        zarr_1dim_bound = np.arange(start_,stop+chunk_size,chunk_size)
        zarr_1dim_bound[0] = start
        if zarr_1dim_bound[-1] > stop: zarr_1dim_bound[-1] = stop
        out_1dim_bound = zarr_1dim_bound-start

        zarr_1chunk_1dim_slice = []
        out_1chunk_1dim_slice = []
        for j in range(zarr_1dim_bound.shape[0]-1):
            zarr_1chunk_1dim_slice.append(slice(zarr_1dim_bound[j],zarr_1dim_bound[j+1]))
            out_1chunk_1dim_slice.append(slice(out_1dim_bound[j],out_1dim_bound[j+1]))

        zarr_1chunk_slice.append(zarr_1chunk_1dim_slice)
        out_1chunk_slice.append(out_1chunk_1dim_slice)

    zarr_1chunk_slice = list(itertools.product(*zarr_1chunk_slice))
    out_1chunk_slice = list(itertools.product(*out_1chunk_slice))

    return zarr_1chunk_slice, out_1chunk_slice, tuple(out_shape)

def _read_one_chunk(data_zarr,out,zarr_slices,out_slices):
    out[out_slices] = data_zarr[zarr_slices]

def parallel_read_zarr(data_zarr,slices,thread_pool_size=None,fill_slice=True):
    if fill_slice:
        slices = _fill_slice(data_zarr.shape,slices)
    zarr_1chunk_slices, out_1chunk_slices, out_shape = \
    _one_chunk_slices_and_out_shape(data_zarr,slices)
    # global out
    out = np.empty(out_shape,dtype=data_zarr.dtype)

    with concurrent.futures.ThreadPoolExecutor(thread_pool_size) as executor:
        futures = [executor.submit(_read_one_chunk,data_zarr,out,zarr_1chunk_slice,out_1chunk_slice)
                   for zarr_1chunk_slice,out_1chunk_slice in zip(zarr_1chunk_slices,out_1chunk_slices)]
    for future in futures:
        future.result() # raise the errors of the reads
    return out

class ZarrDir():
    def __init__(self,zarr_path_list:list):
        self.zarr_path_list = zarr_path_list
        zarr0 = zarr.open(self.zarr_path_list[0])
        self.chunksize = zarr0.chunks
        self.ndim = zarr0.ndim
        self.dtype = zarr0.dtype

        dim0_shape = 0
        dim0_chunks = []
        for zarr_path in self.zarr_path_list:
            data_zarr = zarr.open(zarr_path,mode='r')
            assert data_zarr.chunks[0] == data_zarr.shape[0]
            assert data_zarr.chunks[1:] == (1,)*(data_zarr.ndim-1)
            dim0_shape += data_zarr.shape[0]
            dim0_chunks.append(data_zarr.chunks[0])
        self.shape = (dim0_shape,*zarr0.shape[1:])
        self.dim0_chunks = tuple(dim0_chunks)

    @classmethod
    def from_dir(cls,zarr_dir):
        zarr_dir = Path(zarr_dir)
        zarr_path_list = sorted(zarr_dir.glob('*.zarr'),key=lambda path: int(path.stem)) # if one chunk is missing, it is ok
        return cls(zarr_path_list)

def _parallel_read_pc_dir(
    zarr_dir,
    idx,
    thread_pool_size=None):
    """Parameters
    ----------
    zarr_dir
        zarr_dir object
    idx
        index for dim 1,2,...
    thread_pool_size : optional
    """
    # global out
    if isinstance(idx,int):
        idx = (idx,)
    out = np.empty(zarr_dir.shape[0],dtype=zarr_dir.dtype)
    out_slices = []
    start = 0
    for chunk in zarr_dir.dim0_chunks:
        end = start+chunk
        out_slices.append(slice(start,end))
        start = end

    def _read_one_zarr_one_chunk(zarr_path,out,idx,out_slice):
        data_zarr = zarr.open(zarr_path,mode='r')
        idx = (slice(None),*idx)
        out[out_slice] = data_zarr[idx]

    with concurrent.futures.ThreadPoolExecutor(thread_pool_size) as executor:
        futures = [executor.submit(_read_one_zarr_one_chunk,zarr_path,out,idx,out_slice)
                   for zarr_path,out_slice in zip(zarr_dir.zarr_path_list,out_slices)]
    for future in futures:
        future.result() # raise the errors of the reads
    return out

def _write_one_chunk(data_zarr,data,zarr_slices,data_slices):
    data_zarr[zarr_slices] = data[data_slices]

def parallel_write_zarr(data,data_zarr,slices,thread_pool_size=None,fill_slice=True):
    if fill_slice:
        slices = _fill_slice(data_zarr.shape,slices)
    zarr_1chunk_slices, data_1chunk_slices = \
    _one_chunk_slices_and_out_shape(data_zarr,slices)[:2]

    with concurrent.futures.ThreadPoolExecutor(thread_pool_size) as executor:
        futures = [executor.submit(_write_one_chunk,data_zarr,data,zarr_1chunk_slice,data_1chunk_slice)
                   for zarr_1chunk_slice,data_1chunk_slice in zip(zarr_1chunk_slices,data_1chunk_slices)]
    for future in futures:
        future.result() # raise the errors of the writes
