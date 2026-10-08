"""Deep learning operators: Noise2Fringe (n2f, n2fs3d) and Noise2Fringe Transformer (n2ft) filtering"""

__all__ = ['download_dl_model', 'n2f', 'n2fs3d', 'n2ft']

import numpy as np
from numba import prange
from scipy.spatial import KDTree
import fpsample
import os
import functools

import importlib
from pathlib import Path
import requests
import math
import random
import moraine as mr
from .utils_ import ngjit, ngpjit, mcuda_jit
from .utils_ import is_cuda_available, get_array_module, get_n_cpus_avail
from .chunk_ import chunkwise_slicing_mapping, chunkwise_knn_mapping
if is_cuda_available():
    import cupy as cp

def _fetch_github_file(
    gid:str,
    repo:str,
    branch:str,
    file_path:str,
    out_path:str,
):
    """Parameters
    ----------
    gid : str
        github id
    repo : str
        github repo
    branch : str
        repo branch
    file_path : str
        path to the file
    out_path : str
        output path
    """
    url = '/'.join(['https://raw.githubusercontent.com', gid, repo, 'refs','heads', branch, file_path])
    print(' '.join(['Downloading', url,'to',out_path]))
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File '{out_path}' downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def download_dl_model(
    models:str|list=None,
    path:str=None,
):
    """Parameters
    ----------
    models : str | list, optional
        deep learning models to be downloaded, all available models by default
    path : str, optional
        directory to save these models, inside installed Moraine package by default
    """
    if models is None:
        models = ['n2f','n2fs3d','n2ft']
    else:
        if isinstance(models,str):
            models = [models,]

    if path is None: 
        path = importlib.resources.files('moraine')/'dl_model'
    else:
        path = Path(path)

    if 'n2f' in models:
        _fetch_github_file('kanglcn','n2f','main','n2f.pth',str(path/'n2f.pth'))
    if 'n2fs3d' in models:
        _fetch_github_file('kanglcn','n2f','main','n2fs3d.pth',str(path/'n2fs3d.pth'))
    if 'n2ft' in models:
        _fetch_github_file('kanglcn','n2ft','main','n2ft.pth',str(path/'n2ft.pth'))

def _import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError('PyTorch is required by the deep learning models, '
                          'install it with `pip install torch` or `pip install moraine[dl]`.') from e
    return torch

_model_files = {'n2f':'n2f.pth', 'n2fs3d':'n2fs3d.pth', 'n2ft':'n2ft.pth'}

@functools.lru_cache(maxsize=None)
def _load_model(
    name:str,
    path:str=None,
    device:str='cpu',
    compile:bool=False,
):
    """load a deep learning model for inference. Loaded models are cached.

    Parameters
    ----------
    name : str
        model name, 'n2f', 'n2fs3d' or 'n2ft'
    path : str, optional
        path to the model weights, use the model comes with this package by default
    device : str, default: 'cpu'
        torch device
    compile : bool, default: False
        compile the model with torch.compile
    """
    torch = _import_torch()
    if path is None:
        path = importlib.resources.files('moraine')/'dl_model'/_model_files[name]
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} does not exist, download the models with `moraine.download_dl_model()`.')
    if name == 'n2ft':
        from .n2ft_torch_ import N2FT, PointTransformerBlock
        model = N2FT(PointTransformerBlock,[1,1,1,1,1])
    else:
        from .unet_torch_ import UNet
        model = UNet(2 if name == 'n2f' else 3, 2, depth=4, bilinear=True)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval().to(device)
    if compile:
        model = torch.compile(model)
    return model

def _get_model(name, path=None, device='cpu', compile=False):
    return _load_model(name, None if path is None else str(path), device, compile)

def _cuda_device():
    return f'cuda:{cp.cuda.runtime.getDevice()}'

def _infer_unet(
    model,
    x,
):
    """run the unet model, output is the same kind of array as the input

    Parameters
    ----------
    model
    x
        model input, np.ndarray or cp.ndarray, shape (1, in_channels, nlines, width)
    """
    torch = _import_torch()
    with torch.inference_mode():
        if isinstance(x, np.ndarray):
            device = next(model.parameters()).device
            return model(torch.from_numpy(x).to(device)).cpu().numpy()
        return cp.from_dlpack(model(torch.from_dlpack(cp.ascontiguousarray(x))))

@ngpjit
def _pre_infer_n2f_numba(intf):
    nlines, width = intf.shape
    out = np.empty((1,2,nlines,width),dtype=np.float32)
    mask = np.empty((nlines,width),dtype=np.bool_)
    for i in prange(nlines):
        for j in prange(width):
            intf_i_j = intf[i,j]
            if math.isnan(intf_i_j.real):
                mask[i,j] = True
                random_phase = random.uniform(-math.pi,math.pi)
                out[0,0,i,j] = math.cos(random_phase)
                out[0,1,i,j] = math.sin(random_phase)
            else:
                mask[i,j] = False
                amp = abs(intf_i_j)
                out[0,0,i,j] = intf_i_j.real/amp
                out[0,1,i,j] = intf_i_j.imag/amp
    return out, mask

if is_cuda_available():
    from numba import cuda

    @mcuda_jit()
    def _pre_infer_cuda(intf, adi, out, mask):
        # one thread per pixel: unit interferogram in the last two channels of out, adi (if not empty) in the first
        t = cuda.grid(1)
        nlines, width = intf.shape
        if t >= nlines*width:
            return
        i = t//width; j = t%width
        c0 = out.shape[1]-2
        v = intf[i,j]
        masked = math.isnan(v.real)
        if adi.size > 0:
            masked = masked or math.isnan(adi[i,j])
            out[0,0,i,j] = np.float32(1.0) if masked else adi[i,j]
        mask[i,j] = masked
        if not masked:
            amp = abs(v)
            out[0,c0,i,j] = v.real/amp
            out[0,c0+1,i,j] = v.imag/amp

    def _pre_infer_cp(intf, adi=None):
        """model input (1, 2 or 3, nlines, width) float32 and the mask of pixels without data; random phase there"""
        nlines, width = intf.shape
        out = cp.empty((1,2 if adi is None else 3,nlines,width),dtype=cp.float32)
        mask = cp.empty((nlines,width),dtype=cp.bool_)
        adi = cp.empty((0,0),dtype=cp.float32) if adi is None else adi
        if nlines*width > 0:
            _pre_infer_cuda[(nlines*width+127)//128, 128](intf, adi, out, mask)
        c = out.shape[1]-2
        nan_pos = cp.where(mask)
        random_phase = cp.random.uniform(-cp.pi,cp.pi,len(nan_pos[0]))
        out[0,c,nan_pos[0],nan_pos[1]] = cp.cos(random_phase)
        out[0,c+1,nan_pos[0],nan_pos[1]] = cp.sin(random_phase)
        return out, mask

    def _pre_infer_n2f_cp(intf):
        return _pre_infer_cp(intf)

@ngpjit
def _after_infer_n2f_numba(
    infer_out,
    mask,
):
    nlines, width = mask.shape
    out = np.empty((nlines,width),dtype=np.complex64)
    for i in prange(nlines):
        for j in prange(width):
            if mask[i,j]:
                out[i,j] = np.nan+1j*np.nan
            else:
                out[i,j] = infer_out[0,0,i,j]+1j*infer_out[0,1,i,j]
    return out

if is_cuda_available():
    @mcuda_jit()
    def _after_infer_n2f_cuda(infer_out, mask, out):
        t = cuda.grid(1)
        nlines, width = mask.shape
        if t >= nlines*width:
            return
        i = t//width; j = t%width
        if mask[i,j]:
            out[i,j] = complex(math.nan, math.nan)
        else:
            out[i,j] = complex(infer_out[0,0,i,j], infer_out[0,1,i,j])

    def _after_infer_n2f_cp(infer_out,mask):
        nlines, width = mask.shape
        out = cp.empty((nlines,width),dtype=cp.complex64)
        if nlines*width > 0:
            _after_infer_n2f_cuda[(nlines*width+127)//128, 128](infer_out, mask, out)
        return out

def _infer_n2f_cpu(
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2f_numba(intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_numba(infer_out,mask)

def _infer_n2f_gpu(
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2f_cp(intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_cp(infer_out,mask)

def _nan_where_zero(x):
    """copy of `x` (numpy or cupy) with NaN where |x| < 1e-30, the input is not changed"""
    # GAMMA writes 0 where there are no data; should be done when loading the GAMMA results and removed here
    xp = get_array_module(x)
    return xp.where(xp.abs(x)<1e-30, x.dtype.type(np.nan+1j*np.nan), x)

def n2f(
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Parameters
    ----------
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    xp = get_array_module(intf)
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = xp.empty_like(intf)

    if xp is np:
        model = _get_model('n2f', model, 'cpu')
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2f_cpu(_nan_where_zero(intf[in_slice]),model)[map_slice]
    else:
        model = _get_model('n2f', model, _cuda_device())
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2f_gpu(_nan_where_zero(intf[in_slice]),model)[map_slice]

    return out

def _n2f_np_in_gpu(
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """compare with n2f, input and output are np.ndarray but use gpu for inference

    Parameters
    ----------
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = np.empty_like(intf)

    model = _get_model('n2f', model, _cuda_device())
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        out[out_slice] = _infer_n2f_gpu(_nan_where_zero(cp.asarray(intf[in_slice])),model)[map_slice].get()
    return out

@ngpjit
def _pre_infer_n2fs3d_numba(adi,intf):
    nlines, width = intf.shape
    out = np.empty((1,3,nlines,width),dtype=np.float32)
    mask = np.empty((nlines,width),dtype=np.bool_)
    for i in prange(nlines):
        for j in prange(width):
            intf_i_j = intf[i,j]
            adi_i_j = adi[i,j]
            if math.isnan(intf_i_j.real) or math.isnan(adi_i_j):
                mask[i,j] = True
                out[0,0,i,j] = 1.0
                random_phase = random.uniform(-math.pi,math.pi)
                out[0,1,i,j] = math.cos(random_phase)
                out[0,2,i,j] = math.sin(random_phase)
            else:
                mask[i,j] = False
                out[0,0,i,j] = adi_i_j
                amp = abs(intf_i_j)
                out[0,1,i,j] = intf_i_j.real/amp
                out[0,2,i,j] = intf_i_j.imag/amp
    return out, mask

if is_cuda_available():
    def _pre_infer_n2fs3d_cp(adi,intf):
        return _pre_infer_cp(intf, adi)

def _infer_n2fs3d_cpu(
    adi,
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2fs3d_numba(adi,intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_numba(infer_out,mask)

def _infer_n2fs3d_gpu(
    adi,
    intf,
    model,
):
    input_intf, mask = _pre_infer_n2fs3d_cp(adi,intf)
    infer_out = _infer_unet(model, input_intf)
    return _after_infer_n2f_cp(infer_out,mask)

def n2fs3d(
    adi:np.ndarray,
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """Parameters
    ----------
    adi : np.ndarray
        amplitude dispersion index, 2d np.float32 or cp.float32
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    xp = get_array_module(intf)
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = xp.empty_like(intf)

    if xp is np:
        model = _get_model('n2fs3d', model, 'cpu')
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2fs3d_cpu(adi[in_slice],_nan_where_zero(intf[in_slice]),model)[map_slice]
    else:
        model = _get_model('n2fs3d', model, _cuda_device())
        for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
            out[out_slice] = _infer_n2fs3d_gpu(adi[in_slice],_nan_where_zero(intf[in_slice]),model)[map_slice]

    return out

def _n2fs3d_np_in_gpu(
    adi:np.ndarray,
    intf:np.ndarray,
    chunks:tuple=None,
    depths:tuple=(0,0),
    model:str=None,
):
    """compare with n2f, input and output are np.ndarray but use gpu for inference

    Parameters
    ----------
    adi : np.ndarray
        adi, 2d np.float32
    intf : np.ndarray
        interferogram, 2d np.complex64 or cp.complex64
    chunks : tuple, optional
        chunksize, intf.shape by default
    depths : tuple, default: (0, 0)
        width of the boundary
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    """
    shape = intf.shape
    if chunks is None: chunks = shape
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape,chunks,depths)
    out = np.empty_like(intf)

    model = _get_model('n2fs3d', model, _cuda_device())
    for in_slice, out_slice, map_slice in zip(in_slices, out_slices, map_slices):
        out[out_slice] = _infer_n2fs3d_gpu(cp.asarray(adi[in_slice]),_nan_where_zero(cp.asarray(intf[in_slice])),model)[map_slice].get()
    return out

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

def _sample_and_knn(pos, k=16, workers=-1):  # (N, 2)
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

@ngpjit
def _intf_redim2torch(intf):
    # convert (n, m) complex intf to (m, n, 2)
    n, m = intf.shape
    out = np.empty((m, n, 2),dtype=np.float32)
    for i in prange(n):
        for j in range(m):
            intf_i_j = intf[i,j]
            amp_ = abs(intf_i_j)
            out[j,i,0] = intf_i_j.real/amp_
            out[j,i,1] = intf_i_j.imag/amp_
    return out

@ngpjit
def _intf_redim_back(intf):
    # convert (m, n, 2) to (n, m) complex array
    m, n = intf.shape[:2]
    out = np.empty((n, m),dtype=np.complex64)
    for i in prange(n):
        for j in range(m):
            out[i, j] = intf[j,i,0]+intf[j,i,1]*1j
    return out

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

def _infer_n2ft_structure(
    structure,
    intf,
    model,
):
    """Parameters
    ----------
    structure
        of the points, from `_n2ft_structure` on the device of `model`
    intf
        (n,m)
    model
    """
    torch = _import_torch()
    device = next(model.parameters()).device
    pos, keys = structure
    intf = _intf_redim2torch(intf)
    out = np.empty_like(intf)

    with torch.inference_mode():
        intf = torch.from_numpy(intf).to(device)
        for i in range(intf.shape[0]):
            out[i] = model(pos, intf[i:i+1], *keys).cpu().numpy()[0]
    out = _intf_redim_back(out)
    return out

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
    return _infer_n2ft_structure(_n2ft_structure(x, y, next(model.parameters()).device), intf, model)

def n2ft(
    x:np.ndarray,
    y:np.ndarray,
    intf:np.ndarray,
    chunks:int=None,
    k:int=128,
    model:str=None,
    cuda:bool=False,
    compile:bool=False,
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
        halo size for chunkwise processing
    model : str, optional
        path to the model weights (.pth), use the model comes with this package by default
    cuda : bool, default: False
        use gpu for inference
    compile : bool, default: False
        compile the model with torch.compile, faster on gpu but the first call takes tens of seconds
    """
    model = _get_model('n2ft', model, 'cuda' if cuda else 'cpu', compile)

    single_intf = False
    if len(intf.shape) == 1:
        single_intf = True
        intf = intf[:,None]

    n = intf.shape[0]
    if (chunks is None) or (chunks >= n):
        out = _infer_n2ft(x, y, intf, model)
    else:
        out = np.empty_like(intf)
        in_indices, out_slices, map_indices = chunkwise_knn_mapping(x, y, chunks, k=k)
        for in_idx, out_slice, map_idx in zip(in_indices, out_slices, map_indices):
            out[out_slice] = _infer_n2ft(x[in_idx],y[in_idx],intf[in_idx],model)[map_idx]

    if single_intf:
        out = out[:,0]
    return out
