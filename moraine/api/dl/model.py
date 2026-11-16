"""Deep learning models of moraine: download, loading and caching"""

__all__ = ['download_dl_model']

import functools
import importlib
import os
from pathlib import Path

def _https_context():
    """The SSL context of the downloads: None, urllib's default context, which trusts the certificate authorities of
    the environment variables SSL_CERT_FILE and SSL_CERT_DIR, where one of them is set (not empty); else one with the
    CA bundle of certifi where the package certifi can be imported; else None, the default context with the
    certificates of the Python installation."""
    # SSL_CERT_FILE / SSL_CERT_DIR give certificate authorities of their own, e.g. that of a company's TLS-inspecting
    # proxy: OpenSSL loads them in place of its default file / directory when the default context is made, and a
    # context with the CA bundle of certifi would ignore them. An empty value counts as not set.
    if os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR'):
        return None
    # python.org builds of Python on macOS have no certificates of their own until "Install Certificates" is run;
    # certifi is not a dependency of moraine
    try:
        import certifi
    except ImportError:
        return None
    import ssl
    return ssl.create_default_context(cafile=certifi.where())

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
    import http.client
    import urllib.request
    url = '/'.join(['https://raw.githubusercontent.com', gid, repo, 'refs','heads', branch, file_path])
    print(' '.join(['Downloading', url,'to',out_path]))
    try:
        # an HTTPError for bad status codes (4xx or 5xx); read() raises IncompleteRead for a file cut short; a CA
        # bundle of certifi that cannot be loaded (an OSError) is a failed download too
        with urllib.request.urlopen(url, context=_https_context()) as response:
            content = response.read()
    except (OSError, http.client.HTTPException) as e:   # URLError, connection and read errors: printed, not raised
        print(f"Error downloading file: {e}")
        return
    with open(out_path, 'wb') as f:      # errors of the output file are raised
        f.write(content)
    print(f"File '{out_path}' downloaded successfully.")

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
    half:bool=False,
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
    half : bool, default: False
        run the model in half precision (float16), on a GPU only
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
    model.eval()
    if name == 'n2ft':
        # a multiply-add per channel instead of the cuDNN batch norm kernels, which are slower and whose size
        # thresholds would make torch.compile compile the model once per range of batch sizes
        from .n2ft_torch_ import fold_batch_norms
        fold_batch_norms(model)
    else:
        # the batch norms folded into the convolutions: no batch norm kernels
        from .unet_torch_ import fold_batch_norms
        fold_batch_norms(model)
    model.to(device)
    if half:
        # float16 weights and activations in the channels last layout, the layout of the tensor core convolution kernels
        model = model.half().to(memory_format=torch.channels_last)
    if compile:
        # n2ft is called with a different number of points and interferograms every time: one graph with symbolic
        # shapes instead of a compilation per shape
        model = torch.compile(model, dynamic=True if name == 'n2ft' else None)
    return model

def _get_model(name, path=None, device='cpu', compile=False, half=False):
    return _load_model(name, None if path is None else str(path), device, compile, half)

def _cuda_device():
    import cupy as cp
    return f'cuda:{cp.cuda.runtime.getDevice()}'
