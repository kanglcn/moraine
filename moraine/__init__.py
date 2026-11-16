__version__ = "0.10.0"

import importlib as _importlib
import sys as _sys

# zarr imports cupy for its GPU buffers wherever cupy is installed (cupy then imports pytest), which moraine does not
# use: zarr is imported with cupy hidden (decision 0048), and sys.modules is restored at once, so that the GPU code
# paths import cupy as usual
if 'zarr' not in _sys.modules and 'cupy' not in _sys.modules:
    _sys.modules['cupy'] = None          # `import cupy` raises ImportError
    try:
        _importlib.import_module('zarr')
    finally:
        _sys.modules.pop('cupy', None)
del _importlib, _sys

from .api import *
from .api import __all__
