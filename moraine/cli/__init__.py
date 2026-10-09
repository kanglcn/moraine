from .logging import get_logger
from .utils_ import *
from .zarr_ import *
from .transform import *
from .tnet import *
from .math import *
from .polygon import *
from .load import load_gamma_flatten_rslc, load_gamma_lat_lon_hgt, load_gamma_look_vector, load_gamma_range, load_gamma_metadata
from .pc import *
from .ps import *
from .shp import *
from .co import *
from .pl import *
from .dl import *
from .pqm import *
from .pu import *
from .plot import *
from .tiles import *


def __getattr__(name):
    # the notebook widget imports anywidget / ipywidgets: only when used, so that the command line starts fast
    if name == 'TileView':
        from .viewer import TileView
        return TileView
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
