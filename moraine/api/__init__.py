"""API: algorithms on numpy / cupy arrays in memory, grouped by the stages of the processing chain and the foundations
they share (decision 0041); the internal utilities are in moraine.api.utils_"""

from . import pc, tnet, polygon, slc, ps, ds, dl, pqm, unwrap, ts

# foundations: point clouds, image pairs, polygons, SLC basics
from .pc import *
from .tnet import *
from .polygon import *
from .slc import *

# stages: PS, DS, deep learning filters, pixel quality, phase unwrapping, deformation products
from .ps import *
from .ds import *
from .dl import *
from .pqm import *
from .unwrap import *
from .ts import *

__all__ = [n for m in (pc, tnet, polygon, slc, ps, ds, dl, pqm, unwrap, ts) for n in m.__all__]
