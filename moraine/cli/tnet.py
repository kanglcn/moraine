"""Temporal network (image pairs) for the CLI"""

__all__ = ['image_pairs']

import logging
from pathlib import Path

import numpy as np
import zarr

from ..tnet import TempNet
from .logging import mc_logger


@mc_logger
def image_pairs(
    out:str,
    rslc:str=None,
    nimages:int=None,
    bandwidth:int=None,
):
    """Write the image pairs of a temporal network to a text file, for the `image_pairs` arguments.

    Parameters
    ----------
    out : str
        output: text file with two columns, reference and secondary image index
    rslc : str, optional
        input: rslc stack (..., nimages), used to get the number of images; alternative to `nimages`
    nimages : int, optional
        number of images; alternative to `rslc`
    bandwidth : int, optional
        connect every image with the next `bandwidth` images (1: sequential interferograms); all image pairs
        by default
    """
    logger = logging.getLogger(__name__)
    if (rslc is None) == (nimages is None):
        raise ValueError('give either `rslc` or `nimages`')
    if rslc is not None:
        nimages = zarr.open(rslc, mode='r').shape[-1]
    pairs = TempNet.from_bandwidth(nimages, bandwidth).image_pairs
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, pairs, fmt='%d', header='reference secondary')
    logger.info(f'{len(pairs)} image pairs of {nimages} images written to {out}')
