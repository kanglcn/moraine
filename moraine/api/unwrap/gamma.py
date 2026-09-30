"""GAMMA's point cloud unwrapping"""

__all__ = ['gamma_mcf_pt']

import os
import tempfile
from pathlib import Path

import numpy as np

from ..gamma_ import read_gamma_pdata, write_gamma_image, write_gamma_plist

def gamma_mcf_pt(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
    ph_weight:np.ndarray=None,
    ref_point:int=0,
) -> np.ndarray:
    """A simple wrapper for mcf_pt in GAMMA software, only work if you have access to mcf_pt.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape of (N,)
    pc_y : np.ndarray
        y coordinate, shape of (N,)
    ph : np.ndarray
        wrapped phase, shape of (N,) or (N,M)
    ph_weight : np.ndarray, optional
        point weight, shape of (N,) or (N,M), optional
    ref_point : int, default: 0
        index of the reference point (from 0), the first point by default

    Returns
    -------
    np.ndarray
        unwrapped phase, shape of (N,) or (N,M)
    """
    pc_x = pc_x.astype(np.int32)
    pc_y = pc_y.astype(np.int32)
    pc_xy = np.stack((pc_x,pc_y),axis=-1)
    ph = ph.astype(np.complex64)

    with tempfile.TemporaryDirectory() as tempdir_str:
        temp_dir = Path(tempdir_str)
        pc_path = temp_dir/'pc'
        ph_path = temp_dir/'ph'
        unwrap_ph_path = temp_dir/'unwrap_ph'
        write_gamma_plist(pc_xy,pc_path)
        write_gamma_image(ph,ph_path)
        if ph_weight is None:
            ph_weight_path = '-'
        else:
            ph_weight_path = temp_dir/'ph_weight'
            ph_weight = ph_weight.astype(np.float32)   # mcf_pt reads FLOAT
            write_gamma_image(ph_weight,ph_weight_path)

        mcf_pt_command = f'mcf_pt {str(pc_path)} - {str(ph_path)} - {str(ph_weight_path)} - {str(unwrap_ph_path)} - - {ref_point} &> {temp_dir/"gamma.log"}'
        os.system(mcf_pt_command)

        unwrap_ph = read_gamma_pdata(unwrap_ph_path,dtype='float')
        unwrap_ph = unwrap_ph.reshape(ph.shape)
    return unwrap_ph
