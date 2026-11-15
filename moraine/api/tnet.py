"""utilities holding temporal network info"""


__all__ = ['TempNet', 'check_image_pairs', 'nimage_from_npair']

import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _imagepair_from_bandwidth(nimages,bandwidth=None):
    if bandwidth is None: bandwidth = nimages
    if bandwidth > nimages:
        raise ValueError(f'bandwidth {bandwidth} is larger than the number of images {nimages}')
    ref, sec = np.triu_indices(nimages, 1)
    idx = np.where((sec-ref)<=bandwidth)
    return np.stack((ref[idx], sec[idx]),axis=-1).astype(np.int32)

def check_image_pairs(image_pairs):
    """Raise a ValueError unless the image pairs follow the data conventions.

    The conventions: at least one pair, image indices 0 or larger, every reference index smaller than its
    secondary index, the pairs sorted by reference and then secondary index without duplicates. The error names
    the first pair that breaks them.

    Parameters
    ----------
    image_pairs : np.ndarray
        (n_pairs, 2) int: the reference and secondary image index of each pair
    """
    image_pairs = np.asarray(image_pairs)
    if image_pairs.ndim != 2 or image_pairs.shape[1] != 2:
        raise ValueError(f'image pairs must have shape (n_pairs, 2), got {image_pairs.shape}')
    if len(image_pairs) == 0:
        raise ValueError('no image pairs')
    if image_pairs.min() < 0:
        raise ValueError(f'image indices must be 0 or larger, got {image_pairs.min()}')
    ref, sec = image_pairs[:, 0].astype(np.int64), image_pairs[:, 1].astype(np.int64)
    bad = np.flatnonzero(ref >= sec)
    if len(bad):
        k = bad[0]
        raise ValueError(f'the reference index must be smaller than the secondary index: pair {k} is '
                         f'({ref[k]}, {sec[k]})')
    n_images = int(sec.max()) + 1
    bad = np.flatnonzero(np.diff(ref * n_images + sec) <= 0) + 1
    if len(bad):
        k = bad[0]
        raise ValueError(f'image pairs must be sorted by reference and then secondary index, without duplicates: '
                         f'pair {k} ({ref[k]}, {sec[k]}) follows ({ref[k - 1]}, {sec[k - 1]})')

def _check_linked(image_pairs):
    """ValueError unless the image pairs (n_pairs, 2), checked by `check_image_pairs`, link all the images 0 to the
    largest index."""
    ref, sec = image_pairs[:, 0].astype(np.int64), image_pairs[:, 1].astype(np.int64)
    n_images = int(sec.max()) + 1
    graph = coo_matrix((np.ones(len(ref), np.int8), (ref, sec)), shape=(n_images, n_images))
    _, component = connected_components(graph, directed=False)
    apart = np.flatnonzero(component != component[0])
    if len(apart):
        listed = ', '.join(str(i) for i in apart[:10]) + (', ...' if len(apart) > 10 else '')
        raise ValueError(f'the image pairs do not link all the images 0 to {n_images - 1}: {len(apart)} images are not '
                         f'linked to image 0: {listed}')

class TempNet(object):
    """Temporal network: the image pairs (interferograms) of a stack of images.

    Parameters
    ----------
    image_pairs : np.ndarray
        (n_pairs, 2) int: the reference and secondary image index of each pair, of the images 0 to n_images - 1,
        n_images being the largest index + 1; sorted by reference and then secondary index without duplicates,
        every reference index smaller than its secondary index (`check_image_pairs`)
    check_if_valid : bool, default: True
        raise a ValueError unless the image pairs follow these conventions and link all the images (every image
        to every other one through a chain of pairs)
    """
    def __init__(self,image_pairs,check_if_valid=True):
        image_pairs = np.asarray(image_pairs)
        if check_if_valid:
            check_image_pairs(image_pairs)
            _check_linked(image_pairs)
        self._image_pairs = image_pairs.astype(np.int32)
        # the largest index: the last pair need not hold it, e.g. [[0, 1], [0, 3], [1, 2]]
        self._n_images = int(self._image_pairs.max()) + 1 if len(self._image_pairs) else 0
        self._image_pairs_1d = self._image_pairs[:,0]*self._n_images+self._image_pairs[:,1]

    @classmethod
    def from_bandwidth(cls, nimages, bandwidth=None):
        image_pairs = _imagepair_from_bandwidth(nimages,bandwidth)
        return cls(image_pairs,check_if_valid=False)

    @property
    def image_pairs(self):
        return self._image_pairs

    def image_pairs_idx(
        self,
        image_pairs=None,
        ref=None,
        sec=None,
    ):
        single_input = False
        if image_pairs is not None:
            ref = image_pairs[:,0]
            sec = image_pairs[:,1]
        else:
            if isinstance(ref,int):
                if not isinstance(sec,int):
                    raise TypeError('ref and sec must both be integers or both sequences')
                single_input = True
                ref = np.array([ref,]); sec = np.array([sec,])
            else:
                ref = np.array(ref); sec = np.array(sec)
            if len(ref) != len(sec):
                raise ValueError(f'ref and sec must have the same length, got {len(ref)} and {len(sec)}')

        query_1d = ref*self._n_images+sec
        mask = np.isin(query_1d,self._image_pairs_1d,assume_unique=True,invert=True)
        result = np.searchsorted(self._image_pairs_1d,query_1d)
        result[mask] = -1
        if single_input:
            result = result[0]
        return result

def nimage_from_npair(npair):
    '''infer number of images from number of pairs assuming all pairs are selected.'''
    possible_nimage = math.ceil(math.sqrt(2*npair))
    for i in [0,-1,1,-2,2]:
        possible_nimage = possible_nimage+i
        if possible_nimage*(possible_nimage-1) == 2*npair:
            return possible_nimage
    raise ValueError('input is not correct value of number of all pairs')
