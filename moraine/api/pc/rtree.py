"""Rectangle tree for fast point cloud data query"""


__all__ = ['HilbertRtree']

# Some functions adapted from spatialpandas at https://github.com/holoviz/spatialpandas
# under BSD-2-Clause license,

# Further specialized for 2D point cloud data and optimized for zarr.

import numpy as np
from numba import prange
import math
from ..utils_.jit import ngjit, ngpjit
import zarr

@ngjit
def _left_child(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's left child
    """
    return 2 * node + 1

@ngjit
def _right_child(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's right child
    """
    return 2 * node + 2

@ngjit
def _parent(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's parent
    """
    return (node - 1) // 2

@ngjit
def _is_outside(query_bounds, node_bounds):
    if (query_bounds[2] < node_bounds[0]) or (query_bounds[0] > node_bounds[2]):
        return True
    if (query_bounds[3] < node_bounds[1]) or (query_bounds[1] > node_bounds[3]):
        return True
    return False

@ngjit
def _is_inside(query_bounds, node_bounds):
    if ((query_bounds[0] < node_bounds[0]) and (query_bounds[2] > node_bounds[2]) and
        (query_bounds[1] < node_bounds[1]) and (query_bounds[3] > node_bounds[3])):
        return True
    else:
        return False

@ngpjit
def _is_inside_bf(bounds, x, y):
    out = np.empty(x.shape,np.bool_)
    for i in prange(x.size):
        x_i = x[i]; y_i = y[i]
        if (x_i>bounds[0]) and (x_i<bounds[2]) and (y_i>bounds[1]) and (y_i<bounds[3]):
            out[i] = True
        else:
            out[i] = False
    return out

@ngpjit
def _build_hillbert_rtree(x, y, page_size):
    """
        Construct a hillbert rtree, the input coordinates has already been
        sorted according to the hillbert index. Reture the bounds tree
        Args:
            x: x coordinates (e.g., longitude)
            y: y coordinates (e.g., latitude)
            page_size: Number of elements per leaf of the tree.
    """
    # Init bounds_tree array for storing the binary tree representation
    input_size = x.shape[0]
    num_pages = math.ceil(input_size/page_size)

    # no points: the tree of one page, a leaf without points (nan bounds, as the leaves after the last page), which
    # no query enters
    tree_depth = math.ceil(np.log2(num_pages)) if num_pages > 1 else 0
    next_pow2 = 2 ** tree_depth
    tree_length = next_pow2 * 2 - 1
    bounds_tree = np.full((tree_length, 4), np.nan)
    leaf_start = tree_length - next_pow2

    for page in prange(num_pages):
        start = page * page_size
        stop = start + page_size
        page_x = x[start:stop]; page_y = y[start:stop]
        bounds_tree[leaf_start + page, 0] = np.min(page_x)
        bounds_tree[leaf_start + page, 1] = np.min(page_y)
        bounds_tree[leaf_start + page, 2] = np.max(page_x)
        bounds_tree[leaf_start + page, 3] = np.max(page_y)

    # Populate internal layers of tree
    layer = tree_depth - 1
    start = _parent(tree_length - next_pow2)
    stop = _parent(tree_length - 1)

    while layer >= 0:
        for node in prange(start, stop + 1):
            left_bounds = bounds_tree[_left_child(node), :]
            left_valid = not np.isnan(left_bounds[0])
            right_bounds = bounds_tree[_right_child(node), :]
            right_valid = not np.isnan(right_bounds[0])

            if left_valid:
                if right_valid:
                    bounds_tree[node, 0] = min(left_bounds[0],right_bounds[0])
                    bounds_tree[node, 1] = min(left_bounds[1],right_bounds[1])
                    bounds_tree[node, 2] = max(left_bounds[2],right_bounds[2])
                    bounds_tree[node, 3] = max(left_bounds[3],right_bounds[3])
                else:
                    bounds_tree[node, :] = left_bounds

            elif right_valid:
                bounds_tree[node, :] = right_bounds

        # update layer, start/stop
        start = _parent(start)
        stop = _parent(stop)
        layer -= 1

    return bounds_tree

@ngjit
def _expand_ranges(ranges):
    length = np.sum(ranges[:,1]-ranges[:,0])
    idx = np.zeros(length,dtype=np.int64)
    idx_start = 0
    for i in range(ranges.shape[0]):
        start = ranges[i,0]
        stop = ranges[i,1]
        size = stop-start
        idx[idx_start:idx_start+size] = np.arange(start,stop,dtype=np.int64)
        idx_start += size
    return idx

@ngjit
def _node_range(n_nodes, page_size, node):
    """[start, stop) of the point indices under a node: from its leftmost to its rightmost leaf; stop may exceed the
    number of points (the last leaf is not full)"""
    leaf_start = (n_nodes + 1) // 2 - 1
    first = node
    while _left_child(first) < n_nodes:
        first = _left_child(first)
    last = node
    while _right_child(last) < n_nodes:
        last = _right_child(last)
    return (first - leaf_start) * page_size, (last - leaf_start + 1) * page_size

@ngjit
def _maybe_covered_ranges(bounds_tree, n_points, page_size, query_bounds):
    """ranges [start, stop) of the points of the nodes that intersect the query box, in increasing order, adjacent
    ranges of the same kind merged, and whether all their points are in the box (True) or only some may be (False)"""
    n_nodes = bounds_tree.shape[0]
    ranges = np.empty((64, 2), dtype=np.int64)
    is_covered = np.empty(64, dtype=np.bool_)
    n = 0
    nodes = [0]   # depth first, the left child first: the ranges come in increasing order
    while len(nodes) > 0:
        node = nodes.pop()
        node_bounds = bounds_tree[node]
        if np.all(np.isnan(node_bounds)) or _is_outside(query_bounds, node_bounds):   # no points, or outside the box
            continue
        start, stop = _node_range(n_nodes, page_size, node)
        covered = _is_inside(query_bounds, node_bounds)
        if not covered and stop - start > page_size:
            # interior node partly in the box: look at its children
            nodes.append(_right_child(node))
            nodes.append(_left_child(node))
            continue
        if n > 0 and ranges[n-1, 1] == start and is_covered[n-1] == covered:
            ranges[n-1, 1] = stop
            continue
        if n == ranges.shape[0]:
            ranges = np.concatenate((ranges, np.empty_like(ranges)))
            is_covered = np.concatenate((is_covered, np.empty_like(is_covered)))
        ranges[n, 0] = start
        ranges[n, 1] = stop
        is_covered[n] = covered
        n += 1
    if n > 0 and ranges[n-1, 1] > n_points:
        ranges[n-1, 1] = n_points
    return ranges[:n].copy(), is_covered[:n].copy()

def _query_bounds(bounds):
    """query box as a tuple of 4 floats"""
    bounds = tuple(float(b) for b in bounds)
    if len(bounds) != 4:
        raise ValueError(f'bounds must be [x0, y0, xm, ym], got {len(bounds)} values')
    return bounds

@ngjit
def _merged_idx(ranges, is_covered, maybe_covered_is_inside):
    max_len = np.sum(ranges[:,1]-ranges[:,0])
    idx = np.zeros(max_len,dtype=np.int64)
    idx_start = 0
    idx_maybe_covered = 0
    for i in range(ranges.shape[0]):
        start = ranges[i,0]
        stop = ranges[i,1]
        size = stop-start
        if is_covered[i]:
            idx[idx_start:idx_start+size] = np.arange(start,stop,dtype=np.int64)
            idx_start += size
        else:
            _idx = np.arange(start,stop,dtype=np.int64)[maybe_covered_is_inside[idx_maybe_covered:idx_maybe_covered+size]]
            idx_maybe_covered += size
            size = len(_idx)
            idx[idx_start:idx_start+size] = _idx
            idx_start += size
    return idx[:idx_start]

class HilbertRtree:
    """
    This class provides a read-only Hilbert R-tree for spatial indexing.

    See [Hilbert_R-tree](https://en.wikipedia.org/wiki/Hilbert_R-tree) for more info on the Hilbert R-tree.

    This implementation stores the R-tree as an array representation of a binary tree.
    See [Binary_tree](https://en.wikipedia.org/wiki/Binary_tree#Arrays) for more info on the array
    representation of a binary tree.
    """

    def __init__(self, bounds_tree:np.ndarray,
                 n_points:int,
                 page_size=512,
                ):
        """Note that HilbertRtree should not be instantiated directly, 
        but always through the module-level function `build` and `load`.

        Parameters
        ----------
        bounds_tree : np.ndarray
            an array representation of a binary tree. shape [n_pages, 4]. Every row is a bounding box of a node: [x0, y0, xm, ym]
        n_points : int
            number of total points
        page_size : default: 512
            number of points in every leaf node.
        """
        self._bounds_tree = bounds_tree
        self._n_points = n_points
        self._page_size = page_size

    @classmethod
    def build(cls, x:np.ndarray,
              y:np.ndarray,
              page_size=512,
             ):
        """classmethod to build a HilbertRtree.

        Parameters
        ----------
        x : np.ndarray
            x coordinates, e.g., lon, shape [n_points,]; no points (n_points = 0) give a tree whose queries find
            none
        y : np.ndarray
            y coordinates, e.g., lat, shape [n_points,]
        page_size : default: 512
            number of points in every leaf node. should be same as pc_chunk_size.
        """
        if x.shape != y.shape:
            raise ValueError(f'x and y must have the same shape, got {x.shape} and {y.shape}')
        page_size = max(1, page_size)
        bounds_tree = _build_hillbert_rtree(x.astype(np.float64), y.astype(np.float64), page_size)
        return cls(bounds_tree,x.shape[0], page_size)

    def save(self,path:str,
            ):
        """Save the HilbertRtree.

        Parameters
        ----------
        path : str
            zarr path
        """
        bounds_tree_zarr = zarr.open(path,mode='w',shape=self._bounds_tree.shape,dtype=self._bounds_tree.dtype)
        bounds_tree_zarr[:] = self._bounds_tree[:]
        bounds_tree_zarr.attrs['page_size'] = self._page_size
        bounds_tree_zarr.attrs['n_points'] = self._n_points

    @classmethod
    def load(cls, zarr_path:str,
            ):
        """classmethod to load the saved HilbertRtree.

        Parameters
        ----------
        zarr_path : str
            zarr path
        """
        bounds_tree_zarr = zarr.open(zarr_path,mode='r')
        return cls(bounds_tree_zarr[:], bounds_tree_zarr.attrs['n_points'], bounds_tree_zarr.attrs['page_size'])

    @property
    def bounds_tree(self):
        return self._bounds_tree

    def maybe_covered_ranges(self, bounds:tuple|list,
                            )->tuple:
        """Get the ranges that may be within the query box.

        Parameters
        ----------
        bounds : tuple | list
            data query bounding box, [x0, y0, xm, ym]

        Returns
        -------
        tuple
            - array of ranges that may be covered by the query bounding box, shape[n_range, 2]. Every row [start, stop] is the range of query data
            - Bool array indicating if fully convered, shape[n_range,]. True: fully covered (every point in this range are in the box). False: some of points may be in the box.
        """
        return _maybe_covered_ranges(self._bounds_tree, self._n_points, self._page_size, _query_bounds(bounds))

    def bbox_query(self, bounds:tuple|list,
                   x:np.ndarray|zarr.Array,
                   y:np.ndarray|zarr.Array,
                  ):
        """get the index of points that are in the bounding box.

        Parameters
        ----------
        bounds : tuple | list
            data query bounding box, [x0, y0, xm, ym]
        x : np.ndarray | zarr.Array
            x coordinate
        y : np.ndarray | zarr.Array
            y coordinate
        """
        bounds = _query_bounds(bounds)
        ranges, is_covered = _maybe_covered_ranges(self._bounds_tree, self._n_points, self._page_size, bounds)
        maybe_covered_ranges = ranges[~is_covered]
        maybe_covered_index = _expand_ranges(maybe_covered_ranges)
        maybe_covered_x = x[maybe_covered_index]
        maybe_covered_y = y[maybe_covered_index]
        maybe_covered_is_inside = _is_inside_bf(bounds,maybe_covered_x,maybe_covered_y)
        return _merged_idx(ranges, is_covered, maybe_covered_is_inside)
