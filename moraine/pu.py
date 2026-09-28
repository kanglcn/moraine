"""phase unwrapping"""


__all__ = ['gamma_mcf_pt', 'mcf_pc']

import math
import numpy as np
import tempfile
from pathlib import Path
import os
from .gamma_ import read_gamma_pdata, read_gamma_plist, write_gamma_image, write_gamma_plist

from numba import njit
from scipy.spatial import Delaunay
from scipy.sparse import csgraph as csg
import scipy.sparse as sp
from ortools.graph.python import min_cost_flow

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
        reference point, the first point by default

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
            ph_wieght = ph_weight.astype(np.float32)
            write_gamma_image(ph_weight,ph_weight_path)

        mcf_pt_command = f'mcf_pt {str(pc_path)} - {str(ph_path)} - {str(ph_weight_path)} - {str(unwrap_ph_path)} - - {ref_point} &> {temp_dir/"gamma.log"}'
        os.system(mcf_pt_command)

        unwrap_ph = read_gamma_pdata(unwrap_ph_path,dtype='float')
        unwrap_ph = unwrap_ph.reshape(ph.shape)
    return unwrap_ph

def _delaunay(x, y):
    points = np.vstack((x,y)).T
    tri = Delaunay(points,)
    assert tri.coplanar.shape[0] == 0, "input coordinates are not unique."
    return tri.simplices, tri.neighbors

def _edge_for_graph_and_dual_graph(simplex, simplex_neighbors):

    # simplex: 0， 1， 2
    # edges: (2, 0), (0, 1), (1, 2)
    edges = np.stack((np.roll(simplex, 1, 1), simplex), axis=2).reshape(-1, 2) # directed
    sort_idx = np.argsort(edges, axis=1)
    sorted_edges = np.take_along_axis(edges, sort_idx, axis=1)
    unique_edges, unique_idx  = np.unique(sorted_edges, axis=0, return_index=True)

    num_simplex = simplex.shape[0]
    simplex_edges = np.stack((
        np.broadcast_to(np.arange(num_simplex)[:, None], simplex_neighbors.shape),
        np.roll(simplex_neighbors, -1, 1)), axis=2
    ).reshape(-1, 2) % (num_simplex + 1)
    # this % make -1 become num_simplex, therefore, it connect to the earth simplex

    # simplex_edges: (0,1), (0,2), (0,0)
    # so each simplex_edge exactly match point edge

    unique_simplex_edges = np.take_along_axis(simplex_edges, sort_idx, axis=1)[unique_idx]

    return unique_edges, unique_simplex_edges

@njit(fastmath=True)
def _compute_supplys(psi, simplex):
    n_simplex = simplex.shape[0]
    n_vertex = simplex.shape[1]
    two_pi = 2.0 * np.pi

    supplys = np.empty(n_simplex + 1, dtype=np.int32)

    for i in range(n_simplex):
        s = 0.0
        for j in range(n_vertex):
            a = simplex[i, j]
            b = simplex[i, (j - 1) % n_vertex]
            diff = psi[a] - psi[b]
            # wrap to (-pi, pi]
            diff = (diff + np.pi) % two_pi - np.pi
            s += diff
        supplys[i] = int(np.round(s * 0.5 / np.pi))

    # last element: earth node
    supplys[-1] = -np.sum(supplys[:-1])
    return supplys

def _solve_mcf_or(simplex_edges, supplys, capacity, weights):
    smcf = min_cost_flow.SimpleMinCostFlow()
    smcf.add_arcs_with_capacity_and_unit_cost(simplex_edges[:,0], simplex_edges[:,1], capacity, weights)
    smcf.add_arcs_with_capacity_and_unit_cost(simplex_edges[:,1], simplex_edges[:,0], capacity, weights)

    smcf.set_nodes_supplies(np.arange(supplys.shape[0]), supplys)
    status = smcf.solve()
    assert status == 1

    n_edge = simplex_edges.shape[0]
    flows = smcf.flows(np.arange(smcf.num_arcs()))
    flows = flows[n_edge:] - flows[:n_edge]
    return flows

@njit(fastmath=True, nogil=True)
def _build_adjacency(npts, edges):
    # Count neighbors
    counts = np.zeros(npts, dtype=np.int32)
    for a, b in edges:
        counts[a] += 1
        counts[b] += 1

    indptr = np.empty(npts + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(npts):
        indptr[i + 1] = indptr[i] + counts[i]

    indices = np.empty(indptr[-1], dtype=np.int32)
    # Precompute edge positions
    edge_pos_a = np.empty(len(edges), dtype=np.int32)
    edge_pos_b = np.empty(len(edges), dtype=np.int32)

    start = indptr.copy()
    for k in range(len(edges)):
        a = edges[k,0]
        b = edges[k,1]

        ia = start[a]
        ib = start[b]

        indices[ia] = b
        indices[ib] = a

        edge_pos_a[k] = ia
        edge_pos_b[k] = ib

        start[a] += 1
        start[b] += 1

    return indptr, indices, edge_pos_a, edge_pos_b

@njit(fastmath=True)
def _compute_diff_dual(psi, edges, flows, edge_pos_a, edge_pos_b):
    n_edge = edges.shape[0]
    two_pi = 2.0 * np.pi
    diff_dual = np.empty(n_edge * 2, dtype=psi.dtype)

    for i in range(n_edge):
        a = edges[i, 0]
        b = edges[i, 1]
        diff = psi[b] - psi[a]
        diff = (diff + np.pi) % two_pi - np.pi  # wrap_func
        diff += flows[i] * two_pi
        diff_dual[edge_pos_a[i]] = diff
        diff_dual[edge_pos_b[i]] = -diff

    return diff_dual

@njit(fastmath=True, cache=True)
def _unwrap_points_bfs(phase, indptr, indices, gradients):
    """
    BFS-based flood fill using prebuilt CSR adjacency (fast, Numba-compiled).
    """
    npts = len(phase)
    unwrapped = np.zeros(npts, dtype=np.float64)
    visited = np.zeros(npts, dtype=np.bool_)
    visited[0] = True

    queue = [0]
    while queue:
        i = queue.pop(0)
        start, end = indptr[i], indptr[i + 1]
        for idx in range(start, end):
            j = indices[idx]
            g = gradients[idx]
            if not visited[j]:
                unwrapped[j] = unwrapped[i] + g
                visited[j] = True
                queue.append(j)
    return unwrapped + phase[0]

def _prepare_mcf(x,y):
    simplex, simplex_neighbors = _delaunay(x,y)
    edges, simplex_edges = _edge_for_graph_and_dual_graph(simplex, simplex_neighbors)
    n_points = x.shape[0]
    indptr, indices, edge_pos_a, edge_pos_b = _build_adjacency(n_points, edges)
    return simplex, edges, simplex_edges, indptr, indices, edge_pos_a, edge_pos_b

def _solve_mcf(ph, simplex, edges, simplex_edges, indptr, indices, edge_pos_a, edge_pos_b, capacity=int(1e9), weight=1):
    psi = np.angle(ph).astype(np.float32)
    supplys = _compute_supplys(psi, simplex)
    flows = _solve_mcf_or(simplex_edges, supplys, capacity, weight)
    diff_dual = _compute_diff_dual(psi, edges, flows, edge_pos_a, edge_pos_b)
    unw = _unwrap_points_bfs(psi, indptr, indices, diff_dual)
    return unw

def mcf_pc(
    pc_x:np.ndarray,
    pc_y:np.ndarray,
    ph:np.ndarray,
)-> np.ndarray:
    """Minimum cost flow phase unwrapping solver.
Note that the coordinates (pc_x, pc_y) must be unique.

    Parameters
    ----------
    pc_x : np.ndarray
        x coordinate, shape of (N,)
    pc_y : np.ndarray
        y coordinate, shape of (N,)
    ph : np.ndarray
        wrapped phase, shape of (N,), np.complex64

    Returns
    -------
    np.ndarray
        unwrapped phase, shape of (N,)
    """
    # pc_x = pc_x.astype(np.int32)
    # pc_y = pc_y.astype(np.int32)
    required_data = _prepare_mcf(pc_x, pc_y)
    unw = _solve_mcf(ph, *required_data)
    return unw
