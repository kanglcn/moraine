"""Polygons drawn on results (GeoJSON files) and point in polygon tests"""

__all__ = ['read_polygons', 'write_polygons', 'polygons_contain']

import json
from pathlib import Path

import numpy as np

# coordinates of the polygon vertices: longitude / latitude (GeoJSON default) or the radar grid (range, azimuth)
COORDINATES = ('lonlat', 'radar_grid')


def write_polygons(
    path:str,
    polygons:list,
    coordinates:str='lonlat',
):
    """Write polygons to a GeoJSON file (FeatureCollection of Polygon features).

    Parameters
    ----------
    path : str
        output GeoJSON file
    polygons : list
        polygons, each a list of (x, y) vertices, at least 3; the ring is closed in the file
    coordinates : str, default: 'lonlat'
        'lonlat' (x longitude, y latitude in degrees, standard GeoJSON) or 'radar_grid' (x range, y azimuth
        pixel coordinates, stored as ``moraine_coordinates`` in the file)
    """
    if coordinates not in COORDINATES:
        raise ValueError(f'coordinates must be one of {COORDINATES}, not {coordinates!r}')
    features = []
    for poly in polygons:
        ring = [[float(x), float(y)] for x, y in poly]
        if len(ring) < 3:
            raise ValueError(f'a polygon needs at least 3 vertices, got {len(ring)}')
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append({'type': 'Feature', 'properties': {},
                         'geometry': {'type': 'Polygon', 'coordinates': [ring]}})
    out = {'type': 'FeatureCollection', 'moraine_coordinates': coordinates, 'features': features}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1) + '\n')


def read_polygons(
    path:str,
)->tuple:
    """Read the polygons of a GeoJSON file written by `write_polygons` or another program.

    Parameters
    ----------
    path : str
        GeoJSON file with Polygon or MultiPolygon features (holes are ignored)

    Returns
    -------
    tuple
        list of polygons, each an (n_vertices, 2) float64 array of (x, y) with the closing vertex removed,
        and the coordinates, 'lonlat' or 'radar_grid'
    """
    data = json.loads(Path(path).read_text())
    coordinates = data.get('moraine_coordinates', 'lonlat')
    if coordinates not in COORDINATES:
        raise ValueError(f'{path}: unknown moraine_coordinates {coordinates!r}')
    features = data['features'] if data.get('type') == 'FeatureCollection' else [data]
    polygons = []
    for feature in features:
        geometry = feature.get('geometry', feature)
        if geometry['type'] == 'Polygon':
            rings = [geometry['coordinates'][0]]
        elif geometry['type'] == 'MultiPolygon':
            rings = [poly[0] for poly in geometry['coordinates']]
        else:
            continue
        for ring in rings:
            ring = np.asarray(ring, dtype=np.float64)[:, :2]
            if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
                ring = ring[:-1]
            polygons.append(ring)
    return polygons, coordinates


def polygons_contain(
    polygons:list,
    x:np.ndarray,
    y:np.ndarray,
)->np.ndarray:
    """Whether points are inside any of the polygons.

    Parameters
    ----------
    polygons : list
        polygons, each an (n_vertices, 2) array of (x, y) vertices, in the coordinates of `x`, `y`
    x : np.ndarray
        x coordinates of the points, any shape
    y : np.ndarray
        y coordinates of the points, same shape as `x`

    Returns
    -------
    np.ndarray
        bool, same shape as `x`: True inside at least one polygon; points exactly on an edge may be on
        either side; nan coordinates are outside
    """
    from matplotlib.path import Path as MplPath
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    points = np.stack([x.ravel(), y.ravel()], axis=-1)
    inside = np.zeros(points.shape[0], dtype=bool)
    finite = np.isfinite(points).all(axis=1)
    for poly in polygons:
        inside[finite] |= MplPath(np.asarray(poly, dtype=np.float64)).contains_points(points[finite])
    return inside.reshape(x.shape)
