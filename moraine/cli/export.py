"""Export of point clouds to GIS files (CLI): GeoPackage (decision 0055)"""

__all__ = ['pc_export']

import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import zarr

from ..api.ts import _days, _date_list
from .logging import mc_logger
from .utils_ import ArgumentError

# GeoPackage 1.2 (OGC 12-128r15) written with the sqlite3 of Python, as GDAL writes it: the application id 'GPKG'
# and user_version 10200, the tables of the specification, a feature table with an R-tree spatial index and its
# triggers, and the feature count table of GDAL (gpkg_ogr_contents) for fast counts in QGIS.
_APPLICATION_ID = 0x47504B47
_USER_VERSION = 10200
_WGS84 = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
          'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
          'AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
_SYSTEM_TABLES = [
    'CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL,srs_id INTEGER NOT NULL PRIMARY KEY,organization TEXT '
    'NOT NULL,organization_coordsys_id INTEGER NOT NULL,definition  TEXT NOT NULL,description TEXT)',
    "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY,data_type TEXT NOT NULL,identifier TEXT UNIQUE,"
    "description TEXT DEFAULT '',last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
    'min_x DOUBLE, min_y DOUBLE,max_x DOUBLE, max_y DOUBLE,srs_id INTEGER,CONSTRAINT fk_gc_r_srs_id FOREIGN KEY '
    '(srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))',
    'CREATE TABLE gpkg_ogr_contents(table_name TEXT NOT NULL PRIMARY KEY,feature_count INTEGER DEFAULT NULL)',
    'CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL,column_name TEXT NOT NULL,geometry_type_name TEXT '
    'NOT NULL,srs_id INTEGER NOT NULL,z TINYINT NOT NULL,m TINYINT NOT NULL,CONSTRAINT pk_geom_cols PRIMARY KEY '
    '(table_name, column_name),CONSTRAINT uk_gc_table_name UNIQUE (table_name),CONSTRAINT fk_gc_tn FOREIGN KEY '
    '(table_name) REFERENCES gpkg_contents(table_name),CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES '
    'gpkg_spatial_ref_sys (srs_id))',
    'CREATE TABLE gpkg_extensions (table_name TEXT,column_name TEXT,extension_name TEXT NOT NULL,definition TEXT NOT '
    'NULL,scope TEXT NOT NULL,CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name))',
]
_SRS_ROWS = [
    ('Undefined Cartesian SRS', -1, 'NONE', -1, 'undefined', 'undefined Cartesian coordinate reference system'),
    ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', 'undefined geographic coordinate reference system'),
    ('WGS 84 geodetic', 4326, 'EPSG', 4326, _WGS84,
     'longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid'),
]
# the R-tree triggers of the specification (with GDAL's ST_ functions): made after the points are written, the
# R-tree is filled directly
_RTREE_TRIGGERS = [
    'CREATE TRIGGER "{r}_insert" AFTER INSERT ON {t} WHEN (new."geom" NOT NULL AND NOT ST_IsEmpty(NEW."geom")) BEGIN '
    'INSERT OR REPLACE INTO "{r}" VALUES (NEW."fid",ST_MinX(NEW."geom"), ST_MaxX(NEW."geom"),ST_MinY(NEW."geom"), '
    'ST_MaxY(NEW."geom")); END',
    'CREATE TRIGGER "{r}_update1" AFTER UPDATE OF "geom" ON {t} WHEN OLD."fid" = NEW."fid" AND (NEW."geom" NOTNULL '
    'AND NOT ST_IsEmpty(NEW."geom")) BEGIN INSERT OR REPLACE INTO "{r}" VALUES (NEW."fid",ST_MinX(NEW."geom"), '
    'ST_MaxX(NEW."geom"),ST_MinY(NEW."geom"), ST_MaxY(NEW."geom")); END',
    'CREATE TRIGGER "{r}_update2" AFTER UPDATE OF "geom" ON {t} WHEN OLD."fid" = NEW."fid" AND (NEW."geom" ISNULL '
    'OR ST_IsEmpty(NEW."geom")) BEGIN DELETE FROM "{r}" WHERE id = OLD."fid"; END',
    'CREATE TRIGGER "{r}_update3" AFTER UPDATE ON {t} WHEN OLD."fid" != NEW."fid" AND (NEW."geom" NOTNULL AND NOT '
    'ST_IsEmpty(NEW."geom")) BEGIN DELETE FROM "{r}" WHERE id = OLD."fid"; INSERT OR REPLACE INTO "{r}" VALUES '
    '(NEW."fid",ST_MinX(NEW."geom"), ST_MaxX(NEW."geom"),ST_MinY(NEW."geom"), ST_MaxY(NEW."geom")); END',
    'CREATE TRIGGER "{r}_update4" AFTER UPDATE ON {t} WHEN OLD."fid" != NEW."fid" AND (NEW."geom" ISNULL OR '
    'ST_IsEmpty(NEW."geom")) BEGIN DELETE FROM "{r}" WHERE id IN (OLD."fid", NEW."fid"); END',
    'CREATE TRIGGER "{r}_delete" AFTER DELETE ON {t} WHEN old."geom" NOT NULL BEGIN DELETE FROM "{r}" WHERE id = '
    'OLD."fid"; END',
    'CREATE TRIGGER "trigger_insert_feature_count_{n}" AFTER INSERT ON {t} BEGIN UPDATE gpkg_ogr_contents SET '
    "feature_count = feature_count + 1 WHERE lower(table_name) = lower('{n}'); END",
    'CREATE TRIGGER "trigger_delete_feature_count_{n}" AFTER DELETE ON {t} BEGIN UPDATE gpkg_ogr_contents SET '
    "feature_count = feature_count - 1 WHERE lower(table_name) = lower('{n}'); END",
]
# GeoPackageBinary of a point: header 'GP', version 0, flags 1 (little endian, no envelope), srs_id, then the WKB point
_POINT = np.dtype([('magic', 'S2'), ('version', 'u1'), ('flags', 'u1'), ('srs_id', '<i4'), ('order', 'u1'),
                   ('type', '<u4'), ('x', '<f8'), ('y', '<f8')])
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]*$')
_RESERVED = ('fid', 'geom', 'point')
_ROWS = 16384                      # points per insert: bounds the Python objects of the rows
_P10 = 10.0 ** np.arange(64)


def _decimal(a):
    """float64 of float32 values: the shortest decimal number (at most 9 significant digits) with the same float32
    value, so that 1.6 is written as 1.6 rather than 1.600000023841858; nan stays nan"""
    out = np.asarray(a, dtype=np.float32).astype(np.float64)
    x = out.reshape(-1)
    idx = np.flatnonzero(np.isfinite(x) & (x != 0))
    e = np.floor(np.log10(np.abs(x[idx]))).astype(np.int64)
    # the powers of ten are exact up to 1e22: other magnitudes (rare) through the text of numpy's shortest repr
    far = (e < -14) | (e > 27)
    if far.any():
        x[idx[far]] = np.asarray(a, dtype=np.float32).reshape(-1)[idx[far]].astype(str).astype(np.float64)
        idx, e = idx[~far], e[~far]
    v = x[idx]
    # the nearest decimal of 6 significant digits is the shortest one when one of 6 digits or fewer reads back as
    # the value, those of 7 and 8 next, 9 are always enough for float32
    for digits in (6, 7, 8, 9):
        if not idx.size:
            break
        k = digits - 1 - e
        f = _P10[np.abs(k)]
        up = k >= 0
        if up.all():
            r = np.round(v * f) / f
        else:
            r = np.where(up, np.round(v * f) / f, np.round(v / f) * f)
        ok = r.astype(np.float32) == v.astype(np.float32)
        x[idx[ok]] = r[ok]
        idx, v, e = idx[~ok], v[~ok], e[~ok]
    return out


def _values(a):
    """values of a column of points (1D) or of the columns of a block of points (2D, one list per column) as Python
    objects; nan is written as NULL by SQLite"""
    if a.dtype == np.float32 or a.dtype == np.float16:
        a = _decimal(a)
    return a.T.tolist()


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _sql_type(dtype):
    if dtype == bool:
        return 'BOOLEAN'
    if np.issubdtype(dtype, np.integer):    # the integer types of GeoPackage: 8, 16, 32 and 64 bits
        bits = 8 * dtype.itemsize + (not np.issubdtype(dtype, np.signedinteger))
        return 'TINYINT' if bits <= 8 else 'SMALLINT' if bits <= 16 else 'MEDIUMINT' if bits <= 32 else 'INTEGER'
    if dtype == np.float32 or dtype == np.float16:
        return 'FLOAT'
    if np.issubdtype(dtype, np.floating):
        return 'REAL'
    return None


def _create(con, layer, fields):
    """the GeoPackage tables and the empty feature table and R-tree; `fields`: [(name, sql type)]"""
    con.execute(f'PRAGMA application_id = {_APPLICATION_ID}')
    con.execute(f'PRAGMA user_version = {_USER_VERSION}')
    for sql in _SYSTEM_TABLES:
        con.execute(sql)
    con.executemany('INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)', _SRS_ROWS)
    con.execute('INSERT INTO gpkg_geometry_columns VALUES (?, ?, ?, ?, ?, ?)', (layer, 'geom', 'POINT', 4326, 0, 0))
    con.execute('INSERT INTO gpkg_extensions VALUES (?, ?, ?, ?, ?)',
                (layer, 'geom', 'gpkg_rtree_index', 'http://www.geopackage.org/spec120/#extension_rtree', 'write-only'))
    columns = ', '.join(f'{_quote(n)} {t}' for n, t in fields)
    con.execute(f'CREATE TABLE {_quote(layer)} ("fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "geom" POINT, '
                f'{columns})')
    con.execute(f'CREATE VIRTUAL TABLE {_quote("rtree_" + layer + "_geom")} USING rtree(id, minx, maxx, miny, maxy)')


def _finish(con, layer, n, bbox):
    """the contents of the layer (extent, number of points) and the triggers"""
    con.execute('INSERT INTO gpkg_contents (table_name, data_type, identifier, description, min_x, min_y, max_x, '
                "max_y, srs_id) VALUES (?, 'features', ?, '', ?, ?, ?, ?, 4326)", (layer, layer, *bbox))
    con.execute('INSERT INTO gpkg_ogr_contents VALUES (?, ?)', (layer, n))
    for sql in _RTREE_TRIGGERS:                    # the layer name has no quotes
        con.execute(sql.format(r=f'rtree_{layer}_geom', t=_quote(layer), n=layer))


@mc_logger
def pc_export(
    out:str,
    lon:str,
    lat:str,
    ts:str=None,
    dates:list[str]=None,
    mask:str=None,
    chunks:int=None,
    **columns,
):
    """Write points and their values to a GeoPackage file for GIS software such as QGIS.

    Every point is a feature of one layer with a point geometry at its longitude and latitude (WGS84, EPSG:4326), the
    field `point` with its index in the arrays (0 .. n_points - 1, the `ref_point` of the maps), a field of every
    array given as an extra keyword argument and, with `ts`, a field of every image named D followed by its date (e.g.
    D20190102). The values are written in the units of the arrays (float32 ones as the shortest decimal number of the
    same float32 value), nan as NULL. The layer has a spatial index.

    Parameters
    ----------
    out : str
        output: path of the GeoPackage file, ending with .gpkg; its layer is named after the file (`points` for
        points.gpkg); an existing file is replaced
    lon : str
        input: longitude of the points in degrees (WGS84), shape (n_points,), floating point; finite for the points
        written
    lat : str
        input: latitude of the points in degrees (WGS84), shape (n_points,), floating point; finite for the points
        written
    ts : str, optional
        input: a time series of the points, e.g. the displacement in millimeters (`pc-disp`), shape (n_points,
        nimages), floating point; one field per image; none by default
    dates : list of str, optional
        date of every image of `ts`, YYYYMMDD, in increasing order (`dates` of the metadata of `load-gamma-metadata`);
        needed with `ts`
    mask : str, optional
        input: the points to write, bool, shape (n_points,); all points by default
    chunks : int, optional
        number of points read at a time, the chunk size of `lon` by default; they take about 8 bytes per point and
        field in memory
    **columns
        input: the arrays to write as fields, NAME=zarr path, each of shape (n_points,), bool, integer or floating
        point, e.g. VEL='disp/pc_velocity.zarr' (on the command line: --kw VEL=disp/pc_velocity.zarr); NAME is the name
        of the field: letters, digits and _, starting with a letter, different from fid, geom, point and the fields of
        `ts`, ignoring case
    """
    logger = logging.getLogger(__name__)
    out_path = Path(out)
    if out_path.suffix.lower() != '.gpkg':
        raise ArgumentError(f'out must be a GeoPackage file ending with .gpkg, got {out!r}')
    layer = out_path.stem
    if not layer or layer.lower().startswith(('gpkg_', 'rtree_', 'sqlite_')) or '"' in layer or "'" in layer:
        raise ArgumentError(f'out: the layer is named after the file, whose name must not start with gpkg_, rtree_ or '
                            f'sqlite_ nor have quotes; got {out_path.name!r}')

    lon_zarr = zarr.open(lon, mode='r'); logger.zarr_info(lon, lon_zarr)
    lat_zarr = zarr.open(lat, mode='r'); logger.zarr_info(lat, lat_zarr)
    for name, z in (('lon', lon_zarr), ('lat', lat_zarr)):
        if z.ndim != 1 or not np.issubdtype(z.dtype, np.floating):
            raise ArgumentError(f'{name} must be floating point of shape (n_points,), got {z.dtype} {z.shape}')
    n_points = lon_zarr.shape[0]
    if lat_zarr.shape != (n_points,):
        raise ArgumentError(f'lat must have the shape of lon ({n_points},), got {lat_zarr.shape}')

    fields, zarrs, seen = [('point', 'INTEGER')], [], {}
    for name, path in columns.items():
        if not _NAME.match(name):
            raise ArgumentError(f'field name {name!r}: letters, digits and _, starting with a letter')
        if name.lower() in _RESERVED:
            raise ArgumentError(f'field name {name!r}: fid, geom and point are written by pc-export itself')
        if name.lower() in seen:
            raise ArgumentError(f'field name {name!r} is used twice (field names ignore case): '
                                f'{seen[name.lower()]!r} and {name!r}')
        seen[name.lower()] = name
        z = zarr.open(path, mode='r'); logger.zarr_info(f'{name}: {path}', z)
        if z.shape != (n_points,):
            raise ArgumentError(f'{name}: shape ({n_points},) needed (one value per point), got {z.shape}')
        sql_type = _sql_type(z.dtype)
        if sql_type is None:
            raise ArgumentError(f'{name}: bool, integer or floating point needed, got {z.dtype}')
        fields.append((name, sql_type)); zarrs.append(z)

    ts_zarr = None
    if ts is not None:
        ts_zarr = zarr.open(ts, mode='r'); logger.zarr_info(ts, ts_zarr)
        if ts_zarr.ndim != 2 or ts_zarr.shape[0] != n_points or not np.issubdtype(ts_zarr.dtype, np.floating):
            raise ArgumentError(f'ts must be floating point of shape ({n_points}, nimages), got {ts_zarr.dtype} '
                                f'{ts_zarr.shape}')
        if dates is None:
            raise ArgumentError('dates: the date of every image of ts is needed, YYYYMMDD')
        _days(dates, ts_zarr.shape[1])
        for d in _date_list(dates):
            name = 'D' + d
            if name.lower() in seen:
                raise ArgumentError(f'field name {seen[name.lower()]!r} is also the field of the date {d} of ts')
            seen[name.lower()] = name
            fields.append((name, _sql_type(ts_zarr.dtype)))
    elif dates is not None:
        raise ArgumentError('dates is given without ts: they are the dates of the images of ts')

    mask_zarr = None
    if mask is not None:
        mask_zarr = zarr.open(mask, mode='r'); logger.zarr_info(mask, mask_zarr)
        if mask_zarr.dtype != bool or mask_zarr.shape != (n_points,):
            raise ArgumentError(f'mask must be bool of shape ({n_points},), got {mask_zarr.dtype} {mask_zarr.shape}')
    if chunks is None:
        chunks = lon_zarr.chunks[0]
    if not isinstance(chunks, (int, np.integer)) or chunks < 1:
        raise ArgumentError(f'chunks must be a positive number of points, got {chunks!r}')
    names = [n for n, _ in fields[1:]]
    logger.info(f'layer {layer!r} with the fields point, ' + ', '.join(names[:len(zarrs)] or ['(none)'])
                + (f' and {ts_zarr.shape[1]} dates of ts ({names[len(zarrs)]} .. {names[-1]})'
                   if ts_zarr is not None else ''))

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + '.part')       # renamed when complete
    part.unlink(missing_ok=True)
    con = sqlite3.connect(part, isolation_level=None)
    try:
        con.execute('PRAGMA page_size = 65536')                # 10 % smaller than with 4096 for 100 fields
        con.execute('PRAGMA journal_mode = OFF')               # a new file, deleted on error
        con.execute('PRAGMA synchronous = OFF')
        con.execute('PRAGMA cache_size = -262144')             # 256 MiB of pages, for the R-tree
        _create(con, layer, fields)
        insert = (f'INSERT INTO {_quote(layer)} ("fid", "geom", ' + ', '.join(_quote(n) for n, _ in fields)
                  + f') VALUES ({", ".join("?" * (len(fields) + 2))})')
        insert_rtree = f'INSERT INTO {_quote("rtree_" + layer + "_geom")} VALUES (?, ?, ?, ?, ?)'
        n_written, bbox = 0, [np.inf, np.inf, -np.inf, -np.inf]
        con.execute('BEGIN')
        for start in range(0, n_points, chunks):
            block = slice(start, min(start + chunks, n_points))
            index = np.arange(block.start, block.stop)
            sel = slice(None) if mask_zarr is None else mask_zarr[block]
            index = index[sel]
            if not index.size:
                continue
            x, y = lon_zarr[block][sel], lat_zarr[block][sel]
            bad = ~(np.isfinite(x) & np.isfinite(y))
            if bad.any():
                raise ValueError(f'lon / lat of point {index[bad][0]} is not finite: {x[bad][0]}, {y[bad][0]}')
            if np.abs(y).max() > 90 or np.abs(x).max() > 360:
                raise ValueError(f'lon / lat must be in degrees, got lon {x.min()} .. {x.max()}, lat {y.min()} .. '
                                 f'{y.max()} (not web mercator e / n)')
            bbox = [min(bbox[0], x.min()), min(bbox[1], y.min()), max(bbox[2], x.max()), max(bbox[3], y.max())]
            values = [z[block][sel] for z in zarrs]
            ts_block = None if ts_zarr is None else ts_zarr[block][sel]
            for s in range(0, index.size, _ROWS):
                rows = slice(s, min(s + _ROWS, index.size))
                n = rows.stop - rows.start
                geom = np.zeros(n, dtype=_POINT)
                geom['magic'], geom['flags'], geom['srs_id'], geom['order'], geom['type'] = b'GP', 1, 4326, 1, 1
                geom['x'], geom['y'] = x[rows], y[rows]
                b = geom.tobytes()
                size = _POINT.itemsize
                geoms = [b[i:i + size] for i in range(0, len(b), size)]
                fids = range(n_written + 1, n_written + n + 1)
                cols = [index[rows].tolist()] + [_values(v[rows]) for v in values]
                if ts_block is not None:
                    cols += _values(ts_block[rows])
                con.executemany(insert, zip(fids, geoms, *cols))
                xs, ys = x[rows].tolist(), y[rows].tolist()
                con.executemany(insert_rtree, zip(fids, xs, xs, ys, ys))
                n_written += n
        if not n_written:
            bbox = [None] * 4
        _finish(con, layer, n_written, [None if v is None else float(v) for v in bbox])
        con.execute('COMMIT')
        con.close()
    except BaseException:
        con.close()
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out_path)
    logger.info(f'{n_written} of {n_points} points written to {out} ({out_path.stat().st_size / 2**20:.1f} MiB) in '
                f'{time.time() - t0:.1f} s')
