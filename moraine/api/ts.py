"""Deformation products of point clouds: spatial reference, displacement, DEM error (decision 0048)"""

__all__ = ['pc_reference', 'pc_disp', 'pc_sc_dem_error']

import datetime

import numpy as np
from .utils_.device import get_array_module


def pc_reference(
    data,
    ref,
    method:str='median',
):
    """Values relative to a reference: the reference value of every column is subtracted from the column.

    Parameters
    ----------
    data : np.ndarray
        values of the points, shape (n_points,) or (n_points, n), floating point, numpy or cupy; e.g. the unwrapped
        phase of every image in radians (`ts` of `unwrap_correct_closure_pc`) or displacements
    ref : np.ndarray | int
        the reference: a bool mask of the reference points, shape (n_points,), True for at least one point; or the
        index of one reference point, 0 .. n_points - 1
    method : str, default: 'median'
        reference value of a column from the points of a mask: 'median' or 'mean' of their values in the column,
        nan left out; not used with one reference point (its value is the reference value)

    Returns
    -------
    out : np.ndarray
        `data` minus the reference value of its column, same shape and dtype as `data`; the median (mean, value)
        of the reference points is 0 in every column
    ref_value : np.ndarray
        reference value of every column, shape (n,), or () for data of shape (n_points,), dtype of `data`

    Raises
    ------
    ValueError
        when the reference points have no value (all nan) in a column
    """
    xp = get_array_module(data)
    if method not in ('median', 'mean'):
        raise ValueError(f"method must be 'median' or 'mean', not {method!r}")
    if data.ndim not in (1, 2):
        raise ValueError(f'data must have shape (n_points,) or (n_points, n), got shape {tuple(data.shape)}')
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f'data must be floating point, got {data.dtype}')
    n_points = data.shape[0]
    if isinstance(ref, (bool, np.bool_)):
        raise ValueError('ref must be a bool mask of the points or the index of one point, not a single bool')
    if isinstance(ref, (int, np.integer)):
        if not 0 <= ref < n_points:
            raise ValueError(f'the reference point {ref} is not one of the {n_points} points')
        values = data[int(ref):int(ref) + 1]
    else:
        ref = xp.asarray(ref)
        if ref.dtype != bool or ref.shape != (n_points,):
            raise ValueError(f'the reference mask must be bool of shape ({n_points},), got {ref.dtype} {tuple(ref.shape)}')
        if not bool(ref.any()):
            raise ValueError('the reference mask has no point')
        values = data[ref]
    empty = xp.isnan(values).all(axis=0)
    if bool(empty.any()):
        columns = np.flatnonzero(np.atleast_1d(xp.asnumpy(empty) if xp is not np else empty)).tolist()
        raise ValueError(f'the reference points have no value (all nan) in column(s) {columns[:10]}'
                         f'{" ..." if len(columns) > 10 else ""}')
    if values.shape[0] == 1:
        ref_value = values[0]
    elif method == 'median':
        ref_value = xp.nanmedian(values, axis=0)
    else:
        ref_value = xp.nanmean(values, axis=0)
    ref_value = xp.asarray(ref_value, dtype=data.dtype)
    return data - ref_value, ref_value


def pc_disp(
    phase,
    wavelength:float,
):
    """Line of sight displacement in millimeters from unwrapped phase.

    Parameters
    ----------
    phase : np.ndarray
        unwrapped phase in radians, any shape, floating point, numpy or cupy; the phase of an image relative to a
        reference image, growing when the range gets shorter, e.g. the image phases of `unwrap_correct_closure_pc`
        relative to a reference area (`pc_reference`)
    wavelength : float
        radar wavelength in meters, between 0 and 1 (e.g. 0.0554658 for Sentinel-1)

    Returns
    -------
    disp : np.ndarray
        line of sight displacement in millimeters, same shape as `phase`, float32; positive towards the satellite
        (a shorter range); nan where `phase` is nan
    """
    if not np.issubdtype(phase.dtype, np.floating):
        raise ValueError(f'phase must be floating point (unwrapped phase in radians), got {phase.dtype}')
    wavelength = float(wavelength)
    if not 0 < wavelength < 1:
        raise ValueError(f'wavelength must be in meters, between 0 and 1, got {wavelength}')
    # a two way path: one cycle (2 pi) is half a wavelength of range change
    mm_per_rad = np.float32(1000 * wavelength / (4 * np.pi))
    return phase.astype(np.float32, copy=False) * mm_per_rad


def _days(dates, nimages:int):
    """Days from the first of `dates` (YYYYMMDD; a list, or text with commas or spaces), float64; there must be
    `nimages` dates in increasing order."""
    if isinstance(dates, str):
        dates = dates.replace(',', ' ').split()
    dates = [str(d).strip() for d in dates]
    if len(dates) != nimages:
        raise ValueError(f'{len(dates)} dates for {nimages} images; give the date (YYYYMMDD) of every image, in order')
    days = []
    for d in dates:
        try:
            if len(d) != 8 or not d.isdigit():
                raise ValueError(d)
            days.append(datetime.datetime.strptime(d, '%Y%m%d').toordinal())
        except ValueError:
            raise ValueError(f'dates must be YYYYMMDD, got {d!r}') from None
    days = np.asarray(days, dtype=np.float64)
    back = np.flatnonzero(np.diff(days) <= 0)
    if back.size:
        i = int(back[0])
        raise ValueError(f'dates must increase (the images in date order): {dates[i]} is followed by {dates[i + 1]}')
    return days - days[0]


def _mm_per_m(bperp, slant_range, theta):
    """Line of sight displacement in millimeters of a DEM error of 1 m, elementwise, float64: 1000 B_perp /
    (R cos(theta)), theta the elevation angle of the look vector (cos(theta) = sin(incidence))."""
    xp = get_array_module(bperp)
    f64 = lambda a: xp.asarray(a, dtype=xp.float64)
    return 1000.0 * f64(bperp) / (f64(slant_range) * xp.cos(f64(theta)))


def pc_sc_dem_error(
    disp,
    bperp,
    slant_range,
    theta,
    dates:list,
    ref:int=0,
):
    """Spatially correlated DEM error of points from their displacement time series, and the displacement without it.

    A scatterer dh meters above the height used in the processing (the DEM) adds (c_k - c_ref) dh to the displacement
    of image k, with c_k = 1000 B_perp,k / (R cos(theta)) millimeters per meter and B_perp,k the perpendicular
    baseline of image k. dh is the least squares fit, with equal weights, of the differences of the displacement
    between neighbouring dates by a constant, a constant velocity and the DEM error term: other deformation enters
    the fit only through its changes from one date to the next. After the DEM error of every point relative to its
    neighbourhood was removed from the phase before the unwrapping (`pc_su_dem_error`), dh is the spatially correlated
    part; otherwise it is the whole DEM error.

    Parameters
    ----------
    disp : np.ndarray
        line of sight displacement in millimeters, shape (n_points, nimages), floating point, numpy or cupy; the
        images in date order, relative to the image `ref`, positive towards the satellite (`pc_disp`)
    bperp : np.ndarray
        perpendicular baseline of every image at every point in meters, shape (n_points, nimages), floating point;
        relative to the reference image of the co-registration, positive upwards (`load_gamma_bperp`)
    slant_range : np.ndarray
        slant range of the points in meters, shape (n_points,), floating point
    theta : np.ndarray
        elevation angle of the look vector of the points in radians (90 degrees minus the incidence angle), shape
        (n_points,), floating point
    dates : list of str
        date of every image, YYYYMMDD, in increasing order; at least 4 images. Only the days between them are used
    ref : int, default: 0
        the image `disp` is relative to (its column is 0), 0 .. nimages - 1

    Returns
    -------
    dh : np.ndarray
        DEM error in meters, shape (n_points,), float32: the height of the scatterer minus the height used in the
        processing, positive above it; nan where `disp` or `bperp` of the point has a nan
    out : np.ndarray
        `disp` without the DEM error term (c_k - c_ref) dh, in millimeters, shape (n_points, nimages), float32; the
        column of `ref` is that of `disp`; nan in the rows where `dh` is nan
    """
    xp = get_array_module(disp)
    if disp.ndim != 2:
        raise ValueError(f'disp must have shape (n_points, nimages), got shape {tuple(disp.shape)}')
    n_points, nimages = disp.shape
    for name, a, shape in (('disp', disp, (n_points, nimages)), ('bperp', bperp, (n_points, nimages)),
                           ('slant_range', slant_range, (n_points,)), ('theta', theta, (n_points,))):
        if not np.issubdtype(a.dtype, np.floating):
            raise ValueError(f'{name} must be floating point, got {a.dtype}')
        if tuple(a.shape) != shape:
            raise ValueError(f'{name} must have shape {shape}, got shape {tuple(a.shape)}')
    if nimages < 4:
        raise ValueError(f'the DEM error needs at least 4 images, got {nimages}')
    if isinstance(ref, (bool, np.bool_)) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < nimages:
        raise ValueError(f'ref must be the index of one of the {nimages} images, got {ref!r}')
    dt = np.diff(_days(dates, nimages))

    # least squares of dd = a + v dt + dh dc (dd, dc: differences of the displacement and of c between neighbouring
    # dates): dh from the parts of dd and dc that the constant and dt do not explain. The constant and the
    # orthonormal rest of dt (none for equal time steps) are removed from both. No BLAS: einsum without optimize.
    rest = dt - dt.mean()
    basis = [xp.asarray(rest / np.linalg.norm(rest))] if np.linalg.norm(rest) > 1e-9 * np.linalg.norm(dt) else []
    c = _mm_per_m(bperp, slant_range[:, None], theta[:, None])               # (n_points, nimages)
    dc = c[:, 1:] - c[:, :-1]
    scale = xp.einsum('ij,ij->i', dc, dc)
    dd = disp[:, 1:].astype(xp.float64) - disp[:, :-1]
    for a in (dc, dd):
        a -= a.mean(axis=1, keepdims=True)
        for q in basis:
            a -= xp.einsum('ij,j->i', a, q)[:, None] * q
    den = xp.einsum('ij,ij->i', dc, dc)
    num = xp.einsum('ij,ij->i', dc, dd)
    del dc, dd
    ok = den > 1e-12 * scale             # the baselines of the point change other than as a + v t: dh is defined
    dh = xp.where(ok, num / xp.where(ok, den, 1.0), xp.nan)

    c -= c[:, ref:ref + 1].copy()          # c_k - c_ref: the DEM error term relative to the image `ref`
    c *= dh[:, None]
    xp.subtract(disp, c, out=c)
    return dh.astype(xp.float32), c.astype(xp.float32)
