"""Module for loading result from other software"""


__all__ = ['load_gamma_flatten_rslc', 'load_gamma_lat_lon_hgt', 'load_gamma_look_vector', 'load_gamma_range',
           'load_gamma_metadata', 'load_gamma_bperp']

import glob
from pathlib import Path
import tempfile
import re
import os
import logging
import math
import datetime
from decimal import Decimal
import toml
import zarr
import time

import numpy as np
import numba
from ..api.utils_ import mjit
from scipy.constants import speed_of_light

import moraine as mr
from ..api.utils_.gamma_io import read_gamma_image, write_gamma_image, read_gamma_plist, write_gamma_plist
from ..api.utils_.chunk import all_chunk_slices
from ..api.utils_.orbit import bperp as _bperp
from .logging import mc_logger
from .executor import Executor, Chunk

def _rdc_width_nlines(image_par):
    """get slc width and number of lines.
    """
    with open(image_par) as f:
        for line in f:
            if re.search('range_samples',line):
                rdc_width = int(line.split()[1])
            if re.search('azimuth_lines',line):
                rdc_nlines = int(line.split()[1])
    return rdc_width, rdc_nlines

def _geo_width_nlines(dem_par):
    """get dem width and number of lines.
    """
    with open(dem_par) as f:
        for line in f:
            if re.search('width',line):
                geo_width = int(line.split()[1])
            if re.search('nlines',line):
                geo_nlines = int(line.split()[1])
    return geo_width, geo_nlines

def _cor_pos_dem(dem_par):
    """get corner lat and lon and post lat and lon
    """
    with open(dem_par) as f:
        for line in f:
            if re.search('corner_lat',line):
                cor_lat = float(line.split()[1])
            if re.search('corner_lon',line):
                cor_lon = float(line.split()[1])
            if re.search('post_lat',line):
                pos_lat = float(line.split()[1])
            if re.search('post_lon',line):
                pos_lon = float(line.split()[1])
    return cor_lat, cor_lon, pos_lat, pos_lon

def _fetch_slc_par_date(rslc_dir,
                       ):
    """Parameters
    ----------
    rslc_dir
        str / Path

    Returns
    -------
    tuple
        the dates (str, the names of the rslc files without '.rslc'), the rslc files and their par files (Path), three
        lists sorted by file name
    """
    rslc_dir = Path(rslc_dir)
    rslcs = []
    rslc_pars = []
    dates = []
    for rslc in sorted(rslc_dir.glob('*.rslc')):
        rslc_par = rslc.parent / (rslc.name + '.par')
        if not rslc_par.exists():
            raise FileNotFoundError(f'{rslc_par} does not exist')
        date = rslc.stem

        rslcs.append(rslc)
        rslc_pars.append(rslc_par)
        dates.append(date)
    return dates, rslcs, rslc_pars

def _flatten_rslc_task(rslc, sim_orb, width, out, k):
    """read one gamma rslc and its simulated orbital phase, flatten it and write image `k` of the zarr `out`"""
    flat = _flatten_rslc(read_gamma_image(sim_orb, width, dtype='float'), read_gamma_image(rslc, width, dtype='fcomplex'))
    Chunk(out, (slice(0, flat.shape[0]), slice(0, flat.shape[1]), slice(k, k+1))).write(flat[:, :, None])


@mjit(nopython=True, parallel=True, nogil=True)
def _flatten_rslc(sim_orb,rslc):
    y = np.empty(rslc.shape, rslc.dtype)
    for i in numba.prange(len(rslc)):
        y[i] = np.exp(sim_orb[i]*1j)*rslc[i]
    return y

@mc_logger
def load_gamma_flatten_rslc(
    rslc_dir:str,
    reference:str,
    hgt:str,
    scratch_dir:str,
    rslc:str,
    chunks:tuple[int,int]=(1000,1000),
    gamma_threads:int=min(64, os.cpu_count() or 1),
    processes=False,
    n_workers=1,
    threads_per_worker=1,
):
    """Generate flatten rslc data from gamma command and convert them into zarr format.
    The shape of hgt should be same as one rslc image, i.e. the hgt file is generated with 1 by 1 look geocoding.
    All data equal to 0 are replaced with nan.

    Parameters
    ----------
    rslc_dir : str
        input: GAMMA rslc directory with '????????.rslc' and '????????.rslc.par' files
    reference : str
        reference date, eg: '20200202'
    hgt : str
        input: DEM height in radar coordinates (GAMMA float), same size as one rslc
    scratch_dir : str
        directory for preserve gamma intermediate files
    rslc : str
        output: flattened rslc stack, shape (nlines, width, nimages), complex64
    chunks : tuple[int, int], default: (1000, 1000)
        rslc chunk size
    gamma_threads : int, default: min(64, number of CPU cores)
        number of threads of each GAMMA program run for an image (`phase_sim_orb`, set with OMP_NUM_THREADS)
    processes : default: False
        use processes for the workers instead of threads
    n_workers : default: 1
        number of workers
    threads_per_worker : default: 1
        tasks a worker runs at the same time
    """
    logger = logging.getLogger(__name__)
    rslc_path = rslc
    dates, rslcs, rslc_pars = _fetch_slc_par_date(rslc_dir)
    logger.info('rslc found:\n' + '\n'.join(f'{k:4d}  {date}  {rslc_}  {par}'
                                            for k, (date, rslc_, par) in enumerate(zip(dates, rslcs, rslc_pars))))

    reference_idx = dates.index(reference)
    ref_rslc = rslcs[reference_idx]
    ref_rslc_par = rslc_pars[reference_idx]
    hgt = Path(hgt)

    n_image = len(rslcs)
    width,nlines = _rdc_width_nlines(ref_rslc_par)
    az_chunk_size, r_chunk_size = chunks
    logger.info(f'number of images: {n_image}.')
    logger.info(f'image number of lines: {nlines}.')
    logger.info(f'image width: {width}.')

    logger.info('run gamma command to generate required data for flattened rslcs:')

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sim_orbs = []
    for i,(date,rslc,rslc_par) in enumerate(zip(dates,rslcs,rslc_pars)):
        off_par = scratch_dir/(reference+'_'+date+'.off')
        create_offset_command = f'create_offset {str(ref_rslc_par)} {str(rslc_par)} {str(off_par)} 1 1 1 0 >> {str(scratch_dir/"gamma.log")}'
        logger.info('run command: ' + create_offset_command)
        os.system(create_offset_command)
        sim_orb = scratch_dir/(reference+'_'+date+'.sim_orb')
        # GAMMA uses 8 threads when OMP_NUM_THREADS is not set
        phase_sim_orb_command = f'OMP_NUM_THREADS={gamma_threads} phase_sim_orb {str(ref_rslc_par)} {str(rslc_par)} {str(off_par)} {str(hgt)} {str(sim_orb)} {str(ref_rslc_par)} - - 1 1 >> {str(scratch_dir/"gamma.log")}'
        if sim_orb.exists():
            logger.info(f'{sim_orb} exists. skip runing {phase_sim_orb_command}')
        else:
            logger.info('run command: ' + phase_sim_orb_command)
            os.system(phase_sim_orb_command)
        sim_orbs.append(sim_orb)
    logger.info('gamma command finished.')
    logger.info('reading the gamma binary files, flattening the rslcs and saving them to zarr.')
    rslc_zarr = zarr.open(rslc_path, mode='w', shape=(nlines, width, n_image), dtype=np.complex64, chunks=(*chunks, 1))
    logger.zarr_info(rslc_path, rslc_zarr)
    # one task per image: the rslc and its simulated orbital phase in, the flattened rslc out
    tasks = [(rslc, str(sim_orb), width, rslc_path, k) for k, (rslc, sim_orb) in enumerate(zip(rslcs, sim_orbs))]
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map(_flatten_rslc_task, tasks, desc='images')
    logger.info('done.')

@mc_logger
def load_gamma_lat_lon_hgt(diff_par:str,
                           rslc_par:str,
                           dem_par:str,
                           hgt:str,
                           scratch_dir:str,
                           lat_zarr:str,
                           lon_zarr:str,
                           hgt_zarr:str,
                           chunks:tuple[int,int]=(1000,1000),
                          ):
    """Function to load longitude and latitude from gamma binary format to zarr.
    All data equal to 0 are replaced with nan.

    Parameters
    ----------
    diff_par : str
        input: geocoding diff_par, using the simulated image as reference
    rslc_par : str
        input: par file of the reference rslc
    dem_par : str
        input: DEM par file
    hgt : str
        input: DEM height in radar coordinates (GAMMA float)
    scratch_dir : str
        directory for preserve gamma intermediate files
    lat_zarr : str
        output, latitude zarr
    lon_zarr : str
        output, longitude zarr
    hgt_zarr : str
        output, height zarr
    chunks : tuple[int, int], default: (1000, 1000)
        rslc chunk size
    """
    logger = logging.getLogger(__name__)
    geo_width = _geo_width_nlines(dem_par)[0]
    rdc_width, rdc_nlines = _rdc_width_nlines(rslc_par)
    logger.info(f'image shape: ({rdc_nlines},{rdc_width})')

    lat_data = zarr.open(lat_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float64)
    lon_data = zarr.open(lon_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float64)
    hgt_data = zarr.open(hgt_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float32)

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    logger.info('run gamma command to generate longitude, latitude and height:')

    plist = scratch_dir/'plist'
    command =f"mkgrid {str(plist)} {rdc_width} {rdc_nlines} 1 1 >> {str(scratch_dir/'gamma.log')}"
    if plist.exists():
        logger.info(f'{plist} exists. skip runing {command}')
    else:
        # pt_i = np.arange(rdc_width,dtype=np.int32)
        # pt_j = np.arange(rdc_nlines,dtype=np.int32)
        # pt_ii,pt_jj = np.meshgrid(pt_i,pt_j)
        # pt_ij = np.stack((pt_ii,pt_jj),axis=-1)
        # write_gamma_plist(pt_ij,scratch_dir/'plist')
        logger.info('run command: ' + command)
        os.system(command)
        logger.info('gamma command finished.')

    phgt_wgs84 = scratch_dir/'phgt_wgs84'
    command = f"pt2geo {str(plist)} - {rslc_par} - {hgt} {dem_par} {diff_par} 1 1 - - {str(scratch_dir/'plat_lon')} {str(phgt_wgs84)} >> {str(scratch_dir/'gamma.log')}"
    if phgt_wgs84.exists():
        logger.info(f'{phgt_wgs84} exists. skip runing {command}')
    else:
        logger.info('run command: ' + command)
        os.system(command)
        logger.info('gamma command finished.')
    logger.info('writing zarr file.')
    hgt_data[:] = read_gamma_image(str(phgt_wgs84),width=rdc_width,dtype='float')

    ptlonlat = read_gamma_plist(str(scratch_dir/'plat_lon'),dtype='double').reshape(rdc_nlines, rdc_width, 2)
    lon_data[:], lat_data[:] = ptlonlat[...,0], ptlonlat[...,1]
    logger.info('write done.')

@mc_logger
def load_gamma_look_vector(theta:str,
                           phi:str,
                           lt:str,
                           rslc_par:str,
                           dem_par:str,
                           scratch_dir:str,
                           theta_zarr:str,
                           phi_zarr:str,
                           chunks:tuple[int,int]=(1000,1000),
                          ):
    """Load look vector (elevation angle and orientation angle) in map geometry
    from gamma binary format to look vector in radar geometry zarr file.
    The two input data should be generated with the `look_vector` gamma command.
    All data equal to 0 are replaced with nan.

    Parameters
    ----------
    theta : str
        input: elevation angle in map geometry (GAMMA lv_theta)
    phi : str
        input: orientation angle in map geometry (GAMMA lv_phi)
    lt : str
        input: lookup table (GAMMA lt_fine)
    rslc_par : str
        input: par file of the reference rslc
    dem_par : str
        input: DEM par file
    scratch_dir : str
        directory for preserve gamma intermediate files
    theta_zarr : str
        output, elevation angle zarr
    phi_zarr : str
        output, orientation angle zarr
    chunks : tuple[int, int], default: (1000, 1000)
        rslc chunk size
    """
    logger = logging.getLogger(__name__)
    geo_width = _geo_width_nlines(dem_par)[0]
    rdc_width, rdc_nlines = _rdc_width_nlines(rslc_par)
    logger.info(f'image shape: ({rdc_nlines},{rdc_width})')

    theta_data = zarr.open(theta_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float32)
    phi_data = zarr.open(phi_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float32)

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    theta_rdc = scratch_dir/'theta_rdc'
    command = f'geocode {lt} {theta} {geo_width} {str(theta_rdc)} {rdc_width} {rdc_nlines} >> {scratch_dir/"gamma.log"}'
    if theta_rdc.exists():
        logger.info(f'{theta_rdc} exists. skip runing {command}')
    else:
        logger.info('run gamma command to generate elevation angle in range doppler coordinate:')
        logger.info('run command: ' + command)
        os.system(command)
        logger.info('gamma command finished.')
    logger.info('writing data.')
    theta_data[:] = read_gamma_image(theta_rdc,rdc_width,dtype='float')

    phi_rdc = scratch_dir/'phi_rdc'
    command = f'geocode {lt} {phi} {geo_width} {str(phi_rdc)} {rdc_width} {rdc_nlines} >> {scratch_dir/"gamma.log"}'
    if phi_rdc.exists():
        logger.info(f'{phi_rdc} exists. skip runing {command}')
    else:
        logger.info('run gamma command to generate orientation angle in range doppler coordinate:')
        logger.info('run command: ' + command)
        os.system(command)
        logger.info('gamma command finished.')
    logger.info('writing data.')
    phi_data[:] = read_gamma_image(phi_rdc,rdc_width,dtype='float')
    logger.info('Done.')

@mc_logger
def load_gamma_range(rslc_par:str,
                     range_zarr:str,
                     chunks:tuple[int,int]=(1000,1000),
                    ):
    """Generate slant range distance and save to zarr.

    Parameters
    ----------
    rslc_par : str
        input: par file of one rslc
    range_zarr : str
        output, range distance zarr
    chunks : tuple[int, int], default: (1000, 1000)
        rslc chunk size
    """
    logger = logging.getLogger(__name__)
    rdc_width, rdc_nlines = _rdc_width_nlines(rslc_par)
    logger.info(f'image shape: ({rdc_nlines},{rdc_width})')
    with open(rslc_par) as f:
        for line in f:
            if re.search('near_range_slc',line):
                rho0 = float(line.split()[1])
            if re.search('range_pixel_spacing',line):
                d_rho = float(line.split()[1])

    logger.info('Calculating slant range distance.')
    rho1d = np.arange(rdc_width)*d_rho+rho0
    rho2d = np.tile(rho1d,(rdc_nlines,1))
    range_data = zarr.open(range_zarr,mode='w',shape=(rdc_nlines,rdc_width),chunks = chunks, dtype=np.float32)
    logger.info('writing data.')
    range_data[:] = rho2d
    logger.info('Done.')

def _read_gamma_center_time(par):
    """UTC time of the centre line of a GAMMA SLC par file, aware datetime to the microsecond"""
    day = center = None
    with open(par) as f:
        for line in f:
            key, _, value = line.partition(':')
            key, value = key.strip(), value.split()
            if key == 'date':           # year month day; Sentinel-1 adds the start time of the original product
                day = datetime.datetime(*(int(v) for v in value[:3]), tzinfo=datetime.timezone.utc)
            elif key == 'center_time':  # s of that day, read exactly (no float rounding)
                center = Decimal(value[0])
    if day is None or center is None:
        raise ValueError(f'{par}: no date or center_time')
    return day + datetime.timedelta(microseconds=round(center*1000000))

@mc_logger
def load_gamma_metadata(rslc_dir:str,
                        dem_par:str,
                        reference:str,
                        meta_file:str,
                        ):
    """Load necessary metadata into a toml file.

    Parameters
    ----------
    rslc_dir : str
        input: GAMMA rslc directory with '????????.rslc' and '????????.rslc.par' files
    dem_par : str
        input: DEM par file
    reference : str
        reference date, eg: '20200202'
    meta_file : str
        output: toml file with `dates`, the date of every image (YYYYMMDD, from the file names), nimages strings in
        date order; `times`, the time of the centre line of every image (UTC, to the microsecond), nimages date-times
        in the order of `dates`; `radar_wavelength`, `range_pixel_spacing` and `azimuth_pixel_spacing` (m) of the
        reference image; `lonlat_bbox` (degrees) and `merca_bbox` (web mercator, m) of the DEM, [west, south, east,
        north]; `perpendicular_baseline`, that of every image relative to the reference image at the scene centre
        (m), nimages values in the order of `dates`
    """
    meta = dict()
    rslc_dir = Path(rslc_dir)
    dates, times = [], []
    logger = logging.getLogger(__name__)
    for par_file in sorted(rslc_dir.glob('*.rslc.par')):
        dates.append(par_file.name[:8])
        times.append(_read_gamma_center_time(par_file))
    meta['dates'] = dates
    meta['times'] = times

    logger.info('fetching slc related metadata.')
    with open(rslc_dir/(reference+'.rslc.par')) as f:
        for line in f:
            if re.search('radar_frequency',line):
                radar_f = float(line.split()[1])
                rdr_wavelen = speed_of_light/radar_f
                logger.info('Fetching randar wavelength')
            if re.search('heading',line):
                heading = float(line.split()[1])
                logger.info('Fetching heading angle')
            if re.search('range_pixel_spacing',line):
                dr = float(line.split()[1])
                logger.info('Fetching range pixel spacing')
            if re.search('azimuth_pixel_spacing',line):
                daz = float(line.split()[1])
                logger.info('fetching azimuth pixel spacing')

    meta['radar_wavelength'] = rdr_wavelen
    meta['range_pixel_spacing'] = dr
    meta['azimuth_pixel_spacing'] = daz

    logger.info('fetching dem related metadata')
    geo_width, geo_nlines = _geo_width_nlines(dem_par)
    cor_lat, cor_lon, pos_lat, pos_lon = _cor_pos_dem(dem_par)
    x0, xm = cor_lon, cor_lon+(geo_width-1)*pos_lon
    y0, ym = cor_lat+(geo_nlines-1)*pos_lat, cor_lat
    lonlat_bbox = [x0, y0, xm, ym]
    meta['lonlat_bbox'] = lonlat_bbox
    web_x0, web_xm = (lon*20037508.34/180 for lon in (x0, xm))
    web_y0, web_ym = (math.log(math.tan((90+lat)*math.pi/360))/(math.pi/180)*20037508.34/180 for lat in (y0, ym))
    meta['merca_bbox'] = [web_x0, web_y0, web_xm, web_ym]

    with tempfile.TemporaryDirectory() as temp_dir:        
        temp_dir = Path(temp_dir)
        slc_tab = temp_dir/'slc_tab'
        bperp = temp_dir/'bperp'
        itab = temp_dir/'itab'

        tab_content = ''
        for date in dates:
            tab_content += str(rslc_dir/(date+'.rslc'))
            tab_content += '      '
            tab_content += str(rslc_dir/(date+'.rslc.par'))
            tab_content += '\n'
        slc_tab.write_text(tab_content)

        logger.info('Run gamma command to calculate baseline:')
        command = f"base_calc {str(slc_tab)} {str(rslc_dir/(reference+'.rslc.par'))} {str(bperp)} {str(itab)} - > {temp_dir/'log'}"
        logger.info('run command: ' + command)
        os.system(command)
        logger.info('gamma command finished.')
        base = np.loadtxt(bperp, usecols=3, ndmin=1).astype(np.float32)   # column 4: B_perp (m)

    meta['perpendicular_baseline'] = base

    Path(meta_file).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file,'w') as f:
        a = toml.dump(meta,f,encoder=toml.TomlNumpyEncoder())
    logger.info('All meta data: \n'+a)
    logger.info('writing data in toml file.')
    logger.info('Done.')

def _read_gamma_orbit(par):
    """times (s of the day), positions (m) and velocities (m/s) of the orbit state vectors of a GAMMA par file, shapes
    (n,), (n, 3), (n, 3)"""
    n = t0 = dt = None
    pos, vel = {}, {}
    with open(par) as f:
        for line in f:
            key, _, value = line.partition(':')
            key, value = key.strip(), value.split()
            if key == 'number_of_state_vectors':
                n = int(value[0])
            elif key == 'time_of_first_state_vector':
                t0 = float(value[0])
            elif key == 'state_vector_interval':
                dt = float(value[0])
            elif key.startswith('state_vector_position_'):
                pos[int(key.rsplit('_', 1)[1])] = [float(v) for v in value[:3]]
            elif key.startswith('state_vector_velocity_'):
                vel[int(key.rsplit('_', 1)[1])] = [float(v) for v in value[:3]]
    complete = n is not None and sorted(pos) == list(range(1, n+1)) and sorted(vel) == list(range(1, n+1))
    if not complete or t0 is None or dt is None:
        raise ValueError(f'{par}: incomplete orbit state vectors')
    return (t0 + dt*np.arange(n), np.array([pos[k] for k in range(1, n+1)]),
            np.array([vel[k] for k in range(1, n+1)]))

@mc_logger
def load_gamma_bperp(rslc_dir:str,
                     reference:str,
                     lat_zarr:str,
                     lon_zarr:str,
                     hgt_zarr:str,
                     bperp_zarr:str,
                     chunks:tuple[int,int]=None,
                     processes=None,
                     n_workers=None,
                     threads_per_worker=None,
                    ):
    """Perpendicular baseline of every image relative to the reference image at every pixel, from the orbits.

    Parameters
    ----------
    rslc_dir : str
        input: GAMMA rslc directory with '????????.rslc' and '????????.rslc.par' files; the images are in date order,
        as in `load-gamma-flatten-rslc`
    reference : str
        reference date of the co-registration, e.g. '20200707'; its perpendicular baseline is 0
    lat_zarr : str
        input: latitude of the pixels (WGS84), degrees, shape (nlines, width), float64 (`load-gamma-lat-lon-hgt`)
    lon_zarr : str
        input: longitude of the pixels (WGS84), degrees, shape (nlines, width), float64 (`load-gamma-lat-lon-hgt`)
    hgt_zarr : str
        input: height of the pixels above the WGS84 ellipsoid, meters, shape (nlines, width), float32
        (`load-gamma-lat-lon-hgt`)
    bperp_zarr : str
        output: perpendicular baseline, meters, shape (nlines, width, nimages), float32, chunks (`chunks`, 1): the
        component of the position of the satellite of each image relative to that of the reference image, each when
        it sees the pixel, perpendicular to the line of sight of the reference and to its flight direction, positive
        upwards; 0 for the reference image; nan where the latitude, longitude or height is nan
    chunks : tuple[int, int], optional
        data processing chunk size, same as `lat_zarr` by default
    processes : optional
        use processes (True) or threads (False) for the workers. Default: False
    n_workers : optional
        number of workers. Default: 1
    threads_per_worker : optional
        tasks a worker runs at the same time. Default: 1
    """
    logger = logging.getLogger(__name__)
    dates, _, pars = _fetch_slc_par_date(rslc_dir)
    if reference not in dates:
        raise ValueError(f'reference {reference} is not a date of {rslc_dir}')
    ref = dates.index(reference)
    orbits = [_read_gamma_orbit(par) for par in pars]
    n_sv = {date: len(t) for date, (t, _, _) in zip(dates, orbits)}
    if len(set(n_sv.values())) != 1:
        raise ValueError(f'the images have different numbers of orbit state vectors: {n_sv}')
    orbit_time, orbit_pos, orbit_vel = (np.stack(a) for a in zip(*orbits))
    nimages = len(dates)
    logger.info(f'number of images: {nimages}, reference: {reference}, state vectors per image: {orbit_time.shape[1]}')

    width, nlines = _rdc_width_nlines(pars[ref])
    lat = zarr.open(lat_zarr, mode='r')
    for name, path in (('lat_zarr', lat_zarr), ('lon_zarr', lon_zarr), ('hgt_zarr', hgt_zarr)):
        shape = zarr.open(path, mode='r').shape
        if shape != (nlines, width):
            raise ValueError(f'{name} {path} has the shape {shape}, the images ({nlines}, {width})')
    if chunks is None: chunks = lat.chunks
    chunks = tuple(chunks)
    out = zarr.open(bperp_zarr, mode='w', shape=(nlines, width, nimages), dtype=np.float32, chunks=(*chunks, 1))
    logger.zarr_info(bperp_zarr, out)
    # one task per (azimuth, range) chunk: the coordinates of its pixels in, their perpendicular baselines out
    tasks = [([Chunk(lon_zarr, sl), Chunk(lat_zarr, sl), Chunk(hgt_zarr, sl)],
              [Chunk(bperp_zarr, (*sl, slice(0, nimages)))]) for sl in all_chunk_slices((nlines, width), chunks)]
    logger.info(f'perpendicular baseline of {len(tasks)} chunks of {chunks}')
    with Executor(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
        ex.map_chunks(_bperp, tasks, desc='perpendicular baseline', orbit_time=orbit_time, orbit_pos=orbit_pos,
                      orbit_vel=orbit_vel, ref=ref)
    c = (nlines//2, width//2)
    centre = out[c[0], c[1]]
    logger.info(f'perpendicular baseline at the scene centre {c} (m): ' + ', '.join(f'{b:.2f}' for b in centre))
    logger.info('done.')
