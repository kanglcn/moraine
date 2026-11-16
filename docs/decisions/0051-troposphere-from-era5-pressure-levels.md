# 0051 Tropospheric delay from ERA5 pressure levels, computed by moraine

## Status

Accepted

## Date

2026-11-15

## Context

Step 3 of decision 0048 subtracts per image corrections from models or external data before the spatial reference.
The troposphere is the largest of them for Sentinel-1: centimeters per image, with a part stratified by height that
ERA5 describes, the hourly reanalysis of ECMWF on a 0.25 degree grid. ERA5 comes on 37 pressure levels
(`reanalysis-era5-pressure-levels`, kept on disk by the Copernicus Climate Data Store) or on 137 model levels
(`reanalysis-era5-complete`, from the tape archive MARS, slow queues).

The Climate Data Store needs an account and cuts the area out on its side (52 kB for one hour over the Campi
Flegrei), but every request waits in its queue: about 40 s alone, 70 to 80 s with four at a time (2026-11-15). The
92 images of the Campi Flegrei need 184 hours: about 55 minutes with one request per hour. A request is the product
of its dates and hours (a list of dates is accepted); 30 dates by 2 hours (6 660 fields) took 505 s, 92 dates by 2
hours (20 424 fields) are refused ("cost limits exceeded"). Copies of ERA5 in the cloud need no account and no queue
but store the whole earth per hour: the NSF NCAR archive on AWS (`nsf-ncar-era5`, one netCDF4 file per day and
variable, chunks of one hour, all levels and the whole earth, 57 MB compressed) and ARCO-ERA5 on Google Cloud (zarr,
the same chunks, 85 MB). Either sends about 170 MB per hour of the three fields; the values agree with the Climate
Data Store (temperature within 6e-5 K for NCAR, 7e-4 K for ARCO). The NCAR copy follows ERA5 by about three months
(August 2026 on 2026-11-15).

Two programs compute the delay. PyAPS3, used by MintPy, takes the pressure levels in GRIB (pygrib), the geopotential
height as the height, the hour nearest to the image (MintPy) and the zenith delay divided by the cosine of the
incidence angle. RAiDER takes the model levels by default, converts the geopotential to the geometric height and the
heights of the points to EGM2008 through PROJ grids, interpolates in time (the two hours around the image or the
azimuth time of every pixel) and traces rays from the orbit (isce3); it needs GDAL, rasterio, xarray, pyproj grids and
isce3. With the grid of ERA5 the extras change little: the wet delay lies below 2 to 3 km, where a ray of 40 degrees
incidence is 2 km away from the vertical of its point while the grid nodes are 25 km apart; the azimuth times of a
frame span 25 s of an hour. Their effects are below 1 mm and mostly cancel in the displacement relative to a
reference area and a reference image. ERA5 does not resolve the turbulent water vapour below about 30 km, the larger
error.

## Decision

- moraine computes the delay itself, from the pressure levels; PyAPS3 and RAiDER are not used. The optional extra
  `era5` holds h5py (the netCDF4 files) and cdsapi (the Climate Data Store).
- `download-era5` (`moraine.cli.tropo`): for every image the whole hour at or before its time and, unless the time is
  a whole hour, the hour after it; the geopotential, temperature and specific humidity on the 37 levels, over the bbox
  of the points widened to the 0.25 degree grid and one node more on every side; one file per hour,
  `era5_YYYYMMDD_HH.nc` (UTC) in the layout of the netCDF4 files of the Climate Data Store, written through a temporary
  file. Files that hold their hour over the area are kept, so a download continues. Two sources, `source`:
  - `ncar` (default): the NSF NCAR archive on AWS, without an account. The chunks of the hours are read by HTTP range
    requests (urllib of the standard library, no S3 client) and h5py, and cut to the area; the hours of a day are read
    with one opening of each of its three files. Worker processes, as h5py runs one HDF5 call at a time in a process.
  - `cds`: the Climate Data Store, one request per hour; the user's account (`~/.cdsapirc`) is read by cdsapi only.
- `pc-tropo-era5` (API `pc_tropo_era5`, `era5_zenith_delay` in `moraine.api.tropo`), from the profile of every grid
  node:
  - refractivity constants of PyAPS (k1 0.776 K/Pa, k2 0.716 K/Pa, k3 3750 K²/Pa, Rd 287.05, Rv 461.495 J/(kg K)),
    split into the hydrostatic part and the wet part with k2' = k2 - k1 Rd / Rv;
  - water vapour pressure from the specific humidity, e = q P / (Rd/Rv + (1 - Rd/Rv) q);
  - heights of the levels above the geoid from the geopotential, with gravity and earth radius depending on latitude;
  - hydrostatic delay 1e-6 k1 Rd P(h) / g_m with the mean gravity g_m of Saastamoinen, ln P linear in height between
    levels; wet delay from the wet refractivity exponential in height between levels, integrated exactly up to the
    highest level; below the lowest level the lowest layer goes on;
  - the delays of the four grid nodes around a point at the height of the point, interpolated bilinearly; in time
    linearly between the two whole hours around the time of the image;
  - the line of sight: the zenith delay divided by sin(theta), theta the elevation angle of the look vector;
  - the height datum: one value `geoid`, the height of the geoid above the WGS84 ellipsoid at the scene in meters,
    without a default; `hgt - geoid` is the height above the geoid of ERA5;
  - output: tropo_k = -1000 (SD_k - SD_ref) mm, the apparent displacement positive towards the satellite as `pc-disp`;
    the corrected displacement is `disp - tropo` (`math`).
- `examples/06_products.toml` follows the order of decision 0048: `pc-disp`, `download-era5`, `pc-tropo-era5`, `math`
  (disp - tropo), `pc-reference` (in millimeters), `pc-sc-dem-error`.
- Validation: an exponential atmosphere (exact up to the float32 fields); the standard atmosphere with a humidity
  profile, against its integral on a 0.5 m grid (within 0.3 mm at 0 to 3000 m); bilinear, time interpolation, sign,
  reference image, nan and the files of the Climate Data Store; Campi Flegrei.

## Consequences

- 06 downloads from the Climate Data Store: it needs the user's account and the network, or the files of the hours in
  its `era5` directory. Two requests per image; the reference value of `pc-reference` is in millimeters.
- Leaving out the geoid errs by about ZTD'' dh N relative to the reference area (ZTD'' the curvature of the delay with
  height, dh the height above the reference area, N the geoid): 1.5 mm per image at the Campi Flegrei (450 m, N 46 m),
  less than 1 mm between images, a few millimeters over 3 km of relief; the variation of N over a scene (a few
  meters) less than 1 mm.
- Memory: the fields of all hours, about 450 bytes per grid node and hour; a task 4 bytes per point and image; a
  download worker of `ncar` about 250 MB (one chunk of the whole earth).
- The default sends about 170 MB per hour: for the Campi Flegrei (184 hours, 31 GB) about 15 minutes with four worker
  processes, 10 with eight (36 to 54 MB/s from the cluster to AWS); the Climate Data Store stays for slow networks,
  for the last months that the NCAR copy does not hold yet, and as the original source.
- Campi Flegrei (336 830 points, 92 images, 184 hours): from the NCAR archive 6.1 s per day with four worker
  processes; on 83 hours the fields agree with the Climate Data Store to 0.008 m2/s2, 6e-5 K and 4e-9 kg/kg, the
  troposphere of the images to 0.0001 mm. The troposphere relative to the reference area has a standard deviation
  over the points of 1.9 mm (median of the images), at most 7.9 mm. On the 12 day differences of the displacement the
  slope with height drops from 2.64 to 2.04 mm per 100 m (root mean square), the standard deviation over the points
  from 4.49 to 4.40 mm (median); 40 % of the pairs get worse. The 1 km means of the spatially correlated DEM error
  from the odd and the even dates correlate by 0.42 instead of 0.27. The points cover 17 by 14 km with 450 m of
  relief: the turbulent atmosphere dominates.
- The turbulent part stays (step 8 of decision 0048). Model levels, ray tracing, the azimuth time of every pixel or
  other models (HRES, GACOS) would be another decision.

## Do not

- Do not add PyAPS3, RAiDER, a GRIB reader (pygrib, eccodes) or an S3 client (boto3, s3fs) as dependencies.
- Do not read, write or log the key of the Climate Data Store.
- Do not take `meta.toml` as an input of the commands: the times, the bbox and the geoid are values.
- Do not give `geoid` a default: the datum of the heights depends on how the DEM was made.
