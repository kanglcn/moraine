```python exec="true" source="above" result="text" session="tropo"
import numpy as np
import moraine as mr

levels = 100.0 * np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
                           500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])  # Pa
grid_lat, grid_lon = np.array([41.0, 40.75]), np.array([14.0, 14.25])   # degrees, as ERA5 (latitude decreasing)

def atmosphere(p0, e0, T=270.0):
    """z, t, q of an isothermal atmosphere over the 2 x 2 grid nodes: pressure p0 and water vapour pressure e0 (Pa)
    at sea level, scale heights 8 km and 2 km"""
    h = -8000.0 * np.log(levels / p0)                                    # height of the levels, m
    e = e0 * np.exp(-h / 2000.0)
    q = 0.622 * e / (levels - 0.378 * e)
    nodes = np.ones((levels.size, 2, 2))
    return 9.80665 * h[:, None, None] * nodes, T * nodes, q[:, None, None] * nodes

lon, lat = np.array([14.1, 14.1, 14.2]), np.array([40.8, 40.8, 40.9])
hgt = np.array([0.0, 1000.0, 500.0])                                     # m above the geoid
ztd = mr.era5_zenith_delay(lon, lat, hgt, levels, grid_lon, grid_lat, *atmosphere(101325.0, 1500.0))
print(ztd.round(4))
```

The zenith delay in meters of the atmosphere above three points at sea level, 1000 m and 500 m; at sea level 2.31 m
of it is hydrostatic, from the pressure, and 0.16 m wet, from the water vapour. It falls with the height of the
point.

```python exec="true" source="above" result="text" session="tropo"
times = ['2019-01-02T05:30:00Z', '2019-01-14T05:00:00Z']                # UTC time of the two images
hours = ['2019-01-02T05:00Z', '2019-01-02T06:00Z', '2019-01-14T05:00Z']  # the ERA5 hours they need
fields = [atmosphere(101325.0, 1500.0), atmosphere(101325.0, 1700.0), atmosphere(100800.0, 1200.0)]
z, t, q = (np.stack([f[i] for f in fields]) for i in range(3))           # (n_hours, n_levels, n_lat, n_lon)
theta = np.full(3, 0.96)                                                 # elevation angle of the look vector, rad
tropo = mr.pc_tropo_era5(lon, lat, hgt, theta, times, 0.0, hours, levels, grid_lon, grid_lat, z, t, q)
print(tropo.round(1))
```

Image 0 at 05:30 takes the mean of the hours 05:00 and 06:00. The air of image 1 has a lower pressure and less water
vapour: a smaller delay than in image 0, an apparent motion towards the satellite in millimeters, larger at sea level
than at 1000 m. The displacement corrected for the troposphere is the displacement minus `tropo`.
