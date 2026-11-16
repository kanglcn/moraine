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

```python exec="true" source="above" result="text" session="tropo"
rng = np.random.default_rng(0)
h = rng.uniform(0, 1000, 2000)                                        # heights above the geoid, m
stable = np.arange(2000) < 1000                                       # the first half: an area without deformation
x = (10000.0 - h) ** 1.6                                              # the power law of the defaults
k_true = np.array([0.0, 2e-5, -1e-5])                                 # mm per m^1.6, image 0 the reference
uplift = np.where(stable, 0.0, 0.01 * h)[:, None] * np.arange(3)      # the other half rises with the height
disp = (x[:, None] * k_true + uplift + rng.normal(0, 0.3, (2000, 3)) * [0, 1, 1]).astype(np.float32)
tropo, k = mr.pc_tropo_powerlaw(disp, h.astype(np.float32), 0.0, mask=stable)
print(k[:, 0])                                                         # close to k_true
corrected = disp - tropo
corrected -= np.median(corrected[stable], axis=0)                    # relative to the stable area
print(np.abs(corrected[~stable, 2] - uplift[~stable, 2]).max().round(2))   # the uplift stays, mm
k_all = mr.pc_tropo_powerlaw(disp, h.astype(np.float32), 0.0)[1]
print(k_all[:, 0])                                                     # without the mask the uplift is in K
```

Fitted on the points of the stable area, K of the two images comes back, and the uplift of the other area, up to
20 mm in image 2, stays in the corrected displacement relative to the stable area, within 1 mm (the noise of 0.3 mm
and the error of K). Fitted on all points, the uplift, which grows with the height, is taken for
troposphere: K of both images is far from the truth.
