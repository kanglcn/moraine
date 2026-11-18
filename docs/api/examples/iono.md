```python exec="true" source="above" result="text" session="iono"
import numpy as np
import moraine as mr

lon, lat = np.array([14.05, 14.14, 14.24]), np.array([40.78, 40.83, 40.90])   # degrees
theta = np.full(3, np.radians(55.4))         # elevation angle of the look vector (descending track), rad
phi = np.full(3, np.radians(-10.6))          # orientation angle of the look vector, counterclockwise from east, rad
ipp_lon, ipp_lat, mapping = mr.pierce_point(lon, lat, theta, phi)
print(ipp_lon.round(2), ipp_lat.round(2), mapping.round(4))
```

Three points of the Campi Flegrei: their look vectors cross the layer of the ionosphere at 450 km about 280 km to the
east, towards the satellite of the descending track, with a mapping factor (slant over vertical TEC) of 1.18.

```python exec="true" source="above" result="text" session="iono"
tec_lat = np.arange(87.5, -88, -2.5)         # the grid of the maps of JPL, degrees
tec_lon = np.arange(-180, 181, 5.0)
tec_times = ['2019-01-02T04:00:00Z', '2019-01-02T06:00:00Z']
# vertical TEC: 5 TEC units everywhere at 04:00; at 06:00 9 at 40 N, growing by 1 per degree to the south
tec = np.stack([np.full((tec_lat.size, tec_lon.size), 5.0), 9 + (40 - tec_lat[:, None]) + 0 * tec_lon])
times = ['2019-01-02T04:00:00Z', '2019-01-02T05:00:00Z']
iono = mr.pc_iono_tec(lon, lat, theta, phi, times, 0.0554658, tec, tec_times, tec_lat, tec_lon)
print(iono.round(2))
```

The line of sight displacement in millimeters relative to image 0, positive towards the satellite, from two made up
maps (in practice from `read_ionex` of the files of `download-ionex`): about 2 more TEC units at 05:00 make the line
of sight about 32 mm shorter (13.8 mm per TEC unit times the mapping factor), less at the points whose pierce points
lie further north, where the TEC grew less: the part a spatial reference does not remove. The displacement corrected
for the ionosphere is the displacement minus `iono`.
