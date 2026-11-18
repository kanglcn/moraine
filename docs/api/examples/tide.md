```python exec="true" source="above" result="text" session="tide"
import numpy as np
import moraine as mr

lon, lat = np.array([14.05, 14.14, 14.24]), np.array([40.78, 40.83, 40.90])   # degrees
for t in ['2019-01-02T05:12:13Z', '2019-01-02T11:24:13Z']:
    print(t, (1000 * mr.solid_tide(lon, lat, t)).round(1))                   # east, north, up in mm
```

The east, north and up displacement of three points of the Campi Flegrei at two times six hours apart: the ground
moves by centimeters in that time, nearly the same at the three points, some 20 km apart.

```python exec="true" source="above" result="text" session="tide"
theta = np.full(3, np.radians(55.4))         # elevation angle of the look vector (descending track), rad
phi = np.full(3, np.radians(-10.6))          # orientation angle of the look vector, counterclockwise from east, rad
times = ['2019-01-02T05:12:13Z', '2019-01-14T05:12:13Z', '2019-01-26T05:12:13Z']
tide = mr.pc_solid_tide(lon, lat, theta, phi, times, ref=0)
print(tide.round(2))
```

The line of sight displacement in millimeters relative to image 0, positive towards the satellite: several
centimeters between images twelve days apart, of which a spatial reference removes all but the differences between
the points, about a millimeter here. The displacement corrected for the tides is the displacement minus `tide`.
