```python exec="true" source="above" result="text" session="polygon"
import os
import tempfile
import numpy as np
import moraine as mr

square = [(14.10, 40.80), (14.15, 40.80), (14.15, 40.85), (14.10, 40.85)]      # (longitude, latitude) vertices
path = os.path.join(tempfile.mkdtemp(), 'areas.geojson')
mr.write_polygons(path, [square])                                               # a GeoJSON FeatureCollection
polygons, coordinates = mr.read_polygons(path)
print(coordinates, polygons[0].shape)

lon = np.array([14.12, 14.20, np.nan])
lat = np.array([40.82, 40.82, 40.82])
print(mr.polygons_contain(polygons, lon, lat))                                 # inside, outside, nan
```

Polygons drawn in the viewer are saved in the same format; `polygon-mask` makes masks of rasters or point clouds
from them (`moraine_coordinates` tells longitude / latitude from radar grid coordinates).
