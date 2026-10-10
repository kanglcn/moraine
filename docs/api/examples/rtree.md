```python exec="true" source="above" result="text" session="rtree"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
x, y = rng.random(3000), rng.random(3000)                        # e.g. longitude, latitude of points
tree = mr.HilbertRtree.build(x, y, page_size=64)                 # page_size: points per leaf (the chunk size of the point cloud)
idx = tree.bbox_query([0.2, 0.2, 0.4, 0.4], x, y)                # indices of the points in the box [x0, y0, xm, ym]
print(len(idx), idx[:5])
print(np.all((x[idx] >= 0.2) & (x[idx] <= 0.4) & (y[idx] >= 0.2) & (y[idx] <= 0.4)))
```

`tree.save(path)` / `HilbertRtree.load(path)` keep the tree as a zarr array next to a point cloud; the viewer uses
it to read only the points on screen.
