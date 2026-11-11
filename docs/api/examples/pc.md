```python exec="true" source="above" result="text" session="pc"
import numpy as np
import moraine as mr

shape = (8, 8)                                                   # (nlines, width) of the raster
gix1 = np.array([[0, 1], [2, 3], [5, 5]], dtype=np.int32)         # (azimuth, range) of three points
hix1 = mr.pc_hix(gix1, shape)                                    # hilbert index of each point
print(hix1)
order = mr.pc_sort(hix1)                                         # point clouds are kept in hilbert order
hix1, gix1 = hix1[order], gix1[order]

gix2 = np.array([[2, 3], [7, 0]], dtype=np.int32)
hix2 = np.sort(mr.pc_hix(gix2, shape))
hix, in1, in2, new2 = mr.pc_union(hix1, hix2)                    # sorted inputs, sorted union
print(hix, mr.pc_gix(hix, shape).tolist())
print(in1, in2, new2)                                            # where the points of each input went
```

```python exec="true" source="above" result="text" session="pc"
ras = mr.pc2ras(hix, np.arange(len(hix), dtype=np.float32), shape)   # point data on the raster, nan elsewhere
print(ras)
print(mr.ras2pc(ras, mr.pc_gix(hix, shape)))                          # and back
```

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
