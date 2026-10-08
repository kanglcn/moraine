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
