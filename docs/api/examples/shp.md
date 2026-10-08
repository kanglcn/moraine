```python exec="true" source="above" result="text" session="shp"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
nlines, width, nimages = 12, 16, 30
# two homogeneous regions of different brightness, side by side (exponential intensity: speckle)
rmli = rng.exponential(1.0, (nlines, width, nimages)).astype(np.float32)
rmli[:, 8:] *= 4.0
p = mr.ks_test(rmli, az_half_win=2, r_half_win=2)                 # p value of the KS test, (nlines, width, 5, 5)
is_shp, shp_num = mr.select_shp(p, alpha=0.05)                   # SHP: p >= alpha
print(p.shape, is_shp.shape, shp_num.shape, shp_num.dtype)
print('inside the left region', shp_num[6, 3], ' at the boundary', shp_num[6, 7], ' of', 5 * 5)
```

A pixel with many SHPs is a DS candidate (e.g. `shp_num >= 50` of a 11 x 11 window in the pipelines); `is_shp`
is the input of the coherence estimation.
