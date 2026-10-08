```python exec="true" source="above" result="text" session="co"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
nlines, width, nimages = 20, 30, 10
rslc = (rng.standard_normal((nlines, width, nimages)) + 1j * rng.standard_normal((nlines, width, nimages))).astype(np.complex64)
idx = np.array([[5, 5], [10, 15]], dtype=np.int32)              # (azimuth, range) of two points
pc_is_shp = np.ones((2, 5, 5), dtype=bool)                      # their SHPs in a 5 x 5 window (all here)

coh = mr.emperical_co_pc(rslc, idx, pc_is_shp)                   # compressed coherence, (n_points, n_pairs)
print(coh.shape, coh.dtype)
C = mr.uncompress_coh(coh)                                       # full hermitian matrices, (n_points, nimages, nimages)
print(C.shape, np.allclose(np.diagonal(C, axis1=1, axis2=2).real, 1, atol=1e-5))
print(mr.isPD(np.abs(C)))                                        # is the coherence magnitude matrix positive definite?
```

```python exec="true" source="above" result="text" session="co"
bad = np.abs(C[0]).copy()
bad[0, 1] = bad[1, 0] = 1.5                                      # not a valid coherence matrix any more
print(mr.isPD(bad[None])[0], mr.isPD(mr.nearestPD(bad[None]))[0])
intf = mr.intf(np.ascontiguousarray(rslc[..., 0]), np.ascontiguousarray(rslc[..., 1]))   # 1 x 1 look interferogram of two images
print(intf.shape, intf.dtype)
```

The compressed layout is the upper triangle in the order of `numpy.triu_indices(nimages, 1)` (data contract);
`emperical_co_pc(..., return_n_looks=True)` also gives the effective number of independent looks of each SHP set,
the input of the weighted DS temporal coherence.
