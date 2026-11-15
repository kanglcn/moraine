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
```

The compressed layout is the upper triangle in the order of `numpy.triu_indices(nimages, 1)` (data contract);
`emperical_co_pc(..., return_n_looks=True)` also gives the effective number of independent looks of each SHP set,
the input of the weighted DS temporal coherence.

```python exec="true" source="above" result="text" session="pl"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
nimages, n_points = 8, 3
phi = rng.uniform(-np.pi, np.pi, (n_points, nimages))            # a true phase history per point
phi[:, 0] = 0
ph_true = np.exp(1j * phi)
lag = np.abs(np.arange(nimages)[:, None] - np.arange(nimages)[None, :])
gamma = 0.7 ** lag                                               # coherence magnitude decaying with the time lag
C = gamma * ph_true[:, :, None] * np.conj(ph_true[:, None, :])   # (n_points, nimages, nimages)
i, j = np.triu_indices(nimages, 1)
coh = C[:, i, j].astype(np.complex64)                            # compressed, (n_points, n_pairs)

ph = mr.emi(coh)                                                 # phase linking, (n_points, nimages), unit amplitude
print(ph.shape, ph.dtype)
print('max phase error [rad]', np.abs(np.angle(ph * np.conj(ph_true))).max().round(4))
t_coh = mr.ds_temp_coh(coh, ph)                                  # how well the linked phases explain the coherence
print('temporal coherence', t_coh.round(3))
```

Coherence estimated from few looks is often not positive definite; `emi` regularizes such matrices by default
(`regularize=True`). With `n_looks` (from `emperical_co_pc`) `ds_temp_coh` also returns the weighted temporal
coherence, the effective number of image pairs and the number of connected components of the coherent pairs,
the quantities the DS selection uses.
