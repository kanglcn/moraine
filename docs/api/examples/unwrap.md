```python exec="true" source="above" result="text" session="unwrap"
import numpy as np
import moraine as mr
from moraine.api.unwrap.delaunay_ import delaunay

rng = np.random.default_rng(0)
n = 400
x, y = rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)         # point coordinates in meters, unique
simplices, neighbors = delaunay(x, y)                            # Delaunay triangulation (scipy.spatial.Delaunay format)
print(simplices.shape, neighbors.shape)

phi = 0.004 * x + 0.006 * y                                      # a phase ramp of a few cycles (less than pi between neighbours)
ph = np.exp(1j * phi).astype(np.complex64)                       # the wrapped interferogram
unw = mr.mcf_pc(x, y, ph)                                        # minimum cost flow unwrapping, (n,)
err = (unw - unw[0]) - (phi - phi[0])
print('max error [rad]', np.abs(err).max().round(6))
```

```python exec="true" source="above" result="text" session="unwrap"
nimages = 4
phis = np.stack([k * phi / 3 for k in range(nimages)], axis=1)   # a phase history growing with time
phs = np.exp(1j * phis).astype(np.complex64)                     # (n_points, nimages)
unw, pairs = mr.emcf_pc(x, y, phs, image_pairs=mr.TempNet.from_bandwidth(nimages, bandwidth=2).image_pairs)
print(unw.shape, pairs.shape)
truth = phis[:, pairs[:, 0]] - phis[:, pairs[:, 1]]
err = (unw - unw[:1]) - (truth - truth[:1])
print('max error [rad]', np.abs(err).max().round(6))
```

`unwrap_correct_closure_pc` corrects the interferograms of a network where loops do not close and gives the unwrapped
phase of every image; `gamma_mcf_pt` wraps GAMMA's `mcf_pt` for comparison.
