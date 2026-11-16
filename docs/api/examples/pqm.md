```python exec="true" source="above" result="text" session="pqm"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
n_points, nimages = 4, 6
phi = rng.uniform(-np.pi, np.pi, (n_points, nimages))
ph = np.exp(1j * phi).astype(np.complex64)                       # phase history of the points, (n_points, nimages)
pairs = mr.TempNet.from_bandwidth(nimages, bandwidth=1).image_pairs
clean = (ph[:, pairs[:, 0]] * np.conj(ph[:, pairs[:, 1]])).astype(np.complex64)   # the sequential interferograms
noise = np.exp(1j * 0.6 * rng.standard_normal((n_points, nimages))).astype(np.complex64)
noise[0] = 1                                                     # the first point is clean

t_coh = mr.temp_coh(clean, ph * noise, pairs)                   # agreement of the (filtered) interferograms with the raw phases
print(t_coh.shape, t_coh.dtype)
print(t_coh.round(3))
```

In the pipelines the interferograms are the deep learning filtered ones and `rslc` the raw stack or phase history:
points whose filtered phase agrees with the raw phase are kept.

```python exec="true" source="above" result="text" session="pqm"
bperp = rng.normal(0, 80, (n_points, nimages)).astype(np.float32)     # perpendicular baselines, m
slant_range = np.full(n_points, 850e3, dtype=np.float32)              # m
theta = np.full(n_points, 0.96, dtype=np.float32)                     # elevation of the look vector, rad
wavelength = 0.0554658
dh_true = np.array([0.0, 8.0, -12.0, 15.0])                           # m above the neighbourhood
dem = 4 * np.pi / wavelength * bperp * dh_true[:, None] / (slant_range * np.cos(theta))[:, None]
roof = (ph * np.exp(1j * dem)).astype(np.complex64)                  # the phase of points above their neighbourhood

print(mr.temp_coh(clean, roof, pairs).round(3))                      # taken for noise
dh, t_coh = mr.pc_su_dem_error(clean, roof, bperp, slant_range, theta, wavelength, pairs)
print(dh.round(2))
print(t_coh.round(3))
```

Points above (or below) the neighbourhood that the filtered interferograms describe lose temporal coherence in
proportion to their baselines; `pc_su_dem_error` finds their height relative to the neighbourhood and the temporal
coherence without it.
