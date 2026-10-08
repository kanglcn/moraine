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
