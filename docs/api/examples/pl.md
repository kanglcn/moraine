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
