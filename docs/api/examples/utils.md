```python exec="true" source="above" result="text" session="utils"
import numpy as np
from moraine.api.utils_ import get_array_module, is_cuda_available, get_n_cpus_avail

print(is_cuda_available())                       # True only when CUDA_VISIBLE_DEVICES names a GPU and cupy imports
xp = get_array_module(np.zeros(3))               # numpy for numpy arrays, cupy for cupy arrays
print(xp.__name__)
print(get_n_cpus_avail() >= 1)
```

Functions dispatch on the kind of their input with `get_array_module`; `mjit`, `ngjit` and `ngpjit` are the numba
decorators of moraine (compiled code cached by a hash of the sources, decision 0029).
