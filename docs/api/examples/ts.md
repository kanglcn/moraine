```python exec="true" source="above" result="text" session="ts"
import numpy as np
import moraine as mr

n_points, nimages = 6, 4
offset = np.array([0.0, 0.5, -0.3, 1.2], dtype=np.float32)     # unknown value of every image, common to all points
motion = np.zeros((n_points, nimages), dtype=np.float32)
motion[4:] = np.arange(nimages) * 0.4                            # the last two points move, the first four are stable
ts = motion + offset                                             # e.g. image phases of the closure correction
ts[3, 2] += 2 * np.pi                                            # one stable point with an unwrapping error

stable = np.zeros(n_points, dtype=bool); stable[:4] = True      # the reference area
out, ref_value = mr.pc_reference(ts, stable)                     # median of the reference points per image
print(ref_value)
print(out.round(2))
out_mean = mr.pc_reference(ts, stable, method='mean')[0]         # the unwrapping error shifts the mean
print(out_mean[:, 2].round(2))
out_point = mr.pc_reference(ts, 0)[0]                            # one reference point
print(out_point[0])
```

The median of the reference area removes the value common to all points in every image; one point with an
unwrapping error in the area does not change it, while it shifts the mean of image 2.

```python exec="true" source="above" result="text" session="ts"
disp = mr.pc_disp(out, wavelength=0.0554658)                    # Sentinel-1: 1 rad = 4.41 mm
print(disp.dtype)
print(disp.round(2))
```

The referenced phases in millimeters along the line of sight: the moving points come towards the satellite by
1.77 mm per image (0.4 rad); a whole cycle of unwrapping error is half a wavelength, 27.7 mm.

```python exec="true" source="above" result="text" session="ts"
rng = np.random.default_rng(0)
dates = ['20190102', '20190114', '20190126', '20190207', '20190219', '20190303', '20190315', '20190327']
bperp = rng.normal(0, 80, (3, 8)).astype(np.float32)            # perpendicular baselines, m
slant_range = np.full(3, 850e3, dtype=np.float32)                 # m
theta = np.full(3, 0.96, dtype=np.float32)                        # elevation of the look vector, rad
c = 1000 * bperp / (slant_range * np.cos(theta))[:, None]         # mm per m of DEM error
days = 12 * np.arange(8)
dh_true = np.array([0.0, 10.0, -5.0])                             # m above the DEM
disp = (0.1 * days + (c - c[:, :1]) * dh_true[:, None]).astype(np.float32)
dh, out = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)
print(dh.round(2))
print(out.round(2))
```

Points on the DEM, 10 m above and 5 m below it, moving towards the satellite by 0.1 mm per day: the DEM error comes
back from the differences between neighbouring dates, and the displacement without it is the motion alone, 1.2 mm
per 12 days.

```python exec="true" source="above" result="text" session="ts"
dates = [str(d).replace('-', '') for d in np.datetime64('2019-01-02') + 12 * np.arange(60)]
t = 12 * np.arange(60) / 365.25                                   # years from the first date
v_true = np.array([0.0, 10.0, -5.0])                              # mm per year
disp = v_true[:, None] * t + 2 * np.sin(2 * np.pi * t) + rng.normal(0, 1, (3, 60))
disp[:, 20] += 15                                                 # one date with a large error of every point
disp = disp.astype(np.float32)
velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, dates, periods=[1])
print(velocity.round(2), velocity_std.round(2), rms.round(2))
print(np.sqrt((residual ** 2).mean(axis=0)).argmax(), dates[20])   # the date with the largest residual RMS
velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, dates, exclude=[dates[20]], periods=[1])
print(velocity.round(2), velocity_std.round(2), rms.round(2))
print(coef.round(2))                                              # a, v, cosine and sine of the annual term
print(residual[:, 20].round(1))                                   # the excluded date keeps its residual
```

Three points moving by 0, 10 and -5 mm per year with an annual term of 2 mm and 1 mm of noise: the residual RMS over
the points shows the date with the error; left out of the fit, the velocities come back to within about one standard
deviation (0.24 mm per year) and the residual RMS of the points to the noise, and the residual of that date is its
error.
