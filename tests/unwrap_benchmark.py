"""Synthetic data with known truth for the unwrapping of point cloud interferograms.

- `make`: clustered points with sparse bridges, two seasons with a winter gap, deformation (trend and
  seasonal), DEM error, per image atmosphere, noise growing with low coherence and snow.
- `make_islands`: coherent islands separated by water (no points) with large scale signal differences
  between them (atmosphere, two volcanoes), so that spatial unwrapping gets whole islands wrong in some
  interferograms (the case of the phase closure correction of MintPy); noisy points and point x image
  decorrelation (random phase) inside the islands.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, label


def field(rng, shape, scale):
    f = gaussian_filter(rng.normal(size=shape), scale)
    return f / f.std()


def make(seed=0, side=400, noise=0.6, atmo=1.0, snow=2.0):
    rng = np.random.default_rng(seed)
    coh = field(rng, (side, side), 12)                          # smooth coherence proxy
    keep = (coh > 0.3) & (rng.random((side, side)) < 0.5)       # dense clusters
    keep |= rng.random((side, side)) < 0.004                      # sparse points elsewhere (bridges)
    yy, xx = np.nonzero(keep)
    cp = np.clip(coh[yy, xx], -1, 3)
    q = 1 / (1 + np.exp(-2 * cp))                                  # point quality 0..1
    # two seasons of 8 biweekly images, separated by a winter gap of 224 days
    t = np.r_[np.arange(8) * 14, 98 + 224 + np.arange(8) * 14].astype(float)
    nimg = len(t)
    b = rng.normal(0, 70, nimg)
    u, v = xx / side, yy / side
    vel = 12 * np.exp(-((u - 0.45) ** 2 + (v - 0.55) ** 2) / 0.015) + 2 * u   # rad / year
    seasonal = 1.5 * np.clip(field(rng, (side, side), 30)[yy, xx], 0, None)   # rad, freeze / thaw where > 0
    dem = 0.02 * field(rng, (side, side), 20)[yy, xx]                          # rad / m of baseline
    atm = np.stack([field(rng, (side, side), 40)[yy, xx] for _ in range(nimg)], -1) * atmo
    clean = (vel[:, None] * t[None, :] / 365 + seasonal[:, None] * np.sin(2 * np.pi * t[None, :] / 365)
             + dem[:, None] * b[None, :] + atm)
    season = np.ones(nimg)
    for g in np.flatnonzero(np.diff(t) > 100):                     # images next to every winter gap
        season[[g, g + 1]] = snow
    sigma = noise * season[None, :] * (1.4 - q[:, None])
    true = clean + rng.normal(size=clean.shape) * sigma
    x, y = xx.astype(float), yy.astype(float)
    return dict(x=x, y=y, t=t, b=b, true=true, ph=np.exp(1j * true).astype(np.complex64), quality=q.astype(np.float32))


def make_islands(seed=0, side=400, nimg=30, atmo=0.7, noise=0.5, decorrelated=0.03):
    """`decorrelated`: share of (point, image) with a random phase, on 20 % of the points."""
    rng = np.random.default_rng(seed)
    land = field(rng, (side, side), 25 * side / 400) > 0.9
    lab, n_islands = label(land)
    keep = land & (rng.random((side, side)) < 0.35)
    yy, xx = np.nonzero(keep)
    t = np.arange(nimg) * 12.0
    b = rng.normal(0, 50, nimg)
    u, v = xx / side, yy / side
    vel = np.zeros(len(xx))
    for i, amp in zip(range(1, min(n_islands, 2) + 1), (25.0, -15.0)):     # two volcanoes, rad / year
        iy, ix = np.nonzero(lab == i)
        vel += amp * np.exp(-((u - ix.mean() / side) ** 2 + (v - iy.mean() / side) ** 2) / 0.004)
    dem = 0.02 * field(rng, (side, side), 20)[yy, xx]
    atm = np.stack([field(rng, (side, side), 80 * side / 400)[yy, xx] for _ in range(nimg)], -1) * atmo
    clean = vel[:, None] * t[None, :] / 365 + dem[:, None] * b[None, :] + atm
    q = np.clip(0.5 + 0.5 * field(rng, (side, side), 5)[yy, xx], 0.1, 1)
    true = clean + rng.normal(size=clean.shape) * noise * (1.5 - q[:, None])
    bad = (rng.random(clean.shape) < decorrelated * 5) & (rng.random(len(xx)) < 0.2)[:, None]
    true = np.where(bad, clean + rng.uniform(-np.pi, np.pi, clean.shape), true)
    return dict(x=xx.astype(float), y=yy.astype(float), t=t, b=b, true=true,
                ph=np.exp(1j * true).astype(np.complex64), quality=q.astype(np.float32), island=lab[yy, xx] - 1)


def wrong_cycles(unw, pairs, true):
    """Whole cycles by which every (point, interferogram) is wrong, relative to the most common error of that
    interferogram (so that neither the reference point nor a constant offset counts), (n_points, n_pairs)."""
    tr = true[:, pairs[:, 0]] - true[:, pairs[:, 1]]
    c = np.rint((unw - tr) / (2 * np.pi)).astype(np.int64)
    e = np.empty_like(c)
    for k in range(c.shape[1]):
        vals, cnt = np.unique(c[:, k], return_counts=True)
        e[:, k] = c[:, k] - vals[cnt.argmax()]
    return e


def scores(unw, pairs, true, x, y):
    """(share of wrong (point, interferogram), share of wrong gradients on the edges of the point network)."""
    from moraine.api.unwrap.mcf import _mcf_edges
    tr = true[:, pairs[:, 0]] - true[:, pairs[:, 1]]
    edges = _mcf_edges(x, y)[3]
    p, q = edges[:, 0], edges[:, 1]
    edge = np.mean(np.rint(((unw[q] - unw[p]) - (tr[q] - tr[p])) / (2 * np.pi)) != 0)
    return np.mean(wrong_cycles(unw, pairs, true) != 0), edge
