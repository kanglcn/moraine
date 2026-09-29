"""Synthetic benchmark with known truth for the unwrapping of point cloud interferograms: clustered points with sparse bridges, two seasons with a winter
gap, deformation (trend and seasonal), DEM error, per image atmosphere, noise growing with low coherence and
snow."""
import numpy as np
from scipy.ndimage import gaussian_filter


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


def scores(unw, pairs, true, x, y):
    """(wrong cycles relative to point 0, wrong spatial edges) against the truth."""
    from moraine.pu import _mcf_network, _spatial_edges
    tr = true[:, pairs[:, 0]] - true[:, pairs[:, 1]]
    rel = np.mean(np.rint((unw - unw[0] - (tr - tr[0])) / (2 * np.pi)) != 0)
    tri, half, _ = _mcf_network(x, y)
    rep, _, _ = _spatial_edges(tri, half)
    p, q = tri[rep], tri[rep - rep % 3 + (rep + 1) % 3]
    edge = np.mean(np.rint(((unw[q] - unw[p]) - (tr[q] - tr[p])) / (2 * np.pi)) != 0)
    return rel, edge
