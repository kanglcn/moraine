"""Smoke tests: the plots can be built and rendered with bokeh."""
import numpy as np
import pytest

hv = pytest.importorskip('holoviews')
from moraine.api.plot import ras_plot, pc_plot

hv.extension('bokeh')


def _render(obj):
    if isinstance(obj, tuple):
        obj = obj[0] * obj[1]
    return hv.render(obj, backend='bokeh')


def test_ras_plot_2d(rng):
    _render(ras_plot(rng.random((300, 200)).astype(np.float32)))


@pytest.mark.parametrize('post_proc,n_kdim', [('intf_0', None), ('intf_seq', None), ('intf_all', 2)])
def test_ras_plot_stack(rslc, post_proc, n_kdim):
    plot = ras_plot(rslc[:300, :300, :5], post_proc=post_proc, n_kdim=n_kdim)
    # interferogram index dimensions are unbounded until given a range, as in the tutorials
    plot = plot.redim.range(**{d.name: (0, 3) for d in plot.kdims})
    _render(plot)


def test_pc_plot(rng):
    n = 5000
    x, y = rng.random(n) * 1000, rng.random(n) * 1000
    _render(pc_plot(rng.random(n).astype(np.float32), y, x))
