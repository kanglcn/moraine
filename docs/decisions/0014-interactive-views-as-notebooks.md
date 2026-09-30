# 0014 Interactive views are generated notebooks

## Status

Superseded by 0017

## Date

2026-09-28

## Context

Users check results interactively (zoom into details, step through the images of a stack) all the time.
`ras_plot` / `pc_plot` do this on pyramids, but need a running python process behind the plot. Writing
the plotting code each time is repetitive and a source of mistakes (colours, axes, slider ranges).

A panel / bokeh server was tried: it needs port forwarding, which the VS Code Remote Tunnels connection
of the maintainer's cluster does not allow. A notebook kernel runs on the data machine and talks to the
editor through the notebook channel, so it works over any VS Code or Jupyter connection.

## Decision

`moraine view PYRAMID ... -o view.ipynb` writes a notebook with one cell per pyramid calling
`moraine.command.summary.view_pyramid`, which chooses from the data:

- colours: the cyclic colorcet `colorwheel` over (-pi, pi] for phases (complex data, interferograms),
  `viridis` over the 1 % - 99 % range of the pyramid statistics otherwise, (0, 1) for booleans; one colour
  bar shared by the image and points layers;
- axes: range / azimuth with azimuth down for rasters and point clouds on the radar grid (non negative
  integer bounds), x / y with north up for map coordinates;
- size: the aspect of the scene fitted into 900 x 700 screen pixels, at most 1:4;
- sliders over the images of a stack, with the ranges of the chosen post processing.

`quicklook` uses the same cyclic colour map.

## Consequences

- No server, no port forwarding; agents create the notebook, the user opens it.
- The notebook contains no data, only calls with absolute pyramid paths.

## Do not

- Do not write custom plotting code for checking results; improve `view_pyramid` instead.
- Do not start plot servers for users on remote machines.
