# Design decisions

Architecture decision records (ADR): one file per decision that shapes how moraine is built or used,
with its reasons and what must not be done because of it. Agents and contributors read this index
before changing the design; a decision is changed by a new record, not by editing code against it.

## Index

| # | decision | status |
|---|---|---|
| [0001](0001-record-decisions.md) | Record design decisions as ADRs in docs/decisions | Accepted |
| [0002](0002-pytorch-for-deep-learning.md) | Deep learning inference uses PyTorch only | Accepted |
| [0003](0003-plain-python-no-nbdev.md) | Develop moraine as a plain Python package, not with nbdev | Accepted |
| [0004](0004-no-docs-website.md) | No documentation website for now | Accepted |
| [0005](0005-cli-generated-from-docstrings.md) | The command line is generated from moraine.cli signatures and docstrings | Accepted |
| [0006](0006-toml-pipelines.md) | Processing chains are TOML pipelines that resume | Accepted |
| [0007](0007-visualization-through-pyramids.md) | Visualization and result statistics go through pyramids | Superseded by 0018 |
| [0008](0008-tool-independent-agent-docs.md) | Agent documentation is tool independent and tested | Accepted |
| [0009](0009-verified-examples.md) | Example pipelines are verified on real data | Accepted |
| [0010](0010-versioned-contracts.md) | Formats others depend on are versioned contracts | Accepted |
| [0011](0011-architecture-map.md) | The module map and layer rules are tested | Accepted |
| [0012](0012-network-emcf-inversion.md) | Redundant networks, EMCF unwrapping and time series inversion | Accepted |
| [0013](0013-own-delaunay-and-mcf.md) | Own Delaunay triangulation and successive shortest path MCF | Accepted |
| [0014](0014-interactive-views-as-notebooks.md) | Interactive views are generated notebooks | Superseded by 0018 |
| [0015](0015-api-package.md) | The API layer is the package moraine.api, grouped by topic | Accepted |
| [0016](0016-tile-viewer-anywidget-leaflet.md) | Tile viewer with anywidget and Leaflet, tiles through the notebook channel | Superseded by 0018 |
| [0017](0017-polygons-as-geojson.md) | Polygons are GeoJSON files in longitude / latitude or radar grid coordinates | Accepted |
| [0018](0018-one-viewer.md) | One viewer: `moraine.cli.view` replaces the holoviews plots | Accepted |
| [0019](0019-chunks-space-blocks-one-image.md) | Data are chunked by blocks in space and one image per chunk | Accepted |
| [0020](0020-emcf-any-image-pair-network.md) | EMCF on any network of image pairs, in steps of their own units | Accepted |
| [0021](0021-closure-correction-by-regions.md) | Unwrapping errors are corrected by phase closure per region, as in MintPy | Accepted |
| [0022](0022-closure-correction-outputs-phase-time-series.md) | The phase closure correction outputs the phase time series; no least squares inversion | Accepted |
| [0023](0023-emi-adaptive-regularization.md) | EMI regularizes coherence matrices that are not positive definite, adaptively per point | Accepted |
| [0024](0024-weighted-ds-temporal-coherence.md) | Weighted DS temporal coherence with noise-free squared coherence weights | Accepted |
| [0025](0025-oversampling-from-speckle-correlation.md) | The oversampling of the SLCs is measured from the speckle correlation | Accepted |
| [0026](0026-emi-regularized-by-default.md) | EMI is regularized by default; DS are selected by temporal coherence | Accepted |
| [0027](0027-small-ds-quality-interface.md) | Weighted temporal coherence through `ds_temp_coh`; no EMI quality output; speckle correlation internal | Accepted |
| [0028](0028-ds-selection-by-connected-image-pairs.md) | DS are selected by the connectivity of the coherent image pairs, not by the number of effective pairs | Accepted |

## Writing a record

- File name `NNNN-short-title.md`, the next free number; add it to the index in the same change.
- Sections, in this order: the title line `# NNNN Title`, then `Status`, `Date`, `Context`, `Decision`,
  `Consequences`, `Do not`.
- Status is one of `Proposed`, `Accepted`, `Superseded by NNNN`, `Deprecated`.
- To change a decision, write a new record and mark the old one `Superseded by NNNN`; keep the old file.
- Keep records short: the reason and the rule matter, the history is in git.
- `tests/test_decisions.py` checks the index, the sections and the status of every record.
