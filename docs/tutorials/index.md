# Tutorials

The five notebooks of `nbs/Tutorials/CLI/CampiFlegrei/` in the repository: the same steps as the
[workflows](../workflows/index.md), written as Python calls of `moraine.cli` with interactive views (`mc.view`) of
every result, on the Campi Flegrei data set (Sentinel-1 descending track 22, 92 dates 2019-2021, the uplifting
caldera near Naples: urban, many PS and DS; how the data were prepared is in `data/CampiFlegrei/README.md`).

| notebook | what it does |
|---|---|
| [01 Load data](CampiFlegrei/01_load.md) | GAMMA results to zarr, coordinates, metadata |
| [02 PS processing](CampiFlegrei/02_ps.md) | amplitude dispersion, Noise2Fringe temporal coherence, PS candidates |
| [03 DS processing](CampiFlegrei/03_ds.md) | SHP, DS candidates, coherence matrices, phase linking, DS selection |
| [04 Pixel refinement](CampiFlegrei/04_refine.md) | merge PS and DS, Noise2Fringe Transformer, temporal coherence |
| [05 Phase unwrapping](CampiFlegrei/05_unwrap.md) | EMCF unwrapping and phase closure correction |

!!! info "Text and code only"
    The notebooks are committed without outputs, so these pages show the text and the code but not the figures.
    Run a notebook in Jupyter or VS Code to see the interactive maps, or look at the figures of the real run in the
    [workflow guides](../workflows/index.md) and the [agent session](../agent/index.md). Each page links to its
    notebook on GitHub.

Run the notebooks in order; every notebook reads the results of the previous ones in its own folder.
