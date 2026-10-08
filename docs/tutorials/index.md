# Tutorials

Notebooks in `nbs/Tutorials/` of the repository: the same five steps as the [workflows](../workflows/index.md),
written as Python calls of `moraine.cli` with interactive views (`mc.view`) of every result, for two Sentinel-1
data sets, and two notebooks on the array API.

| data set | notebooks | site |
|---|---|---|
| Campi Flegrei | [01](CampiFlegrei/01_load.md), [02](CampiFlegrei/02_ps.md), [03](CampiFlegrei/03_ds.md), [04](CampiFlegrei/04_refine.md), [05](CampiFlegrei/05_unwrap.md) | descending track 22, 92 dates 2019-2021, the uplifting caldera near Naples: urban, many PS |
| Xinpu | [01](Xinpu/01_load.md), [02](Xinpu/02_ps.md), [03](Xinpu/03_ds.md), [04](Xinpu/04_refine.md), [05](Xinpu/05_unwrap.md) | ascending track 84, 60 dates, a landslide complex in the Three Gorges Reservoir: vegetated, mostly DS |
| array API | [Adaptive multilook](Adaptive_Multilook.md), [DS processing](DS_Processing.md) | `moraine.*` functions on arrays in memory |

!!! info "Text and code only"
    The notebooks are committed without outputs, so these pages show the text and the code but not the figures.
    Run a notebook in Jupyter or VS Code to see the interactive maps, or look at the figures of real runs in the
    [workflow guides](../workflows/index.md) and the [agent session](../agent/index.md). Each page links to its
    notebook on GitHub.

Run the notebooks of one data set in order; every notebook reads the results of the previous ones in its own
folder. The data are GAMMA results prepared as described in the first notebook.
