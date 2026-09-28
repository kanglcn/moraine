---
name: moraine-processing
description: Run InSAR processing with moraine (load GAMMA data, PS / DS selection, phase linking, n2f / n2ft filtering, phase unwrapping) through the `moraine` command line and TOML pipelines, and check the results with `moraine info` / `moraine quicklook`. Use when the user asks to process, re-process, check or tune InSAR data with moraine.
---

# moraine processing

The instructions are tool independent and live in the repository:

1. Read `AGENTS.md` (environment, running pipelines, checking results, data conventions, rules).
2. Read the guide of the requested workflow in `docs/workflows/`:
   `01_load.md`, `02_ps.md`, `03_ds.md`, `04_refine.md`, `05_unwrap.md`.
3. Start from the matching verified pipeline in `examples/`, copied into the user's working directory.

After every run compare `moraine info` of the pyramids with the expected ranges in the guide and look at
the quicklook PNGs before continuing; report anything unusual to the user.
