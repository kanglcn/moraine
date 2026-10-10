# Browser tests of the viewer

`moraine/cli/viewer.js` (the maps of `moraine.cli.view`, decisions 0018 and 0036) cannot be tested by pytest.
Two tests run it outside the notebook, with a fake anywidget model that answers the kernel's messages:

| test | what runs | what it checks |
|---|---|---|
| `smoke.mjs` | the module in jsdom with the real Leaflet and a fake deck.gl | the 2D and 3D code paths: tile requests, slider swaps, probes, markers, polygons, the loading of the deck.gl bundle under an AMD loader, the interpolated elevation tiles |
| `e2e/e2e.mjs` | the module in headless Chromium with the real Leaflet and deck.gl (WebGL through SwiftShader) | what the user does: hover, click (time series), double click (reference), drag, slider, zoom; the probed point is compared with deck.gl's projection to the metre |

They need node (20 or later) and internet access for the Leaflet and deck.gl bundles (jsdelivr); the elevation
and base map tiles are made by the test page itself.

```bash
cd tests/browser
npm install                       # jsdom and playwright, into tests/browser/node_modules
npx playwright install chromium   # once; or point MORAINE_CHROMIUM to a Chromium / Chrome binary
node smoke.mjs
node e2e/e2e.mjs
```

Run them after a change of `viewer.js` or `viewer.css` (the python tests and the pre-commit hook do not).
Extend `smoke.mjs` for new messages or controls, `e2e/e2e.mjs` for new interactions. They print the browser's
console errors; a clean run ends with `all ok`.
