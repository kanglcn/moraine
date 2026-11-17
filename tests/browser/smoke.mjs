// Smoke test of moraine/cli/viewer.js in jsdom: the 2D (real Leaflet) and 3D (fake deck.gl) paths, and the
// loading of the deck.gl bundle under an AMD loader. See README.md; `node smoke.mjs` prints `all ok`.
import { JSDOM } from "jsdom";
import assert from "node:assert/strict";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// the Leaflet module and a copy of viewer.js importing it locally, kept in .cache
const here = dirname(fileURLToPath(import.meta.url)), cache = join(here, ".cache");
mkdirSync(cache, { recursive: true });
const leaflet = join(cache, "leaflet-src.esm.mjs");
if (!existsSync(leaflet)) {
  const response = await fetch("https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet-src.esm.js");
  if (!response.ok) throw new Error(`cannot fetch Leaflet: HTTP ${response.status}`);
  writeFileSync(leaflet, await response.text());
}
const viewerPath = join(cache, "viewer.mjs");
writeFileSync(viewerPath, readFileSync(join(here, "..", "..", "moraine", "cli", "viewer.js"), "utf8")
  .replace(/import \* as L from "[^"]+";/, 'import * as L from "./leaflet-src.esm.mjs";'));

const dom = new JSDOM(`<!doctype html><html><body><div id="root"></div></body></html>`,
                      { pretendToBeVisual: true, url: "https://localhost/" });
const { window } = dom;
for (const k of ["window", "document", "navigator", "HTMLElement", "Element", "Node", "MouseEvent", "Event",
                 "getComputedStyle", "SVGElement", "DOMException"]) {
  globalThis[k] = window[k];
}
// sized elements: the width / height of the style, or 700 x 500
Object.defineProperty(window.HTMLElement.prototype, "clientWidth", { get() { return parseInt(this.style.width) || 700; } });
Object.defineProperty(window.HTMLElement.prototype, "clientHeight", { get() { return parseInt(this.style.height) || 500; } });
Object.defineProperty(window.HTMLElement.prototype, "offsetWidth", { get() { return this.clientWidth; } });
Object.defineProperty(window.HTMLElement.prototype, "offsetHeight", { get() { return this.clientHeight; } });
window.HTMLElement.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, right: this.clientWidth, bottom: this.clientHeight, width: this.clientWidth, height: this.clientHeight };
};
// jsdom has no SVG geometry: let Leaflet detect SVG support and draw polygons with its SVG renderer
const svgProto = (window.SVGSVGElement || window.SVGElement).prototype;
if (!svgProto.createSVGRect) svgProto.createSVGRect = () => ({});
const observers = [];
globalThis.ResizeObserver = class {
  constructor(cb) { this.cb = cb; }
  observe(el) { observers.push([this.cb, el, this]); }
  disconnect() { for (let i = observers.length - 1; i >= 0; i--) if (observers[i][2] === this) observers.splice(i, 1); }
};
const fire_resize = () => observers.forEach(([cb]) => cb());
// blob URLs of javascript become temporary module files, importable by node; node's Blob keeps no text: keep the parts
const NodeBlob = globalThis.Blob;
globalThis.Blob = class extends NodeBlob { constructor(parts, options) { super(parts, options); this.parts = parts; } };
let blobCount = 0;
globalThis.URL.createObjectURL = (blob) => {
  globalThis.lastBlob = blob;
  if (blob.type !== "text/javascript") return "blob:fake";
  const path = join(cache, `blob_${blobCount++}.mjs`);
  writeFileSync(path, blob.parts.join(""));
  return pathToFileURL(path).href;
};
globalThis.URL.revokeObjectURL = () => {};
window.URL.createObjectURL = globalThis.URL.createObjectURL;
window.URL.revokeObjectURL = globalThis.URL.revokeObjectURL;
globalThis.createImageBitmap = async (blob) => ({ width: 256, height: 256, blob, image: blob.image, close() {} });
// canvases keep one image; a blob of a canvas carries the bytes of its image
globalThis.ImageData = class { constructor(w, h) { this.width = w; this.height = h; this.data = new Uint8ClampedArray(4 * w * h); } };
globalThis.OffscreenCanvas = class {
  constructor(w, h) { this.width = w; this.height = h; this.image = null; }
  getContext() {
    const c = this;
    return { drawImage(bitmap) { c.image = bitmap.image; }, getImageData() { return c.image; }, putImageData(img) { c.image = img; } };
  }
  convertToBlob() { const b = new Blob([this.image.data], { type: "image/png" }); b.image = this.image; return b; }
};
globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
window.requestAnimationFrame = globalThis.requestAnimationFrame;
window.cancelAnimationFrame = globalThis.cancelAnimationFrame;

// a fake deck.gl: layers keep their props, Deck keeps setProps, picking gives a point of the terrain
class FakeLayer {
  constructor(...props) { this.props = Object.assign({}, ...props); this.id = this.props.id; }
  clone(p) { return new this.constructor(this.props, p); }
}
// a flat mesh at 300 m over the scene, in "common" coordinates that are longitude / latitude here; the fake
// viewport's rays go from a camera 5000 m up to a point 1000 m below the ground at the pixel's position
const flatMesh = { attributes: { POSITION: { value: new Float32Array([14.0, 40.7, 300, 14.3, 40.7, 300, 14.3, 41.0, 300, 14.0, 41.0, 300]) } },
                   indices: { value: new Uint32Array([0, 1, 2, 0, 2, 3]) } };
const fakeViewport = {
  unproject: ([x, y, z]) => [14.1 + x * 1e-5, 40.8 - y * 1e-5, z === 1 ? -1000 : 1000],
  projectPosition: ([lng, lat, z]) => [lng, lat, z], projectFlat: ([lng, lat]) => [lng, lat], unprojectFlat: ([x, y]) => [x, y],
  distanceScales: { metersPerUnit: [1, 1, 1], unitsPerMeter: [1, 1, 1] },
};
const deck = {
  instances: [],
  Deck: class { constructor(props) { this.props = props; deck.instances.push(this); }
                setProps(p) { Object.assign(this.props, p); }
                pickObject({ x, y }) { return x > 600 ? null : { layer: { props: { mesh: flatMesh } }, index: 0 }; }
                getViewports() { return [fakeViewport]; }
                redraw() {} finalize() { this.finalized = true; } },
  MapView: class { constructor(p) { this.p = p; } },
  WebMercatorViewport: class { constructor(p) { this.p = p; }
                               fitBounds(b) { return { longitude: (b[0][0] + b[1][0]) / 2, latitude: (b[0][1] + b[1][1]) / 2, zoom: 12.3 }; } },
  fetched: [],
  TileLayer: class extends FakeLayer {},
  TerrainLayer: class extends FakeLayer { constructor(...p) { super({ fetch: async (url, options) => { deck.fetched.push(url); return { url, options }; } }, ...p); } },
  BitmapLayer: class extends FakeLayer {},
  ScatterplotLayer: class extends FakeLayer {}, PathLayer: class extends FakeLayer {},
  SimpleMeshLayer: class extends FakeLayer {}, COORDINATE_SYSTEM: { CARTESIAN: 0, LNGLAT: 1 },
  _TerrainExtension: class {},
};
globalThis.deck = deck;

const { default: widget } = await import(pathToFileURL(viewerPath).href);

// a fake anywidget model
function makeModel(state) {
  const handlers = {};
  return {
    state, sent: [],
    get(k) { return this.state[k]; },
    set(k, v) { this.state[k] = v; (handlers[`change:${k}`] || []).forEach((fn) => fn()); },
    save_changes() {},
    on(ev, fn) { (handlers[ev] = handlers[ev] || []).push(fn); },
    off(ev, fn) { handlers[ev] = (handlers[ev] || []).filter((f) => f !== fn); },
    send(content, buffers) { this.sent.push(content); },
    reply(id, msg, buffers = []) { (handlers["msg:custom"] || []).forEach((fn) => fn({ id, ...msg }, buffers)); },
    take(type) { const i = this.sent.findIndex((m) => m.type === type); assert.ok(i >= 0, `no ${type} request`); return this.sent.splice(i, 1)[0]; },
  };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Campi Flegrei like extent in web mercator metres
const base = {
  crs: "web_mercator", view_origin: [0, 0], extent: [1562936, 4979838, 1585244, 4998248], frame: [700, 577], size: [],
  zoom: 11, max_zoom: 19, axis_labels: ["longitude", "latitude"],
  panels: [{ title: "pc_ph", series: true,
             layers: [{ label: "pc_ph", colors: ["#000", "#fff"], clim: [-3.14, 3.14], bar_label: "phase", opacity: 1, sliders: ["image"] },
                      { label: "coh", colors: ["#000", "#fff"], clim: [0, 1], bar_label: "coh", opacity: 0.8, sliders: [] }] }],
  kdims: [{ name: "image", max: 3 }], index: { image: 0 }, dates: ["a", "b", "c", "d"], polygons: [],
  selected: {}, reference: {}, terrain: {},
};
const png = new Uint8Array([137, 80, 78, 71]);

async function test2d() {
  const model = makeModel(structuredClone(base));
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  assert.ok(el.querySelector(".moraine-tv-yaxis"), "2D maps have axes");
  assert.equal(el.querySelectorAll(".moraine-tv-handle").length, 1);
  fire_resize();
  await sleep(50);
  const tiles = model.sent.filter((m) => m.type === "tile");
  assert.ok(tiles.length > 0, "tiles requested after sizing");
  assert.ok(tiles.every((m) => m.index.image === 0 && [0, 1].includes(m.layer)));
  model.reply(tiles[0].id, { type: "tile" }, [png]);
  model.sent = [];
  // slider: only the layer with the slider is requested again
  el.querySelector(".moraine-tv-sliders input[type=range]").value = "2";
  el.querySelector(".moraine-tv-sliders input[type=range]").dispatchEvent(new window.Event("change"));
  await sleep(50);
  const again = model.sent.filter((m) => m.type === "tile");
  assert.ok(again.length > 0 && again.every((m) => m.layer === 0 && m.index.image === 2), "slider swaps layer 0 only");
  // hover -> value request -> status
  const mapEl = el.querySelector(".moraine-tv-map");
  mapEl.dispatchEvent(new window.MouseEvent("mousemove", { clientX: 350, clientY: 250, bubbles: true }));
  await sleep(20);
  const value = model.sent.find((m) => m.type === "value");
  assert.ok(value && typeof value.z === "number", "value request with the zoom");
  model.reply(value.id, { type: "value", x: value.x, y: value.y, values: [{ label: "pc_ph", value: 1.5, point: 7 }] });
  assert.match(el.querySelector(".moraine-tv-status").textContent, /pc_ph \(point 7\): 1.5/);
  // polygons from python are drawn
  model.set("polygons", [[[14.1, 40.8], [14.2, 40.8], [14.2, 40.9]]]);
  assert.ok(mapEl.querySelectorAll("path").length >= 1, "polygon drawn");
  cleanup();
  console.log("2D ok:", tiles.length, "tiles requested, status:", el.querySelector(".moraine-tv-status").textContent.slice(0, 60));
}

async function test3d() {
  const model = makeModel({ ...structuredClone(base),
    terrain: { url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", max_zoom: 15, attribution: "Terrain: test" } });
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  assert.equal(deck.instances.length, 1, "one Deck per map");
  const d = deck.instances[0];
  assert.ok(!el.querySelector(".moraine-tv-yaxis"), "no axes in 3D");
  assert.equal(el.querySelector(".draw").style.display, "none", "no polygon drawing in 3D");
  assert.match(el.querySelector(".hint").textContent, /tilt and rotate/);
  const ids = () => d.props.layers.map((l) => l.id);
  assert.deepEqual(ids(), ["terrain", "base", "data-0-1", "data-1-1", "polygons"], "no markers yet");
  const terrain = d.props.layers[0];
  assert.deepEqual(terrain.props.elevationDecoder, { rScaler: 256, gScaler: 1, bScaler: 1 / 256, offset: -32768 });
  assert.equal(terrain.props.operation, "terrain+draw");
  assert.equal(terrain.props.pickable, false, "nothing is picked on the GPU");
  assert.equal(terrain.props.maxZoom, 21, "terrain tiles at every zoom of the view, two levels beyond 2D");
  assert.equal(terrain.props.elevationData, "moraine-terrain://{z}/{x}/{y}");
  assert.equal(d.props.layers[2].props.maxZoom, 21, "data tiles two levels beyond 2D");
  assert.equal(d.props.layers[2].props.debounceTime, 80);
  assert.match(d.props.layers[1].props.data, /World_Imagery.*\{z\}\/\{y\}\/\{x\}$/);
  // the loaded terrain tiles are what the cursor hits: none yet, so a move probes nothing
  const mapEl = el.querySelector(".moraine-tv-map");
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 100, clientY: 100, bubbles: true }));
  await sleep(30);
  assert.equal(model.sent.filter((m) => m.type === "value").length, 0, "no terrain loaded: nothing probed");
  terrain.props.onTileLoad({ index: { x: 1, y: 2, z: 15 }, content: [flatMesh] });
  assert.equal(d.props.controller.maxZoom, 20, "deck zoom is Leaflet zoom - 1");
  assert.equal(d.props.viewState.pitch, 55);
  assert.equal(el.querySelector(".moraine-tv-ctl select"), null, "satellite only: no base map choice");
  assert.equal(el.querySelector(".moraine-tv-sliders .exaggeration"), null, "no exaggeration slider");
  assert.match(el.querySelector(".moraine-tv-attr").textContent, /^Tiles © Esri.*\| Terrain: test$/);
  // deck's own hover picking is switched off once it is loaded
  const offs = [];
  d.eventManager = { off: (ev) => offs.push(ev) };
  d._onPointerMove = () => {};
  d.props.onLoad();
  assert.deepEqual(offs, ["pointermove", "pointerleave"]);
  // elevation tiles: up to the service's last zoom from the service, deeper ones cut from their parent tile
  deck.fetched = [];
  const got = await terrain.props.fetch("moraine-terrain://15/1/2", { propName: "elevationData" });
  assert.equal(got.url, "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/15/1/2.png");
  assert.equal((await terrain.props.fetch("https://other/x.png", {})).url, "https://other/x.png");
  // the parent tile (15, 2, 3): heights 100 + 0.5 i + 0.25 j, Terrarium encoded
  const parent = new ImageData(256, 256);
  for (let j = 0; j < 256; j++) for (let i = 0; i < 256; i++) {
    const v = Math.round((100 + 0.5 * i + 0.25 * j + 32768) * 256), k = 4 * (j * 256 + i);
    parent.data[k] = (v >> 16) & 255; parent.data[k + 1] = (v >> 8) & 255; parent.data[k + 2] = v & 255; parent.data[k + 3] = 255;
  }
  let parentFetches = 0;
  globalThis.fetch = async (url) => { parentFetches++; assert.equal(url, "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/15/2/3.png"); return { ok: true, blob: async () => ({ image: parent }) }; };
  const subTile = await terrain.props.fetch("moraine-terrain://17/9/13", { propName: "elevationData" });   // n = 4: parent pixels 64.. of (2, 3)
  assert.equal(subTile.url, "blob:fake", "the interpolated tile goes to the loader as a blob");

  // decode the last canvas blob: the parts of the Blob given to createObjectURL are the image bytes
  const img = globalThis.lastBlob.image.data;
  const height = (i, j) => { const k = 4 * (j * 256 + i); return img[k] * 256 + img[k + 1] + img[k + 2] / 256 - 32768; };
  // pixel (i, j) of the sub tile samples the parent at u = 64 + (i + 0.5) / 4 - 0.5, v = 64 + (j + 0.5) / 4 - 0.5
  assert.ok(Math.abs(height(0, 0) - (100 + 0.5 * 63.625 + 0.25 * 63.625)) < 0.01, `height ${height(0, 0)}`);
  assert.ok(Math.abs(height(255, 100) - (100 + 0.5 * 127.375 + 0.25 * 88.625)) < 0.01, `height ${height(255, 100)}`);
  await terrain.props.fetch("moraine-terrain://17/10/13", { propName: "elevationData" });
  assert.equal(parentFetches, 1, "the parent tile is fetched once");
  // sizing fits the scene (in the next frame)
  fire_resize();
  await sleep(20);
  assert.equal(d.props.viewState.zoom, 12.3);
  // the tiles of a data layer come from the kernel
  const data0 = d.props.layers[2];
  assert.deepEqual(data0.props.extent.map((v) => Math.round(v * 1000) / 1000), [14.04, 40.779, 14.24, 40.904]);
  const ctl = new AbortController();
  const promise = data0.props.getTileData({ index: { x: 1, y: 2, z: 3 }, signal: ctl.signal });
  const tile = model.take("tile");
  assert.deepEqual([tile.layer, tile.x, tile.y, tile.z, tile.panel], [0, 1, 2, 3, 0]);
  model.reply(tile.id, { type: "tile" }, [png]);
  const bitmap = await promise;
  assert.equal(bitmap.width, 256);
  const aborted = data0.props.getTileData({ index: { x: 1, y: 2, z: 4 }, signal: ctl.signal });
  const t2 = model.take("tile");
  ctl.abort();
  await aborted.then(() => assert.fail("aborted tile resolved"), (e) => assert.equal(e.name, "AbortError"));
  model.reply(t2.id, { type: "tile" }, [png]);            // dropped: no callback left
  const sub = data0.props.renderSubLayers({ id: "x", tile: { boundingBox: [[14.1, 40.8], [14.2, 40.9]] }, data: bitmap });
  assert.deepEqual(sub.props.bounds, [14.1, 40.8, 14.2, 40.9]);
  assert.equal(sub.props.textureParameters.magFilter, "nearest");
  // hover: the point of the loaded terrain under the cursor and its height, probed once per frame with the
  // last position, never while a button is down
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 50, clientY: 50, bubbles: true }));
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 100, clientY: 100, bubbles: true }));
  assert.equal(model.sent.filter((m) => m.type === "value").length, 0, "not probed at once");
  await sleep(30);
  const value = model.take("value");
  assert.equal(value.z, 13.3, "the kernel gets Leaflet's zoom");
  assert.equal(model.sent.filter((m) => m.type === "value").length, 0, "one probe for two moves");
  model.reply(value.id, { type: "value", x: 1570000, y: 4990000, values: [{ label: "coh", value: 0.9 }] });
  assert.match(el.querySelector(".moraine-tv-status").textContent, /height 300 m.*coh: 0.9/);
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 50000, clientY: 100, bubbles: true }));   // off the terrain
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 120, clientY: 100, bubbles: true, buttons: 1 }));   // a drag
  await sleep(30);
  assert.equal(model.sent.filter((m) => m.type === "value").length, 0);
  terrain.props.onTileUnload({ index: { x: 1, y: 2, z: 15 } });                 // unloaded: nothing to hit
  mapEl.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 100, clientY: 100, bubbles: true }));
  await sleep(30);
  assert.equal(model.sent.filter((m) => m.type === "value").length, 0, "no terrain: nothing probed");
  terrain.props.onTileLoad({ index: { x: 1, y: 2, z: 15 }, content: [flatMesh] });
  // slider: layer 0 swapped, the old layer stays until the new one has its tiles
  model.set("index", { image: 1 });
  assert.deepEqual(ids(), ["terrain", "base", "data-0-1", "data-0-2", "data-1-1", "polygons"]);
  d.props.layers[2].props.onViewportLoad([]);               // the old layer: nothing happens
  assert.equal(d.props.layers.length, 6);
  d.props.layers[3].props.onViewportLoad([]);
  assert.deepEqual(ids(), ["terrain", "base", "data-0-2", "data-1-1", "polygons"]);
  // opacity slider of layer 1, visibility
  const opacity = [...el.querySelectorAll(".moraine-tv-sliders label")].find((l) => l.textContent.startsWith("opacity coh")).querySelector("input");
  opacity.value = "0.3"; opacity.dispatchEvent(new window.Event("input"));
  assert.equal(d.props.layers[3].props.opacity, 0.3);
  const check = el.querySelectorAll(".moraine-tv-ctl input[type=checkbox]")[1];
  check.checked = false; check.dispatchEvent(new window.Event("change"));
  assert.equal(d.props.layers[3].props.visible, false);
  // click: time series of the point of the terrain; double click: reference; both marked on the terrain
  mapEl.dispatchEvent(new window.MouseEvent("pointerdown", { clientX: 200, clientY: 200, bubbles: true }));
  mapEl.dispatchEvent(new window.MouseEvent("click", { clientX: 201, clientY: 200, bubbles: true }));
  await sleep(300);
  const series = model.take("series");
  assert.equal(series.panel, 0);
  model.reply(series.id, { type: "series", layer: 0, label: "pc_ph", key: 5, point: 5, x: 1570000, y: 4990000, values: [0, 1, null, 2] });
  assert.equal(d.props.layers.at(-1).props.data.length, 1, "the clicked point is marked");
  assert.deepEqual(model.get("selected"), { panel: 0, layer: 0, label: "pc_ph", key: 5, x: 1570000, y: 4990000, point: 5 });
  assert.ok(el.querySelector(".moraine-tv-chart svg"), "time series drawn");
  mapEl.dispatchEvent(new window.MouseEvent("dblclick", { clientX: 300, clientY: 200, bubbles: true }));
  const locate = model.take("locate");
  model.reply(locate.id, { type: "locate", layer: 0, label: "pc_ph", key: 9, point: 9, x: 1571000, y: 4991000 });
  assert.equal(d.props.layers.at(-1).props.data.length, 1, "the reference is marked at once");
  assert.deepEqual(model.get("reference"), { panel: 0, layer: 0, label: "pc_ph", key: 9, x: 1571000, y: 4991000, point: 9 });
  const relative = model.take("series");                   // the time series of the clicked point, now relative
  assert.deepEqual(relative.ref, { panel: 0, layer: 0, key: 9 });
  model.reply(relative.id, { type: "series", layer: 0, label: "pc_ph", key: 5, point: 5, x: 1570000, y: 4990000, values: [0, 1, 2, 3], ref: 9 });
  assert.equal(d.props.layers.at(-1).props.data.length, 2, "reference and clicked point marked");
  assert.deepEqual(d.props.layers.at(-1).props.data[0].color, [224, 0, 0]);
  assert.equal(d.props.layers.at(-1).props.terrainDrawMode, "drape", "markers painted on the terrain");
  assert.match(el.querySelector(".moraine-tv-chart .title").textContent, /relative to point 9/);
  // a drag is not a click
  model.sent = [];
  mapEl.dispatchEvent(new window.MouseEvent("pointerdown", { clientX: 200, clientY: 200, bubbles: true }));
  mapEl.dispatchEvent(new window.MouseEvent("click", { clientX: 260, clientY: 200, bubbles: true }));
  await sleep(300);
  assert.equal(model.sent.filter((m) => m.type === "series").length, 0, "a drag asks for nothing");
  // polygons from python are draped paths, closed
  model.set("polygons", [[[14.1, 40.8], [14.2, 40.8], [14.2, 40.9]]]);
  assert.deepEqual(d.props.layers.at(-2).props.data[0].at(-1), [14.1, 40.8]);
  cleanup();
  assert.ok(d.finalized, "deck finalized on cleanup");
  console.log("3D ok");
}

async function test3dLinked() {
  deck.instances = [];
  const model = makeModel({ ...structuredClone(base), terrain: { url: "u/{z}/{x}/{y}.png", max_zoom: 15, attribution: "t" },
    panels: [base.panels[0], { title: "second", series: false, layers: [base.panels[0].layers[1]] }] });
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  assert.equal(deck.instances.length, 2);
  const [a, b] = deck.instances;
  a.props.onViewStateChange({ viewState: { longitude: 14.15, latitude: 40.85, zoom: 13, pitch: 60, bearing: 30 } });
  assert.deepEqual(b.props.viewState, { longitude: 14.15, latitude: 40.85, zoom: 13, pitch: 60, bearing: 30 });
  cleanup();
  console.log("3D linked ok");
}

// the deck.gl bundle is loaded in a module scope where the AMD `define` of the notebook front end is hidden
async function testLoadDeck() {
  deck.instances = [];
  const fake = globalThis.deck;
  delete globalThis.deck;
  globalThis.__fakeDeck = fake;
  globalThis.define = Object.assign(() => { throw new Error("AMD define called: the bundle took the AMD branch"); }, { amd: {} });
  const umd = `(function webpackUniversalModuleDefinition(root, factory) {
    if (typeof exports === 'object' && typeof module === 'object') module.exports = factory();
    else if (typeof define === 'function' && define.amd) define([], factory);
    else if (typeof exports === 'object') exports['deck'] = factory();
    else root['deck'] = factory();})(globalThis, function () { "use strict"; return globalThis.__fakeDeck; });`;
  const model = makeModel({ ...structuredClone(base), terrain: { url: "u/{z}/{x}/{y}.png", max_zoom: 15, attribution: "t" } });
  // a failing fetch is reported in the widget, and tried again by the next widget
  globalThis.fetch = async () => ({ ok: false, status: 404 });
  const el2 = document.createElement("div");
  await widget.render({ model: makeModel(structuredClone(model.state)), el: el2 });
  assert.match(el2.textContent, /HTTP 404/);
  assert.equal(globalThis.deck, undefined);
  let fetched = 0;
  globalThis.fetch = async (url) => { fetched++; assert.match(url, /deck\.gl@9\.1\.12\/dist\.min\.js$/); return { ok: true, text: async () => umd }; };
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  assert.equal(fetched, 1, "the bundle is fetched once");
  assert.equal(globalThis.deck, fake, "globalThis.deck defined although `define` exists");
  assert.equal(deck.instances.length, 1);
  cleanup();
  // a second widget on the page does not fetch again
  const el3 = document.createElement("div");
  (await widget.render({ model: makeModel(structuredClone(model.state)), el: el3 }))();
  assert.equal(fetched, 1);
  console.log("loadDeck ok");
}

// colour bar ticks, the scale bar, slider stepping by buttons and keys, the time axis of the chart
async function testControls() {
  const dates = ["20190102", "20190114", "20190208", "20200101"];
  const model = makeModel({ ...structuredClone(base), dates });
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  fire_resize();
  await sleep(50);
  // two colour bars of 150 px (the map is 500 px high, 300 px for all bars): ticks every 2 over (-3.14, 3.14)
  const bar = el.querySelector(".moraine-tv-bar");
  assert.equal(bar.querySelector(".ramp").style.height, "150px");
  const ticks = [...bar.querySelectorAll(".tick")];
  assert.deepEqual(ticks.map((t) => t.textContent), ["-2", "0", "2"]);
  assert.equal(ticks[1].style.top, "75px", "0 in the middle of the ramp");
  assert.equal(bar.querySelector(".hi").textContent, "3.14");
  // the scale bar of a web mercator map
  assert.ok(el.querySelector(".leaflet-control-scale"), "scale bar");
  // slider buttons: the steps are applied once the stepping pauses, as one change
  model.sent = [];
  let notebookKeys = 0;
  const onDocKey = () => { notebookKeys++; };
  document.addEventListener("keydown", onDocKey);
  const row = el.querySelector(".moraine-tv-sliders .slider"), input = row.querySelector("input");
  row.querySelector(".next").click();
  row.querySelector(".next").click();
  assert.equal(row.querySelector("output").textContent, "2 (20190208)", "the value is shown at once");
  assert.equal(model.get("index").image, 0, "not applied yet");
  await sleep(320);
  assert.equal(model.get("index").image, 2, "applied after the pause");
  const swapped = model.sent.filter((m) => m.type === "tile");
  assert.ok(swapped.length > 0 && swapped.every((m) => m.layer === 0 && m.index.image === 2), "one swap to image 2");
  row.querySelector(".next").click(); row.querySelector(".next").click();
  await sleep(320);
  assert.equal(model.get("index").image, 3, "stops at the last image");
  row.querySelector(".prev").click();
  await sleep(320);
  assert.equal(model.get("index").image, 2);
  // keys: a key changes the value (the browser does), the change is applied after the pause; keys do not reach
  // the notebook
  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
  input.value = "1";
  input.dispatchEvent(new window.Event("input"));
  input.dispatchEvent(new window.Event("change"));
  assert.equal(model.get("index").image, 2, "a key step is not applied at once");
  await sleep(320);
  assert.equal(model.get("index").image, 1);
  assert.equal(notebookKeys, 0, "slider keys stop at the slider");
  // the mouse: applied when released
  input.dispatchEvent(new window.MouseEvent("pointerdown", { bubbles: true }));
  input.value = "3";
  input.dispatchEvent(new window.Event("change"));
  assert.equal(model.get("index").image, 3, "a mouse change is applied at once");
  document.removeEventListener("keydown", onDocKey);
  // the time series against the dates
  const mapEl = el.querySelector(".moraine-tv-map");
  mapEl.dispatchEvent(new window.MouseEvent("click", { clientX: 350, clientY: 250, bubbles: true }));
  await sleep(300);
  const series = model.take("series");
  model.reply(series.id, { type: "series", layer: 0, label: "pc_ph", key: 5, point: 5, x: 1570000, y: 4990000, values: [0, 1, null, 2] });
  const svg = el.querySelector(".moraine-tv-chart svg");
  const cx = [...svg.querySelectorAll("circle")].map((c) => Number(c.getAttribute("cx")));
  const t = dates.map((d) => Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6)));
  const W = Number(svg.getAttribute("width")), x0 = 56, x1 = W - 12;
  const expected = [0, 1, 3].map((i) => x0 + (t[i] - t[0]) / (t[3] - t[0]) * (x1 - x0));
  cx.forEach((x, k) => assert.ok(Math.abs(x - expected[k]) < 1e-6, `point ${k} at ${x}, expected ${expected[k]}`));
  const labels = [...svg.querySelectorAll(".xtick")].map((e) => e.textContent);
  assert.ok(labels.length >= 2 && labels.every((l) => /^\d{4}-\d{2}$/.test(l)), `month ticks: ${labels}`);
  assert.equal(svg.querySelector("circle title").textContent, "20190102: 0");
  cleanup();
  // dates that are not dates: the positions in the series
  const el2 = document.createElement("div");
  document.getElementById("root").appendChild(el2);
  const model2 = makeModel(structuredClone(base));
  const cleanup2 = await widget.render({ model: model2, el: el2 });
  fire_resize();
  await sleep(50);
  el2.querySelector(".moraine-tv-map").dispatchEvent(new window.MouseEvent("click", { clientX: 350, clientY: 250, bubbles: true }));
  await sleep(300);
  const s2 = model2.take("series");
  model2.reply(s2.id, { type: "series", layer: 0, label: "pc_ph", key: 5, point: 5, x: 1570000, y: 4990000, values: [0, 1, 2, 3] });
  const cx2 = [...el2.querySelectorAll(".moraine-tv-chart circle")].map((c) => Number(c.getAttribute("cx")));
  assert.ok(Math.abs((cx2[1] - cx2[0]) - (cx2[3] - cx2[2])) < 1e-6, "evenly spaced");
  assert.deepEqual([...el2.querySelectorAll(".moraine-tv-chart .xtick")].map((e) => e.textContent), ["a", "b", "c", "d"]);
  cleanup2();
  console.log(`controls ok: bar ticks ${ticks.map((t) => t.textContent)}, time ticks ${labels}`);
}

// the dates and the calendar ticks of the chart (functions of the module, taken from its source)
function testTimeTicks() {
  const src = readFileSync(join(here, "..", "..", "moraine", "cli", "viewer.js"), "utf8");
  const fn = (name) => src.slice(src.indexOf(`function ${name}(`), src.indexOf("\n}\n", src.indexOf(`function ${name}(`)) + 2);
  const { parseDate, timeTicks } = new Function(`const DAY = 86400000;\n${fn("parseDate")}\n${fn("timeTicks")}\nreturn { parseDate, timeTicks };`)();
  assert.equal(parseDate("20190102"), Date.UTC(2019, 0, 2));
  assert.equal(parseDate("2020-02-29"), Date.UTC(2020, 1, 29));
  assert.ok(Number.isNaN(parseDate("20190229")), "no 29 February 2019");
  assert.ok(Number.isNaN(parseDate("a")) && Number.isNaN(parseDate("2019")));
  const labels = (a, b, n) => timeTicks(parseDate(a), parseDate(b), n).map((t) => t.label);
  // three years (the Campi Flegrei example) on a chart of 9 ticks: every 6 months
  assert.deepEqual(labels("20190102", "20211229", 9), ["2019-07", "2020-01", "2020-07", "2021-01", "2021-07"]);
  assert.deepEqual(labels("20190102", "20211229", 3), ["2020", "2021"]);
  // days
  assert.deepEqual(labels("20190102", "20190114", 9), ["2019-01-02", "2019-01-04", "2019-01-06", "2019-01-08", "2019-01-10", "2019-01-12", "2019-01-14"]);
  // decades
  assert.deepEqual(labels("19920601", "20221231", 9), ["1995", "2000", "2005", "2010", "2015", "2020"]);
  for (const [a, b, n] of [["20190102", "20211229", 9], ["20190102", "20190114", 4], ["19920601", "20221231", 2]]) {
    const ticks = timeTicks(parseDate(a), parseDate(b), n);
    assert.ok(ticks.length <= n && ticks.every(({ t }) => t >= parseDate(a) && t <= parseDate(b)), `${a} .. ${b}, ${n}`);
  }
  console.log("time ticks ok");
}

// the radar grid: axes in pixels, no scale bar
async function testGrid() {
  const model = makeModel({ ...structuredClone(base), crs: "grid", view_origin: [0.5, 0.5], extent: [0, 0, 4160, 981],
                            frame: [700, 165], axis_labels: ["range", "azimuth"] });
  const el = document.createElement("div");
  document.getElementById("root").appendChild(el);
  const cleanup = await widget.render({ model, el });
  fire_resize();
  await sleep(50);
  assert.ok(el.querySelector(".moraine-tv-xaxis .tick"), "axis ticks");
  assert.equal(el.querySelector(".leaflet-control-scale"), null, "no scale bar on the radar grid");
  cleanup();
  console.log("grid ok");
}

try {
  testTimeTicks();
  await testControls();
  await testGrid();
  await test2d();
  await test3d();
  await test3dLinked();
  await testLoadDeck();
  console.log("all ok");
} finally {
  for (const f of readdirSync(cache)) if (f.startsWith("blob_")) rmSync(join(cache, f));
}
