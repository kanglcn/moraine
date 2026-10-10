// Leaflet maps of moraine layers (moraine/cli/viewer.py, decision 0018). Tiles are requested from
// the kernel with custom widget messages and come back as PNG buffers. Several maps are zoomed and panned
// together; sliders, time series, reference and polygons are shared by all maps.
import * as L from "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet-src.esm.js";

const LEAFLET_CSS = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css";
const MAX_HEIGHT = 700;            // of a map sized automatically (screen pixels)
const MIN_SIZE = [200, 150];       // smallest map (width, height)

function loadCss() {
  if (document.querySelector(`link[href="${LEAFLET_CSS}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = LEAFLET_CSS;
  document.head.appendChild(link);
}

// crs "grid": map position (lng, lat) = data (x, y) - view_origin, y down, 2**z screen pixels per data unit
const GRID = L.extend({}, L.CRS.Simple, { transformation: L.transformation(1, 0, 1, 0) });

// base maps under web mercator layers, loaded by the browser
const BASE_MAPS = {
  "Satellite (Esri)": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxNativeZoom: 18, attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics" }],
  "CARTO light": ["https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    { maxNativeZoom: 20, attribution: "&copy; OpenStreetMap contributors &copy; CARTO" }],
  "OpenStreetMap": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxNativeZoom: 19, attribution: "&copy; OpenStreetMap contributors" }],
};

function fmt(v) {
  if (v === null || v === undefined) return "nan";
  if (typeof v === "boolean") return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return a < 1e-3 || a >= 1e5 ? v.toExponential(3) : String(Number(v.toPrecision(4)));
}

// tick spacing of 1, 2 or 5 * 10**n giving about `n` ticks over `span`
function tickStep(span, n = 6) {
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
}

// ticks of the visible range [v0, v1] at screen positions pos(v), clipped to [0, size]
function drawTicks(container, v0, v1, pos, size, horizontal) {
  const step = tickStep(v1 - v0);
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  container.replaceChildren();
  for (let v = Math.ceil(v0 / step) * step; v <= v1; v += step) {
    const p = pos(v);
    if (p < 0 || p > size) continue;
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.textContent = v.toFixed(digits);
    tick.style[horizontal ? "left" : "top"] = `${p}px`;
    container.appendChild(tick);
  }
}

// a map of `width` x `height` screen pixels; the column of the map (and the axis below it) then follows its width
function setMapSize(mapEl, plot, width, height) {
  mapEl.style.width = `${Math.max(MIN_SIZE[0], Math.round(width))}px`;
  mapEl.style.height = `${Math.max(MIN_SIZE[1], Math.round(height))}px`;
  plot.style.gridTemplateColumns = "64px auto";
}

function render({ model, el }) {
  loadCss();
  const mercator = model.get("crs") === "web_mercator";
  const [vox, voy] = model.get("view_origin");
  const [ex0, ey0, ex1, ey1] = model.get("extent");
  const [fw, fh] = model.get("frame");
  const size = model.get("size");
  const [xlabel, ylabel] = model.get("axis_labels");
  const panels = model.get("panels");
  const dates = model.get("dates");
  const zoom = model.get("zoom"), maxZoom = model.get("max_zoom");
  const anySeries = panels.some((p) => p.series);

  el.classList.add("moraine-tv");
  el.innerHTML = `
    <div class="moraine-tv-toolbar">
      <button class="draw" title="click the vertices, double click to close, Esc to cancel; right click a polygon to remove it">draw polygon</button>
      <button class="clear-polygons">clear polygons</button>
      <button class="clear-ref">clear reference</button>
      <span class="hint"></span>
    </div>
    <div class="moraine-tv-panels"></div>
    <div class="moraine-tv-sliders"></div>
    <div class="moraine-tv-status"></div>
    <div class="moraine-tv-chart"></div>`;
  const status = el.querySelector(".moraine-tv-status");
  const chart = el.querySelector(".moraine-tv-chart");
  const sliders = el.querySelector(".moraine-tv-sliders");

  // requests to the kernel: id -> callback
  let nextId = 0;
  const pending = new Map();
  function request(content, callback) {
    const id = nextId++;
    pending.set(id, callback);
    model.send({ ...content, id, index: model.get("index") });
    return id;
  }
  function onMessage(msg, buffers) {
    const callback = pending.get(msg.id);
    if (!callback) return;          // tile unloaded or value outdated meanwhile
    pending.delete(msg.id);
    callback(msg, buffers);
  }
  model.on("msg:custom", onMessage);

  // coordinates
  const toLatLng = (x, y) => L.CRS.EPSG3857.unproject(L.point(x, y));     // web mercator metres
  const dataLatLng = (x, y) => mercator ? toLatLng(x, y) : L.latLng(y - voy, x - vox);
  const bounds = L.latLngBounds(dataLatLng(ex0, ey0), dataLatLng(ex1, ey1));
  // map position for the kernel (lng / lat on the grid, metres for web mercator), null outside the data
  function position(latlng) {
    let u = latlng.lng, v = latlng.lat;
    if (mercator) ({ x: u, y: v } = L.CRS.EPSG3857.project(latlng));
    const [x, y] = mercator ? [u, v] : [vox + u, voy + v];
    return x < ex0 || y < ey0 || x >= ex1 || y >= ey1 ? null : [u, v];
  }
  function where(x, y) {
    if (mercator) {
      const p = toLatLng(x, y);
      return `${xlabel} ${p.lng.toFixed(6)}, ${ylabel} ${p.lat.toFixed(6)}`;
    }
    return `${xlabel} ${fmt(x)}, ${ylabel} ${fmt(y)}`;
  }
  const pointName = (msg) => msg.point === undefined ? where(msg.x, msg.y) : `point ${msg.point} (${where(msg.x, msg.y)})`;

  // ---------------------------------------------------------------- maps
  const Tiles = L.GridLayer.extend({
    createTile(coords, done) {
      const img = document.createElement("img");
      img.className = "moraine-tv-tile";
      img.alt = "";
      const { panel, layer } = this.options;
      img._moraineId = request({ type: "tile", panel, layer, z: coords.z, x: coords.x, y: coords.y }, (msg, buffers) => {
        if (msg.error) {
          status.textContent = `tile error: ${msg.error}`;
          done(new Error(msg.error), img);
          return;
        }
        const url = URL.createObjectURL(new Blob([buffers[0]], { type: "image/png" }));
        img.onload = () => { URL.revokeObjectURL(url); done(null, img); };
        img.src = url;
      });
      return img;
    },
  });

  // the tile layer of layer `l` of a map (any zoom: the map limits it); the kernel's answers to tiles unloaded
  // meanwhile are dropped
  function makeTiles(view, l, opacity) {
    const layer = new Tiles({ tileSize: 256, bounds, minZoom: -100, maxZoom, panel: view.p, layer: l, opacity,
                              updateWhenZooming: false, keepBuffer: 1, zIndex: 10 + l });
    layer.on("tileunload", (e) => pending.delete(e.tile._moraineId));
    return layer;
  }

  function makePanel(panel, p) {
    const box = document.createElement("div");
    box.className = "moraine-tv-panel";
    box.innerHTML = `
      <div class="moraine-tv-title"></div>
      <div class="moraine-tv-body">
        <div class="moraine-tv-plot">
          <div class="moraine-tv-yaxis"><span class="title"></span><div class="ticks"></div></div>
          <div class="moraine-tv-map"><div class="moraine-tv-handle" title="drag to resize the map"></div></div>
          <div></div>
          <div class="moraine-tv-xaxis"><div class="ticks"></div><span class="title"></span></div>
        </div>
        <div class="moraine-tv-bars"></div>
      </div>`;
    el.querySelector(".moraine-tv-panels").appendChild(box);
    box.querySelector(".moraine-tv-title").textContent = panel.title;
    const mapEl = box.querySelector(".moraine-tv-map"), plot = box.querySelector(".moraine-tv-plot");
    // the size given in python; otherwise the map takes the width of the notebook with the aspect of the scene
    // once it is in the page (below)
    if (size.length === 2) setMapSize(mapEl, plot, size[0], size[1]);
    // one colour bar per layer
    for (const info of panel.layers) {
      const bar = document.createElement("div");
      bar.className = "moraine-tv-bar";
      bar.innerHTML = `<span class="hi"></span><div class="ramp"></div><span class="lo"></span><span class="label"></span>`;
      bar.querySelector(".ramp").style.background = `linear-gradient(to top, ${info.colors.join(",")})`;
      bar.querySelector(".ramp").style.height = `${Math.min(fh, 300) / Math.max(1, panel.layers.length)}px`;
      bar.querySelector(".lo").textContent = fmt(info.clim[0]);
      bar.querySelector(".hi").textContent = fmt(info.clim[1]);
      bar.querySelector(".label").textContent = info.bar_label;
      box.querySelector(".moraine-tv-bars").appendChild(bar);
    }

    // the element may not be in the page yet (size 0): the map starts at the kernel's zoom and is fitted to the
    // data once it has its size (below); the zoom is continuous, tiles are drawn at the nearest integer zoom
    const map = L.map(mapEl, {
      crs: mercator ? L.CRS.EPSG3857 : GRID, maxZoom, minZoom: mercator ? 0 : zoom - 1, zoomSnap: 0,
      attributionControl: mercator, doubleClickZoom: false, maxBounds: mercator ? undefined : bounds.pad(0.5),
    });
    map.setView(bounds.getCenter(), zoom);
    if (map.attributionControl) map.attributionControl.setPrefix(false);
    const baseMaps = {};
    if (mercator) {
      for (const [name, [url, options]] of Object.entries(BASE_MAPS)) {
        baseMaps[name] = L.tileLayer(url, { ...options, maxZoom });
      }
      baseMaps["none"] = L.layerGroup();
      baseMaps["Satellite (Esri)"].addTo(map);
    }
    const view = { p, map, mapEl, box, tiles: [], control: null, markers: null, polygons: null, resize: null, sketch: null };
    const overlays = {};
    view.tiles = panel.layers.map((info, l) => {
      const layer = makeTiles(view, l, info.opacity).addTo(map);
      overlays[info.label] = layer;
      return layer;
    });
    if (mercator || panel.layers.length > 1) {
      view.control = L.control.layers(baseMaps, overlays, { collapsed: true }).addTo(map);
    }

    // axes: range to the right, azimuth down (grid); longitude and latitude in degrees (web mercator)
    const xTicks = box.querySelector(".moraine-tv-xaxis .ticks"), yTicks = box.querySelector(".moraine-tv-yaxis .ticks");
    box.querySelector(".moraine-tv-xaxis .title").textContent = xlabel;
    box.querySelector(".moraine-tv-yaxis .title").textContent = ylabel;
    function drawAxes() {
      const size = map.getSize();
      if (!size.x || !size.y) return;
      const b = map.getBounds();
      const u0 = b.getWest(), u1 = b.getEast(), v0 = b.getSouth(), v1 = b.getNorth();   // v0 < v1
      if (mercator) {
        drawTicks(xTicks, u0, u1, (lng) => map.latLngToContainerPoint([v0, lng]).x, size.x, true);
        drawTicks(yTicks, v0, v1, (lat) => map.latLngToContainerPoint([lat, u0]).y, size.y, false);
        return;
      }
      drawTicks(xTicks, vox + u0, vox + u1, (x) => map.latLngToContainerPoint([v0, x - vox]).x, size.x, true);
      drawTicks(yTicks, voy + v0, voy + v1, (y) => map.latLngToContainerPoint([y - voy, u0]).y, size.y, false);
    }
    map.on("move zoom resize", drawAxes);
    drawAxes();

    // once the element has its size in the page, let Leaflet measure it and fit the data into it; later size
    // changes (the handle below) keep the view
    let shown = false;
    const ramps = box.querySelectorAll(".moraine-tv-bar .ramp");
    view.resize = new ResizeObserver(() => {
      if (!mapEl.clientWidth || !mapEl.clientHeight) return;
      if (!shown && size.length !== 2) {
        // the width of the notebook, at most MAX_HEIGHT high, with the aspect of the scene
        const w = Math.min(mapEl.clientWidth, Math.round(MAX_HEIGHT * fw / fh));
        setMapSize(mapEl, plot, w, w * fh / fw);
      }
      map.invalidateSize({ animate: false });
      if (!shown) {
        if (!mercator) {                        // let the data fit however small the map is, and one zoom more
          map.setMinZoom(-100);
          map.setMinZoom(Math.floor(map.getBoundsZoom(bounds, false, L.point(4, 4))) - 1);
        }
        map.fitBounds(bounds, { animate: false, padding: [4, 4] });
      }
      shown = true;
      ramps.forEach((r) => { r.style.height = `${Math.min(mapEl.clientHeight, 300) / Math.max(1, ramps.length)}px`; });
      drawAxes();
    });
    view.resize.observe(mapEl);
    // drag the lower right corner of the map to resize it
    const handle = mapEl.querySelector(".moraine-tv-handle");
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();                       // no mouse events for Leaflet: not a drag of the map
      e.stopPropagation();
      const start = { x: e.clientX, y: e.clientY, w: mapEl.clientWidth, h: mapEl.clientHeight };
      const move = (ev) => setMapSize(mapEl, plot, start.w + ev.clientX - start.x, start.h + ev.clientY - start.y);
      const stop = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", stop);
        handle.removeEventListener("pointercancel", stop);
      };
      handle.setPointerCapture(e.pointerId);
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", stop);
      handle.addEventListener("pointercancel", stop);
    });

    // values of all layers under the cursor, at most one request in flight
    let valueId = null, lastPos = null;
    function probe() {
      if (valueId !== null || lastPos === null) return;
      const [u, v] = lastPos;
      lastPos = null;
      valueId = request({ type: "value", panel: p, x: u, y: v, z: map.getZoom() }, (msg) => {
        valueId = null;
        if (msg.error) status.textContent = `error: ${msg.error}`;
        else {
          const values = msg.values.map((f) => `${f.label}${f.point === undefined ? "" : ` (point ${f.point})`}: ${fmt(f.value)}`);
          status.textContent = `${where(msg.x, msg.y)}  |  ${values.length ? values.join("  |  ") : "no data"}`;
        }
        probe();
      });
    }
    map.on("mousemove", (e) => {
      const pos = position(e.latlng);
      if (pos === null) { lastPos = null; status.textContent = ""; return; }
      lastPos = pos;
      probe();
    });
    map.on("mouseout", () => { lastPos = null; });

    view.markers = L.layerGroup().addTo(map);
    view.polygons = L.layerGroup().addTo(map);
    return view;
  }
  const views = panels.map(makePanel);

  // linked zoom and pan
  let syncing = false;
  for (const view of views) {
    view.map.on("move", () => {
      if (syncing) return;
      syncing = true;
      for (const other of views) {
        if (other !== view) other.map.setView(view.map.getCenter(), view.map.getZoom(), { animate: false });
      }
      syncing = false;
    });
  }

  // ---------------------------------------------------------------- sliders
  const imageText = (i) => dates.length > i ? `${i} (${dates[i]})` : String(i);
  const sliderInputs = {};
  for (const kdim of model.get("kdims")) {
    const row = document.createElement("label");
    row.innerHTML = `<span></span><input type="range" min="0" max="${kdim.max}" step="1"><output></output>`;
    row.querySelector("span").textContent = kdim.label || kdim.name;
    const input = row.querySelector("input"), output = row.querySelector("output");
    input.value = model.get("index")[kdim.name];
    output.textContent = imageText(Number(input.value));
    input.addEventListener("input", () => { output.textContent = imageText(Number(input.value)); });
    input.addEventListener("change", () => {
      model.set("index", { ...model.get("index"), [kdim.name]: Number(input.value) });
      model.save_changes();
    });
    sliderInputs[kdim.name] = [input, output];
    sliders.appendChild(row);
  }
  // the slider values changed on the map or in python: show them and replace the tiles of the layers that depend
  // on the changed sliders; the old tiles stay until the new ones are drawn, the other layers are not touched
  let shownIndex = { ...model.get("index") };
  function swapTiles(view, l) {
    const old = view.tiles[l], next = makeTiles(view, l, old.options.opacity);
    view.tiles[l] = next;
    if (view.control) {
      view.control.removeLayer(old);
      view.control.addOverlay(next, panels[view.p].layers[l].label);
    }
    if (!view.map.hasLayer(old)) return;          // switched off in the layer control: stays off
    next.once("load", () => old.remove());
    next.addTo(view.map);
  }
  function onIndex() {
    const index = model.get("index");
    const changed = Object.keys(index).filter((k) => index[k] !== shownIndex[k]);
    shownIndex = { ...index };
    for (const [name, [input, output]] of Object.entries(sliderInputs)) {
      input.value = index[name];
      output.textContent = imageText(Number(input.value));
    }
    for (const view of views) {
      panels[view.p].layers.forEach((info, l) => {
        if (info.sliders.some((s) => changed.includes(s))) swapTiles(view, l);
      });
    }
  }
  model.on("change:index", onIndex);
  // opacity of each layer, to see the layers or the base map below
  for (const view of views) {
    if (!mercator && panels[view.p].layers.length < 2) continue;
    panels[view.p].layers.forEach((info, l) => {
      const row = document.createElement("label");
      row.innerHTML = `<span></span><input type="range" min="0" max="1" step="0.05"><output></output>`;
      row.querySelector("span").textContent = `opacity ${info.label}`;
      const input = row.querySelector("input"), output = row.querySelector("output");
      input.value = info.opacity; output.textContent = info.opacity;
      input.addEventListener("input", () => { output.textContent = input.value; view.tiles[l].setOpacity(Number(input.value)); });
      sliders.appendChild(row);
    });
  }

  // ---------------------------------------------------------------- time series and reference
  let target = null, ref = null, clickTimer = null;     // target: {p, pos, z} clicked; ref: locate result
  // what python sees of a pixel / point: map, layer, key (line / column or point index), coordinates
  const pick = (msg, p) => {
    const out = { panel: p, layer: msg.layer, label: msg.label, key: msg.key, x: msg.x, y: msg.y };
    if (msg.point !== undefined) out.point = msg.point;
    return out;
  };
  const marker = (latlng, color) => L.circleMarker(latlng, { radius: 6, color, weight: 2, fill: false, interactive: false });
  function drawMarkers(msg) {
    for (const view of views) {
      view.markers.clearLayers();
      if (ref) marker(dataLatLng(ref.x, ref.y), "#e00").addTo(view.markers);
      if (msg && msg.x !== undefined) marker(dataLatLng(msg.x, msg.y), "#fff").addTo(view.markers);
    }
  }
  function requestSeries() {
    if (target === null) return;
    const { p, pos: [u, v], z } = target;
    const refKey = ref ? { panel: ref.p, layer: ref.layer, key: ref.key } : null;
    request({ type: "series", panel: p, x: u, y: v, z, ref: refKey }, (msg) => {
      if (msg.error) { chart.textContent = `error: ${msg.error}`; return; }
      drawMarkers(msg);
      model.set("selected", msg.key === undefined ? {} : pick(msg, p));
      model.save_changes();
      if (msg.values === undefined) { chart.textContent = "no time series here"; return; }
      const relative = ref && ref.p === p && ref.layer === msg.layer;
      const name = `${msg.label}: ${pointName(msg)}`;
      drawChart(msg.values, relative ? `${name} relative to ${pointName(ref)}` : name);
    });
  }
  function drawChart(values, title) {
    const W = Math.max(views[0].mapEl.clientWidth + 64, 400), H = 200, m = { l: 56, r: 12, t: 22, b: 34 };
    const n = values.length, finite = values.filter((v) => v !== null);
    let y0 = Math.min(...finite), y1 = Math.max(...finite);
    if (!finite.length) { y0 = 0; y1 = 1; } else if (y0 === y1) { y0 -= 1; y1 += 1; }
    const px = (i) => m.l + (n > 1 ? i / (n - 1) : 0.5) * (W - m.l - m.r);
    const py = (v) => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    const add = (tag, attrs, text) => {
      const e = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      if (text !== undefined) e.textContent = text;
      svg.appendChild(e);
      return e;
    };
    add("text", { x: m.l, y: 14, class: "title" }, title);
    add("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b, class: "axis" });
    add("line", { x1: m.l, y1: m.t, x2: m.l, y2: H - m.b, class: "axis" });
    const step = tickStep(y1 - y0, 4), digits = Math.max(0, -Math.floor(Math.log10(step)));
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
      add("line", { x1: m.l, y1: py(v), x2: W - m.r, y2: py(v), class: "grid" });
      add("text", { x: m.l - 4, y: py(v) + 4, class: "ytick" }, v.toFixed(digits));
    }
    const xlab = (i) => dates.length === n ? dates[i] : String(i);
    const every = Math.max(1, Math.ceil(n / Math.floor((W - m.l - m.r) / 72)));
    for (let i = 0; i < n; i += every) add("text", { x: px(i), y: H - m.b + 16, class: "xtick" }, xlab(i));
    let path = "";
    values.forEach((v, i) => { if (v !== null) path += `${path && values[i - 1] !== null ? "L" : "M"}${px(i)},${py(v)}`; });
    add("path", { d: path, class: "line" });
    values.forEach((v, i) => {
      if (v === null) return;
      add("circle", { cx: px(i), cy: py(v), r: 3, class: "dot" });
      svg.lastChild.appendChild(document.createElementNS(ns, "title")).textContent = `${xlab(i)}: ${fmt(v)}`;
    });
    chart.replaceChildren(svg);
  }
  el.querySelector(".clear-ref").addEventListener("click", () => {
    ref = null;
    model.set("reference", {}); model.save_changes();
    drawMarkers(); requestSeries();
  });
  if (!anySeries) el.querySelector(".clear-ref").style.display = "none";

  // ---------------------------------------------------------------- polygons
  // vertices in data coordinates on the radar grid, longitude / latitude for web mercator
  const polyLatLng = ([x, y]) => mercator ? L.latLng(y, x) : dataLatLng(x, y);
  const latLngPoly = (ll) => mercator ? [ll.lng, ll.lat] : [vox + ll.lng, voy + ll.lat];
  function drawPolygons() {
    for (const view of views) {
      view.polygons.clearLayers();
      model.get("polygons").forEach((poly, k) => {
        const shape = L.polygon(poly.map(polyLatLng), { color: "#ff0", weight: 2, fillOpacity: 0.1 });
        shape.on("contextmenu", (e) => {
          L.DomEvent.stop(e);
          model.set("polygons", model.get("polygons").filter((_, i) => i !== k));
          model.save_changes();
        });
        shape.addTo(view.polygons);
      });
    }
  }
  model.on("change:polygons", drawPolygons);
  drawPolygons();
  const hint = el.querySelector(".hint"), drawButton = el.querySelector(".draw");
  let drawing = null;       // vertices (latlng) of the polygon being drawn
  function stopDrawing() {
    drawing = null;
    for (const view of views) { if (view.sketch) view.sketch.remove(); view.sketch = null; }
    drawButton.classList.remove("active");
    hint.textContent = anySeries ? "click: time series, double click: reference" : "";
  }
  stopDrawing();
  drawButton.addEventListener("click", () => {
    if (drawing) { stopDrawing(); return; }
    drawing = [];
    for (const view of views) view.sketch = L.polyline([], { color: "#ff0", weight: 2, dashArray: "4 4" }).addTo(view.map);
    drawButton.classList.add("active");
    hint.textContent = "click the vertices, double click to close, Esc to cancel";
  });
  el.querySelector(".clear-polygons").addEventListener("click", () => {
    model.set("polygons", []); model.save_changes();
  });
  const onKey = (e) => { if (e.key === "Escape" && drawing) stopDrawing(); };
  document.addEventListener("keydown", onKey);

  for (const view of views) {
    const { p, map } = view;
    map.on("click", (e) => {
      if (drawing) {
        drawing.push(e.latlng);
        for (const v of views) v.sketch.setLatLngs(drawing);
        return;
      }
      if (!panels[p].series) return;
      clearTimeout(clickTimer);                // wait: the click may be the first of a double click
      const pos = position(e.latlng);
      clickTimer = setTimeout(() => {
        if (pos === null) return;
        target = { p, pos, z: map.getZoom() };
        requestSeries();
      }, 250);
    });
    map.on("dblclick", (e) => {
      clearTimeout(clickTimer);
      if (drawing) {
        // the two clicks of the double click added the last vertex twice
        const pts = drawing.filter((q, i) => i === 0 ||
          map.latLngToContainerPoint(q).distanceTo(map.latLngToContainerPoint(drawing[i - 1])) > 3);
        if (pts.length >= 3) {
          model.set("polygons", [...model.get("polygons"), pts.map(latLngPoly)]);
          model.save_changes();
        }
        stopDrawing();
        return;
      }
      if (!panels[p].series) return;
      const pos = position(e.latlng);
      if (pos === null) return;
      request({ type: "locate", panel: p, x: pos[0], y: pos[1], z: map.getZoom() }, (msg) => {
        if (msg.error || msg.key === undefined) return;
        ref = { ...msg, p };
        model.set("reference", pick(msg, p)); model.save_changes();
        drawMarkers();
        requestSeries();
      });
    });
  }

  return () => {
    document.removeEventListener("keydown", onKey);
    clearTimeout(clickTimer);
    model.off("msg:custom", onMessage);
    model.off("change:polygons", drawPolygons);
    model.off("change:index", onIndex);
    for (const view of views) { view.resize.disconnect(); view.map.remove(); }
  };
}

export default { render };
