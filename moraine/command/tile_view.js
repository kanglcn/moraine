// Leaflet map of a moraine raster pyramid. Tiles are requested from the kernel with custom widget
// messages and come back as PNG buffers (moraine/command/tile_view.py, decision 0015).
import * as L from "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet-src.esm.js";

const LEAFLET_CSS = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css";

function loadCss() {
  if (document.querySelector(`link[href="${LEAFLET_CSS}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = LEAFLET_CSS;
  document.head.appendChild(link);
}

// map positions are cells of level 0: lat = line, lng = column, lines grow downwards; 2**z screen pixels
// per cell. The data coordinate of position u is origin + u * res.
const CRS = L.extend({}, L.CRS.Simple, { transformation: L.transformation(1, 0, 1, 0) });

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

function render({ model, el }) {
  loadCss();
  const [ny, nx] = model.get("shape");
  const [fw, fh] = model.get("frame");
  const [xlabel, ylabel] = model.get("axis_labels");
  const [ox, oy] = model.get("origin"), res = model.get("res");

  el.classList.add("moraine-tv");
  el.innerHTML = `
    <div class="moraine-tv-title"></div>
    <div class="moraine-tv-body">
      <div class="moraine-tv-plot">
        <div class="moraine-tv-yaxis"><span class="title"></span><div class="ticks"></div></div>
        <div class="moraine-tv-map"></div>
        <div></div>
        <div class="moraine-tv-xaxis"><div class="ticks"></div><span class="title"></span></div>
      </div>
      <div class="moraine-tv-bar"><span class="hi"></span><div class="ramp"></div><span class="lo"></span>
        <span class="label"></span></div>
    </div>
    <div class="moraine-tv-sliders"></div>
    <div class="moraine-tv-status"></div>`;
  el.querySelector(".moraine-tv-title").textContent = model.get("title");
  const mapEl = el.querySelector(".moraine-tv-map");
  mapEl.style.width = `${fw}px`;
  mapEl.style.height = `${fh}px`;
  const status = el.querySelector(".moraine-tv-status");

  // colour bar
  const colors = model.get("colors");
  const [lo, hi] = model.get("clim");
  el.querySelector(".moraine-tv-bar .ramp").style.background = `linear-gradient(to top, ${colors.join(",")})`;
  el.querySelector(".moraine-tv-bar .ramp").style.height = `${Math.min(fh, 300)}px`;
  el.querySelector(".moraine-tv-bar .lo").textContent = fmt(lo);
  el.querySelector(".moraine-tv-bar .hi").textContent = fmt(hi);
  el.querySelector(".moraine-tv-bar .label").textContent = model.get("label");

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

  const bounds = L.latLngBounds([0, 0], [ny, nx]);
  // the element may not be in the page yet (size 0), so the initial zoom comes from the kernel
  const zoom = model.get("zoom");
  const map = L.map(mapEl, {
    crs: CRS, maxZoom: model.get("max_zoom"), minZoom: zoom - 1, zoomSnap: 1,
    attributionControl: false, maxBounds: bounds.pad(0.5),
  });
  map.setView(bounds.getCenter(), zoom);

  const Tiles = L.GridLayer.extend({
    createTile(coords, done) {
      const img = document.createElement("img");
      img.className = "moraine-tv-tile";
      img.alt = "";
      img._moraineId = request({ type: "tile", z: coords.z, x: coords.x, y: coords.y }, (msg, buffers) => {
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
  const layer = new Tiles({ tileSize: 256, bounds, minZoom: zoom - 1, maxZoom: model.get("max_zoom"),
                            updateWhenZooming: false, keepBuffer: 1 });
  layer.on("tileunload", (e) => pending.delete(e.tile._moraineId));
  layer.addTo(map);

  // axes: range to the right (lng), azimuth down (lat)
  const xTicks = el.querySelector(".moraine-tv-xaxis .ticks"), yTicks = el.querySelector(".moraine-tv-yaxis .ticks");
  el.querySelector(".moraine-tv-xaxis .title").textContent = xlabel;
  el.querySelector(".moraine-tv-yaxis .title").textContent = ylabel;
  function drawAxes() {
    const size = map.getSize();
    if (!size.x || !size.y) return;
    const b = map.getBounds();
    const u0 = b.getWest(), u1 = b.getEast(), v0 = b.getSouth(), v1 = b.getNorth();   // v0 < v1
    drawTicks(xTicks, ox + u0 * res, ox + u1 * res,
              (x) => map.latLngToContainerPoint([v0, (x - ox) / res]).x, size.x, true);
    drawTicks(yTicks, oy + v0 * res, oy + v1 * res,
              (y) => map.latLngToContainerPoint([(y - oy) / res, u0]).y, size.y, false);
  }
  map.on("move zoom resize", drawAxes);
  drawAxes();

  // once the element has its size in the page, let Leaflet measure it and centre the scene again
  let shown = false;
  const resize = new ResizeObserver(() => {
    if (!mapEl.clientWidth || !mapEl.clientHeight) return;
    map.invalidateSize();
    if (!shown) map.setView(bounds.getCenter(), zoom);
    shown = true;
    drawAxes();
  });
  resize.observe(mapEl);

  // value under the cursor, at most one request in flight
  let valueId = null, lastPos = null;
  function probe() {
    if (valueId !== null || lastPos === null) return;
    const [u, v] = lastPos;
    lastPos = null;
    valueId = request({ type: "value", x: u, y: v, z: map.getZoom() }, (msg) => {
      valueId = null;
      if (msg.error) status.textContent = `error: ${msg.error}`;
      else if (msg.x === undefined) status.textContent = "no point here";
      else {
        const where = `${xlabel} ${fmt(msg.x)}, ${ylabel} ${fmt(msg.y)}`;
        status.textContent = (msg.point === undefined ? where : `point ${msg.point} (${where})`) +
                             `: ${fmt(msg.value)}`;
      }
      probe();
    });
  }
  map.on("mousemove", (e) => {
    const u = e.latlng.lng, v = e.latlng.lat;
    if (u < 0 || v < 0 || u >= nx || v >= ny) {     // outside the scene: nothing to ask
      lastPos = null;
      status.textContent = "";
      return;
    }
    lastPos = [u, v];
    probe();
  });
  map.on("mouseout", () => { lastPos = null; });

  // sliders over the images of a stack
  const sliders = el.querySelector(".moraine-tv-sliders");
  model.get("kdims").forEach((kdim, k) => {
    const row = document.createElement("label");
    row.innerHTML = `<span>${kdim.name}</span><input type="range" min="0" max="${kdim.max}" step="1">
                     <output></output>`;
    const input = row.querySelector("input"), output = row.querySelector("output");
    input.value = model.get("index")[k];
    output.textContent = input.value;
    input.addEventListener("input", () => { output.textContent = input.value; });
    input.addEventListener("change", () => {
      const index = [...model.get("index")];
      index[k] = Number(input.value);
      model.set("index", index);
      model.save_changes();
      layer.redraw();          // unloads the old tiles, their requests are dropped by "tileunload"
    });
    sliders.appendChild(row);
  });

  return () => {
    resize.disconnect();
    model.off("msg:custom", onMessage);
    map.remove();
  };
}

export default { render };
