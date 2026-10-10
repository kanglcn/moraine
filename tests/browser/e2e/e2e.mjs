// End to end test of moraine/cli/viewer.js in headless Chromium: real Leaflet and deck.gl (WebGL through
// SwiftShader), page.html with a fake model. See ../README.md; `node e2e/e2e.mjs` prints `all ok`.
// MORAINE_CHROMIUM: a Chromium / Chrome binary to use instead of playwright's own.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { chromium } from "playwright";

const root = normalize(join(dirname(fileURLToPath(import.meta.url)), "..", "..", ".."));   // the repository
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const path = normalize(join(root, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  if (!path.startsWith(root)) { res.writeHead(403); res.end(); return; }
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404); res.end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}/tests/browser/e2e/page.html`;

const browser = await chromium.launch({
  headless: true, executablePath: process.env.MORAINE_CHROMIUM || undefined,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl", "--no-sandbox"],
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(query) {
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  page.on("console", (msg) => { if (msg.type() === "error" || msg.type() === "warning") console.log(`  [browser ${msg.type()}] ${msg.text().slice(0, 300)}`); });
  page.on("pageerror", (e) => console.log(`  [page error] ${e.message}`));
  await page.goto(`${base}${query}`, { waitUntil: "load" });
  await page.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
  return page;
}
const requests = (page, type) => page.evaluate((t) => window.requests.filter((r) => r.type === t).length, type);
const status = (page) => page.$eval(".moraine-tv-status", (e) => e.textContent);
const mapBox = (page) => page.$eval(".moraine-tv-map", (e) => { const r = e.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });

async function test2d() {
  const page = await open("");
  await sleep(1500);
  const tiles = await requests(page, "tile");
  assert.ok(tiles > 0, "2D: tiles requested");
  const b = await mapBox(page);
  await page.mouse.move(b.x + b.w / 2, b.y + b.h / 2);
  await sleep(300);
  assert.match(await status(page), /pc_ph \(point 42\): 1.25/, "2D: hover shows the value");
  await page.mouse.click(b.x + b.w / 2, b.y + b.h / 2);
  await sleep(600);
  assert.equal(await requests(page, "series"), 1, "2D: click asks for the time series");
  assert.ok(await page.$(".moraine-tv-chart svg"), "2D: chart drawn");
  console.log(`2D ok: ${tiles} tiles, status "${(await status(page)).slice(0, 60)}"`);
  await page.close();
}

async function test3d() {
  const page = await open("?terrain=1");
  await page.waitForFunction(() => window.requests.filter((r) => r.type === "terrain").length > 4, null, { timeout: 60000 });
  await sleep(3000);
  const [terrain, satellite, tiles] = await Promise.all([requests(page, "terrain"), requests(page, "satellite"), requests(page, "tile")]);
  console.log(`3D: ${terrain} terrain, ${satellite} satellite, ${tiles} data tiles requested`);
  assert.ok(tiles > 0, "3D: data tiles requested");
  const b = await mapBox(page);
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  // hover: the status shows the height and the value once the pointer rests
  await page.mouse.move(cx - 40, cy - 20);
  await page.mouse.move(cx, cy);
  await sleep(500);
  const hovered = await status(page);
  console.log(`3D hover status: "${hovered.slice(0, 90)}"`);
  assert.match(hovered, /height \d+ m.*pc_ph \(point 42\): 1.25/, "3D: hover shows height and value");
  // the probed position is the point of the flat terrain (100 m) under the cursor: compare with deck.gl's
  // projection of the pixel's ray at 100 m, and the height with 100 m
  const probe = await page.evaluate(() => window.requests.filter((r) => r.type === "value").at(-1));
  const R = 6378137, lng = probe.x / R * 180 / Math.PI, lat = (2 * Math.atan(Math.exp(probe.y / R)) - Math.PI / 2) * 180 / Math.PI;
  const expected = await page.evaluate(([px, py]) => {
    const mapEl = document.querySelector(".moraine-tv-map"), r = mapEl.getBoundingClientRect();
    return mapEl.moraine.deck.getViewports()[0].unproject([px - r.left, py - r.top], { targetZ: 100 });
  }, [cx, cy]);
  const metres = Math.hypot((lng - expected[0]) * 111320 * Math.cos(lat * Math.PI / 180), (lat - expected[1]) * 110574);
  const height = Number(hovered.match(/height (\d+) m/)[1]);
  console.log(`3D probe: ${metres.toFixed(1)} m from the ray at 100 m, height ${height} m`);
  assert.ok(metres < 2, `3D: probed position off by ${metres.toFixed(1)} m`);
  assert.ok(Math.abs(height - 100) < 2, `3D: probed height ${height} m, terrain at 100 m`);
  // click: the time series
  const before = await requests(page, "series");
  await page.mouse.click(cx, cy);
  await sleep(800);
  const after = await requests(page, "series");
  console.log(`3D click: series requests ${before} -> ${after}`);
  assert.equal(after, before + 1, "3D: click asks for the time series");
  assert.ok(await page.$(".moraine-tv-chart svg"), "3D: chart drawn");
  const selected = await page.evaluate(() => window.model.state.selected);
  assert.equal(selected.point, 42, "3D: selected point in the model");
  // double click: the reference, then the series again relative to it
  await page.mouse.dblclick(cx + 30, cy + 10);
  await sleep(800);
  const reference = await page.evaluate(() => window.model.state.reference);
  assert.equal(reference.point, 42, "3D: double click sets the reference");
  assert.equal(await requests(page, "locate"), 1, "3D: one locate request");
  assert.match(await page.$eval(".moraine-tv-chart .title", (e) => e.textContent), /relative to/, "3D: series relative to the reference");
  // a drag with the right button rotates: no series asked
  const seriesBefore = await requests(page, "series");
  await page.mouse.move(cx, cy);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(cx + 60, cy + 30, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await sleep(600);
  assert.equal(await requests(page, "series"), seriesBefore, "3D: a drag asks for no series");
  // the slider: new tiles of the layer
  const tilesBefore = await requests(page, "tile");
  await page.$eval(".moraine-tv-sliders input[type=range]", (input) => { input.value = "2"; input.dispatchEvent(new Event("change")); });
  await sleep(1500);
  assert.ok(await requests(page, "tile") > tilesBefore, "3D: slider requests tiles");
  // zoom in with the wheel: tiles beyond the 2D limit (max_zoom 16)
  await page.mouse.move(cx, cy);
  for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, -300); await sleep(150); }
  await sleep(2500);
  const zs = await page.evaluate(() => window.requests.filter((r) => r.type === "tile").map((r) => r.z));
  console.log(`3D tile zoom levels requested: ${[...new Set(zs)].sort((a, b) => a - b).join(" ")}`);
  assert.ok(Math.max(...zs) >= 17, "3D: zooming in requests tiles beyond the 2D limit");
  const log = await page.evaluate(() => window.log);
  console.log(`3D page log: ${log.length ? JSON.stringify(log).slice(0, 600) : "clean"}`);
  assert.equal(log.length, 0, "3D: no errors in the page");
  console.log("3D ok");
  await page.close();
}

try {
  await test2d();
  await test3d();
  console.log("all ok");
} finally {
  await browser.close();
  server.close();
}
