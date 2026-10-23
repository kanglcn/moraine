/* Home page hero: a wrapped-phase field (an uplift bowl on a gentle ramp, growing with time, drawn with a cyclic
   colour map like an interferogram). Computed on a small grid and scaled up by the browser; one static frame
   when the visitor prefers reduced motion. Does nothing on pages without the canvas. */
(function () {
  var canvas = document.getElementById('mo-fringes');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var W = 192, H = 108;
  var buf = document.createElement('canvas');
  buf.width = W; buf.height = H;
  var bctx = buf.getContext('2d');
  var img = bctx.createImageData(W, H);
  var data = img.data;
  var TWO_PI = Math.PI * 2;

  // cyclic colour map: 256 entries of hue around the wheel, fixed saturation and lightness
  var lut = new Uint8ClampedArray(256 * 3);
  for (var i = 0; i < 256; i++) {
    var h = i / 256, s = 0.72, l = 0.56;
    var q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    var rgb = [h + 1 / 3, h, h - 1 / 3].map(function (t) {
      t = (t + 1) % 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    });
    lut[i * 3] = rgb[0] * 255; lut[i * 3 + 1] = rgb[1] * 255; lut[i * 3 + 2] = rgb[2] * 255;
  }

  function frame(t) {
    // the bowl grows and breathes; the ramp (orbital residual) and ripples (topography) are fixed
    var amp = 7.5 + 2.5 * Math.sin(t * 0.25);
    var cx = 0.42 + 0.03 * Math.sin(t * 0.11), cy = 0.55 + 0.02 * Math.cos(t * 0.09);
    var k = 0;
    for (var y = 0; y < H; y++) {
      var v = y / H;
      for (var x = 0; x < W; x++) {
        var u = x / W;
        var dx = (u - cx) * 1.78, dy = v - cy;
        var r2 = dx * dx + dy * dy;
        var phase = amp * Math.exp(-r2 / 0.07)
          + 1.6 * u + 0.9 * v
          + 0.35 * Math.sin(9 * u + 2.1 * v + 0.3 * t) + 0.25 * Math.sin(4.3 * v - 6 * u + 0.2 * t);
        var c = ((phase % TWO_PI) + TWO_PI) % TWO_PI / TWO_PI * 255 | 0;
        data[k++] = lut[c * 3]; data[k++] = lut[c * 3 + 1]; data[k++] = lut[c * 3 + 2]; data[k++] = 255;
      }
    }
    bctx.putImageData(img, 0, 0);
    var w = canvas.clientWidth || 1200, hgt = canvas.clientHeight || 500;
    if (canvas.width !== w || canvas.height !== hgt) { canvas.width = w; canvas.height = hgt; }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(buf, 0, 0, w, hgt);
  }

  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  frame(0);
  if (reduced) return;
  var visible = true, last = 0;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; }).observe(canvas);
  }
  function loop(ms) {
    if (visible && ms - last > 40) { last = ms; frame(ms / 1000); }
    window.requestAnimationFrame(loop);
  }
  window.requestAnimationFrame(loop);
})();
