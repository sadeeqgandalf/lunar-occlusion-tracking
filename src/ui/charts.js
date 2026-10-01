// Tiny dependency-free line charts for telemetry.
const nice = (v) => { const e = Math.pow(10, Math.floor(Math.log10(v || 1))), f = v / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * e; };

export function lineChart(cv, o) {
  const dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  if (cv.width !== Math.round(r.width * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext('2d'), W = r.width, H = r.height, pad = { l: 40, r: 8, t: 18, b: 16 };
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const tMin = o.tMin, tMax = Math.max(o.tMax, tMin + 1);
  let yMax = o.yMax;
  if (!yMax) { yMax = 0; for (const s of o.series) for (const p of s.pts) if (Number.isFinite(p[1])) yMax = Math.max(yMax, p[1]); yMax = nice(yMax * 1.1 || 1); }
  const X = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * (W - pad.l - pad.r), Y = (v) => H - pad.b - (Math.min(v, yMax) / yMax) * (H - pad.t - pad.b);
  c.font = '10px ui-monospace,Menlo,monospace'; c.fillStyle = '#7d8ba3'; c.strokeStyle = '#1d2840'; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = (yMax * i) / 4, y = Y(v); c.beginPath(); c.moveTo(pad.l, y); c.lineTo(W - pad.r, y); c.stroke(); c.fillText(fmt(v), 3, y + 3); }
  for (const b of o.bands || []) { c.fillStyle = b.color; c.fillRect(pad.l, Y(b.y1), W - pad.l - pad.r, Math.max(1, Y(b.y0) - Y(b.y1))); }
  for (const s of o.series) {
    c.strokeStyle = s.color; c.lineWidth = s.width || 1.5; c.setLineDash(s.dash || []); c.beginPath();
    let pen = false;
    for (const p of s.pts) {
      if (!Number.isFinite(p[1])) { pen = false; continue; }
      const x = X(p[0]), y = Y(p[1]);
      if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
    }
    c.stroke(); c.setLineDash([]);
  }
  c.fillStyle = '#9fb0c9'; c.fillText(o.title, pad.l, 11);
}
const fmt = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v >= 10 ? v.toFixed(0) : v.toFixed(v < 1 ? 2 : 1));
