import { TAU } from '../core/linalg.js';
import { covEllipse } from '../filters/base.js';

const STYLE = {
  rover: { hazard: ['#05060a', '#7a4a32'], label: '', terrain: ['#2a1810', '#1c0f0a'], scale: 'm' },
  jet: { hazard: ['#ff3b4a33', '#ff3b4a'], label: 'NFZ', terrain: ['#0c1624', '#08101b'], scale: 'm' },
  spacecraft: { hazard: ['#ffb45433', '#ffb454'], label: 'DEBRIS', terrain: ['#05070d', '#05070d'], scale: 'm' },
};

const GLYPH = {
  rover(c, s) { c.beginPath(); c.roundRect(-s * 0.9, -s * 0.55, s * 1.8, s * 1.1, 3); c.moveTo(s * 0.4, 0); c.lineTo(s * 1.25, 0); },
  jet(c, s) { c.beginPath(); c.moveTo(s * 1.3, 0); c.lineTo(-s, s * 0.85); c.lineTo(-s * 0.45, 0); c.lineTo(-s, -s * 0.85); c.closePath(); },
  spacecraft(c, s) { c.beginPath(); c.rect(-s * 0.45, -s * 0.45, s * 0.9, s * 0.9); c.rect(-s * 1.4, -s * 0.18, s * 0.9, s * 0.36); c.rect(s * 0.5, -s * 0.18, s * 0.9, s * 0.36); },
};

export class MapView {
  constructor(cv) { this.cv = cv; this.c = cv.getContext('2d'); this.zoom = 1; this.pan = { x: 0, y: 0 }; this.follow = false; this.fit = 1; this.W = 1; this.H = 1; this.swap = false; this.pad = 22; this.inset = { t: 34, b: 0, l: 0, r: 0 }; this.ox = 0; this.oy = 0; }

  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.cv.getBoundingClientRect();
    this.W = r.width; this.H = r.height;
    this.cv.width = Math.round(r.width * dpr); this.cv.height = Math.round(r.height * dpr);
    this.c.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.world) this.setWorld(this.world, true);
  }
  /** Fit the whole world into the free box (canvas minus legend inset), with pixel padding, centred. */
  setWorld(world, keepView = false) {
    this.world = world; this.swap = !!world.viewSwap; // RPO convention: along-track (V-bar) horizontal, radial (R-bar) vertical
    const b = world.bounds, I = this.inset, p = this.pad;
    const aw = Math.max(50, this.W - I.l - I.r - 2 * p), ah = Math.max(50, this.H - I.t - I.b - 2 * p);
    const spanX = b.xmax - b.xmin, spanY = b.ymax - b.ymin;
    this.fit = this.swap ? Math.min(aw / spanY, ah / spanX) : Math.min(aw / spanX, ah / spanY);
    this.ox = I.l + (this.W - I.l - I.r) / 2; this.oy = I.t + (this.H - I.t - I.b) / 2;
    if (!keepView) { this.zoom = 1; this.pan = { x: 0, y: 0 }; }
  }
  get s() { return this.fit * this.zoom; }
  center(m) {
    if (this.follow && m) { const p = m.platform.pose(m.truth); return [p.x, p.y]; }
    const b = this.world.bounds; return [(b.xmin + b.xmax) / 2 + this.pan.x, (b.ymin + b.ymax) / 2 + this.pan.y];
  }
  w2s(x, y, ctr) { return this.swap ? [this.ox + (y - ctr[1]) * this.s, this.oy - (x - ctr[0]) * this.s] : [this.ox + (x - ctr[0]) * this.s, this.oy - (y - ctr[1]) * this.s]; }
  s2w(px, py, ctr) { return this.swap ? [ctr[0] + (this.oy - py) / this.s, ctr[1] + (px - this.ox) / this.s] : [ctr[0] + (px - this.ox) / this.s, ctr[1] - (py - this.oy) / this.s]; }
  /** Drag-pan by a pixel delta (content follows the cursor). */
  panBy(dx, dy) { if (this.swap) { this.pan.y -= dx / this.s; this.pan.x += dy / this.s; } else { this.pan.x -= dx / this.s; this.pan.y += dy / this.s; } }

  draw(m, ui) {
    if (this.W < 120 || this.H < 120) return; // hidden or too small to draw into
    const c = this.c, ctr = this.center(m), s = this.s, P = m.platform, st = STYLE[P.id], b = m.world.bounds;
    const S = (x, y) => this.w2s(x, y, ctr);
    c.clearRect(0, 0, this.W, this.H);
    c.fillStyle = st.terrain[0]; c.fillRect(0, 0, this.W, this.H);
    // world bounds + grid
    const pA = S(b.xmin, b.ymax), pB = S(b.xmax, b.ymin);
    const bx0 = Math.min(pA[0], pB[0]), bx1 = Math.max(pA[0], pB[0]), by0 = Math.min(pA[1], pB[1]), by1 = Math.max(pA[1], pB[1]);
    c.fillStyle = st.terrain[1]; c.fillRect(bx0, by0, bx1 - bx0, by1 - by0);
    const gstep = niceStep(60 / s);
    c.strokeStyle = '#ffffff10'; c.lineWidth = 1; c.beginPath();
    for (let x = Math.ceil(b.xmin / gstep) * gstep; x <= b.xmax; x += gstep) { const a0 = S(x, b.ymin), a1 = S(x, b.ymax); c.moveTo(a0[0], a0[1]); c.lineTo(a1[0], a1[1]); }
    for (let y = Math.ceil(b.ymin / gstep) * gstep; y <= b.ymax; y += gstep) { const a0 = S(b.xmin, y), a1 = S(b.xmax, y); c.moveTo(a0[0], a0[1]); c.lineTo(a1[0], a1[1]); }
    c.stroke(); c.strokeStyle = '#ffffff30'; c.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);

    // slip / sand patches
    for (const p of m.world.patches) {
      const [px, py] = S(p.x, p.y), g = c.createRadialGradient(px, py, 0, px, py, p.r * s);
      g.addColorStop(0, '#d9a45a55'); g.addColorStop(1, '#d9a45a00'); c.fillStyle = g; c.beginPath(); c.arc(px, py, p.r * s, 0, TAU); c.fill();
    }
    // hazards
    for (const h of m.world.hazards) {
      const [px, py] = S(h.x, h.y);
      c.beginPath(); c.arc(px, py, h.r * s, 0, TAU);
      if (P.id === 'rover') { const g = c.createRadialGradient(px, py, h.r * s * 0.3, px, py, h.r * s); g.addColorStop(0, '#05060a'); g.addColorStop(1, '#1b0f0a'); c.fillStyle = g; c.fill(); c.strokeStyle = '#9a6a4a'; c.lineWidth = 2; c.stroke(); }
      else { c.fillStyle = st.hazard[0]; c.fill(); c.strokeStyle = st.hazard[1]; c.setLineDash([6, 4]); c.lineWidth = 1.5; c.stroke(); c.setLineDash([]); c.fillStyle = st.hazard[1]; c.font = '10px ui-monospace,monospace'; c.textAlign = 'center'; c.fillText(st.label, px, py + 3); c.textAlign = 'left'; }
    }
    // landmarks and live measurement lines
    const tp = P.pose(m.truth), [tx, ty] = S(tp.x, tp.y);
    const rng = m.sensors.landmarkRange || m.sensors.lidarRange;
    if (ui.showTruth && rng) { c.strokeStyle = '#4cc9f018'; c.setLineDash([3, 6]); c.beginPath(); c.arc(tx, ty, rng * s, 0, TAU); c.stroke(); c.setLineDash([]); }
    if (ui.showTruth) for (const l of m.lastMeas) if (m.t - l.t < 0.6) { const [lx, ly] = S(l.lx, l.ly); c.strokeStyle = '#4cc9f055'; c.beginPath(); c.moveTo(tx, ty); c.lineTo(lx, ly); c.stroke(); }
    for (const l of m.world.landmarks) { const [lx, ly] = S(l.x, l.y); c.fillStyle = P.id === 'spacecraft' ? '#7cf' : '#4cc9f0'; c.beginPath(); c.moveTo(lx, ly - 6); c.lineTo(lx + 4.5, ly); c.lineTo(lx, ly + 6); c.lineTo(lx - 4.5, ly); c.fill(); }
    if (P.id === 'spacecraft') { const [ox, oy] = S(0, 0); c.strokeStyle = '#7cf'; c.lineWidth = 2; c.beginPath(); c.arc(ox, oy, 7, 0, TAU); c.stroke(); c.fillStyle = '#7cf'; c.font = '10px ui-monospace,monospace'; c.fillText('TARGET', ox + 10, oy - 8); }
    // objectives
    for (const t of m.targets) {
      const [px, py] = S(t.x, t.y), pulse = 1 + 0.15 * Math.sin(performance.now() / 250);
      c.strokeStyle = t.done ? '#4ade80' : '#ffd166'; c.fillStyle = t.done ? '#4ade8033' : '#ffd16622'; c.lineWidth = 2;
      c.beginPath(); c.arc(px, py, (t.done ? 8 : 10 * pulse), 0, TAU); c.fill(); c.stroke();
      c.fillStyle = c.strokeStyle; c.font = 'bold 11px ui-monospace,monospace'; c.textAlign = 'center'; c.fillText(t.id, px, py + 4); c.textAlign = 'left';
    }
    // waypoint path from the primary estimate
    const pri = m.fs.get(m.primary), pe = pri && P.pose(pri.filter.getState());
    if (pe && m.waypoints.length) {
      c.strokeStyle = '#ffffff90'; c.setLineDash([5, 5]); c.lineWidth = 1.2; c.beginPath();
      let [sx, sy] = S(pe.x, pe.y); c.moveTo(sx, sy);
      for (const w of m.waypoints) { [sx, sy] = S(w.x, w.y); c.lineTo(sx, sy); }
      c.stroke(); c.setLineDash([]);
      m.waypoints.forEach((w, i) => { const [px, py] = S(w.x, w.y); c.fillStyle = '#fff'; c.beginPath(); c.arc(px, py, 3.5, 0, TAU); c.fill(); c.font = '10px ui-monospace,monospace'; c.fillText(i + 1, px + 6, py - 6); });
    }
    // trails
    if (ui.showTruth) trail(c, m.truthTrail, S, '#ffffff55', 1.5);
    for (const f of m.fs.values()) trail(c, f.trail, S, ui.colors[f.id] + '99', f.id === m.primary ? 2 : 1);
    // particles
    if (ui.showParticles) for (const f of m.fs.values()) {
      const p = f.filter.getParticles(); if (!p) continue;
      const n = f.filter.nx, pi = m.model.posIdx;
      c.fillStyle = ui.colors[f.id] + '66';
      for (let i = 0; i < f.filter.N; i += 2) { const [px, py] = S(p[i * n + pi[0]], p[i * n + pi[1]]); c.fillRect(px - 1, py - 1, 2, 2); }
    }
    // estimates
    const gl = GLYPH[P.id], gs = P.id === 'jet' ? 12 : P.id === 'spacecraft' ? 10 : 10;
    for (const f of m.fs.values()) {
      const x = f.filter.getState(), pose = P.pose(x), [px, py] = S(pose.x, pose.y), col = ui.colors[f.id];
      if (ui.showCov) {
        const e = covEllipse(subPos(f.filter.getCov(), m.model)), k = Math.sqrt(5.991);
        c.strokeStyle = col; c.lineWidth = f.id === m.primary ? 1.8 : 1; c.fillStyle = col + '14';
        c.beginPath(); c.ellipse(px, py, Math.max(k * Math.sqrt(e.l1) * s, 2.5), Math.max(k * Math.sqrt(e.l2) * s, 2.5), this.swap ? e.angle - Math.PI / 2 : -e.angle, 0, TAU); c.fill(); c.stroke();
      }
      c.save(); c.translate(px, py); c.rotate(-(P.id === 'spacecraft' ? 0 : pose.th)); gl(c, gs);
      c.fillStyle = col + '55'; c.strokeStyle = col; c.lineWidth = f.id === m.primary ? 2.5 : 1.5; c.fill(); c.stroke(); c.restore();
      if (f.id === m.primary) { c.fillStyle = col; c.font = 'bold 10px ui-monospace,monospace'; c.fillText('PRIMARY', px + 12, py - 12); }
    }
    // truth
    if (ui.showTruth) {
      c.save(); c.translate(tx, ty); c.rotate(-(P.id === 'spacecraft' ? 0 : tp.th)); GLYPH[P.id](c, gs * 1.25);
      c.fillStyle = '#ffffff22'; c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.fill(); c.stroke(); c.restore();
    }
    if (this.swap) {
      c.fillStyle = '#7cf'; c.font = '11px ui-monospace,monospace';
      c.fillText('V-bar (along-track, +) →', bx0 + 8, by1 - 8); c.fillText('↑ R-bar (radial, +)', bx0 + 8, by0 + 16);
    }
    // scale bar
    const bar = niceStep(100 / s), bw = bar * s, X0 = this.W - 16 - bw, Y0 = this.H - 8; // sits in the padding band, clear of the world border
    c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X0 + bw, Y0); c.stroke();
    c.fillStyle = '#fff'; c.font = '11px ui-monospace,monospace'; c.fillText(bar >= 1000 ? bar / 1000 + ' km' : bar + ' m', X0, Y0 - 5);
  }
}

function trail(c, pts, S, color, w) {
  if (pts.length < 2) return;
  c.strokeStyle = color; c.lineWidth = w; c.beginPath();
  pts.forEach(([x, y], i) => { const [px, py] = S(x, y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
  c.stroke();
}
const niceStep = (v) => { const e = Math.pow(10, Math.floor(Math.log10(v))), f = v / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * e; };
/** 2x2 position block so covEllipse works for any model's posIdx */
function subPos(P, model) {
  const [i, j] = model.posIdx;
  return { get: (a, b) => P.get([i, j][a], [i, j][b]) };
}
