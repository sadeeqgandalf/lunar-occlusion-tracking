// Lunar-surface scene: a rover mast camera watches astronauts / rovers walking among boulders.
import { RNG } from '../core/rng.js';
import { clamp, wrapAngle } from '../core/linalg.js';
import { visibility, detectionProb } from './occlusion.js';

export const MOT_SCENARIOS = {
  boulders: { label: 'Artemis EVA · Boulder field', blurb: 'Five crew/rovers walk through a boulder field. They keep disappearing behind rocks.', targets: 5, boulders: 9, pd: 0.95, clutter: 0.8 },
  crossing: { label: 'Worksite · Crossing paths', blurb: 'Eight targets, paths cross often behind cover. Hardest for keeping identities.', targets: 8, boulders: 7, pd: 0.92, clutter: 1.2 },
  open: { label: 'Open plain · Sparse cover', blurb: 'A single boulder: fewer, shorter occlusions. Shows how results scale with how much cover there is.', targets: 5, boulders: 1, pd: 0.95, clutter: 0.8 },
};

export const CAMERA = {
  x: 0, y: -4, th: Math.PI / 2, fov: (100 * Math.PI) / 180, range: 48,
  sigB: 0.004,                       // bearing noise [rad] (~2.5 px)
  sigR0: 0.08, sigRk: 0.0012,        // stereo depth noise: sigR = sigR0 + sigRk * r^2  [m]
  imgW: 960, imgH: 300, targetH: 1.8, targetR: 0.4,
};
export const sigRange = (cam, r) => cam.sigR0 + cam.sigRk * r * r;

export class MotWorld {
  constructor(scenario = 'boulders', seed = 7, overrides = {}) {
    this.cfg = { ...MOT_SCENARIOS[scenario] || MOT_SCENARIOS.boulders, ...overrides };
    this.id = scenario; this.seed = seed;
    this.cam = { ...CAMERA };
    this.bounds = { xmin: -38, xmax: 38, ymin: 0, ymax: 46 };
    this.rng = new RNG(seed * 9973 + 17);   // world + truth motion
    this.srng = new RNG(seed * 7717 + 5);   // sensor noise (separate stream: same truth regardless of detector settings)
    this.t = 0; this.dt = 0.1; this.frame = 0;
    this._placeBoulders(); this._spawnTargets();
  }

  _placeBoulders() {
    const r = this.rng, cam = this.cam; this.boulders = [];
    for (let k = 0; this.boulders.length < this.cfg.boulders && k < 2000; k++) {
      const range = r.uniform(9, 36), off = r.uniform(-0.42, 0.42) * cam.fov, br = r.uniform(1.4, 3.2);
      const b = { x: cam.x + range * Math.cos(cam.th + off), y: cam.y + range * Math.sin(cam.th + off), r: br };
      if (this.boulders.some((o) => Math.hypot(o.x - b.x, o.y - b.y) < o.r + b.r + 3.5)) continue;
      this.boulders.push(b);
    }
  }

  _spawnTargets() {
    const r = this.rng; this.targets = [];
    for (let i = 0; i < this.cfg.targets; i++) {
      let p;
      for (let k = 0; k < 500; k++) {
        { const rr = r.uniform(12, this.cam.range - 8), oo = r.uniform(-0.38, 0.38) * this.cam.fov; p = { x: this.cam.x + rr * Math.cos(this.cam.th + oo), y: this.cam.y + rr * Math.sin(this.cam.th + oo) }; }
        if (!this.boulders.some((b) => Math.hypot(b.x - p.x, b.y - p.y) < b.r + 1.2)) break;
      }
      const h = r.uniform(-Math.PI, Math.PI), v = r.uniform(0.6, 1.3);
      this.targets.push({ id: i + 1, x: p.x, y: p.y, h, v, trail: [], vis: visibility(this.cam, this.boulders, p.x, p.y, this.cam.targetR) });
    }
  }

  _goal() {
    const r = this.rng, cam = this.cam;
    for (let k = 0; k < 200; k++) {
      const rr = r.uniform(11, cam.range - 6), oo = r.uniform(-0.4, 0.4) * cam.fov;
      const g = { x: cam.x + rr * Math.cos(cam.th + oo), y: cam.y + rr * Math.sin(cam.th + oo) };
      if (!this.boulders.some((b) => Math.hypot(b.x - g.x, b.y - g.y) < b.r + 1.5)) return g;
    }
    return { x: cam.x, y: cam.y + 25 };
  }

  /** Truth motion: goal-directed walking between worksites (EVA-like), small heading noise, steering around boulders. */
  step() {
    const dt = this.dt, r = this.rng;
    for (const t of this.targets) {
      if (!t.goal || Math.hypot(t.goal.x - t.x, t.goal.y - t.y) < 1.5) t.goal = this._goal();
      let sx = Math.cos(Math.atan2(t.goal.y - t.y, t.goal.x - t.x)), sy = Math.sin(Math.atan2(t.goal.y - t.y, t.goal.x - t.x));
      for (const b of this.boulders) {                     // walk around rocks, not through them
        const dx = t.x - b.x, dy = t.y - b.y, d = Math.hypot(dx, dy) - b.r;
        if (d < 2.0) { const w = 1.6 * (2.0 - Math.max(d, 0.05)) / 2.0; sx += (dx / (d + b.r)) * w; sy += (dy / (d + b.r)) * w; }
      }
      t.h += clamp(wrapAngle(Math.atan2(sy, sx) - t.h), -1, 1) * 1.2 * dt + 0.08 * Math.sqrt(dt) * r.randn();
      t.x += t.v * Math.cos(t.h) * dt; t.y += t.v * Math.sin(t.h) * dt;
      for (const b of this.boulders) { // never inside a rock
        const dx = t.x - b.x, dy = t.y - b.y, d = Math.hypot(dx, dy);
        if (d < b.r + 0.4) { t.x = b.x + (dx / d) * (b.r + 0.4); t.y = b.y + (dy / d) * (b.r + 0.4); }
      }
      t.vis = visibility(this.cam, this.boulders, t.x, t.y, this.cam.targetR);
      if (this.frame % 3 === 0) { t.trail.push([t.x, t.y]); if (t.trail.length > 120) t.trail.shift(); }
    }
    this.t += dt; this.frame++;
  }

  /**
   * One camera frame of detections. The tracker gets ONLY `dets` (no identities). `gt[i]` is the true target id
   * of dets[i] (0 = clutter), kept separate for scoring.
   */
  sense() {
    const cam = this.cam, s = this.srng, dets = [], gt = [];
    for (const t of this.targets) {
      const v = t.vis || visibility(cam, this.boulders, t.x, t.y, cam.targetR);
      if (s.next() >= detectionProb(v.inFov, v.visFrac, this.cfg.pd)) continue;
      const sr = sigRange(cam, v.range);
      const sb = cam.sigB * (v.visFrac < 0.9 ? 2 : 1);  // partially hidden: centroid jitters more
      dets.push(this._det(v.range + sr * s.randn(), v.visCenter + sb * s.randn()));
      gt.push(t.id);
    }
    const nFalse = poisson(this.cfg.clutter, s);
    for (let k = 0; k < nFalse; k++) {
      dets.push(this._det(s.uniform(4, cam.range), cam.th + s.uniform(-cam.fov / 2, cam.fov / 2)));
      gt.push(0);
    }
    return { dets, gt };
  }

  _det(range, bearing) {
    const cam = this.cam, f = cam.imgW / 2 / Math.tan(cam.fov / 2), phi = wrapAngle(bearing - cam.th);
    return { range, bearing, u: cam.imgW / 2 - f * Math.tan(phi), hPx: (f * cam.targetH) / Math.max(range, 1) };
  }
}

function poisson(lam, rng) { let k = 0, p = Math.exp(-lam), s = p; const u = rng.next(); while (u > s && k < 50) { k++; p *= lam / k; s += p; } return k; }
