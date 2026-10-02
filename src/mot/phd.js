// Particle PHD filter: a "where could anyone be?" density over the ground (Mahler 2003; SMC-PHD, Vo, Singh & Doucet 2005).
// Same algorithm as pytorch3d/lunar3d/phd.py. N weighted particles carry the intensity; total weight = expected number
// of people. Each frame:
//   predict  constant-velocity motion + noise, weights x p_survive
//   birth    a little mass around each detection, weighted by how visible that spot is (Ristic, Clark, Vo & Vo 2012):
//            uniform birth mass would pile up behind rocks, where nothing can remove it, as phantom people
//   update   w <- w [ (1 - P_D(x)) + sum_z P_D(x) g(z|x) / (kappa + sum_j P_D(x_j) g(z|x_j) w_j) ]
// P_D(x) is the visibility-based detection probability of that spot. Where the camera can see, a miss removes mass;
// behind a rock (P_D ~ 0) mass survives. That is negative information, visible as glow pooling into blind zones.
// For speed in the browser P_D is precomputed once on a 0.5 m ground grid (the camera and rocks never move).
import { RNG } from '../core/rng.js';
import { wrapAngle } from '../core/linalg.js';
import { visibility, detectionProb, shadowFraction } from './occlusion.js';
import { sigRange } from './world.js';

const GX0 = -50, GX1 = 50, GY0 = -6, GY1 = 60, GS = 0.5;

export class ParticlePHD {
  constructor(world, { n = 2500, seed = 1 } = {}) {
    this.W = world; this.n = n; this.rng = new RNG(seed * 31 + 7);
    this.X = new Float64Array(0); this.w = new Float64Array(0); this.pending = null;
    this.ps = 0.995; this.birthMass = 0.03; this.sigA = 0.9; this.dt = world.dt;
    // detection-probability grid for the (fixed) camera
    const cam = world.cam, cfg = world.cfg, sun = cfg.shadows ? world.sun : null;
    this.gw = Math.round((GX1 - GX0) / GS); this.gh = Math.round((GY1 - GY0) / GS);
    this.pd = new Float32Array(this.gw * this.gh);
    for (let j = 0; j < this.gh; j++) for (let i = 0; i < this.gw; i++) {
      const x = GX0 + (i + 0.5) * GS, y = GY0 + (j + 0.5) * GS, v = visibility(cam, world.boulders, x, y, cam.targetR);
      if (!v.inFov) continue;
      this.pd[j * this.gw + i] = detectionProb(true, v.visFrac, cfg.pd, sun ? shadowFraction(world.boulders, x, y, sun) : 0);
    }
  }

  pdAt(x, y) {
    const i = Math.floor((x - GX0) / GS), j = Math.floor((y - GY0) / GS);
    return i < 0 || j < 0 || i >= this.gw || j >= this.gh ? 0 : this.pd[j * this.gw + i];
  }

  _births(dets) {
    const cam = this.W.cam, m = 40, X = [], w = [];
    for (const z of dets) for (let k = 0; k < m; k++) {
      const r = z.range + sigRange(cam, z.range) * this.rng.randn(), b = z.bearing + cam.sigB * 2 * this.rng.randn();
      const x = cam.x + r * Math.cos(b), y = cam.y + r * Math.sin(b);
      X.push(x, y, 0.6 * this.rng.randn(), 0.6 * this.rng.randn());
      w.push((this.birthMass / m) * this.pdAt(x, y) / this.W.cfg.pd);     // unseeable spot => it was a false alarm
    }
    return [X, w];
  }

  step(dets) {
    const dt = this.dt, cam = this.W.cam, r = this.rng;
    // predict
    let X = Array.from(this.X), w = Array.from(this.w);
    for (let i = 0; i < w.length; i++) {
      const o = 4 * i, ax = this.sigA * r.randn(), ay = this.sigA * r.randn();
      X[o] += X[o + 2] * dt + 0.5 * dt * dt * ax; X[o + 1] += X[o + 3] * dt + 0.5 * dt * dt * ay;
      X[o + 2] += dt * ax; X[o + 3] += dt * ay;
      const sp = Math.hypot(X[o + 2], X[o + 3]); if (sp > 1.6) { X[o + 2] *= 1.6 / sp; X[o + 3] *= 1.6 / sp; }   // walking speeds
      w[i] *= this.ps;
    }
    if (this.pending) { X = X.concat(this.pending[0]); w = w.concat(this.pending[1]); }
    this.pending = this._births(dets);                        // used from the next frame (never updated twice)
    // update
    const N = w.length, pd = new Float64Array(N), rr = new Float64Array(N), br = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const x = X[4 * i], y = X[4 * i + 1]; pd[i] = this.pdAt(x, y);
      rr[i] = Math.hypot(x - cam.x, y - cam.y); br[i] = Math.atan2(y - cam.y, x - cam.x);
    }
    const kappa = this.W.cfg.clutter / (cam.fov * (cam.range - 4)), sb = cam.sigB * Math.SQRT2 + 0.002;
    const gain = new Float64Array(N); for (let i = 0; i < N; i++) gain[i] = 1 - pd[i];
    const num = new Float64Array(N);
    for (const z of dets) {
      let den = kappa;
      for (let i = 0; i < N; i++) {
        if (pd[i] === 0) { num[i] = 0; continue; }
        const sr = sigRange(cam, rr[i]), a = (z.range - rr[i]) / sr, b = wrapAngle(z.bearing - br[i]) / sb;
        num[i] = a * a + b * b > 40 ? 0 : pd[i] * Math.exp(-0.5 * (a * a + b * b)) / (2 * Math.PI * sr * sb);
        den += num[i] * w[i];
      }
      for (let i = 0; i < N; i++) gain[i] += num[i] / den;
    }
    let mass = 0; for (let i = 0; i < N; i++) { w[i] *= gain[i]; mass += w[i]; }
    // systematic resampling back to n particles, keeping the total mass
    this.X = new Float64Array(4 * this.n); this.w = new Float64Array(this.n);
    if (!(mass > 0)) { this.X = new Float64Array(0); this.w = new Float64Array(0); return; }
    let c = w[0] / mass, k = 0; const u0 = r.next() / this.n;
    for (let m = 0; m < this.n; m++) {
      const u = u0 + m / this.n;
      while (u > c && k < N - 1) { k++; c += w[k] / mass; }
      for (let q = 0; q < 4; q++) this.X[4 * m + q] = X[4 * k + q] + (q < 2 ? 0.05 : 0.05) * r.randn();
      this.w[m] = mass / this.n;
    }
  }

  get expectedCount() { let s = 0; for (const v of this.w) s += v; return s; }

  /** Density (people per m^2) on a display grid, lightly blurred. Rows from max y (top) to min y (bottom). */
  density(x0 = -34, x1 = 34, y0 = -2, y1 = 46, cell = 0.75) {
    const nx = Math.round((x1 - x0) / cell), ny = Math.round((y1 - y0) / cell), D = new Float32Array(nx * ny);
    for (let m = 0; m < this.w.length; m++) {
      const i = Math.floor((this.X[4 * m] - x0) / cell), j = ny - 1 - Math.floor((this.X[4 * m + 1] - y0) / cell);
      if (i >= 0 && j >= 0 && i < nx && j < ny) D[j * nx + i] += this.w[m] / (cell * cell);
    }
    for (let pass = 0; pass < 3; pass++) {                    // 3x3 box blur, three times (a soft glow)
      const T = D.slice();
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        let s = 0, c = 0;
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const a = i + di, b = j + dj; if (a >= 0 && b >= 0 && a < nx && b < ny) { s += T[b * nx + a]; c++; } }
        D[j * nx + i] = s / c;
      }
    }
    return { D, nx, ny, x0, x1, y0, y1 };
  }
}
