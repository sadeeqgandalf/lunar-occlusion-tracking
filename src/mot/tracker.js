// Multi-object tracker: per-track constant-velocity EKF, chi-square gating, global-nearest-neighbour association
// (Hungarian), and a Bayesian track-existence probability r (Bernoulli-filter update, Ristic et al. 2013;
// same spirit as IPDA, Musicki et al. 1994):
//   detected with measurement z:  r <- r[P_D g(z) + (1-P_D) k] / [r P_D g(z) + (1 - r P_D) k]   (g: EKF likelihood, k: clutter density)
//   missed:                       r <- r (1 - P_D) / (1 - r P_D)
//
// Third variant, negative information (negInfo): a missed detection is evidence about WHERE the object is.
// Bayes: p(x | missed) ∝ p(x) (1 - P_D(x)). We approximate it by moment matching over fixed samples of the
// predicted Gaussian, weighting each by (1 - P_D(x_i)): hypotheses in plain view are down-weighted, hypotheses in a
// shadow are kept, and mean, velocity and covariance all move into the shadow consistently.
// P_D,exp (used by both aware variants) is the sample average of P_D(x_i), the same integral.
//
// THE ONLY DIFFERENCE between the two variants is the detection probability each track expects:
//   naive            P_D(track) = P_D                         (assumes every object is always detectable)
//   occlusion-aware  P_D(track) = P_D x visibility(predicted position | boulder map, camera FOV)
// A missed detection updates existence by Bayes:  r <- r (1 - P_D) / (1 - r P_D).
// So a miss on a track predicted to be hidden carries (almost) no evidence that the object is gone.
import { Mat, wrapAngle } from '../core/linalg.js';
import { hungarian } from './hungarian.js';
import { visibility, detectionProb, shadowFraction } from './occlusion.js';
import { sigRange } from './world.js';
import { RNG } from '../core/rng.js';

const NS = 64; // fixed antithetic samples of a 4-D standard normal (deterministic runs)
const Z = (() => { const r = new RNG(424242), z = []; for (let i = 0; i < NS / 2; i++) { const v = [r.randn(), r.randn(), r.randn(), r.randn()]; z.push(v, v.map((q) => -q)); } return z; })();

export const TRACKER_DEFAULTS = {
  sigA: 0.9,          // white-noise acceleration [m/s^2] while measured: covers ~p99 of walker manoeuvres
  sigACoast: 0.9,     // ... while coasting unseen: unobserved turns (e.g. walking around a rock) need more room
  gate: 11.83,        // chi-square 2-dof, 99.73% (3-sigma equivalent)
  rBirth: 0.25,       // existence prob of a new track
  pSurvive: 0.998,    // per-frame survival
  rConfirm: 0.75,     // confirmed when r >= this (and >= 3 hits)
  rDelete: 0.08,      // deleted when r < this
  maxSigma: 7,        // ... or when 1-sigma position uncertainty exceeds this [m]
  pd: 0.95,
  clutter: 0.8,       // expected false detections per frame (spread uniformly over range x bearing)
};

export class Tracker {
  constructor(cam, boulders, { occlusionAware = true, negInfo = false, name, sun = null, ...params } = {}) {
    this.cam = cam; this.boulders = boulders; this.sun = sun; this.aware = occlusionAware; this.negInfo = occlusionAware && negInfo;
    this.name = name || (this.negInfo ? 'Aware + neg. info' : occlusionAware ? 'Occlusion-aware' : 'Naive');
    this.p = { ...TRACKER_DEFAULTS, ...params };
    this.tracks = []; this.nextId = 1; this.dt = 0.1;
  }

  /** Samples x_i = mean + L z_i of the track's predicted Gaussian, with P_D at each. */
  _samples(t) {
    const L = t.P.chol(), pts = [];
    for (const z of Z) {
      const x = t.x.slice();
      for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) x[r] += L.get(r, c) * z[c];
      const v = visibility(this.cam, this.boulders, x[0], x[1], this.cam.targetR);
      pts.push({ x, pd: detectionProb(v.inFov, v.visFrac, this.p.pd, this.sun ? shadowFraction(this.boulders, x[0], x[1], this.sun) : 0) });
    }
    return pts;
  }

  /** Expected detection probability: P_D x visibility averaged over the track's predicted uncertainty. */
  expectedPd(t) {
    if (!this.aware) return this.p.pd;
    t._pts = this._samples(t);
    return t._pts.reduce((a, q) => a + q.pd, 0) / t._pts.length;
  }

  _predict(t) {
    const dt = this.dt, q = (t.lastSeen > 0 ? this.p.sigACoast : this.p.sigA) ** 2, x = t.x;
    t.x = [x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]];
    const F = Mat.from([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]);
    const a = (dt ** 4) / 4, b = (dt ** 3) / 2, c = dt * dt;
    const Q = Mat.from([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]]).scale(q);
    t.P = F.mul(t.P).mul(F.T()).add(Q).sym();
    t.r *= this.p.pSurvive;
  }

  _meas(t) { // predicted range/bearing, Jacobian, innovation covariance pieces
    const cam = this.cam, dx = t.x[0] - cam.x, dy = t.x[1] - cam.y, q = dx * dx + dy * dy, rr = Math.sqrt(q);
    const H = Mat.from([[dx / rr, dy / rr, 0, 0], [-dy / q, dx / q, 0, 0]]);
    const R = Mat.diag([sigRange(cam, rr) ** 2, cam.sigB ** 2 * 2]); // bearing variance inflated for partial-occlusion jitter
    const S = H.mul(t.P).mul(H.T()).add(R);
    return { zh: [rr, Math.atan2(dy, dx)], H, R, S, Si: S.inv() };
  }

  /** Negative information: moment-match p(x)(1 - P_D(x)) using the samples drawn for this frame. */
  _missUpdate(t) {
    const pts = t._pts; if (!pts) return;
    let W = 0; const m = [0, 0, 0, 0];
    for (const q of pts) { const w = 1 - q.pd; W += w; for (let k = 0; k < 4; k++) m[k] += w * q.x[k]; }
    if (W < 1e-6 || W > pts.length * (1 - 1e-6)) return; // all hypotheses equally (in)visible: nothing learned
    for (let k = 0; k < 4; k++) m[k] /= W;
    const P = Mat.zeros(4, 4);
    for (const q of pts) { const w = (1 - q.pd) / W; for (let a2 = 0; a2 < 4; a2++) for (let b2 = 0; b2 < 4; b2++) P.d[a2 * 4 + b2] += w * (q.x[a2] - m[a2]) * (q.x[b2] - m[b2]); }
    // keep a floor so a sample-collapsed covariance cannot become overconfident
    for (let k = 0; k < 4; k++) P.d[k * 5] += k < 2 ? 0.02 : 0.01;
    t.x = m; t.P = P.sym();
  }

  _birth(d) {
    const cam = this.cam, c = Math.cos(d.bearing), s = Math.sin(d.bearing), r = d.range;
    const J = Mat.from([[c, -r * s], [s, r * c]]), Rm = Mat.diag([sigRange(cam, r) ** 2, cam.sigB ** 2 * 2]);
    const Pp = J.mul(Rm).mul(J.T());
    const P = Mat.diag([0, 0, 1.0, 1.0]);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) P.set(i, j, Pp.get(i, j) + (i === j ? 0.05 : 0));
    this.tracks.push({ id: this.nextId++, x: [cam.x + r * c, cam.y + r * s, 0, 0], P, r: this.p.rBirth, hits: 1, age: 0, lastSeen: 0, confirmed: false, hidden: false, pdExp: 1 });
  }

  /** Process one frame of detections ({range, bearing}). */
  step(dets) {
    const p = this.p;
    for (const t of this.tracks) { this._predict(t); t.age++; t.pdExp = this.expectedPd(t); t.hidden = t.pdExp < 0.3; }
    const kappa = p.clutter / (this.cam.fov * (this.cam.range - 4)); // clutter density per (m x rad)

    // Likelihood-ratio assignment with an explicit "missed" option per track (track-score form used by JPDA/MHT):
    //   take detection j:  cost = -log( P_D,i g_ij / kappa )      miss:  cost = -log( 1 - P_D,i P_G )
    // A track that expects to be hidden (P_D,i ~ 0) therefore prefers "missed" over grabbing a nearby detection,
    // which is what stops coasting tracks being hijacked by clutter or by another object.
    const T = this.tracks.length, D = dets.length, FORBID = 1e9, PG = 0.9973;
    const pre = this.tracks.map((t) => this._meas(t)), g = this.tracks.map(() => new Array(D).fill(0));
    const cost = this.tracks.map((t, i) => {
      const m = pre[i], pdi = Math.max(t.pdExp, 0.02), row = new Array(D + T).fill(FORBID);
      const det = m.S.get(0, 0) * m.S.get(1, 1) - m.S.get(0, 1) * m.S.get(1, 0);
      dets.forEach((d, j) => {
        const y = [d.range - m.zh[0], wrapAngle(d.bearing - m.zh[1])];
        const nis = y[0] * (m.Si.get(0, 0) * y[0] + m.Si.get(0, 1) * y[1]) + y[1] * (m.Si.get(1, 0) * y[0] + m.Si.get(1, 1) * y[1]);
        if (nis > p.gate) return;
        g[i][j] = Math.exp(-0.5 * nis) / (2 * Math.PI * Math.sqrt(det));
        row[j] = -Math.log((pdi * g[i][j]) / kappa);
      });
      row[D + i] = -Math.log(1 - pdi * PG);
      return row;
    });
    const assign = hungarian(cost, FORBID);
    const used = new Set();

    this.tracks.forEach((t, i) => {
      const j = assign[i];
      if (j >= 0 && j < D) { // EKF update
        used.add(j);
        const d = dets[j], m = pre[i], y = Mat.from([[d.range - m.zh[0]], [wrapAngle(d.bearing - m.zh[1])]]);
        const K = t.P.mul(m.H.T()).mul(m.Si), dx = K.mul(y);
        for (let k = 0; k < 4; k++) t.x[k] += dx.d[k];
        const IKH = Mat.eye(4).sub(K.mul(m.H));
        t.P = IKH.mul(t.P).mul(IKH.T()).add(K.mul(m.R).mul(K.T())).sym();
        const pdH = Math.max(t.pdExp, 0.02);
        t.r = (t.r * (pdH * g[i][j] + (1 - pdH) * kappa)) / (t.r * pdH * g[i][j] + (1 - t.r * pdH) * kappa);
        t.hits++; t.lastSeen = 0;
        if (!t.confirmed && t.r >= p.rConfirm && t.hits >= 3) t.confirmed = true;
      } else { // missed: Bayes existence update with the EXPECTED detection probability
        const pd = t.pdExp;
        t.r = (t.r * (1 - pd)) / (1 - t.r * pd); t.lastSeen++;
        if (this.negInfo) this._missUpdate(t);
      }
    });

    const gated = (j) => cost.some((row) => row[j] < FORBID);
    this.tracks = this.tracks.filter((t) => t.r >= p.rDelete && Math.sqrt(Math.max(t.P.get(0, 0), t.P.get(1, 1))) <= p.maxSigma);
    // births only from detections that fall outside every surviving track's gate (prevents duplicate tracks)
    dets.forEach((d, j) => {
      if (used.has(j)) return;
      if (!gated(j)) this._birth(d);
    });
  }

  confirmed() { return this.tracks.filter((t) => t.confirmed); }
  /** What the system would output as boxes: confirmed and seen within the last 0.3 s. Same rule for every variant. */
  reported() { return this.tracks.filter((t) => t.confirmed && t.lastSeen <= 3); }
}
