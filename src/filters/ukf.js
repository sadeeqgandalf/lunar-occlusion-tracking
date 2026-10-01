import { Mat, wrapAngle } from '../core/linalg.js';
import { Filter, CHI2_2DOF_99, gateFor, RECOVER_INFLATE, resid, wrapState } from './base.js';

const circMean = (vals, w) => {
  let s = 0, c = 0;
  for (let i = 0; i < vals.length; i++) { s += w[i] * Math.sin(vals[i]); c += w[i] * Math.cos(vals[i]); }
  return Math.atan2(s, c);
};
const blockDiag = (A, B) => {
  const o = Mat.zeros(A.r + B.r, A.c + B.c);
  for (let i = 0; i < A.r; i++) for (let j = 0; j < A.c; j++) o.set(i, j, A.get(i, j));
  for (let i = 0; i < B.r; i++) for (let j = 0; j < B.c; j++) o.set(A.r + i, A.c + j, B.get(i, j));
  return o;
};

/**
 * Unscented Kalman Filter, augmented-state form (Probabilistic Robotics Table 3.4).
 * Process noise and measurement noise ride inside the sigma points (no additive Q/R shortcut).
 * Angle states / measurement angles are averaged on the circle and differenced with wrapping.
 */
export class UKF extends Filter {
  static id = 'ukf';
  static label = 'Unscented Kalman';
  static color = '#b5f44a';
  static blurb = 'Augmented sigma-point filter (Thrun Table 3.4). No Jacobians; noise pushed through the true nonlinearity.';
  static paramSpec = {
    qScale: { label: 'Process noise ×', min: 0.2, max: 60, step: 0.1, value: 4 },
    rScale: { label: 'Measurement σ ×', min: 0.3, max: 8, step: 0.1, value: 1.5 },
    gate: { label: 'NIS gate (χ² 2-dof)', min: 2, max: 40, step: 0.1, value: CHI2_2DOF_99 },
  };
  static ALPHA = 1; static BETA = 2; static KAPPA = 1;

  init(x0, P0) { this.x = Array.from(x0); this.P = P0.clone(); }

  /** lambda = a^2 (L+k) - L ; Wm0 = lambda/(L+lambda) ; Wc0 = Wm0 + (1 - a^2 + b) ; Wi = 1/(2(L+lambda)) */
  _weights(L) {
    const { ALPHA: a, BETA: b, KAPPA: k } = UKF, lam = a * a * (L + k) - L;
    const Wm = new Array(2 * L + 1).fill(1 / (2 * (L + lam))), Wc = Wm.slice();
    Wm[0] = lam / (L + lam);
    Wc[0] = Wm[0] + (1 - a * a + b);
    return { Wm, Wc, gamma: Math.sqrt(L + lam) };
  }

  _sigma(mean, Pa, gamma) {
    const L = mean.length, C = Pa.chol(), pts = [mean.slice()];
    for (let i = 0; i < L; i++) {
      const p = mean.slice(), q = mean.slice();
      for (let r = 0; r < L; r++) { const v = gamma * C.get(r, i); p[r] += v; q[r] -= v; }
      pts.push(p, q);
    }
    return pts;
  }

  _stateMean(pts, Wm) {
    const M = this.model, mean = new Array(M.nx).fill(0);
    for (let i = 0; i < pts.length; i++) for (let j = 0; j < M.nx; j++) mean[j] += Wm[i] * pts[i][j];
    for (const j of M.angleStates) mean[j] = circMean(pts.map((p) => p[j]), Wm);
    return mean;
  }

  predict(u, dt) {
    const M = this.model, s = this.sensors, n = M.nx, L = n + M.nw;
    const { Wm, Wc, gamma } = this._weights(L);
    const Wn = M.W(u, dt, s).scale(this.params.qScale);
    const pts = this._sigma([...this.x, ...new Array(M.nw).fill(0)], blockDiag(this.P, Wn), gamma)
      .map((p) => wrapState(M, M.f(p.slice(0, n), u, p.slice(n), dt, s)));
    const mean = this._stateMean(pts, Wm);
    let P = Mat.zeros(n, n);
    for (let i = 0; i < pts.length; i++) {
      const d = Mat.from(resid(M.angleStates, pts[i], mean).map((v) => [v]));
      P = P.add(d.mul(d.T()).scale(Wc[i]));
    }
    this.x = mean; this.P = P.sym();
  }

  update(m) {
    const M = this.model, spec = M.meas[m.kind];
    if (!spec) return { accepted: false };
    const n = M.nx, k = spec.dim, L = n + k;
    const R = spec.R(m, this.params.rScale), { Wm, Wc, gamma } = this._weights(L);
    const pts = this._sigma([...this.x, ...new Array(k).fill(0)], blockDiag(this.P, R), gamma);
    const Z = pts.map((p) => {
      const z = spec.h(p.slice(0, n), m);
      for (let i = 0; i < k; i++) z[i] += p[n + i];
      for (const i of spec.angleDims) z[i] = wrapAngle(z[i]);
      return z;
    });
    const zm = new Array(k).fill(0);
    for (let i = 0; i < Z.length; i++) for (let j = 0; j < k; j++) zm[j] += Wm[i] * Z[i][j];
    for (const j of spec.angleDims) zm[j] = circMean(Z.map((q) => q[j]), Wm);

    let S = Mat.zeros(k, k), Pxz = Mat.zeros(n, k);
    for (let i = 0; i < pts.length; i++) {
      const dz = Mat.from(resid(spec.angleDims, Z[i], zm).map((v) => [v]));
      const dx = Mat.from(resid(M.angleStates, pts[i].slice(0, n), this.x).map((v) => [v]));
      S = S.add(dz.mul(dz.T()).scale(Wc[i]));
      Pxz = Pxz.add(dx.mul(dz.T()).scale(Wc[i]));
    }
    const y = Mat.from(resid(spec.angleDims, spec.z(m), zm).map((v) => [v]));
    const Si = S.inv(), nis = y.T().mul(Si).mul(y).d[0];
    const gate = gateFor(this.params.gate, k);
    if (!(nis <= gate)) {
      if (this.noteOutcome(m.kind, false)) this.P = this.P.scale(RECOVER_INFLATE);
      return { accepted: false, nis };
    }
    this.noteOutcome(m.kind, true);
    const K = Pxz.mul(Si), dx = K.mul(y);
    for (let i = 0; i < n; i++) this.x[i] += dx.d[i];
    wrapState(M, this.x);
    this.P = this.P.sub(K.mul(S).mul(K.T())).sym();
    return { accepted: true, nis };
  }

  getState() { return this.x.slice(); }
  getCov() { return this.P; }
}
