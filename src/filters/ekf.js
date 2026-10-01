import { Mat } from '../core/linalg.js';
import { Filter, CHI2_2DOF_99, RECOVER_INFLATE, gateFor, jacobians, measJacobian, resid, wrapState } from './base.js';

/**
 * Extended Kalman Filter, Probabilistic Robotics Table 3.3.
 *   predict:  mu_bar = g(u, mu);  Sigma_bar = G Sigma G^T + V M V^T
 *   update:   K = Sigma_bar H^T (H Sigma_bar H^T + Q)^-1;  mu = mu_bar + K (z - h(mu_bar));  Sigma = (I-KH) Sigma_bar
 * The covariance update uses the algebraically identical Joseph form for numerical positive-definiteness.
 * Mahalanobis innovation gating (chi-square) is the ML data-association test of Ch. 7.5.
 */
export class EKF extends Filter {
  static id = 'ekf';
  static label = 'Extended Kalman';
  static color = '#4cc9f0';
  static blurb = 'Thrun Table 3.3: linearise g and h about the mean; Joseph-form update with χ² innovation gating.';
  static paramSpec = {
    qScale: { label: 'Process noise ×', min: 0.2, max: 60, step: 0.1, value: 4 },
    rScale: { label: 'Measurement σ ×', min: 0.3, max: 8, step: 0.1, value: 1.5 },
    gate: { label: 'NIS gate (χ² 2-dof)', min: 2, max: 40, step: 0.1, value: CHI2_2DOF_99 },
  };

  init(x0, P0) { this.x = Array.from(x0); this.P = P0.clone(); }

  predict(u, dt) {
    const M = this.model, s = this.sensors;
    const { G, V } = jacobians(M, this.x, u, dt, s);
    const Wm = M.W(u, dt, s).scale(this.params.qScale);
    this.P = G.mul(this.P).mul(G.T()).add(V.mul(Wm).mul(V.T())).sym();
    this.x = wrapState(M, M.f(this.x, u, new Array(M.nw).fill(0), dt, s));
  }

  update(m) {
    const M = this.model, spec = M.meas[m.kind];
    if (!spec) return { accepted: false };
    const x = this.x, n = M.nx;
    const R = spec.R(m, this.params.rScale);
    const H = measJacobian(spec, x, m);
    const y = Mat.from(resid(spec.angleDims, spec.z(m), spec.h(x, m)).map((v) => [v]));
    const S = H.mul(this.P).mul(H.T()).add(R), Si = S.inv();
    const nis = y.T().mul(Si).mul(y).d[0];
    if (!(nis <= gateFor(this.params.gate, spec.dim))) {
      if (this.noteOutcome(m.kind, false)) this.P = this.P.scale(RECOVER_INFLATE);
      return { accepted: false, nis };
    }
    this.noteOutcome(m.kind, true);
    const K = this.P.mul(H.T()).mul(Si), dx = K.mul(y);
    for (let i = 0; i < n; i++) x[i] += dx.d[i];
    wrapState(M, x);
    const IKH = Mat.eye(n).sub(K.mul(H));
    this.P = IKH.mul(this.P).mul(IKH.T()).add(K.mul(R).mul(K.T())).sym();
    return { accepted: true, nis };
  }

  getState() { return this.x.slice(); }
  getCov() { return this.P; }
}
