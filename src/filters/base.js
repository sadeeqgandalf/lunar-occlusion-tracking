// Estimator interface, generic over a vehicle Model (see src/models/*).
//
// Model contract:
//   nx, nw, angleStates, f(x,u,w,dt,sensors) -> x', W(u,dt,sensors) -> Mat(nw x nw), P0() -> Mat,
//   jacobians?(x,u,dt,sensors) -> {G, V}   (omitted => numerical)
//   meas[kind] = {dim, angleDims, z(m), h(x,m), H?(x,m), R(m,rScale)}
//
// Measurement objects come from the platform's sensors, e.g.
//   {kind:'landmark', lx, ly, range, bearing, sigR, sigB} | {kind:'fix', x, y, sigma} | ...
import { Mat, wrapAngle } from '../core/linalg.js';
import { numJac } from '../core/numjac.js';

export const CHI2_2DOF_99 = 9.21;
/** 99% chi-square quantiles by measurement dimension (exact, not scaled heuristics). */
export const CHI2_99 = { 1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086, 6: 16.812 };
/** The UI gate slider is expressed as the 2-dof threshold; map it to other dimensions at the same tail ratio. */
export const gateFor = (slider, dim) => (CHI2_99[dim] ?? CHI2_99[2]) * (slider / CHI2_99[2]);
export const RECOVER_STREAK = 10;
export const RECOVER_INFLATE = 1.5;

export const wrapState = (model, x) => { for (const i of model.angleStates) x[i] = wrapAngle(x[i]); return x; };

/** a - b with angle components wrapped to (-pi, pi]. */
export function resid(angleIdx, a, b) {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
  for (const i of angleIdx) r[i] = wrapAngle(r[i]);
  return r;
}

export function jacobians(model, x, u, dt, s) {
  if (model.jacobians) return model.jacobians(x, u, dt, s);
  const zero = new Array(model.nw).fill(0);
  const G = numJac((xx) => model.f(xx, u, zero, dt, s), x, model.angleStates);
  const V = numJac((w) => model.f(x, u, w, dt, s), zero, model.angleStates);
  return { G, V };
}

export const measJacobian = (spec, x, m) => (spec.H ? spec.H(x, m) : numJac((xx) => spec.h(xx, m), x, spec.angleDims));

export function covEllipse(P) {
  const a = P.get(0, 0), b = P.get(0, 1), c = P.get(1, 1);
  const mid = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  return { l1: Math.max(mid + d, 0), l2: Math.max(mid - d, 0), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

export class Filter {
  static id = 'filter';
  static label = 'Filter';
  static color = '#ffffff';
  static blurb = '';
  /** Tunable knobs surfaced as sliders: {key:{label,min,max,step,value}} */
  static paramSpec = {};

  constructor(model, sensors, params = {}, rng = null) {
    this.model = model;
    this.sensors = sensors;
    this.rng = rng;
    this.params = {};
    for (const [k, sp] of Object.entries(this.constructor.paramSpec)) this.params[k] = params[k] ?? sp.value;
  }
  setParam(k, v) { this.params[k] = v; }
  /**
   * Gate lock-out recovery (Bar-Shalom, Estimation with Applications to Tracking and Navigation, 5.4): if every
   * measurement keeps failing the chi-square gate the *filter* is probably wrong, not the sensor. After a streak
   * of rejections, report true so the caller inflates its covariance and re-opens the gate. Tracked per measurement
   * kind: an always-healthy sensor (e.g. airspeed) must not mask a locked-out one (e.g. radar beacons).
   */
  noteOutcome(kind, accepted) {
    this.rejStreak ||= {};
    if (accepted) { this.rejStreak[kind] = 0; return false; }
    this.rejStreak[kind] = (this.rejStreak[kind] || 0) + 1;
    return this.rejStreak[kind] >= RECOVER_STREAK;
  }
  init(x0, P0) {}
  predict(u, dt) {}
  /** @returns {{accepted:boolean, nis?:number}} */
  update(m) { return { accepted: false }; }
  getState() { return new Array(this.model.nx).fill(0); }
  getCov() { return Mat.eye(this.model.nx); }
  getParticles() { return null; }
}
