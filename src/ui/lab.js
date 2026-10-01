// Filter Lab: compile user-written estimators into registered Filter classes.
import { Mat, wrapAngle } from '../core/linalg.js';
import { numJac } from '../core/numjac.js';
import { Filter, resid, jacobians } from '../filters/base.js';
import { registerFilter } from '../filters/index.js';

export const TEMPLATE = `// Write any estimator. It runs live against EKF / UKF / PF on the same sensor stream.
// State layout is model.stateNames. u = control inputs, meas = one measurement:
//   {kind:'landmark'|'lidar', range, bearing, lx, ly, sigR, sigB} | {kind:'fix', x, y, sigma} | {kind:'airspeed', v}
const { Mat, model, sensors, jacobians, resid, wrapAngle } = ctx;
const nx = model.nx, zeros = new Array(model.nw).fill(0);
let x, P;
const gain = 0.25;                       // <- tune me

return {
  name: 'Complementary (starter)',
  init(x0, P0) { x = Array.from(x0); P = P0.clone(); },

  predict(u, dt) {
    // propagate the mean with the vehicle model, grow covariance with its Jacobians: P = G P G' + V W V'
    const { G, V } = jacobians(x, u, dt);
    P = G.mul(P).mul(G.T()).add(V.mul(model.W(u, dt, sensors)).mul(V.T())).sym();
    x = model.f(x, u, zeros, dt, sensors);
  },

  update(meas) {
    // turn the measurement into an implied position, then nudge toward it (no gating, no cross-correlation: that's
    // exactly what a Kalman filter does properly. Replace this with your own idea!)
    let px, py;
    if (meas.kind === 'fix') { px = meas.x; py = meas.y; }
    else if (meas.kind === 'lidar') { px = -meas.range * Math.cos(meas.bearing); py = -meas.range * Math.sin(meas.bearing); }
    else if (meas.kind === 'landmark') {
      const a = x[model.headingIdx] + meas.bearing;
      px = meas.lx - meas.range * Math.cos(a); py = meas.ly - meas.range * Math.sin(a);
    } else return { accepted: false };
    const [i, j] = model.posIdx;
    x[i] += gain * (px - x[i]); x[j] += gain * (py - x[j]);
    P.set(i, i, P.get(i, i) * (1 - gain)); P.set(j, j, P.get(j, j) * (1 - gain));
    return { accepted: true };
  },

  getState() { return x; },
  getCov() { return P; },
};`;

let counter = 0;
const COLORS = ['#ffd166', '#06d6a0', '#ef476f', '#118ab2', '#c77dff'];

/** Compile + smoke-test. Returns {Cls} or throws with a readable message. */
export function compileFilter(source, model, sensors) {
  const id = `custom${++counter}`;
  const factory = new Function('ctx', `"use strict";\n${source}`);
  class Custom extends Filter {
    static id = id;
    static label = 'Custom';
    static color = COLORS[(counter - 1) % COLORS.length];
    static blurb = 'Your filter, compiled in the Lab.';
    static paramSpec = {};
    constructor(m, s, params, rng) {
      super(m, s, params, rng);
      this.impl = factory({
        Mat, wrapAngle, numJac, resid, model: m, sensors: s,
        jacobians: (x, u, dt) => jacobians(m, x, u, dt, s),
        rng: { randn: () => rng.randn(), next: () => rng.next() },
      });
      if (!this.impl || typeof this.impl.predict !== 'function') throw new Error('Lab code must `return {name, init, predict, update, getState, getCov}`.');
    }
    init(x0, P0) { this.impl.init(x0, P0); }
    predict(u, dt) { this.impl.predict(u, dt); }
    update(m) { return this.impl.update(m) || { accepted: false }; }
    getState() { return Array.from(this.impl.getState()); }
    getCov() { return Mat.coerce(this.impl.getCov()); }
  }
  // smoke test on the live model so errors show up here, not mid-mission
  const probe = new Custom(model, sensors, {}, { randn: () => 0, next: () => 0.5 });
  const x0 = model.P0().d.length ? new Array(model.nx).fill(0) : [];
  x0[0] = 1; x0[1] = 1;
  probe.init(x0, model.P0());
  probe.predict(model.id === 'spacecraft' ? { ax: 0, ay: 0 } : model.id === 'jet' ? { a: 0, w: 0 } : { v: 1, w: 0 }, 0.05);
  const s = probe.getState(), P = probe.getCov();
  if (s.length !== model.nx || !s.every(Number.isFinite)) throw new Error(`getState() must return ${model.nx} finite numbers, got ${JSON.stringify(s)}`);
  if (P.r !== model.nx || P.c !== model.nx) throw new Error(`getCov() must be ${model.nx}×${model.nx}`);
  Custom.label = probe.impl.name || Custom.id;
  registerFilter(Custom);
  return Custom;
}
