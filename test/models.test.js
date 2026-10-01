import test from 'node:test';
import assert from 'node:assert/strict';
import { roverModel } from '../src/models/rover.js';
import { jetModel } from '../src/models/jet.js';
import { spacecraftModel } from '../src/models/spacecraft.js';
import { numJac } from '../src/core/numjac.js';
import { ROVER_SENSORS, JET_SENSORS, SPACECRAFT_SENSORS } from '../src/sim/sensors.js';

const maxAbs = (A, B) => A.d.reduce((m, v, i) => Math.max(m, Math.abs(v - B.d[i])), 0);

// A representative interior state, control and landmark per model (every state component non-zero).
const CASES = [
  { model: roverModel, S: ROVER_SENSORS, x: [3, -2, 0.7, 0.01, 0.12], u: { v: 1.1, w: 0.31 }, lm: { lx: 15, ly: 9 }, dt: 0.05 },
  { model: jetModel, S: JET_SENSORS, x: [3000, -2000, 0.7, 220, 0.001, 8, -5], u: { a: 0.5, w: 0.05 }, lm: { lx: 15000, ly: 9000 }, dt: 0.05 },
  { model: spacecraftModel, S: { ...SPACECRAFT_SENSORS }, x: [10, -40, 0.05, -0.1, 0.002, -0.001], u: { ax: 0.01, ay: -0.02 }, lm: {}, dt: 0.1 },
];

test('every model: state/noise dimensions are self-consistent', () => {
  for (const { model, S, x, u, dt } of CASES) {
    assert.equal(x.length, model.nx, model.id);
    assert.equal(model.f(x, u, new Array(model.nw).fill(0), dt, S).length, model.nx);
    assert.equal(model.P0().r, model.nx);
    assert.equal(model.W(u, dt, S).r, model.nw);
    assert.equal(model.stateNames.length, model.nx);
  }
});

test('every model with analytic G, V: matches finite differences (Thrun Ch.5/7), incl. straight-line limit', () => {
  for (const { model, S, x, u, dt } of CASES) {
    if (!model.jacobians) continue;
    const us = model.id === 'rover' ? [u, { v: 0.9, w: -0.02 }, { v: 1.3, w: 0.0005 }] : [u]; // last = near-zero turn rate
    for (const uu of us) {
      const z0 = new Array(model.nw).fill(0), { G, V } = model.jacobians(x, uu, dt, S);
      const Gn = numJac((xx) => model.f(xx, uu, z0, dt, S), x, model.angleStates);
      const Vn = numJac((w) => model.f(x, uu, w, dt, S), z0, model.angleStates);
      assert.ok(maxAbs(G, Gn) < 1e-5, `${model.id} G err ${maxAbs(G, Gn)} u=${JSON.stringify(uu)}`);
      assert.ok(maxAbs(V, Vn) < 1e-5, `${model.id} V err ${maxAbs(V, Vn)} u=${JSON.stringify(uu)}`);
    }
  }
});

test('every analytic measurement Jacobian in every model matches finite differences', () => {
  for (const { model, x, lm } of CASES) {
    for (const [kind, spec] of Object.entries(model.meas)) {
      if (!spec.H) continue;
      const err = maxAbs(spec.H(x, lm), numJac((xx) => spec.h(xx, lm), x, spec.angleDims));
      assert.ok(err < 1e-5, `${model.id}.${kind} H err ${err}`);
    }
  }
});

test('rover: wheel slip is observable in the model (ground speed = odometry x (1 - slip))', () => {
  const x = [0, 0, 0, 0, 0.25], y = roverModel.f(x, { v: 1, w: 0 }, [0, 0, 0, 0], 1);
  assert.ok(Math.abs(y[0] - 0.75) < 1e-9);
});

test('spacecraft: unforced CW conserves the drift-free relative orbit; accelerometer bias is subtracted', () => {
  const S = SPACECRAFT_SENSORS, n = S.meanMotion, z = [0, 0, 0, 0];
  let s = [100, 0, 0, -2 * n * 100, 0, 0], maxX = 0;
  for (let i = 0; i < 20000; i++) { s = spacecraftModel.f(s, { ax: 0, ay: 0 }, z, 0.05, S); maxX = Math.max(maxX, Math.abs(s[0])); }
  assert.ok(maxX < 100.5, `bounded CW orbit drifted: ${maxX}`);
  // a reading equal to the bias means zero true acceleration
  const y = spacecraftModel.f([0, 0, 0, 0, 0.004, -0.002], { ax: 0.004, ay: -0.002 }, z, 1, S);
  assert.ok(Math.abs(y[2]) < 1e-12 && Math.abs(y[3]) < 1e-12);
});

test('jet: no hand Jacobians (numerical fallback finite); wind advects position and airspeed integrates accel', () => {
  assert.equal(jetModel.jacobians, undefined);
  const x = [0, 0, 0, 220, 0, 10, -4], z = new Array(5).fill(0);
  const y = jetModel.f(x, { a: 0.5, w: 0 }, z, 1);
  assert.ok(Math.abs(y[0] - (220.25 + 10)) < 1e-6 && Math.abs(y[1] + 4) < 1e-6 && Math.abs(y[3] - 220.5) < 1e-9);
  const G = numJac((xx) => jetModel.f(xx, { a: 0.5, w: 0.05 }, z, 0.05), x, [2]);
  assert.ok(G.d.every(Number.isFinite));
});
