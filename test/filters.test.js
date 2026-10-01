import test from 'node:test';
import assert from 'node:assert/strict';
import { Mission } from '../src/sim/mission.js';
import { Mat } from '../src/core/linalg.js';
import { EKF } from '../src/filters/ekf.js';
import { UKF } from '../src/filters/ukf.js';
import { RNG } from '../src/core/rng.js';
import { spacecraftModel } from '../src/models/spacecraft.js';
import { SPACECRAFT_SENSORS } from '../src/sim/sensors.js';

const run = (opts, secs, setup) => {
  const m = new Mission({ oracle: true, ...opts });
  setup?.(m);
  m.issue({ type: 'plan' });
  for (let i = 0; i < secs * 20 && m.status === 'running'; i++) m.step();
  return m;
};

test('linear algebra: inverse and Cholesky round-trip', () => {
  const A = Mat.from([[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]]);
  const I = A.mul(A.inv());
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.ok(Math.abs(I.get(i, j) - (i === j ? 1 : 0)) < 1e-12);
  const L = A.chol(), R = L.mul(L.T());
  assert.ok(A.d.every((v, i) => Math.abs(v - R.d[i]) < 1e-12));
});

test('EKF is statistically consistent when the model matches the world (ANEES ~ dof, Bar-Shalom 5.4)', () => {
  // no slip, no faults, filter tuned to the true noise (qScale = rScale = 1) over several seeds
  const vals = [];
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const m = new Mission({ platform: 'rover', scenario: 'nominal', seed, oracle: true, filters: ['ekf'] });
    m.world.patches.length = 0; m.world.hazards.length = 0; // flat sand-free terrain
    m.platform.slipAt = () => 0; // exact ground truth = wheel odometry
    const f = m.fs.get('ekf').filter; f.setParam('qScale', 1); f.setParam('rScale', 1);
    m.issue({ type: 'plan' });
    for (let i = 0; i < 20 * 200 && m.status === 'running'; i++) m.step();
    vals.push(m.summary('ekf').anees);
  }
  const mean = vals.reduce((a, b) => a + b) / vals.length;
  assert.ok(mean > 1.2 && mean < 6, `ANEES ${mean.toFixed(2)} should be near dof=3 (values ${vals.map((v) => v.toFixed(1))})`);
});

test('EKF and UKF agree on the linear-dynamics spacecraft (UKF reduces to KF-like accuracy)', () => {
  const m = run({ platform: 'spacecraft', scenario: 'approach', filters: ['ekf', 'ukf'] }, 150);
  const a = m.summary('ekf'), b = m.summary('ukf');
  assert.ok(Math.abs(a.rmse - b.rmse) < 0.1, `rmse ekf ${a.rmse} ukf ${b.rmse}`);
});

test('every filter beats dead reckoning on every platform', () => {
  for (const [platform, scenario, secs] of [['rover', 'nominal', 200], ['jet', 'corridor', 300], ['spacecraft', 'approach', 150]]) {
    const m = run({ platform, scenario, filters: ['dr', 'ekf', 'ukf', 'pf'] }, secs);
    const dr = m.summary('dr').rmse;
    for (const id of ['ekf', 'ukf', 'pf']) assert.ok(m.summary(id).rmse < dr / 2, `${platform}/${id}: ${m.summary(id).rmse} vs DR ${dr}`);
  }
});

test('EKF survives a sand-drift slip burst without gate lock-out (recovery inflation)', () => {
  const m = run({ platform: 'rover', scenario: 'sand', filters: ['ekf'] }, 250);
  assert.ok(m.summary('ekf').rmse < 3, `rmse ${m.summary('ekf').rmse}`);
});

test('simulation is deterministic for a given seed', () => {
  const a = run({ platform: 'rover', scenario: 'dust', seed: 11, filters: ['ekf', 'pf'] }, 60);
  const b = run({ platform: 'rover', scenario: 'dust', seed: 11, filters: ['ekf', 'pf'] }, 60);
  assert.deepEqual(a.fs.get('pf').filter.getState(), b.fs.get('pf').filter.getState());
  assert.deepEqual(a.truth, b.truth);
});

test('missions complete with an oracle autopilot on all platforms', () => {
  for (const [platform, scenario] of [['rover', 'nominal'], ['jet', 'corridor'], ['spacecraft', 'approach']]) {
    const m = run({ platform, scenario, filters: ['ekf'] }, 700);
    assert.equal(m.status, 'complete', `${platform}: ${m.status} ${m.failReason}`);
  }
});

test('custom vehicle with no Jacobians works through the numerical fallback', () => {
  const noJac = { ...spacecraftModel, jacobians: undefined, meas: { ...spacecraftModel.meas, lidar: { ...spacecraftModel.meas.lidar, H: undefined } } };
  const S = { ...SPACECRAFT_SENSORS };
  const a = new EKF(spacecraftModel, S), b = new EKF(noJac, S);
  const x0 = [20, -100, 0, 0, 0, 0], P0 = spacecraftModel.P0();
  a.init(x0, P0); b.init(x0, P0);
  const u = { ax: 0.01, ay: 0 }, m = { kind: 'lidar', range: 101.5, bearing: Math.atan2(100, -20) + 0.01, sigR: 0.3, sigB: 0.004 };
  for (let i = 0; i < 40; i++) { a.predict(u, 0.05); b.predict(u, 0.05); }
  a.update(m); b.update(m);
  assert.ok(a.getState().every((v, i) => Math.abs(v - b.getState()[i]) < 1e-5));
});
