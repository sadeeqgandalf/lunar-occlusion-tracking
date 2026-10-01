import test from 'node:test';
import assert from 'node:assert/strict';
import { Mission } from '../src/sim/mission.js';
import { coach, healthHtml } from '../src/ui/coach.js';

const run = (m, secs) => { for (let i = 0; i < secs / m.dt; i++) m.step(); };

test('coach: parked vehicle is called out and offers to start', () => {
  const m = new Mission({ platform: 'rover', scenario: 'nominal' });
  const c = coach(m);
  assert.match(c.title, /parked/i);
  assert.ok(c.cta.some(([, a]) => a === 'start'));
});

test('coach: an active fault is explained with what to watch', () => {
  const m = new Mission({ platform: 'rover', scenario: 'sandbox' });
  m.issue({ type: 'plan' }); run(m, 5); m.injectFault('blackout');
  const c = coach(m);
  assert.match(c.title, /Beacon blackout/);
  assert.match(c.body, /Watch/);
});

test('coach: dead-reckoning primary is flagged once it has run a while', () => {
  const m = new Mission({ platform: 'rover', scenario: 'sandbox', primary: 'dr', oracle: false });
  m.issue({ type: 'goto', x: -60, y: -30, kind: 'goto' }); run(m, 12);
  const c = coach(m);
  assert.ok(['warn', 'bad'].includes(c.tone), c.title);
});

test('health: derived from onboard quantities only, goes bad for dead reckoning and stays good for the EKF', () => {
  const m = new Mission({ platform: 'rover', scenario: 'nominal', oracle: true, filters: ['dr', 'ekf'] });
  m.issue({ type: 'plan' }); run(m, 120);
  assert.equal(m.health('ekf').level, 'good');
  assert.equal(m.health('dr').level, 'bad');
  assert.ok(m.health('dr').reasons.some((r) => /bubble/.test(r)));
  assert.match(healthHtml(m), /NAV NOMINAL/);
});

test('health: a blackout long enough drives even the EKF to a stated warning, never silently', () => {
  const m = new Mission({ platform: 'rover', scenario: 'sandbox', oracle: true, filters: ['ekf'] });
  m.issue({ type: 'plan' }); run(m, 10);
  m.injectFault('blackout'); m.faults.set('blackout', m.t + 400); // outlast the 30 s default
  run(m, 120);
  const h = m.health();
  assert.notEqual(h.level, 'good');
  assert.ok(h.reasons.length > 0);
});

test('coach: debrief after the mission compares filters and reports hazard hits', () => {
  const m = new Mission({ platform: 'rover', scenario: 'nominal', oracle: true });
  m.issue({ type: 'plan' }); run(m, 400);
  assert.equal(m.status, 'complete');
  const c = coach(m);
  assert.match(c.title, /Debrief/);
  assert.match(c.body, /best was/);
});

test('health: no false "degraded" while the filter settles from its launch prior', () => {
  const m = new Mission({ platform: 'rover', scenario: 'nominal', filters: ['ekf'] });
  m.issue({ type: 'plan' }); run(m, 6);
  assert.equal(m.health().level, 'good');
});
