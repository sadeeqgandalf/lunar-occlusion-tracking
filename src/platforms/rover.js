import { clamp, wrapAngle } from '../core/linalg.js';
import { arc } from '../core/kinematics.js';
import { roverModel } from '../models/rover.js';
import { ROVER_SENSORS } from '../sim/sensors.js';
import { SCENARIOS, buildRover } from '../sim/scenario.js';

const FAULTS = { gyroStep: 'Gyro bias jump', slip: 'Wheel slip (ice/sand)', blackout: 'Beacon blackout (dust)', outliers: 'Beacon outlier burst', orbiterLoss: 'Orbiter pass missed' };
const smooth = (d, r) => { const q = clamp(1 - (d / r) ** 2, 0, 1); return q * q * (3 - 2 * q); };

export const roverPlatform = {
  id: 'rover', label: 'Mars Rover', unit: 'm', model: roverModel, sensors: ROVER_SENSORS,
  scenarios: SCENARIOS, build: buildRover,
  faults: FAULTS, faultLabel: (t) => FAULTS[t] || t,
  resource: { name: 'Battery', fatal: true },
  hazard: { damage: 20, immobilize: 3, label: 'TILT ALERT · crater rim' },
  objective: { noun: 'Sample', verb: 'Drilling', dwell: 2.5, cost: 3, points: 250, tol: 2 },
  controls: { v: 1.2, w: 0.8, labels: { fwd: 'drive', turn: 'steer' } },

  nominalStart: (w) => [w.start.x, w.start.y, w.start.th, 0, 0],
  initPlatformState: () => ({ v: 0, w: 0 }),
  pose: (x) => ({ x: x[0], y: x[1], th: x[2] }),
  idleCmd: () => ({ v: 0, w: 0 }),
  isActive: (c) => c.v !== 0 || c.w !== 0,
  manualCmd: (k) => ({ v: ((k.up ? 1 : 0) - (k.down ? 1 : 0)) * 1.2, w: ((k.left ? 1 : 0) - (k.right ? 1 : 0)) * 0.8 }),
  safeMargin: () => 1.5,
  safeCmd: () => ({ v: 0, w: 0 }),

  slipAt(m, x, y) {
    let s = 0.015;
    for (const p of m.world.patches) s = Math.max(s, p.slip * smooth(Math.hypot(x - p.x, y - p.y), p.r));
    return s + (m.faultActive('slip') ? 0.4 : 0);
  },

  applyFault(m, type) {
    if (type === 'gyroStep') { m.truth[3] += (m.rng.next() < 0.5 ? -1 : 1) * 0.01; return 0; }
    return { slip: 20, blackout: 30, outliers: 25, orbiterLoss: 120 }[type] || 0;
  },

  step(m, cmd, dt) {
    const s = m.sensors, ps = m.ps, x = m.truth, rng = m.rng;
    if (m.resource <= 0) cmd = { v: 0, w: 0 };
    ps.v += clamp(cmd.v - ps.v, -1.2 * dt, 1.2 * dt);
    ps.w += clamp(cmd.w - ps.w, -2 * dt, 2 * dt);
    x[4] = this.slipAt(m, x[0], x[1]); // true slip is part of the truth state (the filter must estimate it)
    const vg = ps.v * (1 - x[4]);       // ground speed < wheel speed; odometry alone can't see it
    const [nx, ny, nth] = arc(x[0], x[1], x[2], vg, ps.w, dt);
    const B = m.world.bounds;
    x[0] = clamp(nx, B.xmin, B.xmax); x[1] = clamp(ny, B.ymin, B.ymax); x[2] = nth;
    x[3] += s.sigmaBiasWalk * Math.sqrt(dt) * rng.randn();
    const moving = Math.abs(ps.v) > 0.05;
    m.resource = clamp(m.resource + (moving ? -(0.02 + 0.08 * Math.abs(ps.v)) : 0.05) * dt, 0, 100);
    return { v: ps.v + s.sigmaV * rng.randn(), w: ps.w + x[3] + s.sigmaW * rng.randn() };
  },

  sense(m, rng) {
    const s = m.sensors, x = m.truth, out = [];
    const bad = m.faultActive('outliers');
    if (!m.faultActive('blackout')) {
      const vis = m.world.landmarks.map((l) => ({ l, d: Math.hypot(l.x - x[0], l.y - x[1]) })).filter((o) => o.d < s.landmarkRange)
        .sort((a, b) => a.d - b.d).slice(0, s.maxLandmarksPerSweep);
      for (const { l, d } of vis) {
        let range = d + s.sigmaR * rng.randn(), bearing = wrapAngle(Math.atan2(l.y - x[1], l.x - x[0]) - x[2] + s.sigmaB * rng.randn());
        if (bad && rng.next() < 0.3) { range += rng.uniform(3, 8) * (rng.next() < 0.5 ? -1 : 1); bearing = wrapAngle(bearing + rng.uniform(0.15, 0.4) * (rng.next() < 0.5 ? -1 : 1)); }
        out.push({ kind: 'landmark', id: l.id, lx: l.x, ly: l.y, range, bearing, sigR: s.sigmaR, sigB: s.sigmaB });
      }
    }
    if (m.t % s.fixPeriod < s.fixWindow && !m.faultActive('orbiterLoss')) {
      let fx = x[0] + s.sigmaFix * rng.randn(), fy = x[1] + s.sigmaFix * rng.randn();
      if (bad && rng.next() < 0.3) { fx += rng.uniform(8, 15); fy -= rng.uniform(8, 15); }
      out.push({ kind: 'fix', x: fx, y: fy, sigma: s.sigmaFix });
    }
    return out;
  },

  /** Pure-pursuit on the *estimated* pose. */
  autopilot(m, est, wp) {
    const dx = wp.x - est.x, dy = wp.y - est.y, dist = Math.hypot(dx, dy);
    const err = wrapAngle(Math.atan2(dy, dx) - est.th);
    const v = Math.abs(err) > 0.9 ? 0 : 1.2 * clamp(dist / 4, 0.25, 1) * Math.max(0, Math.cos(err));
    return { cmd: { v, w: clamp(2 * err, -0.8, 0.8) }, dist, arrived: dist < (wp.kind === 'objective' ? 1 : 1.5) };
  },

  checkObjective(m, t) {
    const d = Math.hypot(m.truth[0] - t.x, m.truth[1] - t.y);
    return d <= 2 ? { ok: true, text: `drill ${d.toFixed(1)} m from target` } : { ok: false, text: `drilled ${d.toFixed(1)} m off target (needs ≤ 2.0 m): pose belief was wrong` };
  },

  onHazard(m, h) {
    const x = m.truth, a = Math.atan2(x[1] - h.y, x[0] - h.x);
    x[0] = h.x + (h.r + 0.8) * Math.cos(a); x[1] = h.y + (h.r + 0.8) * Math.sin(a);
    m.ps.v = 0;
  },
};
