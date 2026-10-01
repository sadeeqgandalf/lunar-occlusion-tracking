import { clamp, wrapAngle } from '../core/linalg.js';
import { spacecraftModel } from '../models/spacecraft.js';
import { SPACECRAFT_SENSORS } from '../sim/sensors.js';
import { genWorld } from '../sim/scenario.js';

const A_MAX = 0.2;
const FAULTS = { imuBias: 'Accelerometer bias', sunBlind: 'Lidar sun-blind', outliers: 'Lidar multipath ghosts', fixLoss: 'Ground-track fix lost' };

const SCENARIOS = {
  approach: { label: 'RPO 1 · Docking Approach', blurb: 'Hold at 150 m and 60 m gates, then dock. Debris flanks the V-bar.', debris: 3, schedule: [], goal: true },
  blind: {
    label: 'RPO 2 · Sun-Blind Lidar', blurb: 'Lidar dazzled during the final approach, plus an accelerometer bias.',
    debris: 4, goal: true, schedule: [{ t: 45, type: 'imuBias' }, { t: 80, type: 'sunBlind' }, { t: 140, type: 'outliers' }],
  },
  sandbox: { label: 'Sandbox · Free Flyer', blurb: 'Fly the chaser yourself and break its sensors.', debris: 3, schedule: [], goal: false },
};

function build(id, seed = 7) {
  const cfg = SCENARIOS[id] || SCENARIOS.approach;
  const w = genWorld({
    seed, salt: 41, jitter: 1.5, bounds: { xmin: -120, xmax: 120, ymin: -330, ymax: 60 },
    pts: [[18, -290], [10, -220], [0, -150], [0, -60], [0, -6]],
    sampleIdx: [2, 3, 4],
    hazards: { n: cfg.debris, r: [10, 18], gap: [8, 18], clear: 6 },
    landmarks: { n: 0, spacing: 1, margin: 1, keepOut: 1 },
  });
  w.landmarks = [{ id: 0, x: 0, y: 0 }]; // the docking target itself is the lidar's reference
  return { ...w, id, cfg, goal: cfg.goal, schedule: cfg.schedule.map((s) => ({ ...s })), unit: 'm', viewSwap: true };
}

const ff = (n, x) => [-(3 * n * n * x[0] + 2 * n * x[3]), 2 * n * x[2]]; // cancel Clohessy-Wiltshire coupling

export const spacecraftPlatform = {
  id: 'spacecraft', label: 'Spacecraft (RPO)', unit: 'm', model: spacecraftModel, sensors: SPACECRAFT_SENSORS,
  scenarios: SCENARIOS, build,
  faults: FAULTS, faultLabel: (t) => FAULTS[t] || t,
  resource: { name: 'Propellant', fatal: true },
  hazard: { damage: 40, immobilize: 0, label: 'COLLISION · debris strike' },
  objective: { noun: 'Gate', verb: 'Station-keeping at', dwell: 3, cost: 1, points: 200, tol: 4 },

  nominalStart: (w) => [w.start.x, w.start.y, 0, 0, 0, 0],
  initPlatformState: (m) => ({ bias0: [m.truth[4], m.truth[5]] }), // accelerometer bias drawn from the prior
  pose: (x) => ({ x: x[0], y: x[1], th: Math.atan2(x[3], x[2]) }),
  idleCmd: () => ({ ax: 0, ay: 0 }),
  isActive: (c) => c.ax !== 0 || c.ay !== 0,
  manualCmd: (k) => ({ ax: ((k.right ? 1 : 0) - (k.left ? 1 : 0)) * A_MAX, ay: ((k.up ? 1 : 0) - (k.down ? 1 : 0)) * A_MAX }),
  safeMargin: () => 4,

  /** Station-keep on the estimate: kill velocity, cancel orbital coupling. */
  holdCmd(m, est) {
    const x = m.o.oracle ? m.truth : m.fs.get(m.primary).filter.getState(), n = m.sensors.meanMotion, f = ff(n, x);
    return { ax: clamp(-0.5 * x[2] + f[0], -A_MAX, A_MAX), ay: clamp(-0.5 * x[3] + f[1], -A_MAX, A_MAX) };
  },
  safeCmd(m, est) { return this.holdCmd(m, est); },

  applyFault(m, type) { return { imuBias: 60, sunBlind: 40, outliers: 25, fixLoss: 80 }[type] || 0; },

  step(m, cmd, dt) {
    const s = m.sensors, x = m.truth, rng = m.rng;
    if (m.resource <= 0) cmd = { ax: 0, ay: 0 };
    const a = { ax: clamp(cmd.ax, -A_MAX, A_MAX), ay: clamp(cmd.ay, -A_MAX, A_MAX) };
    // truth thrust is exactly the command; the accelerometer reads thrust + bias + noise. Passing w = -imuNoise
    // makes the model's (u - bias + w) equal the commanded acceleration.
    const imu = [s.sigmaAccel * rng.randn(), s.sigmaAccel * rng.randn()];
    const fb = m.faultActive('imuBias') ? 0.006 : 0;
    x[4] = m.ps.bias0[0] + fb; x[5] = m.ps.bias0[1] - fb;
    const u = { ax: a.ax + x[4] + imu[0], ay: a.ay + x[5] + imu[1] };
    const nx = spacecraftModel.f(x, u, [-imu[0], -imu[1], 0, 0], dt, s);
    for (let i = 0; i < 4; i++) x[i] = nx[i];
    m.resource = clamp(m.resource - (Math.abs(a.ax) + Math.abs(a.ay)) * dt * 0.5, 0, 100);
    return u;
  },

  sense(m, rng) {
    const s = m.sensors, x = m.truth, out = [], bad = m.faultActive('outliers');
    const r = Math.hypot(x[0], x[1]);
    if (!m.faultActive('sunBlind') && r < s.lidarRange && r > 1) {
      let range = r + s.sigmaR * rng.randn(), bearing = wrapAngle(Math.atan2(-x[1], -x[0]) + s.sigmaB * rng.randn());
      if (bad && rng.next() < 0.3) { range += rng.uniform(2, 6); bearing = wrapAngle(bearing + rng.uniform(0.05, 0.2)); }
      out.push({ kind: 'lidar', lx: 0, ly: 0, range, bearing, sigR: s.sigmaR, sigB: s.sigmaB });
    }
    if (m.t % s.fixPeriod < s.fixWindow && !m.faultActive('fixLoss')) {
      out.push({ kind: 'fix', x: x[0] + s.sigmaFix * rng.randn(), y: x[1] + s.sigmaFix * rng.randn(), sigma: s.sigmaFix });
    }
    return out;
  },

  /** Velocity-shaped approach to the waypoint with CW feed-forward; speed tapers with range for a soft arrival. */
  autopilot(m, est, wp) {
    const x = m.o.oracle ? m.truth : m.fs.get(m.primary).filter.getState(), n = m.sensors.meanMotion;
    const dx = wp.x - x[0], dy = wp.y - x[1], dist = Math.hypot(dx, dy), speed = Math.hypot(x[2], x[3]);
    const vmax = Math.min(2.5, 0.08 * dist), k = dist > 1e-6 ? vmax / dist : 0;
    const f = ff(n, x);
    const cmd = { ax: clamp(0.35 * (dx * k - x[2]) + f[0], -A_MAX, A_MAX), ay: clamp(0.35 * (dy * k - x[3]) + f[1], -A_MAX, A_MAX) };
    return { cmd, dist, arrived: dist < (wp.kind === 'objective' ? 1.2 : 5) && speed < 0.6 };
  },

  checkObjective(m, t) {
    const x = m.truth, d = Math.hypot(x[0] - t.x, x[1] - t.y), v = Math.hypot(x[2], x[3]), dock = t.id === m.targets.length;
    const maxD = dock ? 2 : 4, maxV = dock ? 0.4 : 0.6;
    return d <= maxD && v <= maxV
      ? { ok: true, text: `${d.toFixed(1)} m off, ${v.toFixed(2)} m/s` }
      : { ok: false, text: `${d.toFixed(1)} m off at ${v.toFixed(2)} m/s (needs ≤ ${maxD} m, ≤ ${maxV} m/s)` };
  },

  onHazard(m, h) {
    const x = m.truth, a = Math.atan2(x[1] - h.y, x[0] - h.x);
    x[0] = h.x + (h.r + 0.8) * Math.cos(a); x[1] = h.y + (h.r + 0.8) * Math.sin(a);
    x[2] *= -0.4; x[3] *= -0.4;
  },
};
