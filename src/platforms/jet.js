import { clamp, wrapAngle } from '../core/linalg.js';
import { arc } from '../core/kinematics.js';
import { jetModel } from '../models/jet.js';
import { JET_SENSORS } from '../sim/sensors.js';
import { genWorld } from '../sim/scenario.js';

const V_CRUISE = 220, W_MAX = 0.08;
const FAULTS = { gyroStep: 'INS gyro bias jump', gnssJam: 'GNSS jammed', wind: 'Unmodelled crosswind (jet-stream)', blackout: 'Radar beacons dark', outliers: 'GNSS spoofing / radar ghosts' };

const SCENARIOS = {
  corridor: { label: 'Sortie 1 · Strike Corridor', blurb: 'GNSS available. Route threads between no-fly zones.', landmarks: 14, zones: 4, schedule: [], goal: true },
  jammed: {
    label: 'Sortie 2 · GNSS Denied', blurb: 'Jamming from t=30 s, gyro bias jump, a jet-stream gust. Beacons + INS only.',
    landmarks: 10, zones: 5, goal: true,
    schedule: [{ t: 30, type: 'gnssJam' }, { t: 70, type: 'gyroStep' }, { t: 110, type: 'wind' }],
  },
  sandbox: { label: 'Sandbox · Free Flight', blurb: 'No objective. Inject jamming, wind, spoofing.', landmarks: 12, zones: 3, schedule: [], goal: false },
};

function build(id, seed = 7) {
  const cfg = SCENARIOS[id] || SCENARIOS.corridor;
  const w = genWorld({
    seed, salt: 29, jitter: 900, bounds: { xmin: -34000, xmax: 34000, ymin: -22000, ymax: 22000 },
    pts: [[-28000, -16000], [-14000, -12000], [0, -17000], [14000, -9000], [26000, 2000], [12000, 14000], [-6000, 9000], [-22000, 14000]],
    sampleIdx: [3, 4, 6],
    hazards: { n: cfg.zones, r: [2200, 3600], gap: [900, 2200], clear: 700 },
    landmarks: { n: cfg.landmarks, spacing: 9000, margin: 2000, keepOut: 500 },
  });
  return { ...w, id, cfg, goal: cfg.goal, schedule: cfg.schedule.map((s) => ({ ...s })), unit: 'km' };
}

export const jetPlatform = {
  id: 'jet', label: 'Fighter Jet', unit: 'm', model: jetModel, sensors: JET_SENSORS,
  scenarios: SCENARIOS, build,
  faults: FAULTS, faultLabel: (t) => FAULTS[t] || t,
  resource: { name: 'Fuel', fatal: true },
  hazard: { damage: 35, immobilize: 0, label: 'NO-FLY ZONE BREACH' },
  objective: { noun: 'Checkpoint', verb: 'Crossing', dwell: 0, cost: 0, points: 300, tol: 500 },

  nominalStart: (w) => [w.start.x, w.start.y, w.start.th, V_CRUISE, 0, 0, 0],
  initPlatformState: (m) => ({ wind0: [m.truth[5], m.truth[6]], gust: 0 }), // background wind drawn from the prior
  pose: (x) => ({ x: x[0], y: x[1], th: x[2] }),
  idleCmd: () => ({ a: 0, w: 0 }),
  holdCmd: (m, est) => ({ a: 0, w: 0.03 }), // racetrack orbit once the route is flown
  isActive: (c) => c.a !== 0 || c.w !== 0,
  manualCmd: (k) => ({ a: ((k.up ? 1 : 0) - (k.down ? 1 : 0)) * 4, w: ((k.left ? 1 : 0) - (k.right ? 1 : 0)) * W_MAX }),
  safeMargin: () => 1200,
  /** Can't stop a jet: break turn away from the zone. */
  safeCmd(m, est, h) {
    const cross = Math.cos(est.th) * (h.y - est.y) - Math.sin(est.th) * (h.x - est.x);
    return { a: 0, w: cross > 0 ? -W_MAX : W_MAX };
  },

  applyFault(m, type) {
    if (type === 'gyroStep') { m.truth[4] += (m.rng.next() < 0.5 ? -1 : 1) * 0.0008; return 0; }
    return { gnssJam: 400, wind: 70, blackout: 30, outliers: 25 }[type] || 0;
  },

  step(m, cmd, dt) {
    const s = m.sensors, x = m.truth, rng = m.rng;
    if (m.resource <= 0) cmd = { a: 0, w: 0 };
    const a = clamp(cmd.a, -6, 6), om = clamp(cmd.w, -W_MAX, W_MAX);
    const aTrue = (x[3] >= 300 && a > 0) || (x[3] <= 120 && a < 0) ? 0 : a;
    const [nx, ny, nth] = arc(x[0], x[1], x[2], x[3] + 0.5 * aTrue * dt, om, dt);
    // jet-stream gust: first-order lag toward the fault wind (tau 8 s), so the shear is a ramp, not a step
    const ps = m.ps;
    ps.gust += ((m.faultActive('wind') ? 1 : 0) - ps.gust) * (dt / 8);
    x[5] = ps.wind0[0] + ps.gust * 22; x[6] = ps.wind0[1] - ps.gust * 14; // INS + airspeed cannot see this directly
    const B = m.world.bounds;
    x[0] = clamp(nx + x[5] * dt, B.xmin, B.xmax); x[1] = clamp(ny + x[6] * dt, B.ymin, B.ymax); x[2] = nth;
    x[3] += aTrue * dt;
    x[4] += s.sigmaBiasWalk * Math.sqrt(dt) * rng.randn();
    m.resource = clamp(m.resource - (0.012 + 0.004 * Math.abs(aTrue)) * dt, 0, 100);
    return { a: aTrue + s.sigmaA * rng.randn(), w: om + x[4] + s.sigmaW * rng.randn() };
  },

  sense(m, rng) {
    const s = m.sensors, x = m.truth, out = [], bad = m.faultActive('outliers');
    out.push({ kind: 'airspeed', v: x[3] + s.sigmaAirspeed * rng.randn(), sigma: s.sigmaAirspeed });
    if (!m.faultActive('blackout')) {
      const vis = m.world.landmarks.map((l) => ({ l, d: Math.hypot(l.x - x[0], l.y - x[1]) })).filter((o) => o.d < s.landmarkRange)
        .sort((a, b) => a.d - b.d).slice(0, s.maxLandmarksPerSweep);
      for (const { l, d } of vis) {
        let range = d + s.sigmaR * rng.randn(), bearing = wrapAngle(Math.atan2(l.y - x[1], l.x - x[0]) - x[2] + s.sigmaB * rng.randn());
        if (bad && rng.next() < 0.3) { range += rng.uniform(200, 600); bearing = wrapAngle(bearing + rng.uniform(0.05, 0.15)); }
        out.push({ kind: 'landmark', id: l.id, lx: l.x, ly: l.y, range, bearing, sigR: s.sigmaR, sigB: s.sigmaB });
      }
    }
    if (!m.faultActive('gnssJam')) {
      let fx = x[0] + s.sigmaFix * rng.randn(), fy = x[1] + s.sigmaFix * rng.randn();
      if (bad && rng.next() < 0.4) { fx += rng.uniform(150, 400); fy -= rng.uniform(150, 400); }
      out.push({ kind: 'fix', x: fx, y: fy, sigma: s.sigmaFix });
    }
    return out;
  },

  autopilot(m, est, wp) {
    const dx = wp.x - est.x, dy = wp.y - est.y, dist = Math.hypot(dx, dy);
    const err = wrapAngle(Math.atan2(dy, dx) - est.th);
    const v = m.fs.get(m.primary).filter.getState()[3];
    return { cmd: { a: clamp(0.5 * (V_CRUISE - v), -4, 4), w: clamp(1.2 * err, -W_MAX, W_MAX) }, dist, arrived: dist < 350 };
  },

  checkObjective(m, t) {
    const d = Math.hypot(m.truth[0] - t.x, m.truth[1] - t.y);
    return d <= 500 ? { ok: true, text: `crossed ${d.toFixed(0)} m from fix` } : { ok: false, text: `passed ${d.toFixed(0)} m off (needs ≤ 500 m)` };
  },

  onHazard() {},
};
