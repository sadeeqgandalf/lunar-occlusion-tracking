// Procedural world generation shared by all platforms: a planned route, objectives on it, hazards flanking it
// (harmless to a good estimator, deadly to a drifting one), friction/drift patches and beacon landmarks.
import { RNG } from '../core/rng.js';
import { clamp } from '../core/linalg.js';

const distSeg = (px, py, a, b) => {
  const dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy || 1;
  const t = clamp(((px - a.x) * dx + (py - a.y) * dy) / l2, 0, 1);
  return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
};

/**
 * @param spec {pts:[[x,y]...] start first, sampleIdx, jitter, bounds, hazards:{n,r:[a,b],gap:[a,b],clear}, patches:{n,r:[a,b],slip},
 *              landmarks:{n,spacing,margin,keepOut}, seed, salt}
 */
export function genWorld(spec) {
  const rng = new RNG(spec.seed * 7919 + (spec.salt || 13)), j = (a) => rng.uniform(-a, a), B = spec.bounds;
  const pts = spec.pts.map(([x, y]) => ({ x: x + j(spec.jitter), y: y + j(spec.jitter) }));
  const targets = spec.sampleIdx.map((i, k) => ({ id: k + 1, x: pts[i].x, y: pts[i].y }));
  const route = pts.slice(1).map((p) => {
    const t = targets.find((tg) => tg.x === p.x && tg.y === p.y);
    return t ? { x: p.x, y: p.y, kind: 'objective', targetId: t.id } : { x: p.x, y: p.y, kind: 'goto' };
  });
  const start = { x: pts[0].x, y: pts[0].y, th: Math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x) };

  const H = spec.hazards, hazards = [];
  for (let tries = 0; hazards.length < H.n && tries < 600; tries++) {
    const si = rng.int(pts.length - 1), a = pts[si], b = pts[si + 1], t = rng.uniform(0.15, 0.85);
    const px = a.x + (b.x - a.x) * t, py = a.y + (b.y - a.y) * t;
    const len = Math.hypot(b.x - a.x, b.y - a.y), nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
    const r = rng.uniform(...H.r), side = rng.next() < 0.5 ? -1 : 1, off = r + rng.uniform(...H.gap);
    const c = { x: px + nx * off * side, y: py + ny * off * side, r };
    const clearOfRoute = pts.slice(0, -1).every((p, i) => distSeg(c.x, c.y, p, pts[i + 1]) > r + H.clear);
    const clearOfOthers = hazards.every((o) => Math.hypot(o.x - c.x, o.y - c.y) > o.r + c.r + H.clear * 1.5);
    const inBounds = c.x > B.xmin + c.r && c.x < B.xmax - c.r && c.y > B.ymin + c.r && c.y < B.ymax - c.r;
    if (clearOfRoute && clearOfOthers && inBounds) hazards.push(c);
  }

  const patches = [];
  const Pt = spec.patches || { n: 0 };
  for (let i = 0; i < Pt.n; i++) {
    const a = pts[1 + ((i * 3 + 1) % (pts.length - 2))];
    patches.push({ x: a.x + j(Pt.r[0] * 0.6), y: a.y + j(Pt.r[0] * 0.6), r: rng.uniform(...Pt.r), slip: Pt.slip * rng.uniform(0.8, 1) });
  }

  const L = spec.landmarks, landmarks = [];
  for (let tries = 0; landmarks.length < L.n && tries < 4000; tries++) {
    const l = { id: landmarks.length, x: rng.uniform(B.xmin + L.margin, B.xmax - L.margin), y: rng.uniform(B.ymin + L.margin, B.ymax - L.margin) };
    if (landmarks.some((o) => Math.hypot(o.x - l.x, o.y - l.y) < L.spacing)) continue;
    if (hazards.some((c) => Math.hypot(c.x - l.x, c.y - l.y) < c.r + L.keepOut)) continue;
    landmarks.push(l);
  }
  return { bounds: B, start, route, targets, hazards, patches, landmarks };
}

// ------------------------------------------------------------------ Mars / Earth ground rover
export const SCENARIOS = {
  nominal: { label: 'Sol 12 · Nominal Traverse', blurb: 'Good beacon coverage. Learn the console, then compare filters.', landmarks: 20, craters: 5, sand: 1, slip: 0.25, schedule: [], goal: true },
  dust: {
    label: 'Sol 87 · Dust Storm', blurb: 'Beacon blackout, a gyro bias jump and a missed orbiter pass. Scripted faults.',
    landmarks: 14, craters: 6, sand: 1, slip: 0.3, goal: true,
    schedule: [{ t: 55, type: 'blackout' }, { t: 95, type: 'gyroStep' }, { t: 140, type: 'orbiterLoss' }, { t: 175, type: 'outliers' }],
  },
  sand: { label: 'Sol 140 · Sand Trap', blurb: 'Deep drifts: the wheels spin and odometry lies. Sparse beacons.', landmarks: 12, craters: 5, sand: 4, slip: 0.45, goal: true, schedule: [{ t: 120, type: 'slip' }] },
  sandbox: { label: 'Sandbox · Free Play', blurb: 'No objective. Break things with the fault panel and watch each filter respond.', landmarks: 18, craters: 5, sand: 2, slip: 0.35, schedule: [], goal: false },
};

export function buildRover(id, seed = 7) {
  const cfg = SCENARIOS[id] || SCENARIOS.nominal;
  const w = genWorld({
    seed, jitter: 3, bounds: { xmin: -90, xmax: 90, ymin: -60, ymax: 60 },
    pts: [[-72, -40], [-45, -32], [-15, -44], [18, -28], [50, -40], [66, -10], [42, 14], [12, 30], [-22, 20], [-54, 36], [-62, 4]],
    sampleIdx: [3, 5, 7, 9],
    hazards: { n: cfg.craters, r: [4, 7.5], gap: [2.2, 5], clear: 1.8 },
    patches: { n: cfg.sand, r: [11, 17], slip: cfg.slip },
    landmarks: { n: cfg.landmarks, spacing: 12, margin: 4, keepOut: 1.5 },
  });
  return { ...w, id, cfg, goal: cfg.goal, schedule: cfg.schedule.map((s) => ({ ...s })) };
}
