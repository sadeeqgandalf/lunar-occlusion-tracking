// The Policy Lab world, expert and episode runner: a port of policy/sim.py (test/policy.test.js checks they agree).
// Rover frame: x forward, y left; metres, seconds, radians.

export const DT = 0.25, H = 8, EXEC = 4;
export const V_MAX = 1.2, W_MAX = 0.8, ROVER_R = 0.5;
export const GOAL_X = 16, GOAL_TOL = 1.5, T_MAX = 40, AMBIGUOUS = 0.35;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** Seeded uniform [0, 1) (mulberry32) with a Gaussian helper. */
export function makeRng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, gauss };
}

export const toUnit = ([v, w]) => [(v / V_MAX) * 2 - 1, w / W_MAX];
export const fromUnit = ([a, b]) => [((clamp(a, -1, 1) + 1) / 2) * V_MAX, clamp(b, -1, 1) * W_MAX];

/** World = {rockX, rockY, rockR, goalY, sigma}. Observation = [rock dx, rock dy, rock r, goal dx, goal dy, sigma]. */
export function observe(pose, w, rng) {
  const c = Math.cos(pose.th), s = Math.sin(pose.th);
  const rel = (px, py) => [(px - pose.x) * c + (py - pose.y) * s, -(px - pose.x) * s + (py - pose.y) * c];
  const [rx, ry] = rel(w.rockX, w.rockY), [gx, gy] = rel(GOAL_X, w.goalY);
  return [rx + rng.gauss() * w.sigma, ry + rng.gauss() * w.sigma, w.rockR, gx, gy, w.sigma];
}

/** Signed distance of the rock from the straight line to the goal (+ = rock on the left). */
export function rockOffset(o) {
  const head = Math.atan2(o[4], o[3]);
  return -o[0] * Math.sin(head) + o[1] * Math.cos(head);
}

/** +1 = pass the rock on its left, -1 = on its right; a coin flip when it is dead ahead. */
export function chooseSide(o, rng) {
  const off = rockOffset(o);
  return Math.abs(off) < AMBIGUOUS ? (rng.uniform() < 0.5 ? 1 : -1) : -Math.sign(off);
}

/** The expert's H actions: head for the goal, or follow the tangent of the rock grown by body radius and margin. */
export function expertChunk(o, side) {
  const [rockX, rockY, r, goalX, goalY, sigma] = o, keepOut = r + ROVER_R + 0.25 + 2 * sigma;
  let x = 0, y = 0, th = 0;
  const out = [];
  for (let i = 0; i < H; i++) {
    const c = Math.cos(th), s = Math.sin(th);
    const rx = (rockX - x) * c + (rockY - y) * s, ry = -(rockX - x) * s + (rockY - y) * c;
    const gx = (goalX - x) * c + (goalY - y) * s, gy = -(goalX - x) * s + (goalY - y) * c;
    const toGoal = Math.atan2(gy, gx);
    const along = rx * Math.cos(toGoal) + ry * Math.sin(toGoal), across = -rx * Math.sin(toGoal) + ry * Math.cos(toGoal);
    const blocked = along > 0 && Math.abs(across) < keepOut && along < Math.hypot(gx, gy);
    const tangent = Math.atan2(ry, rx) + side * Math.asin(Math.min(1, keepOut / Math.hypot(rx, ry)));
    const w = clamp(2 * (blocked ? tangent : toGoal), -W_MAX, W_MAX);
    const v = clamp(1 - 0.8 * sigma - 0.3 * Math.abs(w), 0.3, V_MAX);
    out.push([v, w]);
    th += w * DT; x += v * Math.cos(th) * DT; y += v * Math.sin(th) * DT;
  }
  return out;
}

/** Plan an (H, 2) chunk from the rover's pose into world-frame points (for drawing). */
export function chunkPath(pose, chunk) {
  let { x, y, th } = pose;
  const pts = [[x, y]];
  for (const [v, w] of chunk) { th += w * DT; x += v * Math.cos(th) * DT; y += v * Math.sin(th) * DT; pts.push([x, y]); }
  return pts;
}

/** One closed-loop run: call decide(obs) for a chunk, execute EXEC actions, repeat. status: driving|goal|crash|stuck. */
export class Episode {
  constructor(world, seed) {
    this.w = world; this.rng = makeRng(seed);
    this.pose = { x: 0, y: 0, th: 0 }; this.prev = { ...this.pose };
    this.t = 0; this.k = EXEC; this.chunk = null; this.status = 'driving';
    this.path = [[0, 0]]; this.minGap = Infinity; this.passedY = null;
  }

  get done() { return this.status !== 'driving'; }

  /** Advance one action (DT seconds). decide(obs) is called when a new chunk is needed and returns it. */
  step(decide) {
    if (this.done) return;
    if (this.k >= EXEC) { this.obs = observe(this.pose, this.w, this.rng); this.chunk = decide(this.obs); this.k = 0; }
    const [v, w] = this.chunk[this.k++], p = this.pose, w0 = this.w;
    this.prev = { ...p };
    const th = p.th + clamp(w, -W_MAX, W_MAX) * DT;
    const next = { x: p.x + v * Math.cos(th) * DT, y: p.y + v * Math.sin(th) * DT, th };
    if (p.x < w0.rockX && next.x >= w0.rockX) this.passedY = next.y - w0.rockY;
    this.pose = next; this.t += DT; this.path.push([next.x, next.y]);
    const gap = Math.hypot(next.x - w0.rockX, next.y - w0.rockY) - w0.rockR - ROVER_R;
    this.minGap = Math.min(this.minGap, gap);
    if (gap < 0) this.status = 'crash';
    else if (Math.hypot(next.x - GOAL_X, next.y - w0.goalY) < GOAL_TOL) this.status = 'goal';
    else if (this.t >= T_MAX - 1e-9) this.status = 'stuck';
  }
}

/** The expert as a policy: commits to a side on its first look. */
export function makeExpert(rng) {
  let side = null;
  return (o) => expertChunk(o, (side ??= chooseSide(o, rng)));
}
