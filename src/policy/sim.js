// The Policy Lab world, expert and episode runner: a port of policy/sim.py (test/policy.test.js checks they agree).
// Rover frame: x forward, y left; metres, seconds, radians.

export const DT = 0.25, H = 8, EXEC = 4;
export const V_MAX = 1.2, W_MAX = 0.8, ROVER_R = 0.5;
export const GOAL_X = 16, GOAL_TOL = 1.5, SECONDS_PER_M = 2.5, AMBIGUOUS = 0.35;
export const FIELD_ROCKS = 3, SEEN = 2, FIELD_OBS = 3 * SEEN + 4, LOOK = 5;

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

// World = {rocks: [{x, y, r}], goalX, goalY, sigma}.

export const singleWorld = ({ rockX, rockY, rockR, goalY, sigma }) => ({ rocks: [{ x: rockX, y: rockY, r: rockR }], goalX: GOAL_X, goalY, sigma });

/** Rock field along the x axis, as policy/sim.py sample_fields; onLine puts every rock on the lane centre.
 *  The last slot is the empty marker (size 0, past the goal) that fills the view once fewer than SEEN rocks are left. */
export function fieldWorld(rng, sigma, onLine) {
  const rocks = [];
  let x = 4.5 + rng.uniform();
  for (let i = 0; i < FIELD_ROCKS; i++) {
    if (i) x += 9 + 1.5 * rng.uniform();
    rocks.push({ x, y: onLine ? 0 : (rng.uniform() - 0.5) * 2.4, r: 0.6 + 0.5 * rng.uniform() });
  }
  rocks.push({ x: x + 14, y: 0, r: 0 });
  return { rocks, goalX: x + 4, goalY: 0, sigma };
}

/** [dx, dy, r] of the next rocks not yet passed (noisy positions), then [goal dx, goal dy, sigma]; the rock field adds
 *  the heading before the goal. Single rock: [rock dx, rock dy, rock r, goal dx, goal dy, sigma]. */
export function observe(pose, w, rng) {
  const c = Math.cos(pose.th), s = Math.sin(pose.th);
  const rel = (px, py) => [(px - pose.x) * c + (py - pose.y) * s, -(px - pose.x) * s + (py - pose.y) * c];
  const n = w.rocks.length, seen = Math.min(n, SEEN);
  const first = Math.min(w.rocks.filter((k) => pose.x > k.x + k.r + ROVER_R).length, n - seen);
  const vis = w.rocks.slice(first, first + seen), at = vis.map((k) => rel(k.x, k.y));
  const nx = vis.map(() => rng.gauss() * w.sigma), ny = vis.map(() => rng.gauss() * w.sigma);
  const o = vis.flatMap((k, j) => [at[j][0] + nx[j], at[j][1] + ny[j], k.r]);
  if (n > SEEN) o.push(pose.th);
  return [...o, ...rel(w.goalX, w.goalY), w.sigma];
}

export const seenRocks = (o) => (o.length === FIELD_OBS ? SEEN : 1);

/** Signed distance of seen rock j from the straight line to the goal (rock field: from the lane centre); + = left. */
export function rockOffset(o, j = 0) {
  const L = o.length, gx = o[L - 3], gy = o[L - 2];
  if (seenRocks(o) > 1) { const th = o[L - 4]; return (o[3 * j] - gx) * Math.sin(th) + (o[3 * j + 1] - gy) * Math.cos(th); }
  const head = Math.atan2(gy, gx);
  return -o[3 * j] * Math.sin(head) + o[3 * j + 1] * Math.cos(head);
}

/** +1 = pass the nearest rock on its left, -1 = on its right; a coin flip when it is dead ahead. */
export function chooseSide(o, rng) {
  const off = rockOffset(o);
  return Math.abs(off) < AMBIGUOUS ? (rng.uniform() < 0.5 ? 1 : -1) : -Math.sign(off);
}

/** The expert's H actions: head for the goal (rock field: the lane centre LOOK m ahead), or follow the tangent of the
 *  nearest blocking rock grown by body radius and margin. Nearest rock on `side`, the next one away from the line. */
export function expertChunk(o, side) {
  const L = o.length, seen = seenRocks(o), [goalX, goalY, sigma] = o.slice(L - 3);
  const rocks = Array.from({ length: seen }, (_, j) => ({ x: o[3 * j], y: o[3 * j + 1], keepOut: o[3 * j + 2] + ROVER_R + 0.25 + 2 * sigma,
    side: j ? (rockOffset(o, j) >= 0 ? -1 : 1) : side }));
  const laneX = Math.cos(o[L - 4]), laneY = -Math.sin(o[L - 4]);
  let x = 0, y = 0, th = 0;
  const out = [];
  for (let i = 0; i < H; i++) {
    const c = Math.cos(th), s = Math.sin(th), rel = (px, py) => [(px - x) * c + (py - y) * s, -(px - x) * s + (py - y) * c];
    let aimX = goalX, aimY = goalY;
    if (seen > 1) { const back = Math.max(0, (goalX - x) * laneX + (goalY - y) * laneY - LOOK); aimX = goalX - back * laneX; aimY = goalY - back * laneY; }
    const [gx, gy] = rel(goalX, goalY), [ax, ay] = rel(aimX, aimY), toGoal = Math.atan2(ay, ax);
    let heading = toGoal, nearest = Infinity;
    for (const k of rocks) {
      const [rx, ry] = rel(k.x, k.y);
      const along = rx * Math.cos(toGoal) + ry * Math.sin(toGoal), across = -rx * Math.sin(toGoal) + ry * Math.cos(toGoal);
      if (along > 0 && Math.abs(across) < k.keepOut && along < Math.hypot(gx, gy) && along < nearest) {
        nearest = along;
        heading = Math.atan2(ry, rx) + k.side * Math.asin(Math.min(1, k.keepOut / Math.hypot(rx, ry)));
      }
    }
    const w = clamp(2 * heading, -W_MAX, W_MAX);
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
    this.path = [[0, 0]]; this.minGap = Infinity; this.passedY = world.rocks.map(() => null);   // per rock, + = passed on its left
    this.tMax = SECONDS_PER_M * world.goalX;
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
    w0.rocks.forEach((k, i) => { if (p.x < k.x && next.x >= k.x) this.passedY[i] = next.y - k.y; });
    this.pose = next; this.t += DT; this.path.push([next.x, next.y]);
    const gap = Math.min(...w0.rocks.map((k) => Math.hypot(next.x - k.x, next.y - k.y) - k.r - ROVER_R));
    this.minGap = Math.min(this.minGap, gap);
    if (gap < 0) this.status = 'crash';
    else if (Math.hypot(next.x - w0.goalX, next.y - w0.goalY) < GOAL_TOL) this.status = 'goal';
    else if (this.t >= this.tMax - 1e-9) this.status = 'stuck';
  }
}

/** The expert as a policy: commits to a side for each rock on its first look (rocks told apart by their size). */
export function makeExpert(rng) {
  let side = null, rock = null;
  return (o) => {
    if (o[2] !== rock) { side = chooseSide(o, rng); rock = o[2]; }
    return expertChunk(o, side);
  };
}
