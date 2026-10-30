// Lunar SLAM lab engine: a real EKF-SLAM you can watch, next to wheel odometry alone.
//
// The question SLAM answers: "Dropped somewhere with no GPS and no map, where am I,
// and what does the place look like?" The rover has two sensors:
//   - wheel encoders: say how far the wheels turned. They slip in loose regolith, so the
//     rover's own guess of where it is drifts further off with every metre;
//   - a stereo camera: measures range and bearing to rock features it can see.
// Each time step the filter does three things:
//   1. PREDICT  - move the pose guess by what the wheels said; the uncertainty grows.
//   2. OBSERVE  - find features in the camera image. A new one is added to the map at the
//                 position the rover believes it sees it; a known one is a check.
//   3. CORRECT  - compare where a known feature should appear with where it does appear
//                 (the innovation), and move the pose AND the map so they agree better.
//                 The uncertainty shrinks.
// Returning to features seen long ago is a loop closure: the old, well-known features
// pull the drifted pose back and, through the correlations kept in P, fix the map too.
//
// What is real here: the EKF-SLAM maths (state = rover pose + every landmark, one joint
// covariance P, Jacobians below; Thrun, Burgard & Fox, Probabilistic Robotics, ch. 10).
// The estimators never read the true pose or true landmark positions: they get only
// odometry and noisy range/bearing measurements, and the occupancy map is drawn from the
// pose the rover believes. Truth is used to simulate the sensors and to score the result.
// Simplifications (stated in the page): 2-D (x, y, heading), and the camera is told which
// landmark it sees (known data association; real systems must recognise features).

import { RNG } from '../core/rng.js';

export const SLAM_SCENARIOS = {
  loop_home: {
    key: 'loop_home',
    label: '1. Drive a loop and come home',
    sunElDeg: 20,
    wheelSlip: 0.06,
    pathMode: 'route',
    desc: 'The rover leaves the lander, drives a 150 m loop through the boulder field and comes back. Watch the uncertainty ellipse grow on the far side, then snap small when it sees the rocks near the lander again (a loop closure).',
  },
  low_sun: {
    key: 'low_sun',
    label: '2. Low sun (3°): shadows hide features',
    sunElDeg: 3,
    wheelSlip: 0.06,
    pathMode: 'route',
    desc: 'The same loop at a 3° sun, as at the lunar south pole. Long shadows hide many rock features, so the rover has fewer checks and leans on its wheels for longer.',
  },
  slippery: {
    key: 'slippery',
    label: '3. Slippery ground: wheels slip 15%',
    sunElDeg: 20,
    wheelSlip: 0.15,
    pathMode: 'route',
    desc: 'Loose regolith: the wheels report up to 15% more motion than really happened, and the heading creeps. Wheels alone get badly lost; SLAM keeps correcting.',
  },
  explore: {
    key: 'explore',
    label: '4. Explore on its own',
    sunElDeg: 12,
    wheelSlip: 0.06,
    pathMode: 'frontier',
    desc: 'No route: the rover drives to the edge of what its own map has seen (a frontier). When its uncertainty gets large it heads back to mapped ground to close a loop.',
  },
};

export const SLAM_ESTIMATORS = [
  {
    id: 'ekf_slam',
    name: 'SLAM (wheels + camera)',
    shortName: 'SLAM',
    color: '#38bdf8',
    plain: 'Predicts with the wheels, then corrects with every rock feature the camera recognises. Builds the landmark map at the same time.',
  },
  {
    id: 'wheels_only',
    name: 'Wheels only (no SLAM)',
    shortName: 'Wheels only',
    color: '#ef4444',
    plain: 'Adds up wheel motion and never checks it. It can still draw a map, but every mistake in its position goes into the map.',
  },
];

// Route for the 'route' scenarios, in metres (x east, y north). Starts and ends at the lander.
const ROUTE = [[0, 9], [-11, 17], [-15, 31], [-6, 40], [8, 39], [16, 28], [13, 15], [4, 5], [0, -1]];

const ROVER_R = 0.8; // half-width of the rover body (m): rocks must stay at least this far from its centre
// Hazard avoidance: rocks within `range` are sensed (hazard cameras); a turn is safe if the next `horizon` metres
// of its arc keep `margin` metres between the body and every rock.
const HAZARD = { range: 9, horizon: 3.0, margin: 0.3 };
const CAMERA = { hfov: (100 * Math.PI) / 180, minRange: 1.5, maxRange: 14, budget: 6, everySteps: 2 };

export function wrapAngle(a) {
  // atan2 form: never loops, even if a diverged filter hands it Infinity or NaN.
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/** Axes of the 2-sigma-style ellipse of a 2x2 covariance: semi-axes a >= b (at k sigma) and the angle of a. */
export function covEllipse(xx, xy, yy, k = 2) {
  const tr = (xx + yy) / 2;
  const d = Math.sqrt(Math.max(0, ((xx - yy) / 2) ** 2 + xy * xy));
  return {
    a: k * Math.sqrt(Math.max(0, tr + d)),
    b: k * Math.sqrt(Math.max(0, tr - d)),
    angle: 0.5 * Math.atan2(2 * xy, xx - yy),
  };
}

/**
 * EKF-SLAM with range-bearing landmarks (Probabilistic Robotics, Table 10.1, known correspondences).
 * State mu = [x, y, th, m1x, m1y, m2x, m2y, ...]; P is the full joint covariance (row-major n x n).
 * With no landmarks it is plain odometry dead reckoning with a growing 3x3 covariance.
 */
export class EkfSlam {
  constructor(x, y, th, sigmaPos = 0.02, sigmaTh = 0.01) {
    this.n = 3;
    this.mu = [x, y, th];
    this.P = new Float64Array(9);
    this.P[0] = this.P[4] = sigmaPos * sigmaPos;
    this.P[8] = sigmaTh * sigmaTh;
    this.index = new Map(); // landmark id -> offset of its x in mu
  }

  get(i, j) { return this.P[i * this.n + j]; }

  /** Motion step with odometry (v m/s, w rad/s) and its 1-sigma noise (sv, sw). */
  predict(v, w, dt, sv, sw) {
    const { mu, P, n } = this;
    const tm = mu[2] + (w * dt) / 2;
    const c = Math.cos(tm), s = Math.sin(tm);
    mu[0] += v * dt * c;
    mu[1] += v * dt * s;
    mu[2] = wrapAngle(mu[2] + w * dt);

    // G = d(pose')/d(pose): identity except the heading column.
    const g02 = -v * dt * s, g12 = v * dt * c;
    // Q = V M V^T, V = d(pose')/d(v, w).
    const V = [[dt * c, -0.5 * v * dt * dt * s], [dt * s, 0.5 * v * dt * dt * c], [0, dt]];
    const m0 = sv * sv, m1 = sw * sw;

    // Pose-landmark block: P_rm <- G P_rm (and its mirror).
    for (let j = 3; j < n; j++) {
      const p2 = P[2 * n + j];
      P[j] += g02 * p2;
      P[n + j] += g12 * p2;
      P[j * n] = P[j];
      P[j * n + 1] = P[n + j];
    }
    // Pose block: P_rr <- G P_rr G^T + Q.
    const a = [[P[0], P[1], P[2]], [P[n], P[n + 1], P[n + 2]], [P[2 * n], P[2 * n + 1], P[2 * n + 2]]];
    const G = [[1, 0, g02], [0, 1, g12], [0, 0, 1]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let acc = 0;
        for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) acc += G[i][k] * a[k][l] * G[j][l];
        acc += V[i][0] * V[j][0] * m0 + V[i][1] * V[j][1] * m1;
        P[i * n + j] = acc;
      }
    }
  }

  /** Where landmark `id` should appear from the current pose: range, bearing, and the Jacobian pieces. */
  expected(id) {
    const j = this.index.get(id);
    const dx = this.mu[j] - this.mu[0], dy = this.mu[j + 1] - this.mu[1];
    const q = Math.max(dx * dx + dy * dy, 1e-9), r = Math.sqrt(q);
    return { j, dx, dy, q, r, b: wrapAngle(Math.atan2(dy, dx) - this.mu[2]) };
  }

  /** Correct with a range-bearing measurement of a known landmark. Returns the innovation and its size. */
  update(id, rMeas, bMeas, sr, sb) {
    const { mu, P, n } = this;
    const e = this.expected(id);
    const { j, dx, dy, q, r } = e;
    // H is zero except columns 0,1,2 (pose) and j,j+1 (this landmark).
    const cols = [0, 1, 2, j, j + 1];
    const H = [
      [-dx / r, -dy / r, 0, dx / r, dy / r],
      [dy / q, -dx / q, -1, -dy / q, dx / q],
    ];
    // PHt = P H^T (n x 2)
    const PHt = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      let s0 = 0, s1 = 0;
      for (let k = 0; k < 5; k++) {
        const p = P[i * n + cols[k]];
        s0 += p * H[0][k];
        s1 += p * H[1][k];
      }
      PHt[2 * i] = s0;
      PHt[2 * i + 1] = s1;
    }
    // S = H P H^T + R
    let s00 = sr * sr, s01 = 0, s11 = sb * sb;
    for (let k = 0; k < 5; k++) {
      const c = cols[k];
      s00 += H[0][k] * PHt[2 * c];
      s01 += H[0][k] * PHt[2 * c + 1];
      s11 += H[1][k] * PHt[2 * c + 1];
    }
    const det = s00 * s11 - s01 * s01;
    const i00 = s11 / det, i01 = -s01 / det, i11 = s00 / det;
    const nu0 = rMeas - r, nu1 = wrapAngle(bMeas - e.b);
    // K = PHt S^-1; mu += K nu; P -= K PHt^T
    const K = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const a0 = PHt[2 * i], a1 = PHt[2 * i + 1];
      K[2 * i] = a0 * i00 + a1 * i01;
      K[2 * i + 1] = a0 * i01 + a1 * i11;
      mu[i] += K[2 * i] * nu0 + K[2 * i + 1] * nu1;
    }
    mu[2] = wrapAngle(mu[2]);
    for (let i = 0; i < n; i++) {
      const k0 = K[2 * i], k1 = K[2 * i + 1];
      for (let m = i; m < n; m++) {
        const v = P[i * n + m] - (k0 * PHt[2 * m] + k1 * PHt[2 * m + 1]);
        P[i * n + m] = v;
        P[m * n + i] = v;
      }
    }
    return {
      expectedRange: r,
      expectedBearing: e.b,
      innovRange: nu0,
      innovBearing: nu1,
      // Mahalanobis size of the surprise (about 2 on average when the filter is honest).
      nis: nu0 * (i00 * nu0 + i01 * nu1) + nu1 * (i01 * nu0 + i11 * nu1),
    };
  }

  /** Put a newly seen landmark on the map where the current pose says it is; grow mu and P. */
  add(id, rMeas, bMeas, sr, sb) {
    const { mu, P, n } = this;
    const a = mu[2] + bMeas, c = Math.cos(a), s = Math.sin(a);
    const Gr = [[1, 0, -rMeas * s], [0, 1, rMeas * c]];
    const Gz = [[c, -rMeas * s], [s, rMeas * c]];
    const n2 = n + 2;
    const P2 = new Float64Array(n2 * n2);
    for (let i = 0; i < n; i++) P2.set(P.subarray(i * n, i * n + n), i * n2);
    // Cross terms: P_new,x = Gr P_pose,x
    for (let k = 0; k < n; k++) {
      for (let row = 0; row < 2; row++) {
        const v = Gr[row][0] * P[k] + Gr[row][1] * P[n + k] + Gr[row][2] * P[2 * n + k];
        P2[(n + row) * n2 + k] = v;
        P2[k * n2 + n + row] = v;
      }
    }
    // New block: Gr P_rr Gr^T + Gz R Gz^T
    const R = [sr * sr, sb * sb];
    for (let r0 = 0; r0 < 2; r0++) {
      for (let r1 = 0; r1 < 2; r1++) {
        let acc = 0;
        for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) acc += Gr[r0][k] * P[k * n + l] * Gr[r1][l];
        acc += Gz[r0][0] * R[0] * Gz[r1][0] + Gz[r0][1] * R[1] * Gz[r1][1];
        P2[(n + r0) * n2 + n + r1] = acc;
      }
    }
    mu.push(mu[0] + rMeas * c, mu[1] + rMeas * s);
    this.P = P2;
    this.n = n2;
    this.index.set(id, n);
  }

  poseCov() {
    const n = this.n, P = this.P;
    return { xx: P[0], xy: P[1], yy: P[n + 1], thth: P[2 * n + 2] };
  }

  landmarkCov(id) {
    const j = this.index.get(id), n = this.n, P = this.P;
    return { xx: P[j * n + j], xy: P[j * n + j + 1], yy: P[(j + 1) * n + j + 1] };
  }
}

/**
 * Umeyama (1991) 3-D similarity / rigid alignment:
 * Finds (s, R, t) minimizing sum_i || dst_i - (s * R * src_i + t) ||^2.
 * Used strictly for computing published paper benchmark metrics (ATE_Sim3 / ATE_SE3),
 * NEVER for steering or modifying the live SLAM estimators.
 */
export function umeyamaAlign3D(srcPts, dstPts, allowScale = true) {
  const n = Math.min(srcPts.length, dstPts.length);
  if (n === 0) {
    return {
      s: 1.0,
      R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      t: [0, 0, 0],
      aligned: [],
    };
  }
  if (n === 1) {
    const t = [
      dstPts[0][0] - srcPts[0][0],
      dstPts[0][1] - srcPts[0][1],
      dstPts[0][2] - srcPts[0][2],
    ];
    return {
      s: 1.0,
      R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      t,
      aligned: [[dstPts[0][0], dstPts[0][1], dstPts[0][2]]],
    };
  }

  let mx = 0, my = 0, mz = 0;
  let dx = 0, dy = 0, dz = 0;
  for (let i = 0; i < n; i++) {
    mx += srcPts[i][0]; my += srcPts[i][1]; mz += srcPts[i][2];
    dx += dstPts[i][0]; dy += dstPts[i][1]; dz += dstPts[i][2];
  }
  mx /= n; my /= n; mz /= n;
  dx /= n; dy /= n; dz /= n;

  let sxx = 0, sxz = 0, szx = 0, szz = 0, syy = 0;
  let srcVar = 0;
  for (let i = 0; i < n; i++) {
    const sx = srcPts[i][0] - mx;
    const sy = srcPts[i][1] - my;
    const sz = srcPts[i][2] - mz;
    const tx = dstPts[i][0] - dx;
    const ty = dstPts[i][1] - dy;
    const tz = dstPts[i][2] - dz;
    srcVar += sx * sx + sy * sy + sz * sz;
    sxx += sx * tx;
    sxz += sx * tz;
    szx += sz * tx;
    szz += sz * tz;
    syy += sy * ty;
  }
  srcVar /= n;
  sxx /= n; sxz /= n; szx /= n; szz /= n; syy /= n;

  const num = sxz - szx;
  const den = sxx + szz;
  const theta = Math.atan2(num, den);
  const c = Math.cos(theta), sn = Math.sin(theta);

  const R = [
    [c, 0, -sn],
    [0, 1, 0],
    [sn, 0, c],
  ];

  const trRS = c * (sxx + szz) + sn * (sxz - szx) + syy;
  let s = 1.0;
  if (allowScale && srcVar > 1e-8) {
    s = Math.max(0.25, Math.min(4.0, trRS / srcVar));
  }

  const tx = dx - s * (R[0][0] * mx + R[0][2] * mz);
  const ty = dy - s * my;
  const tz = dz - s * (R[2][0] * mx + R[2][2] * mz);

  const aligned = srcPts.map((p) => [
    s * (R[0][0] * p[0] + R[0][2] * p[2]) + tx,
    s * p[1] + ty,
    s * (R[2][0] * p[0] + R[2][2] * p[2]) + tz,
  ]);

  return { s, R, t: [tx, ty, tz], aligned };
}

/**
 * Compute Sturm (2012) ATE RMSE and Geiger (2012) KITTI RPE (% drift per 10 m segment),
 * plus raw unaligned live pose error.
 */
export function evaluateTrajectoryMetrics(truthKeyframes, estKeyframes, allowScale = true, segM = 10.0) {
  const n = Math.min(truthKeyframes.length, estKeyframes.length);
  if (n < 2) {
    return {
      ateRmse: 0,
      atePct: 0,
      ateMax: 0,
      rpePct: 0,
      rawPoseErr: 0,
      scaleEst: 1.0,
      aligned: estKeyframes.map((k) => [...k.pos]),
    };
  }

  const src = estKeyframes.slice(0, n).map((k) => k.pos);
  const dst = truthKeyframes.slice(0, n).map((k) => k.pos);
  const { s, aligned } = umeyamaAlign3D(src, dst, allowScale);

  let sumSq = 0, maxErr = 0;
  const cumDist = [0];
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      const dx = dst[i][0] - dst[i - 1][0];
      const dy = dst[i][1] - dst[i - 1][1];
      const dz = dst[i][2] - dst[i - 1][2];
      cumDist.push(cumDist[i - 1] + Math.hypot(dx, dy, dz));
    }
    const ex = aligned[i][0] - dst[i][0];
    const ey = aligned[i][1] - dst[i][1];
    const ez = aligned[i][2] - dst[i][2];
    const e = Math.hypot(ex, ey, ez);
    sumSq += e * e;
    if (e > maxErr) maxErr = e;
  }

  const lastSrc = src[n - 1];
  const lastDst = dst[n - 1];
  const rawPoseErr = Math.hypot(lastSrc[0] - lastDst[0], lastSrc[2] - lastDst[2]);

  const totalDist = Math.max(cumDist[cumDist.length - 1], 0.1);
  const ateRmse = Math.sqrt(sumSq / n);
  const atePct = (ateRmse / totalDist) * 100.0;

  const rpeErrors = [];
  for (let a = 0; a < n; a++) {
    const targetD = cumDist[a] + segM;
    let b = a + 1;
    while (b < n && cumDist[b] < targetD) b++;
    if (b >= n) break;
    const dTrue = cumDist[b] - cumDist[a];
    if (dTrue < 1.0) continue;
    const dEstX = src[b][0] - src[a][0];
    const dEstY = src[b][1] - src[a][1];
    const dEstZ = src[b][2] - src[a][2];
    const dTrueX = dst[b][0] - dst[a][0];
    const dTrueY = dst[b][1] - dst[a][1];
    const dTrueZ = dst[b][2] - dst[a][2];
    const drift = Math.hypot(dEstX - dTrueX, dEstY - dTrueY, dEstZ - dTrueZ) / dTrue;
    rpeErrors.push(drift * 100.0);
  }
  const rpePct = rpeErrors.length
    ? rpeErrors.reduce((acc, v) => acc + v, 0) / rpeErrors.length
    : atePct * 0.9;

  return {
    ateRmse,
    atePct,
    ateMax: maxErr,
    rpePct,
    rawPoseErr,
    scaleEst: s,
    totalDist,
    aligned,
  };
}

function makeGrid() {
  // 2-D log-odds occupancy grid: x -30..30 m, y -10..50 m, 1.25 m cells.
  return { cols: 48, rows: 48, cellSize: 1.25, xMin: -30, yMin: -10, logOdds: new Float32Array(48 * 48), exploredCount: 0, coveragePct: 0 };
}

/**
 * One SLAM lab run. Sim coordinates: (x, y) metres on the ground; Three.js maps (x, y, up) -> (x, up, -y).
 * `rover` is the truth (only the simulator and the scoring read it); `estimators[i].filter` is what the rover believes.
 */
export class LunarSlamSession {
  constructor(opts = {}) {
    this.scenarioKey = SLAM_SCENARIOS[opts.scenario] ? opts.scenario : 'loop_home';
    const sc = SLAM_SCENARIOS[this.scenarioKey];
    this.seed = opts.seed ?? 3;
    this.rng = new RNG(this.seed * 9973 + 17); // world layout
    this.noise = new RNG(this.seed * 7919 + 5); // sensor and wheel noise

    this.sunElDeg = opts.sunElDeg ?? sc.sunElDeg;
    this.sunAzDeg = opts.sunAzDeg ?? 22;
    this.wheelSlip = opts.wheelSlip ?? sc.wheelSlip;
    this.cameraNoise = opts.cameraNoise ?? 1.0; // multiplies the camera's range and bearing noise
    this.useCamera = opts.useCamera ?? true; // false = the SLAM filter gets no corrections
    this.pathMode = sc.pathMode;
    this.slipPhase = this.noise.next() * Math.PI * 2;
    this.yawPhase = this.noise.next() * Math.PI * 2;

    this.t = 0;
    this.dt = 0.1;
    this.stepCount = 0;
    this.distanceDriven = 0;
    this.bumps = 0; // time steps the body was pressed against a rock
    this.lastBumpT = -99;

    this.rover = { x: 0, y: -1, z: 0, th: Math.PI / 2, v: 0, w: 0 };
    this.manualControl = { v: 0, w: 0, active: false };
    this.waypoint = null;
    this.routeIdx = 0;
    this.lap = 1;
    this.frontierTarget = this.pathMode === 'frontier' ? { x: 0, y: 12, mode: 'explore' } : null;
    this.frontierHoldSteps = 0;

    this.boulders = [];
    this.landmarks = [];
    this._buildLunarField();

    this.truthPath = [];
    this.truthKeyframes = [];
    this.loopClosures = [];
    this.events = [];
    this.mappedPointCloud = [];
    this.maxPointCloud = 1800;
    this.activeObservations = [];
    this.lastCamera = null; // what the last camera frame did, for the "SLAM loop" panel
    this.lastPredict = null;

    this.estimators = SLAM_ESTIMATORS.map((meta) => ({
      ...meta,
      filter: new EkfSlam(this.rover.x, this.rover.y, this.rover.th),
      grid: makeGrid(),
      pose: { x: this.rover.x, y: this.rover.y, th: this.rover.th },
      poseCov: { xx: 0, xy: 0, yy: 0, thth: 0 },
      mapLandmarks: new Map(),
      keyframes: [],
      rawPath: [],
      trackingStatus: '',
      metrics: { rawPoseErr: 0, poseSigmaM: 0, insideRate: 1, ateRmse: 0, atePct: 0, mapRmse: 0, mappedLandmarks: 0, loopCount: 0 },
      nees: { inside: 0, total: 0 },
      ateHistory: [],
      sigmaHistory: [],
    }));
    // The map the page shows is the one SLAM builds.
    this.grid = this.estimators[0].grid;

    this._syncEstimators();
    this._camera();
    this._mapFromBeliefs(this._castRays());
    this._recordKeyframe();
    this.pushEvent('Landed. No map yet, no GPS: the rover only knows it starts here.', 'good');
  }

  get slam() { return this.estimators[0]; }

  _buildLunarField() {
    const rockCoords = [
      [-8, 5, 1.8, 2.2], [8, 6, 1.6, 2.0], [-16, 12, 2.3, 2.7], [15, 13, 2.1, 2.5],
      [0, 14, 1.8, 2.2], [-10, 22, 2.0, 2.4], [10, 21, 1.9, 2.3], [-20, 26, 2.5, 2.9],
      [19, 27, 2.2, 2.6], [-4, 29, 1.7, 2.1], [5, 31, 1.7, 2.1], [-13, 36, 2.2, 2.5],
      [13, 36, 2.0, 2.4], [0, 41, 2.4, 2.8], [-21, 8, 1.9, 2.2], [21, 9, 1.8, 2.1],
      [-6, -3, 1.3, 1.5], [7, -2, 1.4, 1.6], [-18, -3, 2.0, 2.3], [18, -2, 1.9, 2.2],
      [-23, 19, 2.4, 2.8], [23, 19, 2.3, 2.7], [-8, 44, 1.9, 2.2], [9, 44, 1.8, 2.1],
    ];
    this.boulders = rockCoords.map(([x, y, r, h], idx) => ({
      id: idx + 1,
      x: x + (this.rng.next() - 0.5) * 1.2,
      y: y + (this.rng.next() - 0.5) * 1.2,
      r,
      h,
      discovered: false,
    }));
    // Landmarks: distinct features on each boulder's surface plus a few small rocks on the ground.
    let id = 1;
    for (const b of this.boulders) {
      for (let k = 0; k < 2; k++) {
        const ang = (k / 2) * Math.PI * 2 + this.rng.next() * 0.8;
        this.landmarks.push({
          id: id++, x: b.x + Math.cos(ang) * b.r * 0.95, y: b.y + Math.sin(ang) * b.r * 0.95,
          z: b.h * (0.3 + 0.5 * this.rng.next()), boulderId: b.id, kind: 'rock_corner', discovered: false,
        });
      }
    }
    while (this.landmarks.length < 62) {
      const x = (this.rng.next() - 0.5) * 52, y = -6 + this.rng.next() * 52;
      if (this.boulders.some((b) => Math.hypot(x - b.x, y - b.y) < b.r + 1)) continue;
      this.landmarks.push({ id: id++, x, y, z: 0.15, boulderId: null, kind: 'pebble', discovered: false });
    }
  }

  pushEvent(msg, level = 'info') {
    this.events.unshift({ t: this.t, msg, level });
    if (this.events.length > 60) this.events.pop();
  }

  /** Is a ground point inside a boulder's shadow for the current sun? */
  isPointInPolarShadow(x, y, z = 0.15) {
    const tanEl = Math.tan((Math.max(0.5, this.sunElDeg) * Math.PI) / 180);
    const az = (this.sunAzDeg * Math.PI) / 180, ux = Math.cos(az), uy = Math.sin(az);
    for (const b of this.boulders) {
      if (b.h <= z) continue;
      const dx = x - b.x, dy = y - b.y;
      const along = -(dx * ux + dy * uy);
      if (along > 0 && along < (b.h - z) / Math.max(tanEl, 0.015) && Math.abs(-dx * uy + dy * ux) < b.r * 1.05) return true;
    }
    return false;
  }

  /** Chance the camera finds a feature. Lab assumption: shadowed features get hard to find as the sun drops. */
  detectProb(inShadow) {
    if (!inShadow) return 0.95;
    return 0.95 * Math.min(1, Math.max(0.1, (this.sunElDeg - 1) / 14));
  }

  _lineOfSight(lm) {
    const ox = this.rover.x, oy = this.rover.y, dx = lm.x - ox, dy = lm.y - oy, L2 = dx * dx + dy * dy;
    for (const b of this.boulders) {
      if (b.id === lm.boulderId) {
        // Only the side of its own rock that faces the rover is visible.
        if ((lm.x - b.x) * (ox - lm.x) + (lm.y - b.y) * (oy - lm.y) < 0) return false;
        continue;
      }
      const t = Math.max(0, Math.min(1, ((b.x - ox) * dx + (b.y - oy) * dy) / L2));
      if (Math.hypot(ox + t * dx - b.x, oy + t * dy - b.y) < b.r * 0.9 && t < 0.98) return false;
    }
    return true;
  }

  /** Camera noise (1 sigma) for a feature at range r: stereo depth error grows with r^2. */
  cameraSigma(r) {
    return { sr: this.cameraNoise * (0.03 + 0.0024 * r * r), sb: this.cameraNoise * 0.008 };
  }

  // ---------------------------------------------------------------- driving

  _beliefPose() {
    return this.slam.pose;
  }

  /** Steering. The rover steers by where it BELIEVES it is; rocks right in front are avoided from live sensing. */
  _autopilotCommand() {
    if (this.manualControl.active) return { v: this.manualControl.v, w: this.manualControl.w };
    const me = this._beliefPose();
    if (this.waypoint) {
      if (Math.hypot(this.waypoint.x - me.x, this.waypoint.y - me.y) < 1.3) {
        this.pushEvent(`Reached the waypoint (${this.waypoint.x.toFixed(1)}, ${this.waypoint.y.toFixed(1)}) m, as far as the rover knows.`, 'good');
        this.waypoint = null;
      } else return this._steer(this.waypoint.x, this.waypoint.y);
    }
    if (this.pathMode === 'route') {
      const [tx, ty] = ROUTE[this.routeIdx];
      if (Math.hypot(tx - me.x, ty - me.y) < 2.5) {
        this.routeIdx = (this.routeIdx + 1) % ROUTE.length;
        if (this.routeIdx === 0) {
          const err = Math.hypot(this.rover.x - me.x, this.rover.y - me.y);
          this.pushEvent(`Lap ${this.lap} done. SLAM thinks it is home; it is really ${err.toFixed(2)} m away.`, 'good');
          this.lap++;
        }
      }
      const [nx, ny] = ROUTE[this.routeIdx];
      this.frontierTarget = { x: nx, y: ny, mode: 'route' };
      return this._steer(nx, ny);
    }
    if (this.pathMode === 'frontier') {
      this.frontierHoldSteps++;
      if (Math.hypot(this.frontierTarget.x - me.x, this.frontierTarget.y - me.y) < 2.8 || this.frontierHoldSteps > 120) {
        this._pickFrontierTarget();
      }
      return this._steer(this.frontierTarget.x, this.frontierTarget.y);
    }
    return { v: 0, w: 0 };
  }

  /** Frontier exploration on the rover's own map: go to known-free ground next to unknown ground. */
  _pickFrontierTarget() {
    const me = this._beliefPose();
    const sigma = this.slam.metrics.poseSigmaM;
    const kfs = this.slam.keyframes;
    if (sigma > 0.8 && kfs.length > 30 && this.frontierTarget.mode !== 'homing') {
      const anchor = kfs[Math.floor(kfs.length * 0.15)].pos;
      this.frontierTarget = { x: anchor[0], y: -anchor[2], mode: 'homing' };
      this.frontierHoldSteps = 0;
      this.pushEvent(`Uncertainty is ±${(2 * sigma).toFixed(1)} m: heading back to mapped ground to close a loop.`, 'info');
      return;
    }
    const g = this.grid;
    let best = null;
    for (let r = 3; r < g.rows - 3; r += 2) {
      for (let c = 3; c < g.cols - 3; c += 2) {
        if (g.logOdds[r * g.cols + c] > -0.25) continue;
        let unknown = 0, blocked = false;
        for (let dr = -2; dr <= 2; dr++) {
          for (let dc = -2; dc <= 2; dc++) {
            const lo = g.logOdds[(r + dr) * g.cols + c + dc];
            if (Math.abs(lo) < 0.15) unknown++;
            if (lo > 0.45) blocked = true;
          }
        }
        if (unknown < 3 || blocked) continue;
        const fx = g.xMin + (c + 0.5) * g.cellSize, fy = g.yMin + (r + 0.5) * g.cellSize;
        const d = Math.hypot(fx - me.x, fy - me.y);
        if (d < 6 || d > 28) continue;
        const turn = Math.abs(wrapAngle(Math.atan2(fy - me.y, fx - me.x) - me.th));
        const score = unknown * 2.2 - d * 0.35 - turn * 2.4;
        if (!best || score > best.score) best = { x: fx, y: fy, score };
      }
    }
    const fallback = ROUTE[Math.floor(this.stepCount / 120) % ROUTE.length];
    this.frontierTarget = best ? { x: best.x, y: best.y, mode: 'explore' } : { x: fallback[0], y: fallback[1], mode: 'explore' };
    this.frontierHoldSteps = 0;
  }

  _steer(targetX, targetY) {
    const me = this._beliefPose();
    const err = wrapAngle(Math.atan2(targetY - me.y, targetX - me.x) - me.th);
    return this._avoid(Math.max(0.25, Math.cos(err * 0.75)), Math.max(-0.58, Math.min(0.58, err * 1.75)));
  }

  /** Smallest gap (m) between the rover body and any nearby rock along the next HAZARD.horizon metres of an arc. */
  _arcGap(v, w, near) {
    let { x, y, th } = this.rover, gap = Infinity;
    const ds = 0.25, k = w / Math.max(v, 0.1);
    for (let s = ds; s <= HAZARD.horizon + 1e-9; s += ds) {
      th += k * ds;
      x += Math.cos(th) * ds;
      y += Math.sin(th) * ds;
      for (const b of near) gap = Math.min(gap, Math.hypot(x - b.x, y - b.y) - b.r - ROVER_R);
    }
    return gap;
  }

  /**
   * Hazard avoidance from live sensing (rock positions relative to the rover, as hazard cameras measure them;
   * the same in any map frame, so no belief error enters here). Keep the wanted turn if its arc is clear; else the
   * clear turn nearest to it, a little slower; if none is clear, stop and turn on the spot toward the roomier side.
   */
  _avoid(v, w) {
    const near = this.boulders.filter((b) => Math.hypot(b.x - this.rover.x, b.y - this.rover.y) < HAZARD.range + b.r);
    if (!near.length || this._arcGap(v, w, near) > HAZARD.margin) return { v, w };
    let best = null;
    for (let i = 0; i <= 12; i++) {
      const ww = -0.58 + (i * 1.16) / 12;
      if (this._arcGap(v, ww, near) > HAZARD.margin && (!best || Math.abs(ww - w) < Math.abs(best - w))) best = ww;
    }
    if (best !== null) return { v: 0.7 * v, w: best };
    return { v: 0, w: this._arcGap(0.5, 0.58, near) >= this._arcGap(0.5, -0.58, near) ? 0.58 : -0.58 };
  }

  // ---------------------------------------------------------------- one time step

  step() {
    this.t = +(this.t + this.dt).toFixed(2);
    this.stepCount++;

    const cmd = this._autopilotCommand();
    this.rover.v = cmd.v;
    this.rover.w = cmd.w;

    // Truth moves (the rover really goes where the command takes it) - unless a rock is in the way: then the
    // body stops against it (the wheels still turn, so odometry still counts the motion, as a stuck wheel would).
    const tm = this.rover.th + (cmd.w * this.dt) / 2;
    const nx = this.rover.x + cmd.v * Math.cos(tm) * this.dt, ny = this.rover.y + cmd.v * Math.sin(tm) * this.dt;
    const hit = this.boulders.find((b) => Math.hypot(nx - b.x, ny - b.y) < b.r + ROVER_R &&
      Math.hypot(nx - b.x, ny - b.y) < Math.hypot(this.rover.x - b.x, this.rover.y - b.y));
    if (hit) {
      this.bumps++;
      if (this.t - this.lastBumpT > 3) this.pushEvent(`Bumped into rock ${hit.id} and stopped.`, 'warn');
      this.lastBumpT = this.t;
    } else {
      this.rover.x = nx;
      this.rover.y = ny;
      this.distanceDriven += Math.abs(cmd.v) * this.dt;
    }
    this.rover.th = wrapAngle(this.rover.th + cmd.w * this.dt);
    this.truthPath.push([this.rover.x, this.rover.z, -this.rover.y]);
    if (this.truthPath.length > 4000) this.truthPath.shift();

    // What the wheels report: slip makes them over-count, and the heading creeps. Slow-varying, so it adds up.
    const slipNow = this.wheelSlip * (0.5 + 0.5 * Math.sin(0.045 * this.t + this.slipPhase));
    const yawBias = 0.12 * this.wheelSlip * Math.sin(0.03 * this.t + this.yawPhase);
    const odoV = cmd.v * (1 + slipNow) + this.noise.randn() * (0.01 + 0.3 * this.wheelSlip * Math.abs(cmd.v));
    const odoW = cmd.w + yawBias + this.noise.randn() * 0.01;
    // How much the filter is told to distrust the wheels (1 sigma).
    const sv = 0.02 + 4.0 * this.wheelSlip * Math.abs(odoV);
    const sw = 0.015 + 0.5 * this.wheelSlip;

    const before = this.slam.filter.poseCov();
    for (const est of this.estimators) est.filter.predict(odoV, odoW, this.dt, sv, sw);
    const after = this.slam.filter.poseCov();
    this.lastPredict = {
      dist: odoV * this.dt,
      turnDeg: (odoW * this.dt * 180) / Math.PI,
      sigmaBefore: Math.sqrt(before.xx + before.yy),
      sigmaAfter: Math.sqrt(after.xx + after.yy),
    };

    if (this.stepCount % CAMERA.everySteps === 0) {
      this._camera();
      this._mapFromBeliefs(this._castRays());
    }
    this._syncEstimators();
    if (this.stepCount % 8 === 0) this._recordKeyframe();
  }

  /** One camera frame: find features, add new ones to the map, correct with known ones. */
  _camera() {
    const seen = [];
    let missed = 0, shadowed = 0, inView = 0;
    for (const lm of this.landmarks) {
      const dx = lm.x - this.rover.x, dy = lm.y - this.rover.y, r = Math.hypot(dx, dy);
      if (r < CAMERA.minRange || r > CAMERA.maxRange) continue;
      const bearing = wrapAngle(Math.atan2(dy, dx) - this.rover.th);
      if (Math.abs(bearing) > CAMERA.hfov / 2 || !this._lineOfSight(lm)) continue;
      inView++;
      const inShadow = this.isPointInPolarShadow(lm.x, lm.y, lm.z);
      if (inShadow) shadowed++;
      if (this.noise.next() > this.detectProb(inShadow)) {
        missed++;
        seen.push({ lm, rng: r, bearing, inShadow, status: 'missed' });
        continue;
      }
      seen.push({ lm, rng: r, bearing, inShadow, status: 'seen' });
    }
    // Feature budget: use the nearest ones (better depth), like a real front end keeps its best tracks.
    const usable = seen.filter((o) => o.status === 'seen').sort((a, b) => a.rng - b.rng).slice(0, CAMERA.budget);
    const usedIds = new Set(usable.map((o) => o.lm.id));
    for (const o of seen) if (o.status === 'seen' && !usedIds.has(o.lm.id)) o.status = 'skipped';

    const est = this.slam, f = est.filter;
    const cov0 = f.poseCov(), sigma0 = Math.sqrt(cov0.xx + cov0.yy);
    const pose0 = [f.mu[0], f.mu[1]];
    let added = 0, corrected = 0, biggest = null, oldestSeen = null;
    for (const o of usable) {
      const { sr, sb } = this.cameraSigma(o.rng);
      const rMeas = o.rng + this.noise.randn() * sr;
      const bMeas = wrapAngle(o.bearing + this.noise.randn() * sb);
      o.lm.discovered = true;
      if (o.lm.boulderId) this.boulders[o.lm.boulderId - 1].discovered = true;
      if (!this.useCamera) { o.status = 'ignored'; continue; }
      const meta = est.mapLandmarks.get(o.lm.id);
      if (!meta) {
        f.add(o.lm.id, rMeas, bMeas, sr, sb);
        est.mapLandmarks.set(o.lm.id, { id: o.lm.id, z: o.lm.z, obsCount: 1, firstT: this.t, lastT: this.t, firstKf: est.keyframes.length - 1 });
        o.status = 'new';
        added++;
      } else {
        const gap = this.t - meta.lastT;
        if (gap > 25 && (!oldestSeen || gap > oldestSeen.gap)) oldestSeen = { gap, meta };
        const u = f.update(o.lm.id, rMeas, bMeas, sr, sb);
        meta.obsCount++;
        meta.lastT = this.t;
        o.status = 'known';
        corrected++;
        const size = Math.abs(u.innovRange) / sr + Math.abs(u.innovBearing) / sb;
        if (!biggest || size > biggest.size) biggest = { size, id: o.lm.id, measRange: rMeas, measBearing: bMeas, ...u };
      }
    }
    const cov1 = f.poseCov(), sigma1 = Math.sqrt(cov1.xx + cov1.yy);
    this.activeObservations = seen;
    this.cameraTelemetry = { inView, seen: usable.length, missed, added, corrected, shadowFraction: inView ? shadowed / inView : 0 };
    this.lastCamera = {
      t: this.t, inView, used: usable.length, missed, added, corrected, biggest,
      sigmaBefore: Math.sqrt(cov0.xx + cov0.yy), sigmaAfter: sigma1,
      moved: Math.hypot(f.mu[0] - pose0[0], f.mu[1] - pose0[1]),
      cameraOff: !this.useCamera,
    };

    // Loop closure: a feature not seen for a long time is recognised, and the pose snaps back.
    if (oldestSeen && sigma0 > 0.25 && sigma1 < 0.75 * sigma0) {
      const last = this.loopClosures[this.loopClosures.length - 1];
      if (!last || this.t - last.t > 8) {
        const from = est.keyframes[Math.max(0, oldestSeen.meta.firstKf)] || est.keyframes[0];
        this.loopClosures.push({
          t: this.t, lmId: oldestSeen.meta.id, gapS: oldestSeen.gap, sigmaBefore: sigma0, sigmaAfter: sigma1,
          fromPos: [...from.pos], toPos: [f.mu[0], 0, -f.mu[1]], fromId: from.id, toId: this.truthKeyframes.length,
        });
        est.metrics.loopCount = this.loopClosures.length;
        this.pushEvent(
          `Loop closure: recognised landmark #${oldestSeen.meta.id}, last seen ${oldestSeen.gap.toFixed(0)} s ago. ` +
          `Position uncertainty ±${(2 * sigma0).toFixed(2)} m → ±${(2 * sigma1).toFixed(2)} m; the rover moved ${this.lastCamera.moved.toFixed(2)} m.`,
          'good'
        );
      }
    }
    if (added && est.mapLandmarks.size === added) this.pushEvent(`First landmarks on the map: ${added} rock features.`, 'info');
  }

  /** Cast 41 sensor rays from the TRUE rover (that is what a range sensor physically measures). */
  _castRays() {
    const rays = [];
    for (let i = 0; i < 41; i++) {
      const rel = (i / 40 - 0.5) * CAMERA.hfov;
      const a = this.rover.th + rel, c = Math.cos(a), s = Math.sin(a);
      let hit = null, dist = 21;
      for (let d = 1.0; d <= 21; d += 0.65) {
        const x = this.rover.x + c * d, y = this.rover.y + s * d;
        const b = this.boulders.find((bb) => Math.hypot(x - bb.x, y - bb.y) <= bb.r * 0.98);
        if (b) { hit = b; dist = d; break; }
      }
      rays.push({ rel, dist, hit });
    }
    return rays;
  }

  /** Draw each estimator's map from the pose IT believes. A wrong pose puts rocks in the wrong place. */
  _mapFromBeliefs(rays) {
    for (const est of this.estimators) {
      const g = est.grid, p = est.filter.mu;
      for (const ray of rays) {
        const a = p[2] + ray.rel, c = Math.cos(a), s = Math.sin(a);
        for (let d = 1.0; d <= ray.dist + 1e-6; d += 0.65) {
          const col = Math.floor((p[0] + c * d - g.xMin) / g.cellSize);
          const row = Math.floor((p[1] + s * d - g.yMin) / g.cellSize);
          if (col < 0 || col >= g.cols || row < 0 || row >= g.rows) break;
          const k = row * g.cols + col;
          if (ray.hit && d + 0.65 > ray.dist) g.logOdds[k] = Math.min(3.5, g.logOdds[k] + 0.85);
          else g.logOdds[k] = Math.max(-2.5, g.logOdds[k] - 0.28);
        }
      }
      let explored = 0;
      for (let i = 0; i < g.logOdds.length; i++) if (Math.abs(g.logOdds[i]) > 0.2) explored++;
      g.exploredCount = explored;
      g.coveragePct = (explored / g.logOdds.length) * 100;
    }
    // 3-D points on rock faces, placed from SLAM's believed pose.
    if (this.stepCount % 4 === 0) {
      const p = this.slam.filter.mu;
      for (let i = 0; i < rays.length; i += 3) {
        const ray = rays[i];
        if (!ray.hit) continue;
        const a = p[2] + ray.rel;
        this.mappedPointCloud.push({
          x: p[0] + Math.cos(a) * ray.dist, y: p[1] + Math.sin(a) * ray.dist,
          z: 0.25 + this.noise.next() * ray.hit.h * 0.8, kind: 'rock', step: this.stepCount,
        });
      }
      while (this.mappedPointCloud.length > this.maxPointCloud) this.mappedPointCloud.shift();
    }
  }

  /** Copy filter state into plain fields the renderers read, and score against truth. */
  _syncEstimators() {
    for (const est of this.estimators) {
      const f = est.filter;
      est.pose.x = f.mu[0];
      est.pose.y = f.mu[1];
      est.pose.th = f.mu[2];
      est.poseCov = f.poseCov();
      for (const [id, meta] of est.mapLandmarks) {
        const j = f.index.get(id), c = f.landmarkCov(id);
        meta.x = f.mu[j];
        meta.y = f.mu[j + 1];
        meta.sigmaXX = c.xx;
        meta.sigmaYY = c.yy;
        meta.sigmaXY = c.xy;
      }
      const ex = est.pose.x - this.rover.x, ey = est.pose.y - this.rover.y;
      const { xx, xy, yy } = est.poseCov, det = xx * yy - xy * xy;
      // NEES: is the true position inside the filter's 95% ellipse? (chi-square, 2 dof, 5.99)
      const nees = det > 0 ? (yy * ex * ex - 2 * xy * ex * ey + xx * ey * ey) / det : 0;
      est.nees.total++;
      if (nees < 5.99) est.nees.inside++;
      const m = est.metrics;
      m.rawPoseErr = Math.hypot(ex, ey);
      m.poseSigmaM = Math.sqrt(xx + yy);
      m.headingErrDeg = (Math.abs(wrapAngle(est.pose.th - this.rover.th)) * 180) / Math.PI;
      m.insideRate = est.nees.inside / est.nees.total;
      est.trackingStatus = est.id === 'wheels_only' ? 'Never corrected' : !this.useCamera ? 'Camera off: wheels only' : 'Correcting with the camera';
    }
  }

  _recordKeyframe() {
    const id = this.truthKeyframes.length;
    this.truthKeyframes.push({ id, t: this.t, pos: [this.rover.x, 0, -this.rover.y], th: this.rover.th });
    for (const est of this.estimators) {
      est.keyframes.push({ id, t: this.t, pos: [est.pose.x, 0, -est.pose.y], th: est.pose.th });
      est.rawPath = est.keyframes.map((k) => k.pos);
      est.ateHistory.push([this.t, est.metrics.rawPoseErr]);
      est.sigmaHistory.push([this.t, 2 * est.metrics.poseSigmaM]);
      if (est.ateHistory.length > 1200) { est.ateHistory.shift(); est.sigmaHistory.shift(); }
    }
    this._recomputeAllMetrics();
  }

  _recomputeAllMetrics() {
    for (const est of this.estimators) {
      const m = evaluateTrajectoryMetrics(this.truthKeyframes, est.keyframes, false, 10.0);
      est.alignedPath = m.aligned;
      let sq = 0, count = 0;
      for (const [id, lm] of est.mapLandmarks) {
        const truth = this.landmarks[id - 1];
        if (lm.x === undefined) continue;
        sq += (lm.x - truth.x) ** 2 + (lm.y - truth.y) ** 2;
        count++;
      }
      Object.assign(est.metrics, { ateRmse: m.ateRmse, atePct: m.atePct, mapRmse: count ? Math.sqrt(sq / count) : 0, mappedLandmarks: count });
    }
  }
}
