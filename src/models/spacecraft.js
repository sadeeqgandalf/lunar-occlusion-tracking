import { Mat, wrapAngle } from '../core/linalg.js';

/**
 * Proximity operations: chaser relative to a target in a circular orbit, in-plane Clohessy-Wiltshire (Hill) frame.
 *   x = [x_radial, y_alongtrack, vx, vy, accelBiasX, accelBiasY]   u = {ax, ay}: accelerometer reading
 *   true accel = u - bias;                                          w = [n_ax, n_ay, n_bx, n_by]
 * Exact CW state-transition over dt + zero-order-hold acceleration. Linear dynamics, nonlinear (range/bearing) sensors.
 * `s.meanMotion` n [rad/s]; time-compressed for gameplay (real LEO n ~ 1.1e-3).
 */
function phi(n, dt) {
  const s = Math.sin(n * dt), c = Math.cos(n * dt);
  return Mat.from([
    [4 - 3 * c, 0, s / n, (2 / n) * (1 - c)],
    [6 * (s - n * dt), 1, (2 / n) * (c - 1), (4 * s - 3 * n * dt) / n],
    [3 * n * s, 0, c, 2 * s],
    [6 * n * (c - 1), 0, -2 * s, 4 * c - 3],
  ]);
}
const B = (dt) => Mat.from([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]]);

export const spacecraftModel = {
  id: 'spacecraft',
  nx: 6, nw: 4,
  stateNames: ['x_R', 'y_T', 'vx', 'vy', 'b_ax', 'b_ay'],
  angleStates: [],
  neesIdx: [0, 1, 2, 3],
  posIdx: [0, 1],
  headingIdx: -1,

  f(x, u, w, dt, s) {
    const n = s?.meanMotion ?? spacecraftModel.n;
    const P = phi(n, dt), a = [u.ax - x[4] + w[0], u.ay - x[5] + w[1]], Bm = B(dt), o = [0, 0, 0, 0, x[4] + w[2], x[5] + w[3]];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) o[i] += P.get(i, j) * x[j];
      o[i] += Bm.get(i, 0) * a[0] + Bm.get(i, 1) * a[1];
    }
    return o;
  },
  n: 0.02,
  W(u, dt, s) { return Mat.diag([s.sigmaAccel ** 2, s.sigmaAccel ** 2, s.sigmaAccelBiasWalk ** 2 * dt, s.sigmaAccelBiasWalk ** 2 * dt]); },
  jacobians(x, u, dt, s) {
    const Pm = phi(s?.meanMotion ?? spacecraftModel.n, dt), Bm = B(dt), G = Mat.eye(6), V = Mat.zeros(6, 4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) G.set(i, j, Pm.get(i, j));
      G.set(i, 4, -Bm.get(i, 0)); G.set(i, 5, -Bm.get(i, 1));
      V.set(i, 0, Bm.get(i, 0)); V.set(i, 1, Bm.get(i, 1));
    }
    V.set(4, 2, 1); V.set(5, 3, 1);
    return { G, V };
  },
  P0: () => Mat.diag([4, 4, 0.01, 0.01, 9e-6, 9e-6]), // sigma_bias 3 mm/s^2

  meas: {
    lidar: { // relative range + bearing (LVLH frame, attitude known from star tracker) to the target at the origin
      dim: 2, angleDims: [1],
      z: (m) => [m.range, m.bearing],
      h: (x) => [Math.hypot(x[0], x[1]), Math.atan2(-x[1], -x[0])],
      H: (x) => {
        const q = x[0] * x[0] + x[1] * x[1], r = Math.sqrt(q);
        return Mat.from([[x[0] / r, x[1] / r, 0, 0, 0, 0], [-x[1] / q, x[0] / q, 0, 0, 0, 0]]);
      },
      R: (m, rs) => Mat.diag([(m.sigR * rs) ** 2, (m.sigB * rs) ** 2]),
    },
    fix: { // ground-tracked / relative-GNSS position
      dim: 2, angleDims: [],
      z: (m) => [m.x, m.y],
      h: (x) => [x[0], x[1]],
      H: () => Mat.from([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]),
      R: (m, rs) => Mat.diag([(m.sigma * rs) ** 2, (m.sigma * rs) ** 2]),
    },
  },
};
