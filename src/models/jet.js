import { Mat, wrapAngle } from '../core/linalg.js';
import { arc } from '../core/kinematics.js';

/**
 * Fixed-wing aircraft in the horizontal plane: coordinated turn driven by INS + airspeed, with wind estimated.
 *   x = [px, py, heading, airspeed, gyroBias, windE, windN]   u = {a: along-track accel (IMU), w: yaw rate (gyro)}
 *   ground velocity = airspeed along heading + wind;          w = [n_a, n_w, n_bias, n_wx, n_wy]
 * Airspeed-based dead reckoning is blind to wind, so wind is a state (random walk) like any real INS/ADC blend.
 * Jacobians intentionally omitted -> the framework differentiates numerically (core/numjac.js).
 */
export const jetModel = {
  id: 'jet',
  nx: 7, nw: 5,
  stateNames: ['x', 'y', 'ψ', 'v_air', 'b_gyro', 'wind_E', 'wind_N'],
  angleStates: [2],
  neesIdx: [0, 1, 2, 3],
  posIdx: [0, 1],
  headingIdx: 2,

  f(x, u, w, dt) {
    const a = u.a + w[0], om = u.w + w[1] - x[4];
    const vm = x[3] + 0.5 * a * dt; // mean airspeed over the step
    const [nx, ny, nth] = arc(x[0], x[1], x[2], vm, om, dt);
    return [nx + x[5] * dt, ny + x[6] * dt, nth, x[3] + a * dt, x[4] + w[2], x[5] + w[3], x[6] + w[4]];
  },
  W(u, dt, s) { return Mat.diag([s.sigmaA ** 2, s.sigmaW ** 2, s.sigmaBiasWalk ** 2 * dt, s.sigmaWindWalk ** 2 * dt, s.sigmaWindWalk ** 2 * dt]); },
  P0: () => Mat.diag([25, 25, 0.0004, 1, 2.5e-9, 25, 25]), // sigma_bias 5e-5 rad/s (INS-grade); sigma_wind 5 m/s

  meas: {
    landmark: { // ground radar / TACAN beacon: range + bearing relative to nose
      dim: 2, angleDims: [1],
      z: (m) => [m.range, m.bearing],
      h: (x, m) => { const dx = m.lx - x[0], dy = m.ly - x[1]; return [Math.hypot(dx, dy), wrapAngle(Math.atan2(dy, dx) - x[2])]; },
      R: (m, rs) => Mat.diag([(m.sigR * rs) ** 2, (m.sigB * rs) ** 2]),
    },
    fix: { // GNSS position
      dim: 2, angleDims: [],
      z: (m) => [m.x, m.y],
      h: (x) => [x[0], x[1]],
      H: () => Mat.from([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]),
      R: (m, rs) => Mat.diag([(m.sigma * rs) ** 2, (m.sigma * rs) ** 2]),
    },
    airspeed: { // pitot
      dim: 1, angleDims: [],
      z: (m) => [m.v],
      h: (x) => [x[3]],
      H: () => Mat.from([[0, 0, 0, 1, 0, 0, 0]]),
      R: (m, rs) => Mat.diag([(m.sigma * rs) ** 2]),
    },
  },
};
