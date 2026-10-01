import { Mat, wrapAngle } from '../core/linalg.js';
import { arc, arcPartials } from '../core/kinematics.js';

/**
 * Ground rover (Mars or Earth): unicycle on the exact arc model with gyro-bias and odometry-scale (slip) states.
 *   x = [px, py, heading, gyroBias, slip]     u = {v: wheel-odometry speed, w: gyro rate (incl. bias)}
 *   ground speed = u.v (1 - slip);  w = [n_v, n_w, n_bias, n_slip]  (control noise + two random walks)
 * Estimating the odometry scale factor is the standard remedy for wheel slip: the filter learns the sand
 * instead of mistaking it for sensor error.
 */
export const roverModel = {
  id: 'rover',
  nx: 5, nw: 4,
  stateNames: ['x', 'y', 'θ', 'b_gyro', 'slip'],
  angleStates: [2],
  neesIdx: [0, 1, 2],
  posIdx: [0, 1],
  headingIdx: 2,

  f(x, u, w, dt) {
    const om = u.w + w[1] - x[3];
    const [nx, ny, nth] = arc(x[0], x[1], x[2], (u.v + w[0]) * (1 - x[4]), om, dt);
    return [nx, ny, nth, x[3] + w[2], x[4] + w[3]];
  },
  W(u, dt, s) {
    return Mat.diag([s.sigmaV ** 2 + (s.sigmaVRel * u.v) ** 2, s.sigmaW ** 2, s.sigmaBiasWalk ** 2 * dt, s.sigmaSlipWalk ** 2 * dt]);
  },
  jacobians(x, u, dt) {
    const vEff = u.v * (1 - x[4]), p = arcPartials(x[2], vEff, u.w - x[3], dt);
    return {
      G: Mat.from([
        [1, 0, p.dxdth, -p.dxdw, -u.v * p.dxdv],
        [0, 1, p.dydth, -p.dydw, -u.v * p.dydv],
        [0, 0, 1, -dt, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
      V: Mat.from([
        [p.dxdv * (1 - x[4]), p.dxdw, 0, 0],
        [p.dydv * (1 - x[4]), p.dydw, 0, 0],
        [0, dt, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    };
  },
  P0: () => Mat.diag([0.25, 0.25, 0.0025, 4e-6, 9e-4]), // sigma_bias 0.002 rad/s (tactical MEMS), sigma_slip 3 %
  meas: {
    landmark: {
      dim: 2, angleDims: [1],
      z: (m) => [m.range, m.bearing],
      h: (x, m) => { const dx = m.lx - x[0], dy = m.ly - x[1]; return [Math.hypot(dx, dy), wrapAngle(Math.atan2(dy, dx) - x[2])]; },
      H: (x, m) => {
        const dx = m.lx - x[0], dy = m.ly - x[1], q = dx * dx + dy * dy, r = Math.sqrt(q);
        return Mat.from([[-dx / r, -dy / r, 0, 0, 0], [dy / q, -dx / q, -1, 0, 0]]);
      },
      R: (m, rs) => Mat.diag([(m.sigR * rs) ** 2, (m.sigB * rs) ** 2]),
    },
    fix: {
      dim: 2, angleDims: [],
      z: (m) => [m.x, m.y],
      h: (x) => [x[0], x[1]],
      H: () => Mat.from([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),
      R: (m, rs) => Mat.diag([(m.sigma * rs) ** 2, (m.sigma * rs) ** 2]),
    },
  },
};
