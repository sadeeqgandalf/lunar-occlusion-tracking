import { EKF } from './ekf.js';

/** Pure propagation (odometry/IMU only) with covariance growth, no corrections. The baseline to beat. */
export class DeadReckoning extends EKF {
  static id = 'dr';
  static label = 'Dead Reckoning';
  static color = '#9aa4b2';
  static blurb = 'Integrates inertial/odometry inputs only. Watch the uncertainty bubble swell as it drifts.';
  static paramSpec = { qScale: { label: 'Process noise ×', min: 0.2, max: 60, step: 0.1, value: 4 } };
  update() { return { accepted: false }; }
}
