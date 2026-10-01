// Nominal sensor suites per platform. The filters are told these values; faults make reality worse than the model.
export const ROVER_SENSORS = {
  sensePeriod: 0.5,
  sigmaV: 0.03,          // wheel odometry speed noise [m/s]
  sigmaVRel: 0.03,       // residual speed-proportional odometry noise (slip itself is now a state)
  sigmaSlipWalk: 0.02,   // slip random walk [1/sqrt(s)]
  sigmaW: 0.01,          // gyro white noise [rad/s]
  sigmaBiasWalk: 0.00005, // gyro bias random walk [rad/s/sqrt(s)]
  landmarkRange: 25, maxLandmarksPerSweep: 3,
  sigmaR: 0.25, sigmaB: 0.01,
  sigmaFix: 1.5, fixPeriod: 90, fixWindow: 5, // orbiter overhead pass
};
export const DEFAULT_SENSORS = ROVER_SENSORS;

export const JET_SENSORS = {
  sensePeriod: 0.5,
  sigmaWindWalk: 0.6,   // wind random walk [m/s/sqrt(s)]
  sigmaA: 0.15, sigmaW: 0.002, sigmaBiasWalk: 0.000003,
  landmarkRange: 40000, maxLandmarksPerSweep: 3, sigmaR: 20, sigmaB: 0.004,
  sigmaFix: 6, fixPeriod: 1, fixWindow: 1,   // GNSS essentially always on
  sigmaAirspeed: 1.5,
};

export const SPACECRAFT_SENSORS = {
  sensePeriod: 0.5,
  meanMotion: 0.01, // rad/s, time-compressed (LEO is ~1.1e-3) so relative-orbit dynamics play out in minutes
  sigmaAccel: 0.002, sigmaAccelBiasWalk: 0.0005,
  lidarRange: 320, sigmaR: 0.3, sigmaB: 0.004,
  sigmaFix: 3, fixPeriod: 20, fixWindow: 2,
};
