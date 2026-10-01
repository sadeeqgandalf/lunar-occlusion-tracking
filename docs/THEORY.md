# Theory ↔ code map

Reference: S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics* (MIT Press, 2005).

| Piece | Book | Code | How it is checked |
|---|---|---|---|
| Velocity motion model (exact circular arc) | Ch. 5.3, eq. 5.9 | `src/core/kinematics.js`, `src/models/rover.js` | Analytic Jacobians vs finite differences, incl. the straight-line limit (`test/models.test.js`) |
| Jacobians `G = ∂g/∂x`, `V = ∂g/∂(controls)` | Ch. 7.4 | `models/*.js` (`jacobians`), numerical fallback `core/numjac.js` | same, including the extended slip / wind / accel-bias states |
| EKF predict / correct | Table 3.3 | `src/filters/ekf.js` | NEES consistency test (ANEES ≈ dof when model matches world) |
| χ² innovation gating | Ch. 7.5 (ML data association) | `ekf.js`, `ukf.js` | scoreboard "Rej" column |
| UKF (augmented state, scaled sigma points) | Table 3.4 | `src/filters/ukf.js` | agrees with EKF on the linear CW spacecraft model |
| Particle filter | Table 4.3 | `src/filters/pf.js` | beats dead reckoning on all platforms |
| Low-variance resampling | Table 4.4 | `pf.js` `_resample` | – |
| Augmented MCL (random-particle recovery) | Table 8.3 | `pf.js` | – |
| Clohessy-Wiltshire relative motion | (orbital mechanics; Vallado / Clohessy & Wiltshire 1960) | `src/models/spacecraft.js` | drift-free relative orbit stays bounded; Jacobians vs finite differences |
| NEES / NIS consistency | Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, §5.4 | `src/sim/metrics.js` | – |

## Deliberate deviations from the textbook
* **Joseph-form covariance update** in the EKF: algebraically identical to `(I−KH)Σ̄`, numerically safer.
* **Particle filter** resamples only when `N_eff < N/2` (not every step) and adds small roughening noise (Gordon et al.).
* **Gate lock-out recovery** (EKF/UKF): after 10 consecutive gate rejections of one measurement kind the covariance is inflated ×1.5 (Bar-Shalom §5.4). This is a heuristic, not from Thrun, and it is not always helpful (see README results).
* **Disturbances are estimated, not handed over.** Wheel slip (odometry scale factor), crosswind and accelerometer bias are random-walk states in the rover, jet and spacecraft models respectively. The simulator injects them as real disturbances; the filters must learn them from the measurements. Residual speed-proportional noise (`sigmaVRel`) remains for what the random walk misses.
* **Time compression:** the rover moves at ~1 m/s and the spacecraft's mean motion is 0.01 rad/s, far faster than reality, so dynamics are visible in minutes.

## Honest limitations
* Planar (2-D) models only. The jet is a horizontal-plane coordinated-turn model, the spacecraft is in-plane CW; there is no attitude estimation, no 3-D, no quaternions yet.
* Sensor noise is Gaussian except where a fault injects outliers; real sensors have heavier tails and correlation.
* Disturbances outside the model still hurt: the rover "Dust Storm" gyro-bias jump (~16σ beyond the prior) leaves EKF/UKF overconfident while the PF does better. That is a property of the filters' model assumptions, not a claim about real flight filters.
