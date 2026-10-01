# EKF Mission Control

An interactive state-estimation lab: fly a **Mars rover, a fighter jet, or a spacecraft** through faults while EKF, UKF, particle-filter and your own estimators race on the *same* sensor stream. Built to learn estimation properly (consistency, gating, divergence), not just to look at it.

```bash
npm start            # http://localhost:8080  (zero dependencies, Node >= 18)
                     # http://localhost:8080/?demo=1 starts a self-running demo
npm test             # 14 tests: Jacobians, consistency, determinism, UI code paths
npm run bench        # multi-seed benchmark table -> bench/results.md
node bench/xlang/run.mjs   # JS vs C++ vs Python equivalence + speed
```

## What it is
* **Three platforms, one estimation core.** A vehicle is a `Model` (state, `f(x,u,w)`, process noise, measurement models, optional Jacobians). EKF/UKF/PF are written once against that interface. Add a vehicle by supplying a model; if you skip the Jacobians they are differentiated numerically.
* **Mission game.** Waypoint autopilot driven by the *estimate*, hazards (craters / no-fly zones / debris), objectives (sample / checkpoint / dock), battery/fuel, uplink latency, uncertainty-aware safing (autopilot holds when the 3σ bubble overlaps a hazard).
* **Fault injection:** gyro bias jump, wheel slip, beacon blackout, outlier bursts, GNSS jamming, wind, sun-blind lidar, accelerometer bias, missed orbiter pass.
* **Estimator details:** the particle filter uses 1,500 particles by default (500 collapsed in 2 of 20 random streams on the rover sand trap, 1,500 in 0 of 20). Sensor noise parameters live in `src/sim/sensors.js` (e.g. `sigmaVRel`, `sigmaSlipWalk`, `sigmaWindWalk`, `sigmaAccelBiasWalk`).
* **Honest metrics:** error vs. the filter's own 3σ, NEES against the χ² band, ANEES vs degrees of freedom, gate rejections, CPU per step.
* **Filter Lab:** write an estimator in the browser, compile, and watch it race live. Built-in **Learn** panel with guided experiments.

**New here?** Read the operator's guide: [`docs/GUIDE.md`](docs/GUIDE.md) (first-run briefing, what every panel means, guided experiments). Theory mapping to *Probabilistic Robotics*: [`docs/THEORY.md`](docs/THEORY.md).

## Measured results (5 seeds, identical sensor streams; full table in `bench/results.md`)
RMSE in metres. Filters estimate the disturbances that used to break them as extra states: rover `[x,y,θ,gyro bias,slip]`, jet `[x,y,ψ,v,gyro bias,wind E,wind N]`, spacecraft `[x,y,vx,vy,accel bias x,y]`.

| scenario | DR | EKF | UKF | PF (1,500 particles) |
|---|---:|---:|---:|---:|
| rover · nominal | 17.0 | 0.27 | 0.26 | 0.26 |
| rover · sand trap | 27 | 2.8 | 2.7 | 3.2 |
| rover · dust storm | 64 | 4.0 | 4.9 | 1.6 |
| jet · corridor | 1336 | 3.9 | 3.9 | 3.8 |
| jet · GNSS denied | 4350 | 9.1 | 9.1 | 9.3 |
| spacecraft · approach | 40 | 0.23 | 0.23 | 0.32 |
| spacecraft · sun-blind | 59 | 0.57 | 0.57 | 0.82 |

Reading it honestly:
* EKF and UKF are statistically consistent almost everywhere (ANEES ≈ 1–2 against 3–4 degrees of freedom), and nearly identical because the nonlinearity is mild.
* Putting slip, wind and accelerometer bias in the state is what fixed the earlier divergences; it is the standard remedy, not a tuning hack.
* The particle filter is now competitive and consistent, but costs ~200× the CPU of the EKF (~550–780 µs vs ~3 µs per step). It only clearly wins in the rover **dust storm** (1.6 m vs 4.0 m for the EKF and 4.9 m for the UKF), where the injected gyro-bias jump is ~16σ beyond the EKF/UKF prior (their ANEES there: 15 and 8, i.e. overconfident). A good Filter Lab challenge: make an EKF that handles the jump.
* The NIS gate matters: with it off, the rover EKF under outlier bursts goes from ANEES ≈ 1 to 38–86 (see the guide).

## Which language? Measured, not asserted
The same rover EKF (identical equations, gating, recovery) was implemented in JS, C++17 and Python/numpy and replayed on one recorded trace (`bench/xlang/`):

| impl | final state vs JS | µs / filter step |
|---|---|---:|
| JS (V8) | – | ~2.5 |
| C++17 -O2 | 4e-10 | ~0.1 |
| Python + numpy | 4e-10 | ~9 |

* **The math is language-independent**: all three agree to ~4e-10 on the 5-state rover EKF.
* **Speed is not the deciding factor at this size.** A 5-state EKF needs microseconds against a 50 ms (20 Hz) budget in every language. C++ is ~20–30× faster than JS and ~70–100× faster than numpy (run-to-run spread is large at this scale), which matters for big state vectors, many filters or flight CPUs, not for this.
* **So the choice follows the job**: browser JS for the learning UI (shipped here); C++ for flight/embedded and ROS 2 nodes (NASA flight software is C/C++: cFS, F Prime); Python for offline analysis and notebooks. `bench/xlang/ekf.cpp` is a dependency-free reference port of the rover filter (re-ported when the model gained the slip state), and the test suite is the spec any port must match.

## Not done / not verified (so you can trust the rest)
* **No ROS 2 node was built.** ROS 2 isn't installed on the development machine, so none was written or run. The natural path is a C++ `rclcpp` node wrapping the port in `bench/xlang/ekf.cpp` and subscribing to IMU/odometry/landmark topics.
* **The UI was not visually inspected.** The Chrome automation was unavailable in the build session and no browser was installed. The UI is covered only by a stub-DOM test that executes every code path (all platforms, scenarios, faults, filter toggles, Lab) without errors. Expect cosmetic issues on first open; please report them.
* Planar models only; no attitude/3-D; Gaussian-noise sensors; time-compressed dynamics (see `docs/THEORY.md`).
* This is an educational/portfolio simulator, not NASA software and not flight-qualified.

## Layout
```
src/core        rng, linear algebra, arc kinematics, numerical Jacobians
src/models      rover · jet · spacecraft estimation models
src/filters     base interface, DR, EKF, UKF, particle filter, registry
src/platforms   truth dynamics + sensors + autopilot + rules per vehicle
src/sim         mission engine, scenarios, metrics
src/ui          map renderer, charts, Filter Lab, app
test/  bench/   tests, benchmarks, cross-language proof
```
