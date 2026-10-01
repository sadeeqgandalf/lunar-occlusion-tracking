# Operator's Guide

How to use this console, how to read it, and what carries over to real vehicles.

## 0. What this is, and what it is not

**It is** a simulator and algorithm lab for *state estimation*: answering "where is my vehicle, and how sure am I?" from noisy sensors. Four estimators (dead reckoning, EKF, UKF, particle filter) run side by side on the *same* sensor stream, on three vehicles (Mars/Earth rover, fighter jet, spacecraft proximity-ops). The filters are implemented from *Probabilistic Robotics* (Thrun, Burgard, Fox); see `docs/THEORY.md`.

**It is not** flight software, a certified navigation system, or a model of any specific NASA vehicle. Sensor numbers are plausible, not characterised from hardware. Time is compressed (a rover that moves at centimetres per second in real life moves at about a metre per second here; the spacecraft's orbital rate is sped up ~9× (0.01 rad/s vs ~1.1e-3 rad/s in LEO)) so that things happen in minutes. Treat it as a place to build intuition and test ideas, not as proof that a design is flight-ready.

## 1. The first five minutes

1. Run `npm start`, open <http://localhost:8080>. The briefing appears on first load. For the quickest way to understand everything, click **▶ Guided tour** (also the **▶ Tour** button in the header): 11 short steps that spotlight each control and let you press the action yourself (upload a plan, inject a fault, stop the vehicle, hide ground truth).
2. Click **▶ Start the mission**. The nominal plan is uplinked and the vehicle starts driving at 4×.
3. Watch three places, in this order:
   - **Navigation Health** (top-left): the only verdict you would have in real life.
   - **The map**: coloured shapes are what each filter *believes*; the thin ring around each is that filter's own 95% uncertainty. The white outline is ground truth (training only).
   - **The Coach** (bottom of the map): says what is happening and what to do.
4. In **Fault injection** (left), click **Beacon blackout** (rover). Watch the Coach, then Health, then the rings.
5. Press **T** to hide ground truth. That is what real operations look like: you only have the beliefs and their uncertainty.

## 2. Reading the screen

### The map
| Thing | Meaning |
|---|---|
| White outline | Ground truth. Does not exist in real life. |
| Coloured vehicle + ring | One filter's belief and its 95% position uncertainty. |
| Grey ring (Dead Reckoning) | Never corrected, so its ring only grows. |
| Cyan diamonds | Known beacons the vehicle can range to (jet: radar beacons; spacecraft: the target). |
| Dark circles | Hazards (craters, no-fly zones, debris). |
| Yellow numbered rings | Objectives. |
| Dashed white path | Waypoints the autopilot will fly, steering by the **PRIMARY** filter's belief. |

### Navigation Health (what to trust)
Computed from **onboard quantities only**; it never looks at ground truth:

- **3σ bubble vs "needs ≤"**: how big the uncertainty is compared with the accuracy the next objective requires. If the bubble is larger than the tolerance, you cannot promise success.
- **Last outside fix / % rejected**: how long since a beacon, GNSS or orbiter fix was accepted, and how many recent ones the filter threw away. A high rejection rate means the filter and the world disagree.

States: **NAV NOMINAL** (carry on) · **NAV DEGRADED** (keep clear of hazards) · **DO NOT TRUST** (stop, re-localise).

### The scoreboard
| Column | Meaning |
|---|---|
| Err | Current position error vs truth (training only). |
| 3σ | The filter's own claim about its error. |
| RMSE | Average error over the run. |
| **ANEES** | Average normalised estimation error squared. Should be near the degrees of freedom (shown under the table). **Red = overconfident** (dangerous), amber = conservative (wasteful, safe). |
| Rej | Measurements rejected by the χ² gate. |
| µs | CPU time per step. |

A good filter is **accurate and honest**: error inside its own 3σ, ANEES near the dof. A filter that is accurate but overconfident will walk you into a hazard with a smile.

### Charts
1. Position error for all filters.
2. Primary: error vs its own 3σ bound (the orange dashed line). Error should live under it.
3. Primary: NEES against the green 95% band. Frequent excursions above the band mean overconfidence.

## 3. Decision rules (what you would actually do)

| You see | It means | Do |
|---|---|---|
| NAV NOMINAL, error inside 3σ | Estimator healthy | Continue. |
| 3σ bubble growing, no recent fix | Flying on inertial/odometry only | Slow down; head toward beacons; avoid hazards. |
| Many rejections after a fault | Model and world disagree | Pause driving. Wait for the filter to re-converge or retarget. |
| DO NOT TRUST | Cannot guarantee safety | **Stop**, wait for a fix, then continue. |
| SAFE-HOLD | Bubble overlaps a hazard ahead | Retarget around it, wait for the bubble to shrink, or accept the risk by unticking safing. |
| Anything looks wrong and you are unsure | Take the safe action | Press the red **■ STOP** (header, or `Esc`). It clears waypoints and manual drive: a rover stops, a jet enters a holding orbit, a spacecraft station-keeps. **Pause** is different: it freezes the simulation clock. |
| Primary is Dead Reckoning | Autopilot follows a belief that only gets worse | Switch primary to EKF/UKF/PF. |
| ANEES red on your primary | Filter is overconfident | Raise process noise or investigate an unmodelled effect. |

## 4. Experiments (each takes under two minutes)

Numbers below were measured in this repository with the headless runner (seed 7 unless stated). Yours will differ in detail but not in direction.

1. **Dead reckoning fails.** Make DR primary, press **P**, set 8×. Its ring swells and the vehicle follows its wrong belief toward hazards. On the nominal rover run DR averaged about 24 m RMSE; EKF/UKF/PF about 0.1 m.
2. **Honesty vs accuracy.** Drag the EKF's **Process noise ×** down from 4 to 0.2. On the sandbox rover (seed 7) ANEES goes from 1.2 to 11.6 and RMSE from 0.6 m to 1.2 m: the filter trusts its model too much, and its ring no longer covers the truth.
3. **Unmodelled physics.** Jet → *Sortie 2 · GNSS Denied*. After the crosswind fault, EKF/UKF hold (≈ 9 m) because wind is a state; dead reckoning is thousands of metres off. Earlier in this project the same scenario broke the EKF (≈ 1.5 km error) *before* wind was added to the model. That failure is the point: a filter is only as good as the effects it models.
4. **What the gate buys you.** Rover → *Sandbox*, inject **Beacon outlier burst** a few times. The EKF/UKF reject the ghost readings (watch **Rej** climb) and stay accurate and honest: over four seeds, RMSE 0.12–0.79 m and ANEES ≈ 1. Now drag the EKF's **NIS gate** slider to its maximum (40, effectively no gate) and repeat: RMSE rises to 0.75–1.17 m (up to ~6× worse) and ANEES jumps to 38–86, i.e. the filter swallows bad data and becomes badly overconfident. (The particle filter, which has no gate, copes through a heavy-tailed likelihood and stays near the gated filters, but is not better than them here.)
5. **Latency.** Slide **Uplink latency** to 5 s and drive with WASD. Commands arrive late; this is why real vehicles run on-board autonomy plus waypoints.
6. **Other vehicles.** Switch **Platform**. Same filter classes, different physics: the spacecraft uses Clohessy-Wiltshire relative motion with lidar range/bearing; the jet uses a coordinated-turn model with GNSS, airspeed and radar beacons.

## 5. Write your own estimator (Filter Lab)

Open **Filter Lab** (bottom bar). Edit the template and press **Compile & add to mission**. Your filter joins the race on the same measurements.

Return an object `{ name, init(x0, P0), predict(u, dt), update(meas), getState(), getCov() }`. `ctx` provides `Mat`, `model` (with `f`, `W`, `meas`), `jacobians()`, `numJac`, `resid`, `wrapAngle`, `sensors`, `rng`. The state vector layout is `model.stateNames` and differs per platform:

| Platform | State |
|---|---|
| Rover | `[x, y, heading, gyro bias, slip]` |
| Jet | `[x, y, heading, airspeed, gyro bias, wind E, wind N]` |
| Spacecraft | `[x radial, y along-track, vx, vy, accel bias x, accel bias y]` |

Judge your filter the same way: **RMSE low, ANEES near the dof, Err inside 3σ.** Adding a new vehicle means writing one model file (`src/models/*.js`: `f`, `W`, `P0`, measurement specs). Jacobians are optional; the framework differentiates numerically if you omit them.

## 6. From the sandbox to real life

**What transfers**
- The filter mathematics and the consistency methodology (NEES/NIS, χ² gating, lock-out recovery). These are what you would apply to real logged data.
- The idea of an *integrity monitor* (Navigation Health): decide from onboard quantities whether to trust the solution, and attach a decision to it.
- Modelling the unmodelled: slip, wind and bias as estimated states instead of hoping they stay small.

**What does not transfer by itself**
- Sensor noise values, fault behaviour and timing here are assumptions. For a real vehicle they must come from characterisation of the actual hardware.
- There is no real-time scheduling, sensor latency model beyond command uplink delay, or fault-management logic of a flight system.
- The 3-platform models are 2-D (planar) simplifications. Real aircraft and spacecraft navigation is 6-DOF with attitude estimation.

**Implementation language.** A cross-language benchmark replays one recorded rover trace through the same EKF in JavaScript, C++ and Python/numpy (`node bench/xlang/run.mjs`, results in `bench/xlang/results.md`). On one machine (Apple M-series) it measured roughly: C++ ≈ 0.1 µs (0.08–0.13 across runs), JavaScript ≈ 2.5 µs, numpy ≈ 9 µs per step. Take these as approximate (single trace, single machine, run-to-run spread). Two honest conclusions: (1) for a 4–7-state filter all three are far faster than any real sensor rate, so raw speed is not the deciding factor here; (2) C++ wins where you need deterministic timing, embedded targets and flight-software integration (NASA's open-source flight frameworks such as cFS and F Prime are C/C++), while Python is best for offline analysis and JavaScript for this interactive tool. The recommended split is: this app for intuition and design, C++ for an implementation you would deploy, Python for post-flight analysis.

**ROS.** ROS 2 is not installed in this development environment, so no ROS node has been built or tested here. The filter classes are plain functions of `(model, u, measurement)`, which makes a ROS 2 wrapper straightforward in principle, but that remains to be done and verified.

**Next step for real data (not built yet).** Replaying your own logged sensor data (for example a CSV exported from a bag file) through these filters, with the same scoreboard, is the natural next feature. It is not implemented.

## 7. Reference

Keys: `Space` pause · `R` reset · `P` upload plan · `Esc` stop · `Backspace` / right-click undo leg · `T` toggle truth · `F` follow vehicle · `WASD` / arrows manual drive · click map to add a waypoint (click a numbered ring to target the objective).

URL: `?demo=1` skips the briefing and starts the nominal plan immediately.

Tests: `npm test` (models, filters, consistency, coach/health, UI smoke). Benchmarks: `npm run bench`.
