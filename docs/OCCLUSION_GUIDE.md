# Learning guide: tracking people through occlusion

This guide walks through the Occlusion Lab (`track.html`) from first principles. Each idea comes in five layers:

1. **In plain words**: the idea.
2. **The maths**: the equation that implements it.
3. **In the code**: the file and function.
4. **From the literature**: the paper behind it. Bracketed tags such as [R2] point to [`REFERENCES.md`](REFERENCES.md).
5. **Try it**: something to do in the lab.

---

## 0. The question

A rover's mast camera watches astronauts on the Moon. When someone walks behind a boulder, or into a boulder's shadow, the camera stops seeing them. **When they reappear, does the tracker give them back the same ID?** A new ID means the system now believes there are two different people, which is an *ID switch*.

Three trackers see identical camera detections. They differ in exactly one respect: how they reason about not seeing someone.

| Tracker | Belief when a person is not detected |
|---|---|
| Naive | "Everyone is always visible, so a miss is evidence they are gone." |
| Occlusion-aware | "I know where the camera is blind. A miss there is expected, not evidence of absence." |
| Aware + negative information | "…and since I can't see them, they are probably *inside* the blind zone." |

**Try it:** open the lab, click **▶ Tour**, and watch the "What just happened" feed. Every reappearance is scored ✔ or ✘ for each tracker.

---

## 1. What the camera can and cannot see

**In plain words.** The camera sits on a 2.2 m mast. A person is a 1.8 m body. Boulders are rounded rocks: about a third are low (0.7–1.4 m), so you can see heads over them, and the rest are tall (2.4–3.6 m) and hide everything. A person counts as "visible" in proportion to how much of their body the camera can see.

**The maths.** The lab draws 16 sight lines from the camera eye to the person: 4 heights (0.3, 0.8, 1.3, 1.7 m) × 4 widths across the body (±0.09, ±0.28 m). Each boulder is a half-buried ellipsoid with horizontal radius `r` and height `h`. A sight line from eye `E` to body point `P` is blocked if the segment `E + t(P − E)`, `t ∈ (0,1)`, enters the ellipsoid. After scaling by `(r, r, h)` that is a quadratic in `t`.

```
visible fraction  v = (unblocked sight lines) / 16
```

The detected bearing is the centre of the *visible* part. A half-hidden person therefore appears shifted toward their visible side, as a real detector's box would be.

**In the code:** `src/mot/occlusion.js`, `visibility()` and `segmentHits()`.

**From the literature:** occlusion has to enter the observation model, because an occluded target's likelihood vanishes even when its location is known [R4].

**Try it:** switch to **Top-down map**. Dark wedges are blind zones; lighter wedges are low rocks that only hide legs.

---

## 2. The second kind of occlusion: shadows

**In plain words.** At the lunar south pole the sun stays near the horizon, so boulders cast very long shadows. A person standing in one can be in plain line of sight yet too dark for an optical camera to detect.

**The maths.** From three heights on the body (0.4, 0.9, 1.4 m), rays go toward the sun at elevation ε ≈ 7.9° (azimuth set to match the 3-D render). If a ray hits a boulder within 80 m, that part of the body is shadowed.

```
shadow fraction  s = (shadowed body points) / 3
```

**In the code:** `occlusion.js`, `shadowFraction()`. The sun direction is `MotWorld.sun` in `src/mot/world.js`.

**From the literature:** the Moon's spin axis is tilted only ~1.5° to the ecliptic, so polar illumination is grazing and shadows are long [G2]. We raise the sun to ~8° so the scene stays readable.

**Try it:** choose **South pole · Long shadows**. The feed now says "walked into deep shadow (in line of sight, but too dark to detect)".

---

## 3. From visibility to a detection

**In plain words.** Seeing more of a person makes detection more likely. Darkness makes it less likely. The camera also produces false alarms (glints, dust).

**The maths.**

```
P_D(v, s) = 0                                        if v < 0.25 (too little visible)
          = P_D,max · min(1, (v − 0.25)/0.45) · (1 − 0.8 s)   otherwise
```

with `P_D,max` = 0.95. Each detection measures range and bearing with noise:

```
σ_bearing = 0.004 rad (×2 when partly occluded)
σ_range   = 0.08 + 0.0012 · r²   (metres; stereo depth error grows with r², from z = fB/d ⇒ δz ≈ z²δd/(fB) [G1])
```

False alarms per frame are Poisson with mean λ = 0.8, spread uniformly over the field of view. Their density in measurement space is

```
κ = λ / (FOV × (range span))   (per metre·radian)
```

**In the code:** `occlusion.js`, `detectionProb()`; `src/mot/world.js`, `MotWorld.view()` and `MotWorld.sense()`. The tracker receives only `{range, bearing}`, never identities. A test enforces this.

---

## 4. Following one person: the constant-velocity Kalman filter

**In plain words.** Each track is a guess of position and velocity, plus how unsure it is. Each frame (0.1 s) the guess moves forward at its current velocity and becomes a little less certain. When a detection is matched, the guess is pulled toward it.

**The maths.** State `x = [x, y, vx, vy]`, white-noise-acceleration model with σ_a = 0.9 m/s²:

```
x' = F x,   F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
P' = F P Fᵀ + Q,   Q = σ_a² · [[dt⁴/4 I, dt³/2 I],[dt³/2 I, dt² I]]
```

The measurement `z = [range, bearing]` is nonlinear in `x`, so the update linearises it (an extended Kalman filter) with Jacobian `H = ∂h/∂x`. The covariance update uses the Joseph form:

```
S = H P Hᵀ + R,   K = P Hᵀ S⁻¹,   x ← x + K(z − h(x)),   P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
```

**In the code:** `src/mot/tracker.js`, `_predict()`, `_meas()`, and the update block in `step()`.

**From the literature:** [R5] (models, gating); [R9] (the Kalman-plus-assignment family).

---

## 5. Which detection belongs to which person?

**In plain words.** Each frame there are several tracks and several detections. Each track either takes one detection or says "I missed this frame". The cheapest overall pairing wins.

**The maths.** A pairing is allowed only inside the χ² gate: normalised innovation squared NIS = yᵀS⁻¹y ≤ 11.83, the 99.73% point for 2 dof. Costs are negative log likelihood ratios:

```
take detection j:  c_ij = −log( P_D,i · g_ij / κ )      g_ij = N(z_j; h(x_i), S_i)
miss this frame:   c_i0 = −log( 1 − P_D,i · P_G )        P_G = 0.9973
```

The Hungarian algorithm finds the minimum-cost assignment.

**Why the miss option matters.** If a track *expects* to be hidden (`P_D,i ≈ 0`), its miss cost is ≈ 0 while taking a detection is expensive. Hidden tracks therefore do not get **hijacked** by clutter or by a passer-by. This was a real failure in an earlier version, found by tracing a coasting track frame by frame.

**In the code:** `tracker.js`, the cost matrix in `step()`; `src/mot/hungarian.js`, tested against brute force.

**From the literature:** [R6] (likelihood-ratio scoring with a missed-detection option); [R8] (Hungarian algorithm).

---

## 6. Does this track still exist? (where occlusion awareness happens)

**In plain words.** Each track carries a probability `r` that a real person is behind it. Detections raise `r`; misses lower it. The key question is **how much a miss should lower it**.

**The maths** (Bernoulli filter [R2], the same spirit as IPDA [R1]):

```
detected:  r ← r [P_D g + (1 − P_D) κ] / [r P_D g + (1 − r P_D) κ]
missed:    r ← r (1 − P_D) / (1 − r P_D)
```

Look at the miss update. With `P_D = 0.95` (naive), three misses take `r` from 0.99 to about 0.01: the track dies. With `P_D ≈ 0` (the occlusion-aware tracker, for a person it predicts is behind a rock), a miss leaves `r` unchanged: the track waits.

**What "occlusion-aware" computes.** The expected detection probability is averaged over where the person might be, using the track's own uncertainty:

```
P_D,i = (1/64) Σ_k P_D( v(x_k), s(x_k) ),   x_k ~ N(x̂_i, P_i)  (64 fixed antithetic samples)
```

**Track lifecycle:** born at `r = 0.25` from a detection outside every gate; confirmed when `r ≥ 0.75` with ≥ 3 hits; deleted when `r < 0.08` or position σ > 7 m. Reported as a box only if seen in the last 0.3 s, the same rule for all trackers.

**In the code:** `tracker.js`, `expectedPd()`, `_samples()`, and the existence updates in `step()`.

**From the literature:** state-dependent detection probability, "expected but missing" measurements [R3]; existence recursion [R1, R2].

**Try it:** watch a naive track (red card) as someone steps behind a rock. It vanishes within a few frames. Switch the map to the green tracker: its track turns into a dashed "lost sight · searching" area and waits.

---

## 7. Negative information: using the miss to locate the person

**In plain words.** "I don't see them" is also a clue to *where* they are: probably in a blind zone, not in open view. The third tracker moves its estimate accordingly.

**The maths** (Bayes with a missed detection [R3]):

```
p(x | missed) ∝ p(x) · (1 − P_D(x))
```

This posterior is not Gaussian, so it is moment-matched: weight the same 64 samples by `w_k = 1 − P_D(x_k)`, then set the new mean and covariance to the weighted mean and covariance. Velocity moves too, through the samples' correlations. A small floor keeps the covariance from collapsing.

**In the code:** `tracker.js`, `_missUpdate()`.

**Try it:** compare the green and blue cards on the same scenario and seed. The measured effect is in the results below.

---

## 8. How we score, and what the results say

**Metrics** (`src/mot/metrics.js`):

* **ID kept through occlusion.** A person is tracked, becomes hidden for at least 1 s, and reappears. Success means the first reported track matched to them within 2 s has their old ID. Results are binned by occlusion length.
* **ID switches, MOTA, MOTP** [E1]; **IDF1** [E2]; **GOSPA** [E3] (p = 2, c = 2 m, α = 2).
* **Hidden-track survival and coverage.** While a person is hidden, is their track still alive, and is the truth inside its 95% ellipse?

**Statistics:** 20 seeds × 300 s per scenario. Proportions carry Wilson 95% intervals [E5]; per-run metrics are given as mean ± 1.96·SE. Regenerate everything with `node experiments/occlusion_ablation.mjs`; the output is stamped with the git commit.

**Findings** (full tables in [`experiments/results.md`](../experiments/results.md)):

* Short occlusions (1–5 s): **0%** of IDs kept by the naive tracker vs **32–63%** by the aware trackers, depending on the scene.
* ID switches fall by **37%** (boulder field) to **60%** (polar shadows).
* Negative information is better than plain awareness in every scene, but each interval overlaps slightly, so it is "consistently better", not "proven better".
* Long occlusions (≥ 5 s) defeat all motion-only variants (≤ 12% kept). People turn while unseen. This is where **appearance re-identification** is needed, and it motivates the SAM-based work in `ComputerVision_Research`.

---

## 9. Exercises

Each comes with what to look for, not a promised answer. Measure it.

1. **Seed dependence.** Change the seed in the header and watch how much "ID kept" moves between single runs. Then read the 20-seed intervals. Why are single runs untrustworthy?
2. **Shadows versus rocks.** Run the same seed on *Boulder field* and *South pole · Long shadows*. Which tracker's ID-switch count rises most, and why (§6)?
3. **Sparse cover.** In *Open plain · Sparse cover*, occlusions are rarer. How does the gap between green and blue change, and how wide are the intervals?
4. **Read the code.** In `tracker.js`, find where `pdExp` enters (a) the assignment cost and (b) the miss update. Predict what happens if you set `occlusionAware: false` for only one of them.

## 10. Limitations (read before citing any number)

* Simulated walkers and a sensor-model detector, not a neural detector on real images.
* Locally flat ground in the experiment. The 3-D view adds gentle relief and keeps scenery craters out of the camera's field.
* Shadow darkness is a fixed detectability penalty, not a photometric camera model. The sun is raised to ~8° for readability.
* Positioning against the literature is in [`REFERENCES.md`](REFERENCES.md). No novelty is claimed over the cited methods.
