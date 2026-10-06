# The trackers: one step at a time toward tracking through occlusion

Every tracker here gets **the same camera detections** (range, bearing and, for re-ID, a colour signature). None of
them ever sees who anyone really is. They differ only in how they reason about a person they **cannot see**. Each
step up the ladder fixes one specific way occlusion breaks tracking.

## Ranking

Web lab, 20 seeds × 300 s per row. Numbers come from `node experiments/tracker_ladder.mjs 20 300` (raw results in
`experiments/tracker_ladder.json`). Brackets are 95% Wilson intervals. Sensors: **camera** = rover mast camera,
**lidar** = 3-D scanning lidar on the same mast, **lander** = a second camera on a lander, 6 m up.

**Top of the table, both scenarios.** Ranked by ID switches per run (lower is better); IDF1 is higher-is-better.

| Rank | Tracker + sensors | Boulders: ID switches | Boulders: IDF1 | Polar shadows: ID switches | Polar shadows: IDF1 | Same ID after hides > 5 s (boulders / polar) |
|---|---|---|---|---|---|---|
| 1 | **Aware + NI + re-ID · camera + lidar + lander** | **6.6** | **0.756** | **6.0** | **0.773** | 71% / 78% |
| 2 | Aware + NI + re-ID · camera + lander | 9.0 | 0.744 | 11.6 | 0.696 | 87% / 86% |
| 3 | Aware + NI · camera + lidar + lander | 15.1 | 0.580 | 15.3 | 0.592 | 38% / 34% |
| 4 | Aware + NI · camera + lander | 19.6 | 0.534 | 27.9 | 0.454 | 31% / 26% |
| 5 | Aware + NI + re-ID · camera | 23.3 | 0.540 | 24.4 | 0.518 | 50% / 50% |
| 6 | Aware + NI + re-ID · camera + lidar | 26.8 | 0.442 | 27.3 | 0.433 | 36% / 36% |
| 7 | Aware + NI · camera + lidar | 37.9 | 0.320 | 36.3 | 0.333 | 13% / 16% |
| 8 | Aware + NI · camera | 47.8 | 0.279 | 51.3 | 0.260 | 7% / 8% |
| 9 | Occlusion-aware · camera | 54.0 | 0.264 | 60.6 | 0.241 | 4% / 5% |
| 10 | Naive · camera | 75.6 | 0.229 | 129.4 | 0.159 | 0% / 0% |
| – | Naive · camera + lidar + lander | 254.1 | 0.184 | 316.6 | 0.148 | 0% / 0% |

**Short hides (1–5 s), same ID kept, camera only:** Naive 0%, Occlusion-aware 36%, Aware + NI 46%, + re-ID 89%.

**Cost per 0.1 s frame** (browser engine, 5 people): Naive 0.1 ms; Occlusion-aware 1.1 ms; Aware + NI 1.5 ms;
+ re-ID 1.6 ms; with the lander 2.2 ms; PHD heat map 2–3 ms. All run far inside the 100 ms real-time budget.

### What the table teaches

1. **Each ladder step helps:** 75.6 → 54.0 → 47.8 → 23.3 ID switches (camera only, boulders).
2. **More sensors only help a tracker that knows what each sensor can see.** With all three sensors Naive gets far
   worse (75.6 → 254), because every sensor that cannot see a person counts as evidence the person left. The aware
   trackers improve at every step.
3. **A second viewpoint beats a better sensor at the same spot.** The lander (another angle round the rocks) helps
   more than the lidar (same mast, better range and darkness vision): 47.8 → 19.6 vs 47.8 → 37.9.
4. **Lidar helps motion-only tracking and hurts re-ID slightly.** Lidar has no colour. People that only the lidar
   keeps hold of (in shadow, say) carry no colour signature, so they cannot be re-identified later (re-ID: 23.3 →
   26.8 switches). The fix would be lidar re-ID from body shape or reflectivity. With the lander added, lidar helps
   re-ID again (9.0 → 6.6).
5. **Not every number moves the same way.** Occlusion-aware *without* negative information keeps fewer short hides
   with lidar (36% → 23%): sharper lidar position estimates drift out of the blind zone faster when nothing pulls
   them back in. Negative information is what pulls them back.

---

## The ladder

### Step 0 · Naive: "if I can't see you, you're gone"

**What it does.** Each person is a constant-velocity extended Kalman filter in range/bearing. A χ² gate decides
which detections could belong to which track, and Hungarian assignment picks the best one-to-one matching. Each
track carries an **existence probability** `r` (is this a real person?), updated by Bayes on every hit and every
miss, using a constant detection probability `P_D = 0.95`.

**Why it fails behind rocks.** A miss multiplies `r` by `(1 − P_D) / (1 − r P_D)`. Two missed frames (0.2 s) take a
confident track below the deletion threshold, so the person comes back with a new ID.

**Code.** `src/mot/tracker.js` (`occlusionAware: false`). **Papers.** EKF / CV model [R5]; Hungarian [R8];
existence probability [R1, R2]; the SORT family [R9].

### Step 1 · Occlusion-aware: "I expected not to see you there"

**What changes.** The detection probability depends on where the track is predicted to be:

```
P_D(track) = P_D · visibility(predicted position)       visibility = 3-D line of sight past rocks, plus shadows
```

It is averaged over 64 samples of the track's uncertainty. A miss behind a rock then carries almost no evidence
that the person left, so the track coasts instead of dying. The same `P_D` enters the assignment cost, so a hidden
track does not grab someone else's detection.

**Fixes.** Short hides: 0% → 36% of IDs kept (1–5 s), ID switches 76 → 54.
**Still fails.** While coasting, the track drifts out of the blind zone and its search area grows, so it often misses
the person when they reappear.
**Code.** `occlusionAware: true`, `expectedPd`. **Papers.** State-dependent detection probability [R3, R4].

### Step 2 · Negative information: "you must be somewhere I can't see"

**What changes.** A miss also moves the estimate. By Bayes, `p(x | missed) ∝ p(x) · (1 − P_D(x))`: possibilities in
plain view are ruled out, and possibilities in the blind zone survive. This is done by moment-matching the 64
weighted samples (mean, velocity and covariance), so the search area stays inside the blind zone.

**Fixes.** 1–5 s hides kept: 36% → 46%; ID switches 54 → 48.
**Still fails.** Hides longer than about 5 s: 7% kept. Motion alone cannot tell where someone went after 10 seconds
behind a boulder.
**Code.** `negInfo: true`, `_missUpdate`. **Paper.** Koch 2007, negative information [R3].

### Step 3 · Appearance re-identification: "I recognise your suit stripes"

**What changes.** Each detection carries a colour signature of the person's visible pixels (an 18-bin hue
histogram of the saturated suit-stripe colours). Tracks keep a running average of theirs. Two uses:

1. **Association.** The likelihood ratio of every detection-to-track match is multiplied by
   `exp(8 · (similarity − 0.86))` (DeepSORT-style).
2. **Re-identification.** A detection that no gate claims is compared with tracks that were lost (deleted within the
   last 30 s) or are coasting. It takes the old ID if similarity > 0.86 **and** the person could have walked there
   in the time elapsed (≤ 1.6 m/s × gap + 3 m).

**Fixes.** Long hides kept: 7% → **50%**; ID switches 48 → 23; IDF1 0.28 → 0.54.
**Limit.** With identical suits (PyTorch3D control experiment) re-ID can only pick the nearest plausible lost track.
Long hides kept falls to 28%, still better than motion alone (10%).
**Code.** `reid: true`, `_reidentify`; signatures in `src/mot/appearance.js` (modelled, calibrated to PyTorch3D
renders) and `pytorch3d/lunar3d/detector.py` (measured from rendered pixels). **Paper.** DeepSORT [P3].

### Step 4 · A second camera: "the lander can see behind your rock"

**What changes.** A lander camera at 6 m looks back across the worksite. Each frame there is one prediction, then a
gated update per camera, each with **that camera's** visibility-based `P_D`. That is centralised
measurement-level sensor fusion. A person is hidden only if both cameras miss them.

**Fixes.** With re-ID: ID switches 23 → **9** and IDF1 0.54 → **0.74** (boulders); long hides kept rise to 87% (boulders) and 86% (polar). Without awareness it backfires (Naive, above).
to 90%. Without awareness it backfires (Naive, above).
**Not modelled yet.** Camera calibration errors, time offsets, joint multi-camera assignment, track-to-track fusion.
**Code.** `tracker.stepFrames`; lander in `src/mot/world.js`.

### Step 5 · Lidar: "I bring my own light"

**What changes.** A 64-beam scanning lidar on the rover mast (140°, 45 m). It is an active sensor, so the shadow
penalty does not apply: `P_D = 0.97 · visibility`, fading from 32 m to 45 m as returns thin out. Its centroids are
precise (about 5 cm), it has fewer false alarms, and it has no colour. It is fused like the lander. Two fusion
lessons were needed to make it help rather than hurt:

1. **Model the real error, not the spec sheet.** A partly hidden person's returns come only from the visible side,
   so the centroid shifts sideways. The tracker's lidar noise model is doubled to cover that; otherwise precise
   detections fail the gate and spawn duplicate tracks.
2. **Fuse negative information across sensors.** Push a person into a blind zone only if **every** sensor missed
   them, weighted by `Π_s (1 − P_D,s(x))`. Applying the camera's miss before the lidar's hit moved estimates
   behind rocks the lidar could see past.

**Fixes.** Aware + NI with camera + lidar: 47.8 → 37.9 switches (boulders) and 51.3 → 36.3 under polar shadows.
**Code.** `sensorPd` in `src/mot/occlusion.js`, `LIDAR` in `src/mot/world.js`, `_missUpdateFused` in `tracker.js`.
The 3-D view casts the real beams for its point cloud and sensor images (`lidarScan` in `src/mot/lidarscan.js`): a range image and a reflectivity image with surface shading, noise and dropouts. Each object in the image gets an instance mask found from the ranges alone (`segmentScan`; mean IoU 0.94 against the true person masks, 3 seeds), and each person mask takes the colour and ID of its track.

### Alongside: "where could anyone be?" (particle PHD filter)

This is not an identity tracker. It is a density over the ground whose total equals the expected number of people.
The update uses the same visibility-based `P_D`, so mass pools into blind zones when people hide: negative
information you can see. Birth is measurement-driven and weighted by visibility, so no phantom people build up
behind rocks. It counts correctly (median about 5.4 expected for 5 real) and costs 2–3 ms per frame.
**Code.** `src/mot/phd.js`, `pytorch3d/lunar3d/phd.py`. **Papers.** [P4, P5, P6].

---

## Where the same trackers run

| Version | Detections come from | Trackers | Verified |
|---|---|---|---|
| Web lab (`track.html`) | Probabilistic camera model, live | All four, PHD, optional lander | Reference implementation |
| MATLAB (`matlab/lunar`) | Same model, same random numbers | Naive / Aware / Aware + NI, plus MathWorks trackers (below) | Reproduces the web lab exactly |
| PyTorch3D (`pytorch3d/`) | **Rendered images**: visible pixels, depth, colour | All, plus re-ID measured from pixels, PHD, lander | Core reproduces the web lab exactly; re-ID results in `pytorch3d/experiments/results.md` |

## MathWorks trackers (MATLAB Sensor Fusion and Tracking Toolbox, R2026b)

Built-in multi-object trackers in the installed MATLAB: `trackerGNN`, `trackerJPDA` (JIPDA with
`TrackLogic='Integrated'`), `trackerTOMHT`, `trackerPHD`, `trackerGridRFS`, plus `multiSensorTargetTracker`,
`trackFuser` and `multiObjectTracker`. MATLAB has **no built-in MOTS** (segmentation-mask tracking); MOTS methods
are deep-learning research models.

The three that fit this problem run on our exact detections through `matlab/lunar/ToolboxTracker.m`: plain, and
"told what is visible" (detectable track or branch IDs with our visibility-based `P_D`).

Results: 10 seeds × 300 s per scenario, our trackers on the same detections (`matlab/lunar/results_toolbox.md`,
via `lunar_ablation(Toolbox="all")`).

| Tracker | Boulders: same ID 1–5 s | Boulders: ID switches per run | Boulders: IDF1 | Polar: same ID 1–5 s | Polar: ID switches per run | Polar: IDF1 |
|---|---|---|---|---|---|---|
| **trackerJPDA (JIPDA) + visibility** | **57%** | **33.3** | **0.318** | 48% | **42.3** | 0.261 |
| Ours: Aware + neg. info | 49% | 44.2 | 0.285 | 42% | 50.6 | **0.272** |
| Ours: Occlusion-aware | 38% | 50.9 | 0.274 | 37% | 57.6 | 0.258 |
| trackerTOMHT + visibility | 37% | 53.1 | 0.305 | 48% | 64.8 | 0.248 |
| trackerGNN + visibility | 12% | 57.0 | 0.276 | 14% | 82.6 | 0.231 |
| trackerJPDA (JIPDA) | 0% | 56.2 | 0.270 | 0% | 130.7 | 0.185 |
| trackerGNN | 0% | 62.3 | 0.250 | 0% | 102.5 | 0.170 |
| trackerTOMHT | 0% | 72.6 | 0.225 | 0% | 109.9 | 0.158 |
| Ours: Naive | 0% | 72.9 | 0.235 | 0% | 126.5 | 0.166 |

**What it shows.**
- **The visibility idea is confirmed independently.** None of the three MathWorks trackers keeps an ID through
  occlusion unaided (0%). Told our visibility-based detection probability, all three improve.
- **JIPDA beats our motion-only tracker.** JIPDA is joint probabilistic data association with an existence
  probability; told visibility, it beats our Aware + neg. info (33 vs 44 switches, boulders). Its soft association
  weighs every nearby detection instead of committing to one, and that matters when people walk close together.
  Bringing JPDA-style soft association into our tracker is the clear next improvement.
- **Re-ID still leads overall.** Our re-ID tracker (23 switches per run, camera only) uses appearance, which none
  of these trackers has. The best combination would be JIPDA association + negative information + re-ID.

