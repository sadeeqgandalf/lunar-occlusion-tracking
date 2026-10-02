# MATLAB: Lunar Occlusion Tracking Lab

A MATLAB version of the web lab: rover-mast-camera tracking of astronauts walking among boulders on the Moon.
It compares how trackers keep each person's identity when a rock or a cast shadow hides them.

```matlab
cd matlab/lunar
lunar_lab                                  % live 3-D lab (boulder field, seed 8)
lunar_lab(Scenario="polar")                % south pole: long shadows hide people too
lunar_lab(Video="lunar.mp4", Seconds=90)   % record instead of showing a window
lunar_crosscheck                           % same seed in MATLAB and in the web lab's JavaScript: compare
lunar_ablation(Seeds=1:5, Seconds=300)     % Monte Carlo comparison with 95% intervals -> results.md
```

**What you see in `lunar_lab`.** A 3-D scene: lit regolith, boulders, cast shadows, Earth, and the rover with its 2.2 m
camera mast. The rover camera's own rendered view sits underneath, and a scoreboard, People table and event feed sit
on the right.
- **P1–P5** (suit-stripe colour) are the real astronauts. NASA marks EVA suits with coloured stripes so cameras can
  tell crew apart.
- **#11, #12, …** are the ID the selected tracker gave each person: ring and tag in 3-D, mask colour and box in the
  camera view.
- **#11?** with a shaded area means the person is hidden, and the area is the tracker's best guess of where they are.
- Purple wedges are where tall rocks block the camera. Orange rings are this frame's detections, the only thing the
  trackers receive.

Drag to orbit and scroll to zoom. **Space** pauses. **Show** chooses which tracker's view is drawn.

## Trackers

| Tracker | What it does when a person is not detected |
|---|---|
| Aware + neg. info | Expects the miss behind a rock and moves its guess into the blind spot (Koch 2007, negative information) |
| Occlusion-aware | Expects the miss: detection probability = P_D x 3-D visibility of the predicted position |
| Naive | Treats every miss as evidence the person left, and deletes them within ~0.2 s |
| trackerGNN + visibility | MathWorks Sensor Fusion and Tracking Toolbox `trackerGNN`, told each track's visibility-based detection probability every frame (`HasDetectableTrackIDsInput`). An independent implementation of the same idea. |

## Files

| File | Role |
|---|---|
| `LunarWorld.m` | Boulders (half-buried ellipsoids with heights), goal-directed walkers, camera detections with stereo range noise and false alarms |
| `occ_visibility.m`, `occ_shadow.m`, `occ_pd.m` | 3-D line of sight (16 sight lines per person), cast shadows under a low sun, detection probability (vectorised) |
| `OccTracker.m` | Our trackers: constant-velocity EKF, chi-square gate, Hungarian assignment with a "missed" option, Bernoulli existence, negative information |
| `ToolboxGNN.m` | `trackerGNN` wrapped so it runs on the same detections and is scored the same way |
| `MotMetrics.m` | ID switches, IDs kept through occlusion, MOTA, IDF1, GOSPA |
| `Mulberry32.m`, `hungarian.m` | Bit-identical ports of the web lab's random generator and assignment, so runs match exactly |
| `lunar_lab.m` | The visual lab |
| `lunar_crosscheck.m`, `js_reference.mjs` | Run the same scenario in MATLAB and JavaScript and compare |
| `lunar_ablation.m`, `lunar_ablation_merge.m` | Many-seed comparison, written to `results.md` |

## Verified

`lunar_crosscheck` reproduces the web lab exactly: same boulders, walks and detections (14,985 detections over 600 s,
seed 7), and identical ID switches, IDs kept, IDF1 and MOTA for all three trackers, on the boulder field and polar
scenarios. MATLAB runs the simulation about 30x faster than real time.

Needs MATLAB R2026b (tested). `trackerGNN` needs the Sensor Fusion and Tracking Toolbox; without it, run
`lunar_lab(Toolbox=false)`. The cross-check needs Node.js.
