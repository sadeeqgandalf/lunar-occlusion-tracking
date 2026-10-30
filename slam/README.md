# Lunar Polar SLAM Lab & Offline Benchmark Suite (`VGGT-SLAM` vs. `ORB-SLAM3` vs. `LiDAR-Inertial`)

This module pairs the **Interactive 3-D WebGL Lunar Polar SLAM Lab** ([`/slam.html`](../slam.html), powered by [`src/slam/engine.js`](../src/slam/engine.js) and [`src/ui/slam3d.js`](../src/ui/slam3d.js)) with an offline ground-truth recording and evaluation pipeline for **Monocular VGGT-SLAM** and **Stereo ORB-SLAM3** under lunar south-pole illumination (`30°`, `7°`, and `2°` solar elevation).

---

## 1. The SLAM lab (`/slam.html`)

```bash
npm start
open http://localhost:8080/slam.html
```

A real EKF-SLAM ([`src/slam/engine.js`](../src/slam/engine.js)) runs live on a simulated rover, next to wheel odometry alone, so you can see what SLAM buys you.

- **The loop, explained with live numbers.** The left panel shows each step as it happens: **predict** (wheels move the pose guess, uncertainty grows), **observe** (the camera finds rock features; new ones go on the map, known ones become checks), **correct** (expected vs measured range and bearing, the innovation, how far the pose moved and how much the uncertainty shrank). *One step* pauses and runs a single camera frame.
- **Localization you can see.** The rover's believed pose is drawn with its ±2σ ellipse next to the true rover. The right panel scores belief against truth: error, the filter's own ±2σ, and how often the truth is inside its 95% ellipse (an honest filter: about 95%).
- **Real-time mapping you can see.** *The rover's mind* shows its own occupancy map and landmark map, each landmark with its ±2σ ellipse. The map is drawn from the pose the rover believes, so switching *Map by* to *Wheels only* shows the same sensor data smeared by a wrong position.
- **Loop closure.** Recognising a landmark not seen for 25 s or more pulls the drifted pose back; the page logs it with the uncertainty before and after (scenario 1: about 92 s in, ±1.6 m → ±1.05 m).
- **Knobs:** camera corrections on/off, wheel slip, camera noise, sun elevation. Scenarios: loop and come home, 3° sun, 15% slip, frontier exploration.

What is real and what is simplified:
- The estimators never read the true pose or landmark positions: they get odometry and noisy range/bearing only. Truth is used to simulate the sensors and to score.
- EKF-SLAM with known data association (the camera is told which landmark it sees), in 2-D (x, y, heading). Thrun, Burgard & Fox, *Probabilistic Robotics*, ch. 10.
- Lab assumption: features in shadow are found less often as the sun drops (95% when lit, falling to about 10% at 2°).
- `npm test` checks the filter maths and that the tests fail when the map is drawn from the true pose or the innovation sign is flipped.

## 2. Offline Recording & Evaluation Pipeline

| Script | Purpose & Usage |
| :--- | :--- |
| [`record_drive.mjs`](record_drive.mjs) | Drives the RASSOR rover in headless Chromium (`Three.js`), applying physical CMOS shot+read noise ($10,000\,e^-$ full well, $3\,e^-$ read noise) and ground-metered auto-exposure (`0.25×..16×`). Outputs rectified stereo/mono PNGs and exact TUM ground-truth poses (`poses_truth.txt`). |
| [`score_ate.py`](score_ate.py) | Computes Sturm (2012) **ATE RMSE** (via Umeyama `Sim(3)` for monocular or `SE(3)` for stereo) and Geiger (2012) KITTI **RPE (% drift per 10 m)**. Includes built-in self-tests. |
| [`plot_compare.py`](plot_compare.py) | Generates side-by-side top-down comparison plots of Ground Truth vs. Monocular VGGT-SLAM (`Sim(3)`) vs. Stereo ORB-SLAM3 (`SE(3)`). |
| [`plot_paths.py`](plot_paths.py) | Highlights pure-rotation circle loops ($\ge 330^\circ$ within $30\text{ m}$) and keyframe drop gaps ($\ge 8\text{ m}$). |
| [`plot_summary.py`](plot_summary.py) | Plots log-scale trajectory error (% of distance driven) across solar elevations (`30°`, `7°`, `2°`). |
| [`make_video.py`](make_video.py) | Renders synchronized MP4 videos of the rover mast camera input alongside the growing 3-D SLAM point cloud (`map_small.npy`) and aligned keyframe trajectory. |
| [`check_groundtruth.mjs`](check_groundtruth.mjs) | Independent pinhole quaternion projection verifier confirming recorded `poses_truth.txt` aligns pixel-accurately with 3-D boulder centers. |

---

## 3. Measured results: camera SLAM on rendered lunar drives

5 drives per sun angle, about 90 m each, scored against exact simulator ground truth (error as % of distance driven, middle of 5 drives):

| Sun above horizon | VGGT-SLAM, 1 camera (Sim(3) aligned) | ORB-SLAM3, 2 cameras 0.42 m apart (SE(3) aligned) |
|---|---|---|
| 30° | 4.5% | 0.1% |
| 7° | 3.1% | 0.2% |
| 2° | 8.7% | 0.2% |

Per-drive tables: `runs/v5/scores.md` (VGGT-SLAM) and `runs/v6/scores_orb_stereo.md` (ORB-SLAM3). Chart: [`../assets/slam/summary_mono_vs_stereo.png`](../assets/slam/summary_mono_vs_stereo.png). These are simulated images: the stereo result is an upper bound, since the renderer has no lens distortion, blur or calibration error.
