# PyTorch3D lab: seeing, re-identifying and imagining people on the Moon

The lunar occlusion tracker, now fed by a **rendered camera**. PyTorch3D draws the scene from the rover's 2.2 m mast,
and optionally from a lander. Detections come from the pixels:
- **visible fraction** = visible pixels / full silhouette;
- **range** from the depth map, plus stereo noise;
- **bearing** from where the visible pixels sit in the image;
- **appearance**: a colour signature of the person's visible pixels.

Three additions over the web and MATLAB labs:

1. **Re-identification.** Tracks remember what their person looks like: the suit-stripe colours, which is why NASA
   paints them on. Someone who reappears after a long hide, where they could plausibly have walked, gets their old ID
   back. Appearance also adds a likelihood ratio to every detection-to-track match (DeepSORT style).
2. **"Where could anyone be?"** A particle PHD filter. Its density over the ground pools into blind zones when someone
   disappears, because the update uses each spot's visibility-based detection probability (negative information,
   made visible). Its total mass is the expected number of people.
3. **Two cameras.** A lander camera at 6 m looks back across the worksite. Each frame is fused into the same tracker,
   one gated update per camera with that camera's own visibility.

## Run

```bash
cd pytorch3d
source .venv/bin/activate
python -m lunar3d.viz --seconds 90 --out lunar_reid.mp4                     # re-ID video
python -m lunar3d.viz --seconds 90 --phd --two-cameras --out lunar_all.mp4  # + PHD map + lander camera
python experiments/run_reid.py --seed 1 --seconds 300                       # one experiment run
python experiments/summarize.py                                             # pooled table -> experiments/results.md
python tests/test_crosscheck.py                                             # Python port == web lab, exactly
```

## How it is built (and why)

| File | What it does |
|---|---|
| `lunar3d/world.py`, `occlusion.py`, `tracker.py`, `metrics.py` | Python port of the web lab. With one camera and re-ID off it reproduces the JavaScript **exactly** (`tests/test_crosscheck.py`: same detections, ID switches, IDF1, MOTA on the boulder, polar and crossing scenarios) |
| `lunar3d/shapes.py` | Meshes: terrain with craters, faceted boulders, striped astronauts, rover, lander |
| `lunar3d/render.py` | PyTorch3D rendering (pattern from *3D Deep Learning with Python*, ch. 2). The CPU rasteriser tests every pixel against every triangle, so the static scenery is rendered **once** per camera and each frame only small windows around each astronaut are rendered with the same pinhole, then composited by depth. That is exact and ~20x faster (0.13 s per frame) |
| `lunar3d/detector.py` | Pixels to detections and appearance signatures. Adds sensor noise and exposure changes. False alarms take the appearance of whatever rock is there |
| `lunar3d/phd.py` | Particle PHD filter with visibility-dependent P_D and measurement-driven birth |
| `lunar3d/session.py`, `viz.py` | One experiment (rigs, trackers, metrics); the video |

## Honest notes

- **Re-ID here is close to best case.** The stripes were chosen to be distinguishable: same person 0.99 cosine
  similarity, different people about 0.10. The web lab's gold and orange stripes were confused with the gold visor,
  so the stripe palette was changed. Real imagery is harder. The **identical suits** experiment shows the other end,
  where appearance cannot help.
- The tracker's boulder map is smooth ellipsoids while the rendered rocks are bumpy. That map error is realistic and
  deliberate.
- PyTorch3D does not cast shadows. In the polar scenario, shadowed people are darkened in proportion to their
  geometric shadow fraction.
- The detector is the renderer's own instance mask plus noise (an "oracle segmentation" stand-in for a Mask R-CNN),
  not a trained network.

## Setup on Apple Silicon

PyTorch3D has no macOS wheel. It was built from source (`~/projects/_reference/pytorch3d-src`) with one patch:
`setup.py` passes `-std=c++17`, but PyTorch 2.14 headers need C++20. CPU only; there are no MPS kernels.

Results: see `experiments/results.md`.
