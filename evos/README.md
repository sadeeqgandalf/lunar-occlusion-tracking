# EVOS: real event-camera data behind the lab's event camera

The lab's event camera (`Eyes: event camera` in the Occlusion Lab) detects motion, not presence. How strongly it
responds to motion is not a guess: it is fitted to a real recording from the EVOS dataset.

**Dataset.** Crain, A., and Ulrich, S. (2025). *EVent-based Observation of Spacecraft (EVOS): a neuromorphic dataset
for spacecraft proximity operations.* Federated Research Data Repository. doi:10.20383/103.01538. Fifteen recordings
from an event camera (IniVation DVXplorer Micro, 640 x 480) on a stationary chaser spacecraft model watching a moving
target, with motion-capture ground truth, in nominal, dark (0.6 lux) and sun-glare lighting.

**The data is not in this repository.** Download it from the source:

```bash
cd evos
python fetch.py CC-T-NOM CIRC-TR-NOM CIRC-TR-DARK CIRC-TR-SG   # about 680 MB
```

## Scripts

Run them from this folder, next to the downloaded data (`uv run --with aedat --with numpy --with pandas ...`).

| Script | What it does |
|---|---|
| `fit_motion.py CC-T-NOM` | Fits the event camera's response to apparent motion. Result used in `src/mot/occlusion.js`: `EVENT_S0 = 0.0077 rad/s`, `EVENT_LOOM = 0.10` (correlation 0.97 with the measured event rate). |
| `explore.py` | First look: event pictures in three lightings, sensor activity, camera vs motion capture. |
| `evos_video.py CC-T-NOM` | Renders a recording as video: the event camera's view next to the motion-capture view of the test bed. `--lightings CIRC-TR` puts three lightings side by side. |

## What the fit says

`signal = 1 - exp(-(omega + 0.10 * loom) / 0.0077)`, where `omega` is how fast the target crosses the view (rad/s) and
`loom` is how fast it grows or shrinks (1/s). The signal reaches 63% at 0.44 degrees per second across the view, and
motion straight toward or away from the camera counts for a tenth as much.

## Limits

- One recording (`CC-T-NOM`, target moving without spinning). On a spinning run the same model fits poorly
  (correlation 0.61), because spin creates events the model does not describe.
- A 0.3 m foil-wrapped spacecraft model at about 2 m; the lab applies the fit to people at 10-45 m.
- About half the events in a recording come from faulty "hot" pixels; the scripts remove the top 0.5% of pixels by count.
- The dataset's `LICENSE.txt` says CC BY 4.0 and its `README.txt` says CC BY-NC-SA 4.0. Check with the authors before
  any use beyond research.

Lab results with the event camera as the rover's only sensor: [`experiments/event_camera.md`](../experiments/event_camera.md).
