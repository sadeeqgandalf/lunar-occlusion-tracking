# The event camera as the rover's only eyes

The ordinary mast camera is replaced by an event camera: same position, view and depth noise, but it detects motion, not presence, has no colour, and is barely affected by shadow. Its response to motion is fitted to a real recording (EVOS, Crain & Ulrich 2025; see `src/mot/occlusion.js`). Same people and paths for both sensors.

20 seeds × 300 s per row. Brackets are Wilson 95% intervals; ± is 1.96 × standard error over seeds. Regenerate with `node experiments/event_camera.mjs 20 300`.

A "hide" is any spell where the sensor in use cannot detect a person who is in its field of view. For the event camera that includes standing still.

## Boulder field (people keep walking)

| Eyes | Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches / run | IDF1 | MOTA |
|---|---|---|---|---|---|---|
| Camera | Aware + neg. info | 46% [40–52] (269) | 7% [6–10] (619) | 47.8 ± 5.3 | 0.279 | 0.90 |
| Camera | Occlusion-aware | 36% [31–42] (270) | 4% [3–6] (632) | 54.0 ± 5.3 | 0.264 | 0.93 |
| Camera | Naive | 0% [0–1] (267) | 0% [0–1] (635) | 75.6 ± 9.1 | 0.229 | 0.94 |
| Camera | Aware + neg. info + re-ID | 89% [85–92] (275) | 50% [47–54] (626) | 23.3 ± 3.2 | 0.540 | 0.80 |
| Event camera | Aware + neg. info | 49% [43–55] (264) | 8% [6–10] (606) | 51.4 ± 5.8 | 0.266 | 0.88 |
| Event camera | Occlusion-aware | 43% [38–49] (277) | 5% [3–7] (632) | 57.5 ± 6.3 | 0.248 | 0.91 |
| Event camera | Naive | 0% [0–1] (262) | 0% [0–1] (626) | 134.4 ± 9.1 | 0.151 | 0.89 |
| Event camera | Aware + neg. info + re-ID | 49% [43–55] (264) | 8% [6–10] (606) | 51.4 ± 5.8 | 0.266 | 0.88 |

## South pole, long shadows

| Eyes | Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches / run | IDF1 | MOTA |
|---|---|---|---|---|---|---|
| Camera | Aware + neg. info | 45% [40–51] (264) | 8% [6–10] (600) | 51.3 ± 5.4 | 0.260 | 0.84 |
| Camera | Occlusion-aware | 36% [31–42] (271) | 5% [4–7] (616) | 60.6 ± 5.7 | 0.241 | 0.86 |
| Camera | Naive | 0% [0–2] (242) | 0% [0–1] (563) | 129.4 ± 13.0 | 0.159 | 0.80 |
| Camera | Aware + neg. info + re-ID | 87% [82–90] (274) | 50% [46–54] (610) | 24.4 ± 4.3 | 0.518 | 0.75 |
| Event camera | Aware + neg. info | 43% [37–49] (263) | 7% [5–9] (626) | 52.0 ± 4.4 | 0.245 | 0.88 |
| Event camera | Occlusion-aware | 40% [35–46] (272) | 7% [5–9] (642) | 55.3 ± 5.3 | 0.249 | 0.91 |
| Event camera | Naive | 0% [0–1] (265) | 0% [0–1] (615) | 139.0 ± 9.9 | 0.143 | 0.88 |
| Event camera | Aware + neg. info + re-ID | 43% [37–49] (263) | 7% [5–9] (626) | 52.0 ± 4.4 | 0.245 | 0.88 |

## Stop and work (people pause about 6 s at each site)

| Eyes | Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches / run | IDF1 | MOTA |
|---|---|---|---|---|---|---|
| Camera | Aware + neg. info | 56% [47–64] (138) | 8% [6–11] (510) | 38.3 ± 3.7 | 0.332 | 0.93 |
| Camera | Occlusion-aware | 30% [23–39] (138) | 3% [2–5] (517) | 44.5 ± 4.2 | 0.315 | 0.94 |
| Camera | Naive | 0% [0–3] (130) | 0% [0–1] (508) | 61.8 ± 7.6 | 0.289 | 0.95 |
| Camera | Aware + neg. info + re-ID | 89% [83–93] (142) | 51% [47–56] (514) | 19.4 ± 2.4 | 0.573 | 0.84 |
| Event camera | Aware + neg. info | 19% [16–24] (339) | 3% [3–5] (974) | 72.0 ± 5.3 | 0.157 | 0.84 |
| Event camera | Occlusion-aware | 14% [11–18] (358) | 2% [1–3] (1007) | 77.5 ± 5.5 | 0.155 | 0.88 |
| Event camera | Naive | 0% [0–1] (347) | 0% [0–0] (975) | 133.0 ± 12.3 | 0.127 | 0.86 |
| Event camera | Aware + neg. info + re-ID | 19% [16–24] (339) | 3% [3–5] (974) | 72.0 ± 5.3 | 0.157 | 0.84 |

## What it shows

- **A naive tracker does much worse with an event camera** (boulder field: 75.6 → 134.4 ID switches per run). Every dip in motion counts as evidence the person left.
- **An occlusion-aware tracker with an event camera is about level with one using the ordinary camera while people keep moving** (47.8 vs 51.4 switches; the intervals overlap). The same "did I expect to see you?" reasoning covers low motion as well as rocks.
- **Under polar shadow the event camera detects more** (MOTA 0.88 vs 0.84 for Aware + neg. info), because shadow costs it little.
- **No colour means no re-identification.** The camera with re-ID gets 23.3 switches and keeps 89% of short hides; the event camera cannot use it and stays at 51.4 and 49%.
- **People who stop are the weak point** (stop-and-work: 38.3 → 72.0 switches; short hides kept 56% → 19%). The tracker assumes constant velocity, so it expects a stopped person to keep walking and to be visible. A motion model with a "stopped" mode is the fix to try next.

## Limits

- The motion response comes from one EVOS recording of a 0.3 m spacecraft model at about 2 m, applied to people at 10–45 m. It ignores limb motion, which would make walking people easier to see than this model says.
- The shadow factor (10% fewer detections in full shadow) is from event counts in dark vs nominal light on one manoeuvre, not from a detection study.
- The event camera is modelled as a stereo pair with the ordinary camera's depth noise. A single event camera gives direction only.
- Simulated walkers and a sensor model, as everywhere in this lab.
