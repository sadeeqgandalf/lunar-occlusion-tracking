# PyTorch3D results: re-identification on rendered detections

Every detection comes from a PyTorch3D render (visible pixels, depth). All trackers in a row group see the same detections.
"Same ID after a hide": of people who were tracked, hid for >= 1 s and came back, the share that came back with the same ID.

## boulders · distinct suit stripes · 1 camera · seeds [1, 2, 3, 4, 5] × 300 s

| Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches per run | IDF1 |
|---|---|---|---|---|
| Naive | 1% (0–7), 1/83 | 0% (0–3), 0/123 | 103.4 | 0.174 |
| Aware + neg. info | 61% (51–71), 51/83 | 8% (5–14), 10/122 | 77.2 | 0.193 |
| Aware + neg. info + re-ID | 84% (74–90), 71/85 | 50% (41–58), 61/123 | 26.4 | 0.431 |

## boulders · identical suits · 1 camera · seeds [1, 2, 3] × 300 s

| Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches per run | IDF1 |
|---|---|---|---|---|
| Naive | 0% (0–8), 0/44 | 0% (0–6), 0/59 | 99.7 | 0.180 |
| Aware + neg. info | 56% (41–70), 24/43 | 10% (5–20), 6/61 | 72.3 | 0.207 |
| Aware + neg. info + re-ID | 73% (58–84), 32/44 | 28% (18–40), 17/61 | 45.7 | 0.332 |
