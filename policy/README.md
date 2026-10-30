# Multimodal Policy Lab: imitation learning when the expert has two answers

**Question.** An expert rover drives around a rock, on the left or the right. When the rock is dead ahead it picks
either side, 50/50. Three policies learn from the same demonstrations. Which ones copy a *choice*, and which one
copies the *average* of the two choices, which points straight at the rock?

![Rock dead ahead, run 37: the expert passes right, BC heads for the middle and crashes, the autoregressive policy passes left, diffusion passes right](../docs/img/policy_lab_dead_ahead.gif)

*Run 37 with the rock dead ahead (replay it: `policy.html?run=37`). Thin lines: plans each policy is considering;
dashed ring: where the vehicle's centre may not go.*

| Policy | Output | Loss | Reference |
|---|---|---|---|
| BC (regression) | 8 actions directly | squared error | behaviour cloning baseline |
| Autoregressive | 8 action tokens, one at a time | cross-entropy | RT-2 (Brohan et al. 2023), FAST tokens (Pertsch et al. 2025) |
| Diffusion | 8 actions, denoised from noise in 50 steps | noise-prediction error | Diffusion Policy (Chi et al. 2023) |

Live, in the browser: `npm start`, then open `http://localhost:8080/policy.html`, or the
[hosted page](https://sadeeqgandalf.github.io/lunar-occlusion-tracking/policy.html). The page runs the exported networks
themselves (float16 weights, ported forward passes), checked against PyTorch by `test/policy.test.js`. A Moon/Road switch
changes only the look: the stalled car on the road is the same circular obstacle as the rock.

## Results

Closed loop: each policy drives every test world from start to goal, re-planning every 1 s from noisy perception and
executing 4 of its 8 actions. A crash is the vehicle body (radius 0.5 m) touching the obstacle. All policies see the same
400 worlds per test and the same perception noise. Five training seeds (each with its own demonstrations, initial weights
and batch order); mean [min–max] after 60 epochs.

| Policy | Random worlds: crash | Rock dead ahead, σ = 0.05 m: crash | Dead ahead: passed on the left |
|---|---|---|---|
| Expert | 0% | 0% | 51% |
| BC (regression) | 0.6% [0.0–1.3] | **19.2% [10.2–33.0]** | 54% [5–88] |
| Autoregressive | 0.3% [0.0–0.7] | 3.5% [1.8–6.0] | 60% [51–76] |
| Diffusion | 0.1% [0.0–0.5] | 0.5% [0.0–2.0] | 56% [41–80] |

Every run that did not crash reached the goal. Per-seed values: `policy/results.json` (`seeds`).

What the numbers say:

- **Averaging only hurts when the data has two answers.** In random worlds all three policies are about as safe as the
  expert. With the rock dead ahead, BC crashes in 19% of runs, more than any other policy in every seed: it steers
  between the two choices it saw and commits late.
- **Sharp perception exposes it.** At σ = 0.4 m the noisy rock position decides the side for BC as well
  ("Blurry perception" in the lab).
- **BC's side is a fixed bias, not a choice.** Its share of left passes ranges from 5% to 88% across seeds: each fit
  breaks the tie the same way every time. The autoregressive and diffusion policies sample a side, but neither keeps the
  expert's 50/50 balance exactly (51–76% and 41–80% left).
- **Training loss does not predict safety.** BC's loss falls as steadily as the others' while its dead-ahead crash rate
  stays high (loss chart in the lab).

## How it is built

| File | What it does |
|---|---|
| [`sim.py`](sim.py) | world, perception noise, the expert (tangent to the rock's keep-out circle, side fixed per run), closed-loop rollout |
| [`dataset.py`](dataset.py) | 800 expert runs, about 15,900 (observation, 8-action chunk) pairs; steering noise on executed actions (DART) |
| [`tokenizer.py`](tokenizer.py) | uniform (RT-2) and DCT (FAST without BPE) action tokenizers |
| [`model.py`](model.py) | BC regression MLP and the autoregressive transformer (2 layers, d = 64) |
| [`diffusion.py`](diffusion.py) | DDPM policy, cosine schedule, FiLM-conditioned MLP |
| [`train_eval.py`](train_eval.py) | trains all three (same data, 60 epochs, AdamW), scores them closed loop, exports seed 0's weights at epochs 0, 2, 10, 60 |
| [`test_policy.py`](test_policy.py) | expert never crashes, takes both sides, tokenizers round-trip, schedule ends in noise |

Browser port: [`src/policy/sim.js`](../src/policy/sim.js), [`src/policy/policies.js`](../src/policy/policies.js);
page: [`src/ui/policy-app.js`](../src/ui/policy-app.js). Every run on the page has a link (`?run=…`) that replays it exactly.

## Reproduce

```bash
pip install torch numpy pytest
python policy/train_eval.py      # 5 seeds, about 4 min on a laptop CPU; writes results.json and weights/
python -m pytest policy          # Python checks
npm test                         # includes the JavaScript-vs-PyTorch parity tests
```

## Limits

One obstacle on flat ground; the policy sees the obstacle's position and size, not pixels; unicycle motion without
slip; a geometric expert, not a human driver; small networks trained for one task.
