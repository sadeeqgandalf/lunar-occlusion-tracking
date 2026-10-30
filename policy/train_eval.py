"""Train the three policies on the same demonstrations, score them closed loop, export what the page runs.

    python policy/train_eval.py            # 5 training seeds, about 5 min on a laptop CPU

Writes policy/results.json (losses and closed-loop scores per checkpoint) and policy/weights/ (float16 weights per
checkpoint, manifest, parity vectors for test/policy.test.js). Scores are computed with the float16 weights, so
they describe exactly the networks the browser runs.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import dataset
import sim
from diffusion import DiffusionPolicy
from model import ARPolicy, BCPolicy
from tokenizer import DCTTokenizer

EPOCHS = 60
CHECKPOINTS = (0, 2, 10, EPOCHS)
TEST_EPISODES = 400
SHARP_SIGMA = 0.05   # perception noise of the "dead ahead" test: sharp eyes, so nothing breaks the tie
TEMPERATURE = 1.0
HERE = Path(__file__).parent
STATUS = {1: "goal", 2: "crash", 3: "stuck"}


def half_rounded(model):
    """Copy of the model with every exported tensor rounded to float16, as shipped to the browser."""
    m = copy.deepcopy(model).eval()
    with torch.no_grad():
        for name, t in m.state_dict().items():
            if name not in ("betas", "abar"):
                t.copy_(t.half().float())
    return m


def exported(model):
    return [(n, t) for n, t in model.state_dict().items() if n not in ("betas", "abar")]


def test_suites():
    """'random': rock, goal and noise drawn as in training. 'dead_ahead': the rock sits exactly on the straight line
    to the goal and perception is sharp; the expert goes left or right 50/50, so an average of the two hits the rock."""
    rnd = sim.sample_worlds(TEST_EPISODES, torch.Generator().manual_seed(1))
    ahead = sim.sample_worlds(TEST_EPISODES, torch.Generator().manual_seed(2))
    ahead["rock_y"] = ahead["goal_y"] * ahead["rock_x"] / sim.GOAL_X
    ahead["sigma"] = torch.full((TEST_EPISODES,), SHARP_SIGMA)
    return {"random": rnd, "dead_ahead": ahead}


def score(make_policy, suites):
    out = {}
    for name, worlds in suites.items():
        run = sim.rollout(make_policy(), worlds, torch.Generator().manual_seed(7))   # same perception noise for all
        st, py = run["status"], run["passed_y"]
        pct = lambda x: round(100 * x.float().mean().item(), 1)
        out[name] = {**{label: pct(st == code) for code, label in STATUS.items()},
                     "left": pct(py[~torch.isnan(py)] > 0) if (~torch.isnan(py)).any() else None,   # of runs that got past the rock
                     "min_gap_median": round(run["min_gap"].median().item(), 2)}
    return out


def train(seed: int, export: bool) -> dict:
    """Train the three policies from one seed (demonstrations, initial weights, batch order) and score them.
    export: also score the intermediate checkpoints and write the weights the page runs."""
    t0 = time.time()
    torch.manual_seed(seed)
    data = dataset.build(seed=seed)
    obs, actions = data["obs"], data["actions"]
    tok = DCTTokenizer()
    unit, tokens = sim.to_unit(actions), tok.encode(actions)
    bc, ar, dp = BCPolicy(), ARPolicy(length=tok.length, vocab=tok.vocab), DiffusionPolicy()
    models = {"bc": bc, "ar": ar, "dp": dp}
    targets = {"bc": unit, "ar": tokens, "dp": unit}
    for m in models.values():
        m.norm.fit(obs)
    opts = {k: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for k, m in models.items()}
    loader = DataLoader(TensorDataset(torch.arange(len(obs))), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    suites = test_suites()
    results = {"config": {"dt": sim.DT, "horizon": sim.H, "executed": sim.EXEC, "rover_radius": sim.ROVER_R,
                          "temperature": TEMPERATURE, "diffusion_steps": dp.K, "epochs": EPOCHS},
               "dataset": {"pairs": len(obs), "episodes": data["episodes"]},
               "test": {"episodes": TEST_EPISODES, "sharp_sigma": SHARP_SIGMA},
               "expert": score(lambda: sim.Expert(torch.Generator().manual_seed(3)), suites),
               "loss": [], "checkpoints": {}}
    out = HERE / "weights"
    out.mkdir(exist_ok=True)

    for epoch in range(EPOCHS + 1):
        sums = dict.fromkeys(models, 0.0)
        batches = list(loader) if epoch else [(torch.arange(len(obs))[:2048],)]
        for (idx,) in batches:
            for k, m in models.items():
                m.train()
                loss = m.loss(obs[idx], targets[k][idx])
                if epoch:
                    opts[k].zero_grad(); loss.backward(); opts[k].step()
                sums[k] += loss.item()
        results["loss"].append({"epoch": epoch, **{k: round(v / len(batches), 4) for k, v in sums.items()}})

        if epoch == EPOCHS or (export and epoch in CHECKPOINTS):
            h = {k: half_rounded(m) for k, m in models.items()}
            g = torch.Generator().manual_seed(11)
            policies = {"bc": h["bc"].act,
                        "ar": lambda o: tok.decode(h["ar"].sample(o, TEMPERATURE, g)),
                        "dp": lambda o: h["dp"].sample(o, g)}
            results["checkpoints"][str(epoch)] = {k: score(lambda: p, suites) for k, p in policies.items()}
            if export:
                blob = np.concatenate([t.numpy().astype("<f2").ravel() for m in h.values() for _, t in exported(m)])
                (out / f"epoch_{epoch}.bin").write_bytes(blob.tobytes())
            r = results["checkpoints"][str(epoch)]
            print(f"seed {seed} epoch {epoch:2d}  crashes (random / dead ahead):  "
                  + "  ".join(f"{k} {r[k]['random']['crash']:5.1f}% / {r[k]['dead_ahead']['crash']:5.1f}%" for k in models)
                  + f"   [{time.time() - t0:.0f} s]")

    if export:
        (out / "manifest.json").write_text(json.dumps({
            "format": "float16 little-endian; per checkpoint, the tensors below concatenated in this order",
            "checkpoints": list(CHECKPOINTS),
            "tensors": {k: [[n, list(t.shape)] for n, t in exported(m)] for k, m in models.items()}}))
        write_parity(out / "parity.json", h, tok, suites["random"])
    return results


def main(seeds: int):
    """Seed 0 is the run the page shows (weights, every checkpoint); the other seeds give the spread of the final scores."""
    results = train(0, export=True)
    finals = [results["checkpoints"][str(EPOCHS)]] + [train(s, export=False)["checkpoints"][str(EPOCHS)] for s in range(1, seeds)]
    results["seeds"] = {"n": seeds, "final": {k: {suite: {m: [f[k][suite][m] for f in finals] for m in ("goal", "crash", "left")}
                                                     for suite in ("random", "dead_ahead")} for k in finals[0]}}
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nepoch {EPOCHS}, {seeds} seeds: crash rate mean [min, max]")
    for k, v in results["seeds"]["final"].items():
        print(f"  {k}: " + "   ".join(f"{suite} {np.mean(x['crash']):5.1f}% [{min(x['crash'])}, {max(x['crash'])}]" for suite, x in v.items()))
    print("-> policy/results.json, policy/weights/")


def write_parity(path, h, tok, worlds):
    """Reference inputs and outputs of every network and of the expert, for the JavaScript port's tests."""
    g = torch.Generator().manual_seed(5)
    obs = sim.observe(torch.tensor([[0.0, 0, 0], [2, 0.5, 0.2], [4, -0.8, -0.3], [1, 0.2, 0.1]]),
                      {k: v[:4] for k, v in worlds.items()}, g)
    side = torch.tensor([1.0, -1, 1, -1])
    chunk = sim.expert_chunk(obs, side)
    x, k = torch.randn(4, sim.H, 2, generator=g), torch.tensor([0, 10, 25, 49])
    with torch.no_grad():
        ref = {"obs": obs, "side": side, "expert": chunk, "tokens": tok.encode(chunk), "decoded": tok.decode(tok.encode(chunk)),
               "bc": h["bc"].act(obs), "ar_logits": h["ar"].logits(obs, tok.encode(chunk)[:, :-1]),
               "dp_x": x, "dp_k": k, "dp_eps": h["dp"].eps(x, k, obs), "abar": h["dp"].abar}
    path.write_text(json.dumps({n: v.tolist() for n, v in ref.items()}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the three policies, score them closed loop, export the page's weights.")
    ap.add_argument("--seeds", type=int, default=5, help="training seeds (seed 0 is the one exported)")
    main(ap.parse_args().seeds)
