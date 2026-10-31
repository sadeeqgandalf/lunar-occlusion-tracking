"""Expert demonstrations: the expert drives many worlds closed loop; every decision is one training pair
(observation, chunk of H actions). The executed steering gets a little noise so the data also covers states just
off the expert's path (DART, Laskey et al. 2017); the recorded labels stay the expert's clean actions.
Only runs that reach the goal are kept: all of them on the single-rock course, all but a rare one at high perception
noise in the rock field."""
import torch

import sim


def build(episodes: int = 800, seed: int = 0, steer_noise: float = 0.15, field: bool = False) -> dict:
    gen = torch.Generator().manual_seed(seed)
    worlds = (sim.sample_fields if field else sim.sample_worlds)(episodes, gen)
    run = sim.rollout(sim.Expert(gen), worlds, gen, steer_noise=steer_noise, record=True)
    reached = run["status"] == 1
    assert reached.float().mean() > 0.99, "the expert must reach the goal in nearly every demonstration"
    keep = reached[run["episode"]]
    return {"obs": run["obs"][keep], "actions": run["actions"][keep], "episodes": episodes, "reached": int(reached.sum())}


if __name__ == "__main__":
    import sys
    d = build(field="field" in sys.argv)
    obs = d["obs"]
    head_on = sim.rock_offset(obs).abs() < sim.AMBIGUOUS
    print(f"{len(obs)} pairs from {d['reached']} of {d['episodes']} episodes; nearest rock dead ahead in {head_on.float().mean():.0%} of them")
