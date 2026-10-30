"""Expert demonstrations: the expert drives many worlds closed loop; every decision is one training pair
(observation, chunk of H actions). The executed steering gets a little noise so the data also covers states just
off the expert's path (DART, Laskey et al. 2017); the recorded labels stay the expert's clean actions."""
import torch

import sim


def build(episodes: int = 800, seed: int = 0, steer_noise: float = 0.15) -> dict:
    gen = torch.Generator().manual_seed(seed)
    worlds = sim.sample_worlds(episodes, gen)
    run = sim.rollout(sim.Expert(gen), worlds, gen, steer_noise=steer_noise, record=True)
    assert (run["status"] == 1).all(), "the expert must reach the goal in every demonstration"
    return {"obs": run["obs"], "actions": run["actions"], "episodes": episodes}


if __name__ == "__main__":
    d = build()
    obs = d["obs"]
    head_on = sim.rock_offset(obs).abs() < sim.AMBIGUOUS
    print(f"{len(obs)} pairs from {d['episodes']} episodes; rock dead ahead in {head_on.float().mean():.0%} of them")
