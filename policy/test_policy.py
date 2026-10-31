"""Fast checks of the parts every result depends on.    python -m pytest policy/test_policy.py"""
import torch

import sim
from diffusion import cosine_schedule
from tokenizer import DCTTokenizer, UniformTokenizer, dct_matrix


def test_expert_never_crashes_even_with_steering_noise():
    gen = torch.Generator().manual_seed(42)
    run = sim.rollout(sim.Expert(gen), sim.sample_worlds(500, gen), gen, steer_noise=0.15)
    assert (run["status"] == 1).all()
    assert run["min_gap"].min() > 0


def test_expert_takes_both_sides_when_the_rock_is_dead_ahead():
    gen = torch.Generator().manual_seed(0)
    w = sim.on_the_line(sim.sample_worlds(400, gen))
    left = (sim.rollout(sim.Expert(gen), w, gen)["passed_y"] > 0).float().mean()
    assert 0.4 < left < 0.6


def test_tokenizers_round_trip_and_dct_is_shorter():
    chunk = sim.expert_chunk(sim.observe(torch.zeros(64, 3), sim.sample_worlds(64, torch.Generator().manual_seed(1)),
                                         torch.Generator().manual_seed(2)), torch.ones(64))
    dct, uni = DCTTokenizer(), UniformTokenizer()
    assert torch.allclose(dct_matrix(sim.H) @ dct_matrix(sim.H).T, torch.eye(sim.H), atol=1e-6)
    assert dct.encode(chunk).shape[1] == uni.encode(chunk).shape[1] // 2
    assert (dct.decode(dct.encode(chunk)) - chunk).abs().mean() < 0.06
    assert (uni.decode(uni.encode(chunk)) - chunk).abs().max() < 0.02


def test_diffusion_schedule_ends_in_pure_noise():
    _, abar = cosine_schedule(50)
    assert abar[0] > 0.99 and abar[-1] < 1e-3


def test_field_expert_reaches_the_goal_and_chooses_afresh_at_every_rock():
    gen = torch.Generator().manual_seed(0)
    run = sim.rollout(sim.Expert(gen), sim.on_the_line(sim.sample_fields(500, gen)), gen)
    assert (run["status"] == 1).all()
    side = torch.sign(run["passed_y"][:, :sim.FIELD_ROCKS])
    assert 0.4 < (side[:, 1:] == side[:, :-1]).float().mean() < 0.6
