"""The Policy Lab world: one rover, one rock, one goal; the expert; closed-loop rollouts.

Frames: rover frame x forward, y left (metres, radians). Every function is batched over episodes.
Mirrored in src/policy/sim.js; test/policy.test.js checks that both give the same numbers.
"""
import math

import torch

DT = 0.25          # s per action
H = 8              # actions per chunk (2 s of driving)
EXEC = 4           # actions executed before the policy is asked again (receding horizon, 1 s)
V_MAX, W_MAX = 1.2, 0.8   # speed (m/s) and turn rate (rad/s) limits
ROVER_R = 0.5      # rover body radius (m); a crash is the body touching the rock
GOAL_X, GOAL_TOL = 16.0, 1.5
T_MAX = 40.0       # s; still driving after this = stuck
AMBIGUOUS = 0.35   # rock this close to the straight line to the goal: the expert picks a side 50/50


def sample_worlds(n: int, gen: torch.Generator) -> dict:
    u = lambda: torch.rand(n, generator=gen)
    return {
        "rock_x": 5.0 + 4.0 * u(),
        "rock_y": (u() - 0.5) * 2.4,
        "rock_r": 0.8 + 0.7 * u(),
        "goal_y": (u() - 0.5) * 3.0,
        "sigma": 0.05 + 0.35 * u(),   # perception noise (1 std, m) on the rock position
    }


def to_unit(a: torch.Tensor) -> torch.Tensor:
    """(..., 2) [v, w] -> [-1, 1] per dimension."""
    return torch.stack([a[..., 0] / V_MAX * 2 - 1, a[..., 1] / W_MAX], -1)


def from_unit(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(-1, 1)
    return torch.stack([(u[..., 0] + 1) / 2 * V_MAX, u[..., 1] * W_MAX], -1)


def observe(pose: torch.Tensor, w: dict, gen: torch.Generator) -> torch.Tensor:
    """What the rover sees: [rock dx, rock dy, rock r, goal dx, goal dy, sigma] in its own frame; rock position noisy."""
    x, y, th = pose.unbind(-1)
    c, s = torch.cos(th), torch.sin(th)
    rel = lambda px, py: ((px - x) * c + (py - y) * s, -(px - x) * s + (py - y) * c)
    rx, ry = rel(w["rock_x"], w["rock_y"])
    gx, gy = rel(torch.full_like(x, GOAL_X), w["goal_y"])
    noise = torch.randn(2, len(x), generator=gen) * w["sigma"]
    return torch.stack([rx + noise[0], ry + noise[1], w["rock_r"], gx, gy, w["sigma"]], -1)


def rock_offset(obs: torch.Tensor) -> torch.Tensor:
    """Signed distance of the rock from the straight line to the goal (+ = rock on the left)."""
    head = torch.atan2(obs[:, 4], obs[:, 3])
    return -obs[:, 0] * torch.sin(head) + obs[:, 1] * torch.cos(head)


def choose_side(obs: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """+1 = pass the rock on its left, -1 = on its right. Away from the rock, or a coin flip when it is dead ahead."""
    off = rock_offset(obs)
    coin = torch.where(torch.rand(len(off), generator=gen) < 0.5, 1.0, -1.0)
    return torch.where(off.abs() < AMBIGUOUS, coin, -torch.sign(off))


def expert_chunk(obs: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """H actions of the expert: head for the goal; if the rock blocks that line, steer along the tangent of the rock
    grown by the body radius and a margin that widens with perception noise. Planned on the unicycle model."""
    n = len(obs)
    rock_x, rock_y, r, goal_x, goal_y, sigma = obs.unbind(-1)
    keep_out = r + ROVER_R + 0.25 + 2.0 * sigma
    x, y, th = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    out = []
    for _ in range(H):
        c, s = torch.cos(th), torch.sin(th)
        rx, ry = (rock_x - x) * c + (rock_y - y) * s, -(rock_x - x) * s + (rock_y - y) * c
        gx, gy = (goal_x - x) * c + (goal_y - y) * s, -(goal_x - x) * s + (goal_y - y) * c
        to_goal = torch.atan2(gy, gx)
        along = rx * torch.cos(to_goal) + ry * torch.sin(to_goal)
        across = -rx * torch.sin(to_goal) + ry * torch.cos(to_goal)
        blocked = (along > 0) & (across.abs() < keep_out) & (along < torch.hypot(gx, gy))
        dist = torch.hypot(rx, ry)
        tangent = torch.atan2(ry, rx) + side * torch.asin((keep_out / dist).clamp(max=1.0))
        heading = torch.where(blocked, tangent, to_goal)
        w = (2.0 * heading).clamp(-W_MAX, W_MAX)
        v = (1.0 - 0.8 * sigma - 0.3 * w.abs()).clamp(0.3, V_MAX)
        out.append(torch.stack([v, w], -1))
        th = th + w * DT
        x, y = x + v * torch.cos(th) * DT, y + v * torch.sin(th) * DT
    return torch.stack(out, 1)


class Expert:
    """Commits to one side per episode, chosen from its first look at the rock."""

    def __init__(self, gen: torch.Generator):
        self.gen, self.side = gen, None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.side is None:
            self.side = choose_side(obs, self.gen)
        return expert_chunk(obs, self.side)


@torch.no_grad()
def rollout(policy, w: dict, gen: torch.Generator, steer_noise: float = 0.0, record: bool = False) -> dict:
    """Drive every episode closed loop: observe, plan a chunk, execute EXEC actions, repeat.
    steer_noise perturbs the executed turn rate only (DART-style data collection; labels stay clean)."""
    n = len(w["rock_x"])
    pose = torch.zeros(n, 3)
    status = torch.zeros(n, dtype=torch.long)       # 0 driving, 1 goal, 2 crash, 3 stuck
    min_gap = torch.full((n,), math.inf)
    passed_y = torch.full((n,), math.nan)           # rover y relative to the rock when passing it
    path = [pose[:, :2].clone()]
    obs_log, act_log = [], []
    t = 0.0
    while (status == 0).any() and t < T_MAX:
        live = status == 0
        obs = observe(pose, w, gen)
        chunk = policy(obs)
        if record:
            obs_log.append(obs[live]); act_log.append(chunk[live])
        for k in range(EXEC):
            v, om = chunk[:, k, 0], chunk[:, k, 1] + steer_noise * torch.randn(n, generator=gen)
            th = pose[:, 2] + om.clamp(-W_MAX, W_MAX) * DT
            step = torch.stack([pose[:, 0] + v * torch.cos(th) * DT, pose[:, 1] + v * torch.sin(th) * DT, th], -1)
            was_before = pose[:, 0] < w["rock_x"]
            pose = torch.where(live[:, None], step, pose)
            crossing = live & was_before & (pose[:, 0] >= w["rock_x"])
            passed_y = torch.where(crossing, pose[:, 1] - w["rock_y"], passed_y)
            gap = torch.hypot(pose[:, 0] - w["rock_x"], pose[:, 1] - w["rock_y"]) - w["rock_r"] - ROVER_R
            min_gap = torch.where(live, torch.minimum(min_gap, gap), min_gap)
            status = torch.where(live & (gap < 0), 2, status)
            at_goal = torch.hypot(pose[:, 0] - GOAL_X, pose[:, 1] - w["goal_y"]) < GOAL_TOL
            status = torch.where((status == 0) & at_goal, 1, status)
            live = status == 0
            t += DT
            path.append(pose[:, :2].clone())
    status = torch.where(status == 0, 3, status)
    out = {"status": status, "min_gap": min_gap, "passed_y": passed_y, "path": torch.stack(path, 1)}
    if record:
        out["obs"], out["actions"] = torch.cat(obs_log), torch.cat(act_log)
    return out
