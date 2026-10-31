"""The Policy Lab world: one rover, rocks, one goal; the expert; closed-loop rollouts.

Two courses share this code. Single rock: one rock between start and goal, one left-or-right choice. Rock field:
three rocks along a lane, three choices; the rover sees the next two rocks it has not passed and its own heading,
and the expert returns to the lane centre after each rock, so a rock on the centre line is a fresh 50/50 choice.
Frames: rover frame x forward, y left (metres, radians). Every function is batched over episodes; rock arrays are
(episodes, rocks). Mirrored in src/policy/sim.js; test/policy.test.js checks that both give the same numbers.
"""
import math

import torch

DT = 0.25          # s per action
H = 8              # actions per chunk (2 s of driving)
EXEC = 4           # actions executed before the policy is asked again (receding horizon, 1 s)
V_MAX, W_MAX = 1.2, 0.8   # speed (m/s) and turn rate (rad/s) limits
ROVER_R = 0.5      # rover body radius (m); a crash is the body touching a rock
GOAL_X, GOAL_TOL = 16.0, 1.5   # GOAL_X: goal distance on the single-rock course
SECONDS_PER_M = 2.5            # still driving after 2.5 s per metre to the goal = stuck (40 s on the single-rock course)
AMBIGUOUS = 0.35   # rock this close to the straight line to the goal: the expert picks a side 50/50
FIELD_ROCKS, SEEN = 3, 2       # rock field: rocks per course; rocks the rover sees at once
FIELD_OBS = 3 * SEEN + 4       # rock field observation length (single rock: 6)
LOOK = 5.0                     # rock field: the expert steers for the lane centre this far ahead (m)


def sample_worlds(n: int, gen: torch.Generator) -> dict:
    """Single-rock course."""
    u = lambda: torch.rand(n, generator=gen)
    rock_x, rock_y, rock_r = 5.0 + 4.0 * u(), (u() - 0.5) * 2.4, 0.8 + 0.7 * u()
    return {"rock_x": rock_x[:, None], "rock_y": rock_y[:, None], "rock_r": rock_r[:, None],
            "goal_x": torch.full((n,), GOAL_X), "goal_y": (u() - 0.5) * 3.0,
            "sigma": 0.05 + 0.35 * u()}   # perception noise (1 std, m) on rock positions


def sample_fields(n: int, gen: torch.Generator) -> dict:
    """Rock field: FIELD_ROCKS rocks 9-10.5 m apart along a lane (the x axis), the goal on the lane 4 m past the last.
    One more slot holds an empty marker (size 0, 10 m past the goal) that fills the rover's view once fewer than SEEN
    real rocks are left."""
    u = lambda *shape: torch.rand(n, *shape, generator=gen)
    rock_x = torch.cat([4.5 + u(1), 9.0 + 1.5 * u(FIELD_ROCKS - 1)], 1).cumsum(1)
    goal_x, goal_y = rock_x[:, -1] + 4.0, torch.zeros(n)
    pad = lambda real, last: torch.cat([real, last[:, None]], 1)
    return {"rock_x": pad(rock_x, goal_x + 10.0), "rock_y": pad((u(FIELD_ROCKS) - 0.5) * 2.4, goal_y),
            "rock_r": pad(0.6 + 0.5 * u(FIELD_ROCKS), torch.zeros(n)),
            "goal_x": goal_x, "goal_y": goal_y, "sigma": 0.05 + 0.35 * u()}


def on_the_line(w: dict) -> dict:
    """Move every real rock onto the straight line from start to goal: each one is a 50/50 choice for the expert."""
    line = w["goal_y"][:, None] * w["rock_x"] / w["goal_x"][:, None]
    return {**w, "rock_y": torch.where(w["rock_r"] > 0, line, w["rock_y"])}


def to_unit(a: torch.Tensor) -> torch.Tensor:
    """(..., 2) [v, w] -> [-1, 1] per dimension."""
    return torch.stack([a[..., 0] / V_MAX * 2 - 1, a[..., 1] / W_MAX], -1)


def from_unit(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(-1, 1)
    return torch.stack([(u[..., 0] + 1) / 2 * V_MAX, u[..., 1] * W_MAX], -1)


def obs_dim(w: dict) -> int:
    return FIELD_OBS if w["rock_x"].shape[1] > SEEN else 6


def observe(pose: torch.Tensor, w: dict, gen: torch.Generator) -> torch.Tensor:
    """What the rover sees, in its own frame: [dx, dy, r] of the next rocks it has not passed (positions noisy), then
    [goal dx, goal dy, sigma]. Single rock: [rock dx, rock dy, rock r, goal dx, goal dy, sigma]. Rock field: the
    heading relative to the lane comes before the goal, [rock 1, rock 2, heading, goal dx, goal dy, sigma]."""
    x, y, th = (v[:, None] for v in pose.unbind(-1))
    c, s = torch.cos(th), torch.sin(th)
    rel = lambda px, py: ((px - x) * c + (py - y) * s, -(px - x) * s + (py - y) * c)
    n, rocks = w["rock_x"].shape
    seen = min(rocks, SEEN)
    passed = (x > w["rock_x"] + w["rock_r"] + ROVER_R).sum(1, keepdim=True).clamp(max=rocks - seen)
    pick = lambda key: w[key].gather(1, passed + torch.arange(seen))
    rx, ry = rel(pick("rock_x"), pick("rock_y"))
    gx, gy = rel(w["goal_x"][:, None], w["goal_y"][:, None])
    noise = torch.randn(2, n, seen, generator=gen) * w["sigma"][:, None]
    lane = [th] if rocks > SEEN else []
    return torch.cat([torch.stack([rx + noise[0], ry + noise[1], pick("rock_r")], -1).flatten(1),
                      *lane, gx, gy, w["sigma"][:, None]], 1)


def seen_rocks(obs: torch.Tensor) -> int:
    return SEEN if obs.shape[1] == FIELD_OBS else 1


def rock_offset(obs: torch.Tensor, j: int = 0) -> torch.Tensor:
    """Signed distance of seen rock j from the straight line to the goal (rock field: from the lane centre line,
    which runs through the goal); + = rock on the left."""
    if seen_rocks(obs) > 1:
        th = obs[:, -4]
        return (obs[:, 3 * j] - obs[:, -3]) * torch.sin(th) + (obs[:, 3 * j + 1] - obs[:, -2]) * torch.cos(th)
    head = torch.atan2(obs[:, -2], obs[:, -3])
    return -obs[:, 3 * j] * torch.sin(head) + obs[:, 3 * j + 1] * torch.cos(head)


def choose_side(obs: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """+1 = pass the nearest rock on its left, -1 = on its right. Away from the rock, or a coin flip when it is dead ahead."""
    off = rock_offset(obs)
    coin = torch.where(torch.rand(len(off), generator=gen) < 0.5, 1.0, -1.0)
    return torch.where(off.abs() < AMBIGUOUS, coin, -torch.sign(off))


def expert_chunk(obs: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """H actions of the expert: head for the goal (rock field: for the lane centre LOOK m ahead); if a rock blocks that
    line, steer along the tangent of the nearest blocking rock, grown by the body radius and a margin that widens with
    perception noise. The nearest seen rock is passed on `side`, the next one away from the line. Planned on the
    unicycle model."""
    n, seen = len(obs), seen_rocks(obs)
    rock_x, rock_y, r = obs[:, :3 * seen].view(n, seen, 3).unbind(-1)          # (n, seen)
    goal_x, goal_y, sigma = (v[:, None] for v in obs[:, -3:].unbind(-1))      # (n, 1)
    keep_out = r + ROVER_R + 0.25 + 2.0 * sigma
    sides = torch.stack([side] + [torch.where(rock_offset(obs, j) >= 0, -1.0, 1.0) for j in range(1, seen)], 1)
    lane_x, lane_y = torch.cos(obs[:, -4:-3]), -torch.sin(obs[:, -4:-3])      # lane direction (rock field)
    x, y, th = torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1)
    out = []
    for _ in range(H):
        c, s = torch.cos(th), torch.sin(th)
        aim_x, aim_y = goal_x, goal_y
        if seen > 1:
            back = ((goal_x - x) * lane_x + (goal_y - y) * lane_y - LOOK).clamp(min=0)
            aim_x, aim_y = goal_x - back * lane_x, goal_y - back * lane_y
        rx, ry = (rock_x - x) * c + (rock_y - y) * s, -(rock_x - x) * s + (rock_y - y) * c
        gx, gy = (goal_x - x) * c + (goal_y - y) * s, -(goal_x - x) * s + (goal_y - y) * c
        ax, ay = (aim_x - x) * c + (aim_y - y) * s, -(aim_x - x) * s + (aim_y - y) * c
        to_goal = torch.atan2(ay, ax)
        along = rx * torch.cos(to_goal) + ry * torch.sin(to_goal)
        across = -rx * torch.sin(to_goal) + ry * torch.cos(to_goal)
        blocked = (along > 0) & (across.abs() < keep_out) & (along < torch.hypot(gx, gy))
        tangent = torch.atan2(ry, rx) + sides * torch.asin((keep_out / torch.hypot(rx, ry)).clamp(max=1.0))
        first = torch.where(blocked, along, math.inf).argmin(1, keepdim=True)
        heading = torch.where(blocked.any(1, keepdim=True), tangent.gather(1, first), to_goal)
        w = (2.0 * heading).clamp(-W_MAX, W_MAX)
        v = (1.0 - 0.8 * sigma - 0.3 * w.abs()).clamp(0.3, V_MAX)
        out.append(torch.cat([v, w], -1))
        th = th + w * DT
        x, y = x + v * torch.cos(th) * DT, y + v * torch.sin(th) * DT
    return torch.stack(out, 1)


class Expert:
    """Commits to a side for each rock, chosen from its first look at that rock. Rocks are told apart by their size,
    which the rover sees exactly."""

    def __init__(self, gen: torch.Generator):
        self.gen, self.side, self.rock = gen, None, None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        new = torch.ones(len(obs), dtype=torch.bool) if self.rock is None else obs[:, 2] != self.rock
        if new.any():
            pick = choose_side(obs, self.gen)
            self.side = pick if self.side is None else torch.where(new, pick, self.side)
        self.rock = obs[:, 2]
        return expert_chunk(obs, self.side)


@torch.no_grad()
def rollout(policy, w: dict, gen: torch.Generator, steer_noise: float = 0.0, record: bool = False) -> dict:
    """Drive every episode closed loop: observe, plan a chunk, execute EXEC actions, repeat.
    steer_noise perturbs the executed turn rate only (DART-style data collection; labels stay clean)."""
    n = len(w["goal_y"])
    t_max = SECONDS_PER_M * w["goal_x"].max().item()
    pose = torch.zeros(n, 3)
    status = torch.zeros(n, dtype=torch.long)       # 0 driving, 1 goal, 2 crash, 3 stuck
    min_gap = torch.full((n,), math.inf)
    passed_y = torch.full_like(w["rock_x"], math.nan)   # rover y relative to each rock when passing it
    path = [pose[:, :2].clone()]
    obs_log, act_log, ep_log = [], [], []
    t = 0.0
    while (status == 0).any() and t < t_max:
        live = status == 0
        obs = observe(pose, w, gen)
        chunk = policy(obs)
        if record:
            obs_log.append(obs[live]); act_log.append(chunk[live]); ep_log.append(live.nonzero().flatten())
        for k in range(EXEC):
            v, om = chunk[:, k, 0], chunk[:, k, 1] + steer_noise * torch.randn(n, generator=gen)
            th = pose[:, 2] + om.clamp(-W_MAX, W_MAX) * DT
            step = torch.stack([pose[:, 0] + v * torch.cos(th) * DT, pose[:, 1] + v * torch.sin(th) * DT, th], -1)
            was_before = pose[:, :1] < w["rock_x"]
            pose = torch.where(live[:, None], step, pose)
            crossing = live[:, None] & was_before & (pose[:, :1] >= w["rock_x"])
            passed_y = torch.where(crossing, pose[:, 1:2] - w["rock_y"], passed_y)
            gap = (torch.hypot(pose[:, :1] - w["rock_x"], pose[:, 1:2] - w["rock_y"]) - w["rock_r"] - ROVER_R).min(1).values
            min_gap = torch.where(live, torch.minimum(min_gap, gap), min_gap)
            status = torch.where(live & (gap < 0), 2, status)
            at_goal = torch.hypot(pose[:, 0] - w["goal_x"], pose[:, 1] - w["goal_y"]) < GOAL_TOL
            status = torch.where((status == 0) & at_goal, 1, status)
            live = status == 0
            t += DT
            path.append(pose[:, :2].clone())
    status = torch.where(status == 0, 3, status)
    out = {"status": status, "min_gap": min_gap, "passed_y": passed_y, "path": torch.stack(path, 1)}
    if record:
        out["obs"], out["actions"], out["episode"] = torch.cat(obs_log), torch.cat(act_log), torch.cat(ep_log)
    return out
