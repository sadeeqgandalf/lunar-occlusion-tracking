"""Flow-matching transformer policy (the action expert of pi0, Black et al. 2024).

Learns a velocity field that carries random noise to an expert chunk along straight lines (flow matching,
Lipman et al. 2023; rectified flow, Liu et al. 2023). Each of the H actions is a token, and a transformer looks at
all of them at once, conditioned on the observation and the time t through adaptive LayerNorm (DiT, Peebles & Xie 2023).
Sampling: 10 Euler steps from noise (t = 0) to a chunk (t = 1), against the diffusion policy's 50.

    python policy/flow.py      # trains on seed 0's demonstrations and scores closed loop
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import StepEmbedding
from model import Normalizer
from sim import H, from_unit, to_unit


class AdaLNBlock(nn.Module):
    """Transformer block whose LayerNorms are scaled and shifted by the condition c. Each branch has a gate that
    starts at zero, so every block starts as the identity (adaLN-Zero)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):                      # x (B, H, d) action tokens, c (B, d) condition
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        h = self.ln1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        return x + g2 * self.mlp(self.ln2(x) * (1 + s2) + b2)


class FlowPolicy(nn.Module):
    def __init__(self, steps: int = 10, d: int = 64, heads: int = 4, layers: int = 3, obs_dim: int = 6):
        super().__init__()
        self.steps = steps
        self.norm = Normalizer(obs_dim)
        self.obs_in = nn.Sequential(nn.Linear(obs_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.time = nn.Sequential(StepEmbedding(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(H, d) * 0.02)
        self.blocks = nn.ModuleList(AdaLNBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Linear(d, 2)

    def velocity(self, x, t, obs):                # x (B, H, 2) unit chunk on its way from noise, t (B,) in [0, 1]
        c = self.obs_in(self.norm(obs)) + self.time(1000 * t)
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h, c)
        return self.out(self.ln(h))

    def loss(self, obs, unit_actions, gen: torch.Generator = None):
        """A point on the straight line from noise to the expert chunk; the target is that line's direction."""
        noise = torch.randn(unit_actions.shape, generator=gen)
        t = torch.rand(len(obs), generator=gen)
        x = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * unit_actions
        return F.mse_loss(self.velocity(x, t, obs), unit_actions - noise)

    @torch.no_grad()
    def sample(self, obs, gen: torch.Generator = None):
        """Euler integration from noise; different noise can end on a different side of the rock."""
        x = torch.randn(len(obs), H, 2, generator=gen)
        for i in range(self.steps):
            x = x + self.velocity(x, torch.full((len(obs),), i / self.steps), obs) / self.steps
        return from_unit(x)


if __name__ == "__main__":
    import dataset
    from train_eval import EPOCHS, score, test_suites

    torch.manual_seed(0)
    data = dataset.build(seed=0)
    obs, unit = data["obs"], to_unit(data["actions"])
    policy = FlowPolicy()
    policy.norm.fit(obs)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, EPOCHS + 1):
        for idx in torch.randperm(len(obs)).split(256):
            loss = policy.loss(obs[idx], unit[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:2d}  loss {loss.item():.4f}")

    g = torch.Generator().manual_seed(11)
    for suite, r in score(lambda: lambda o: policy.sample(o, g), test_suites()).items():
        print(f"{suite:10s}  crash {r['crash']:4.1f}%   goal {r['goal']:5.1f}%   passed left {r['left']}%")
