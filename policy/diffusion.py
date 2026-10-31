"""Diffusion policy (Chi et al., RSS 2023): start from random noise and remove it step by step, guided by o.

Training: add noise of a random strength k to an expert chunk and learn to predict that noise.
Sampling: K denoising steps from pure noise. Each run can land on a different mode (left or right) because it
starts from different noise. Mirrored for the browser in src/policy/policies.js.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Normalizer
from sim import H, from_unit


def cosine_schedule(K: int):
    """Noise schedule of Nichol & Dhariwal (2021): at k = K-1 almost no signal is left, matching the N(0, I) start."""
    f = lambda t: math.cos((t / K + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.tensor([min(1 - f(k + 1) / f(k), 0.999) for k in range(K)])
    return betas, torch.cumprod(1 - betas, 0)


class StepEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, k):
        f = torch.exp(-math.log(10000.0) * torch.arange(self.dim // 2) / (self.dim // 2))
        a = k.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class FiLMBlock(nn.Module):
    """Residual MLP block; the condition (o, k) scales and shifts its features (FiLM)."""

    def __init__(self, hidden: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(cond, 2 * hidden)
        self.fc1, self.fc2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)

    def forward(self, x, c):
        scale, shift = self.film(c).chunk(2, -1)
        h = F.silu(self.norm(x) * (1 + scale) + shift)
        return x + self.fc2(F.silu(self.fc1(h)))


class DiffusionPolicy(nn.Module):
    def __init__(self, K: int = 50, hidden: int = 128, blocks: int = 3, obs_dim: int = 6):
        super().__init__()
        self.K = K
        betas, abar = cosine_schedule(K)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", abar)
        self.norm = Normalizer(obs_dim)
        self.obs_in = nn.Sequential(nn.Linear(obs_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.step = StepEmbedding(64)
        self.inp = nn.Linear(2 * H, hidden)
        self.blocks = nn.ModuleList(FiLMBlock(hidden, 128) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 2 * H))

    def eps(self, x, k, obs):                     # predicted noise in a noisy unit chunk x (B, H, 2) at step k
        c = torch.cat([self.obs_in(self.norm(obs)), self.step(k)], -1)
        h = self.inp(x.flatten(1))
        for b in self.blocks:
            h = b(h, c)
        return self.out(h).view(-1, H, 2)

    def loss(self, obs, unit_actions):
        k = torch.randint(0, self.K, (len(obs),))
        noise = torch.randn_like(unit_actions)
        a = self.abar[k].view(-1, 1, 1)
        return F.mse_loss(self.eps(a.sqrt() * unit_actions + (1 - a).sqrt() * noise, k, obs), noise)

    @torch.no_grad()
    def sample(self, obs, gen: torch.Generator = None):
        """DDPM ancestral sampling with the clean-chunk estimate clipped to the action range."""
        x = torch.randn(len(obs), H, 2, generator=gen)
        for k in reversed(range(self.K)):
            a, a_prev, beta = self.abar[k], self.abar[k - 1] if k else torch.tensor(1.0), self.betas[k]
            x0 = ((x - (1 - a).sqrt() * self.eps(x, torch.full((len(obs),), k), obs)) / a.sqrt()).clamp(-1, 1)
            mean = (a_prev.sqrt() * beta * x0 + (1 - beta).sqrt() * (1 - a_prev) * x) / (1 - a)
            x = mean + ((beta * (1 - a_prev) / (1 - a)).sqrt() * torch.randn(x.shape, generator=gen) if k else 0)
        return from_unit(x)
