"""Two of the three policies: regression (BC-MSE) and an autoregressive (AR) transformer over action tokens.

Both map an observation o (6 numbers, see sim.observe) to a chunk of H actions. Mirrored for the browser in
src/policy/policies.js, which runs these exact weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import H, from_unit


class Normalizer(nn.Module):
    """Fixed per-feature scaling of the observation, measured on the training set."""

    def __init__(self, dim: int = 6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, obs: torch.Tensor):
        self.mean.copy_(obs.mean(0)); self.std.copy_(obs.std(0).clamp(min=1e-3))

    def forward(self, obs):
        return (obs - self.mean) / self.std


class BCPolicy(nn.Module):
    """Behaviour cloning with a squared-error loss: predicts the AVERAGE expert chunk for each observation."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.norm = Normalizer()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * H))

    def forward(self, obs):                       # -> (B, H, 2) in unit range
        return self.net(self.norm(obs)).view(-1, H, 2)

    def loss(self, obs, unit_actions):
        return F.mse_loss(self(obs), unit_actions)

    @torch.no_grad()
    def act(self, obs):
        return from_unit(self(obs))


class Block(nn.Module):
    """Pre-norm causal self-attention + MLP (GPT-2 layout)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ARPolicy(nn.Module):
    """Predicts action tokens one at a time, each a categorical choice over the vocabulary, so it can put
    probability on 'left' AND 'right' instead of averaging them. Sequence: [o, c_1 .. c_{L-1}] -> logits c_1 .. c_L."""

    def __init__(self, length: int = 8, vocab: int = 65, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.length, self.vocab = length, vocab
        self.norm = Normalizer()
        self.obs_in = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(length, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def logits(self, obs, prev):                  # prev: (B, n < length) tokens so far -> (B, n + 1, vocab)
        x = torch.cat([self.obs_in(self.norm(obs))[:, None], self.tok(prev)], 1)
        x = x + self.pos[: x.shape[1]]
        mask = torch.triu(torch.full((x.shape[1],) * 2, float("-inf")), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

    def loss(self, obs, tokens):
        return F.cross_entropy(self.logits(obs, tokens[:, :-1]).flatten(0, 1), tokens.flatten())

    @torch.no_grad()
    def sample(self, obs, temperature: float = 1.0, gen: torch.Generator = None):
        toks = torch.zeros(len(obs), 0, dtype=torch.long)
        for _ in range(self.length):
            p = F.softmax(self.logits(obs, toks)[:, -1] / temperature, -1)
            toks = torch.cat([toks, torch.multinomial(p, 1, generator=gen)], 1)
        return toks
