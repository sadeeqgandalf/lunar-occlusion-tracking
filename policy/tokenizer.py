"""Action-chunk tokenizers: turn an (H, 2) chunk of [v, w] into integers a transformer can predict.

Uniform: one bin per action per step (RT-2, Brohan et al. 2023) -> 2H tokens.
DCT: a cosine transform over time, keep the K slowest frequencies, bin those (FAST, Pertsch et al. 2025, without
its byte-pair-encoding step) -> 2K tokens. Smooth chunks need few frequencies, so the sequence is shorter.
An odd vocabulary puts a coefficient of exactly 0 (no change over the chunk) in the middle of a bin, not on an edge.
Token order is [v_0, w_0, v_1, w_1, ...]: the first two tokens are the chunk's average speed and average turn.
"""
import math

import torch

from sim import H, from_unit, to_unit


class UniformTokenizer:
    def __init__(self, vocab: int = 64):
        self.vocab = vocab

    def encode(self, actions: torch.Tensor) -> torch.Tensor:          # (B, H, 2) -> (B, 2H)
        u = (to_unit(actions) + 1) / 2
        return (u * (self.vocab - 1)).round().long().clamp(0, self.vocab - 1).flatten(1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:           # (B, 2H) -> (B, H, 2)
        return from_unit(tokens.view(len(tokens), -1, 2).float() / (self.vocab - 1) * 2 - 1)


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II: row k is the k-th cosine over n steps; D @ D.T = I."""
    k, t = torch.arange(n)[:, None], torch.arange(n)[None, :]
    d = torch.cos(math.pi * (t + 0.5) * k / n) * math.sqrt(2 / n)
    d[0] /= math.sqrt(2)
    return d


class DCTTokenizer:
    def __init__(self, k_freq: int = 4, vocab: int = 65, bound: float = 2.5):
        self.k, self.vocab, self.bound = k_freq, vocab, bound
        self.D = dct_matrix(H)

    @property
    def length(self) -> int:
        return 2 * self.k

    def encode(self, actions: torch.Tensor) -> torch.Tensor:          # (B, H, 2) -> (B, 2K)
        coeff = torch.einsum("kh,bhd->bkd", self.D, to_unit(actions))[:, : self.k]
        q = (coeff + self.bound) / (2 * self.bound) * (self.vocab - 1)
        return q.round().long().clamp(0, self.vocab - 1).flatten(1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:           # (B, 2K) -> (B, H, 2)
        coeff = torch.zeros(len(tokens), H, 2)
        coeff[:, : self.k] = tokens.view(len(tokens), self.k, 2).float() / (self.vocab - 1) * 2 * self.bound - self.bound
        return from_unit(torch.einsum("kh,bkd->bhd", self.D, coeff))
