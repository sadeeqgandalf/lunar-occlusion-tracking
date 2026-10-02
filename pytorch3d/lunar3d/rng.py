"""Mulberry32 + Box-Muller, bit-identical to src/core/rng.js (same seed -> same world as the web lab)."""
import math


class Mulberry32:
    def __init__(self, seed):
        self.s = int(seed) & 0xFFFFFFFF
        self.spare = None

    @staticmethod
    def _imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def next(self):
        self.s = (self.s + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.s
        t = self._imul(t ^ (t >> 15), t | 1)
        t ^= (t + self._imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def uniform(self, a=0.0, b=1.0):
        return a + (b - a) * self.next()

    def randn(self):
        if self.spare is not None:
            s, self.spare = self.spare, None
            return s
        u = 0.0
        while u == 0.0:
            u = self.next()
        v = self.next()
        m = math.sqrt(-2 * math.log(u))
        self.spare = m * math.sin(2 * math.pi * v)
        return m * math.cos(2 * math.pi * v)
