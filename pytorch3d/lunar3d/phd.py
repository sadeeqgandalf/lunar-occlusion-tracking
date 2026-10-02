"""Particle PHD filter: a "where could anyone be?" density over the ground (Mahler 2003; SMC-PHD, Vo, Singh & Doucet 2005).

The intensity D(x) is carried by N weighted particles; the total weight is the expected number of people. Each frame:
  predict  x <- constant-velocity motion + noise; weights x p_survive
  birth    small mass around each detection (measurement-driven birth, Ristic, Clark, Vo & Vo 2012): spreading birth
           mass uniformly would pile up behind rocks, where nothing can ever remove it, as phantom people
  update   w <- w [ (1 - P_D(x)) + sum_z  P_D(x) g(z|x) / (kappa(z) + sum_j P_D(x_j) g(z|x_j) w_j) ]
P_D(x) is the visibility-based detection probability of that very spot, per camera. Where the camera can see, a miss
removes mass; behind a rock (P_D ~ 0) mass survives untouched. That is negative information, made visible: when someone
walks out of sight the density pools into the blind zone they went into."""
import math
import numpy as np
from .occlusion import visibility, shadow_fraction, detection_prob, wrap
from .world import sig_range


class ParticlePHD:
    def __init__(self, cams, boulders, sun=None, pd=0.95, clutter=0.8, n=6000, seed=1):
        self.cams, self.boulders, self.sun, self.pd, self.clutter, self.n = cams, boulders, sun, pd, clutter, n
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((0, 4)); self.w = np.zeros(0)
        self.dt, self.ps, self.birth_mass, self.sigA = 0.1, 0.995, 0.03, 0.9

    def _pd(self, cam, X):
        v = visibility(cam, self.boulders, X[:, 0], X[:, 1])
        sh = shadow_fraction(self.boulders, X[:, 0], X[:, 1], self.sun) if self.sun is not None else 0.0
        return detection_prob(v['inFov'], v['visFrac'], self.pd, sh)

    def _births(self, cam, dets, m=60):
        """Birth particles around each detection of this frame, used from the next frame on (not updated twice)."""
        if len(dets) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        Xs = []
        for z in dets:
            r = z[0] + sig_range(cam, z[0]) * self.rng.normal(size=m); b = z[1] + cam['sigB'] * 2 * self.rng.normal(size=m)
            Xs.append(np.c_[cam['x'] + r * np.cos(b), cam['y'] + r * np.sin(b), self.rng.normal(0, 0.6, (m, 2))])
        X = np.concatenate(Xs)
        # a detection from a spot the camera cannot see must be a false alarm: weight births by visibility there
        return X, (self.birth_mass / m) * self._pd(cam, X) / self.pd

    def step(self, frames):
        dt = self.dt
        if len(self.X):
            self.X[:, :2] += self.X[:, 2:] * dt + 0.5 * dt * dt * self.rng.normal(0, self.sigA, (len(self.X), 2))
            self.X[:, 2:] += dt * self.rng.normal(0, self.sigA, (len(self.X), 2))
            self.X[:, 2:] *= np.clip(1.6 / np.maximum(np.linalg.norm(self.X[:, 2:], axis=1), 1e-9), None, 1)[:, None]  # walking speeds
            self.w *= self.ps
        if getattr(self, '_pending', None) is not None:
            self.X = np.r_[self.X, self._pending[0]]; self.w = np.r_[self.w, self._pending[1]]
        born = [self._births(cam, dets) for cam, dets, _ in frames]
        self._pending = (np.concatenate([b[0] for b in born]), np.concatenate([b[1] for b in born]))
        for cam, dets, _ in frames:
            pd = self._pd(cam, self.X)
            kappa = self.clutter / (cam['fov'] * (cam['range'] - 4))
            dx, dy = self.X[:, 0] - cam['x'], self.X[:, 1] - cam['y']
            rr, br = np.hypot(dx, dy), np.arctan2(dy, dx)
            sr, sb = sig_range(cam, rr), cam['sigB'] * math.sqrt(2) + 0.002
            gain = 1 - pd
            for z in dets:
                g = np.exp(-0.5 * (((z[0] - rr) / sr) ** 2 + (wrap(z[1] - br) / sb) ** 2)) / (2 * math.pi * sr * sb)
                num = pd * g
                gain = gain + num / (kappa + np.sum(num * self.w))
            self.w = self.w * gain
        # resample to n particles, preserving the total mass (expected number of people)
        mass = self.w.sum()
        keep = self.w > 0
        self.X, self.w = self.X[keep], self.w[keep]
        if mass <= 0 or len(self.w) == 0:
            self.X, self.w = np.zeros((0, 4)), np.zeros(0); return
        u = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w / mass), u), len(self.w) - 1)
        self.X = self.X[idx] + np.c_[self.rng.normal(0, 0.05, (self.n, 2)), self.rng.normal(0, 0.05, (self.n, 2))]
        self.w = np.full(self.n, mass / self.n)

    @property
    def expected_count(self):
        return float(self.w.sum())

    def density(self, xlim=(-34, 34), ylim=(-2, 46), cell=0.5, blur=2):
        """People per square metre on a ground grid, lightly smoothed (for display)."""
        H, xe, ye = np.histogram2d(self.X[:, 0], self.X[:, 1], bins=[np.arange(*xlim, cell), np.arange(*ylim, cell)], weights=self.w)
        H = H.T / cell ** 2
        k = np.exp(-0.5 * (np.arange(-2 * blur, 2 * blur + 1) / blur) ** 2); k /= k.sum()
        H = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 0, H)
        H = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, H)
        return H, (xe[0], xe[-1], ye[0], ye[-1])
