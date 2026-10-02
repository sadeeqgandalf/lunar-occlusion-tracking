"""Artemis EVA scene: rover mast camera, boulders, goal-directed walkers. Port of src/mot/world.js (same seed -> same world).
Two detection sources:
  * sense()            the web lab's probabilistic sensor model (used to verify this port against the JS)
  * camera detections  from PyTorch3D renders (see camera.py / detector.py)"""
import math
import numpy as np
from .rng import Mulberry32
from .occlusion import visibility, shadow_fraction, detection_prob, wrap

SCENARIOS = {
    'boulders': dict(label='Artemis EVA · Boulder field', targets=5, boulders=9, pd=0.95, clutter=0.8, shadows=False, lowFrac=0.35),
    'crossing': dict(label='Worksite · Crossing paths', targets=8, boulders=7, pd=0.92, clutter=1.2, shadows=False, lowFrac=0.35),
    'polar': dict(label='South pole · Long shadows', targets=5, boulders=9, pd=0.95, clutter=0.8, shadows=True, lowFrac=0.35),
    'open': dict(label='Open plain · Sparse cover', targets=5, boulders=1, pd=0.95, clutter=0.8, shadows=False, lowFrac=0.35),
}


def camera():
    return dict(x=0.0, y=-4.0, h=2.2, th=math.pi / 2, fov=math.radians(100), range=48.0,
                sigB=0.004, sigR0=0.08, sigRk=0.0012, targetH=1.8, targetR=0.4)


def sig_range(cam, r):
    return cam['sigR0'] + cam['sigRk'] * r * r      # stereo depth noise grows with range squared


class LunarWorld:
    def __init__(self, scenario='boulders', seed=7, **overrides):
        self.cfg = {**SCENARIOS[scenario], **overrides}
        self.id, self.seed = scenario, seed
        self.cam = camera()
        self.rng = Mulberry32(seed * 9973 + 17)     # world + truth motion
        self.srng = Mulberry32(seed * 7717 + 5)     # sensor noise
        self.sun = dict(az=math.atan2(-25, -60), el=math.atan2(9, math.hypot(60, 25)))
        self.t, self.dt, self.frame = 0.0, 0.1, 0
        self._place_boulders(); self._spawn_targets()

    def _near_rock(self, x, y, m):
        return any(math.hypot(b['x'] - x, b['y'] - y) < b['r'] + m for b in self.boulders)

    def _place_boulders(self):
        r, c, B, k = self.rng, self.cam, [], 0
        while len(B) < self.cfg['boulders'] and k < 2000:
            k += 1
            rg, off, br = r.uniform(9, 36), r.uniform(-0.42, 0.42) * c['fov'], r.uniform(1.4, 3.2)
            h = r.uniform(0.7, 1.4) if r.next() < self.cfg['lowFrac'] else r.uniform(2.4, 3.6)
            b = dict(x=c['x'] + rg * math.cos(c['th'] + off), y=c['y'] + rg * math.sin(c['th'] + off), r=br, h=h)
            if any(math.hypot(o['x'] - b['x'], o['y'] - b['y']) < o['r'] + b['r'] + 3.5 for o in B):
                continue
            B.append(b)
        self.boulders = B

    def _spawn_targets(self):
        r, c, self.targets = self.rng, self.cam, []
        for i in range(self.cfg['targets']):
            for _ in range(500):
                rr, oo = r.uniform(12, c['range'] - 8), r.uniform(-0.38, 0.38) * c['fov']
                p = (c['x'] + rr * math.cos(c['th'] + oo), c['y'] + rr * math.sin(c['th'] + oo))
                if not self._near_rock(*p, 1.2):
                    break
            h, v = r.uniform(-math.pi, math.pi), r.uniform(0.6, 1.3)
            self.targets.append(dict(id=i + 1, x=p[0], y=p[1], h=h, v=v, goal=None, trail=[], vis=self.view(p[0], p[1])))

    def _goal(self):
        r, c = self.rng, self.cam
        for _ in range(200):
            rr, oo = r.uniform(11, c['range'] - 6), r.uniform(-0.4, 0.4) * c['fov']
            g = (c['x'] + rr * math.cos(c['th'] + oo), c['y'] + rr * math.sin(c['th'] + oo))
            if not self._near_rock(*g, 1.5):
                return g
        return (c['x'], c['y'] + 25)

    def step(self):
        dt, r = self.dt, self.rng
        for t in self.targets:
            if t['goal'] is None or math.hypot(t['goal'][0] - t['x'], t['goal'][1] - t['y']) < 1.5:
                t['goal'] = self._goal()
            a = math.atan2(t['goal'][1] - t['y'], t['goal'][0] - t['x'])
            sx, sy = math.cos(a), math.sin(a)
            for b in self.boulders:                     # walk around rocks, not through them
                dx, dy = t['x'] - b['x'], t['y'] - b['y']
                d = math.hypot(dx, dy) - b['r']
                if d < 2.0:
                    w = 1.6 * (2.0 - max(d, 0.05)) / 2.0
                    sx += dx / (d + b['r']) * w; sy += dy / (d + b['r']) * w
            turn = float(wrap(math.atan2(sy, sx) - t['h']))
            t['h'] += min(1, max(-1, turn)) * 1.2 * dt + 0.08 * math.sqrt(dt) * r.randn()
            t['x'] += t['v'] * math.cos(t['h']) * dt; t['y'] += t['v'] * math.sin(t['h']) * dt
            for b in self.boulders:                     # never inside a rock
                dx, dy = t['x'] - b['x'], t['y'] - b['y']; d = math.hypot(dx, dy)
                if d < b['r'] + 0.4:
                    t['x'] = b['x'] + dx / d * (b['r'] + 0.4); t['y'] = b['y'] + dy / d * (b['r'] + 0.4)
            t['vis'] = self.view(t['x'], t['y'])
            if self.frame % 3 == 0:
                t['trail'].append((t['x'], t['y'])); t['trail'] = t['trail'][-120:]
        self.t += dt; self.frame += 1

    def view(self, x, y, cam=None):
        cam = cam or self.cam
        v = {k: float(np.asarray(val).ravel()[0]) for k, val in visibility(cam, self.boulders, x, y, self.cam['targetR']).items()}
        v['inFov'] = bool(v['inFov'])
        v['shadow'] = float(shadow_fraction(self.boulders, x, y, self.sun)[0]) if self.cfg['shadows'] else 0.0
        v['pd'] = float(detection_prob(v['inFov'], v['visFrac'], self.cfg['pd'], v['shadow']))
        return v

    def sense(self):
        """Web-lab sensor model: one frame of [range, bearing] detections + true ids (0 = false alarm)."""
        c, s, dets, gt = self.cam, self.srng, [], []
        for t in self.targets:
            v = t['vis']
            if s.next() >= v['pd']:
                continue
            sr = sig_range(c, v['range']); sb = c['sigB'] * (2 if v['visFrac'] < 0.9 else 1)
            rg = v['range'] + sr * s.randn(); br = v['visCenter'] + sb * s.randn()
            dets.append((rg, br)); gt.append(t['id'])
        k, p = 0, math.exp(-self.cfg['clutter']); acc, u = p, s.next()
        while u > acc and k < 50:
            k += 1; p *= self.cfg['clutter'] / k; acc += p
        for _ in range(k):
            rg = s.uniform(4, c['range']); br = c['th'] + s.uniform(-c['fov'] / 2, c['fov'] / 2)
            dets.append((rg, br)); gt.append(0)
        return np.array(dets, float).reshape(-1, 2), np.array(gt, int)

    @staticmethod
    def hidden(v):
        return v['inFov'] and v['pd'] < 0.15
