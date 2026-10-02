"""Image-based detector: turns a rendered camera frame into [range, bearing] detections plus an appearance signature.

Per person: visible fraction = visible pixels / full silhouette (measured from the render, not from a formula).
Detected with probability P_D * min(1, (vis - 0.25) / 0.45) * (1 - 0.8 * shadow), the same detector curve as the web
lab, now driven by pixels. Range comes from the depth map (median over visible pixels) plus stereo noise
sigma = 0.08 + 0.0012 r^2; bearing comes from the visible pixels' mean column plus pixel noise. False alarms are drawn
at random places, and their appearance comes from whatever rock or regolith is there in the image."""
import math
import numpy as np
from matplotlib.colors import rgb_to_hsv
from .world import sig_range

HUE_BINS = 18


def appearance(px):
    """Colour signature of a set of pixels [N,3]: hue histogram of saturated pixels (the suit stripes, excluding the gold
    visor every suit shares), weighted by saturation, plus a lightly weighted brightness histogram. Unit length."""
    if len(px) == 0:
        return np.zeros(HUE_BINS + 4)
    hsv = rgb_to_hsv(np.clip(px, 0, 1))
    sat = (hsv[:, 1] > 0.35) & (hsv[:, 2] > 0.12)
    visor = (hsv[:, 0] > 0.08) & (hsv[:, 0] < 0.15)        # every suit has the same gold visor: not identity
    keep = sat & ~visor
    hue = np.histogram(hsv[keep, 0], bins=HUE_BINS, range=(0, 1), weights=hsv[keep, 1])[0]
    val = np.histogram(hsv[~sat, 2], bins=4, range=(0, 1))[0] * 0.05   # white suit: same for everyone, small weight
    f = np.concatenate([hue, val]).astype(float)
    return f / (np.linalg.norm(f) + 1e-12)


class Detector:
    def __init__(self, world, cam, rig, shadows=None):
        """cam: the simulator camera dict (fov, range, noise); rig: the render.Camera that produced the frame."""
        self.world, self.cam, self.rig, self.rng = world, cam, rig, world.srng

    def detect(self, rgb, depth, owner, stats):
        W, s, c, rig = self.world, self.rng, self.cam, self.rig
        dets, feats, gt, boxes = [], [], [], []
        # camera realism: per-frame exposure change and per-pixel sensor noise (deterministic per frame)
        nz = np.random.default_rng(W.frame * 7919 + W.seed)
        rgb = np.clip(rgb * nz.uniform(0.85, 1.15) + nz.normal(0, 0.035, rgb.shape), 0, 1)
        for t in W.targets:
            n, sil, box = stats[t['id']]
            if sil == 0 or n < 6:
                continue
            vis = n / sil
            shadow = t['vis']['shadow']
            pd = W.cfg['pd'] * min(1.0, (vis - 0.25) / 0.45) * (1 - 0.8 * shadow) if vis >= 0.25 else 0.0
            if s.next() >= pd:
                continue
            m = owner == t['id']
            vv, uu = np.nonzero(m)
            px = rgb[m] * (1 - 0.7 * shadow)                     # a shadowed person really is darker to the camera
            ucol = uu.mean() + (s.randn() * 0.6)                 # centroid jitter, ~0.6 px
            phi = math.atan((ucol - rig.W / 2) / rig.f)          # angle right of the optical axis
            rng_ = float(np.median(depth[m])) / math.cos(phi)    # depth along the axis -> range
            rng_ += sig_range(c, rng_) * s.randn()
            dets.append((rng_, c['th'] - phi)); feats.append(appearance(px)); gt.append(t['id'])
            boxes.append((uu.min(), vv.min(), uu.max() + 1, vv.max() + 1))
        lam = W.cfg['clutter']; k, p = 0, math.exp(-lam); acc, u = p, s.next()
        while u > acc and k < 50:
            k += 1; p *= lam / k; acc += p
        for _ in range(k):                                       # false alarms: rocks / regolith that fooled the detector
            r = s.uniform(4, c['range']); b = c['th'] + s.uniform(-c['fov'] / 2, c['fov'] / 2)
            P = np.array([[c['x'] + r * math.cos(b), c['y'] + r * math.sin(b), 0.6]])
            uu, vv, zz = rig.project(P)
            ui, vi = int(np.clip(uu[0], 3, rig.W - 4)), int(np.clip(vv[0], 6, rig.H - 7))
            dets.append((r, b)); feats.append(appearance(rgb[vi - 6:vi + 6, ui - 3:ui + 4].reshape(-1, 3))); gt.append(0)
            boxes.append((ui - 3, vi - 6, ui + 4, vi + 6))
        return np.array(dets, float).reshape(-1, 2), np.array(feats).reshape(len(dets), HUE_BINS + 4), np.array(gt, int), boxes
