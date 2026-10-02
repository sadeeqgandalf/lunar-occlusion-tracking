"""Multi-object tracker (port of src/mot/tracker.js) plus two extensions used by the PyTorch3D lab:
  * several cameras per frame: one predict, then a gated update per camera with THAT camera's visibility
  * appearance re-identification: tracks carry a colour signature; a new detection that matches a lost or coasting
    track's signature (and is reachable in the elapsed time) gets the old ID back instead of a new one.
With one camera and re-ID off it reproduces the JS tracker exactly (tests/test_crosscheck.py).

Variants differ only in the detection probability a track expects:
  naive    P_D everywhere           aware   P_D x visibility(predicted position)
  negInfo  aware + p(x | miss) ∝ p(x)(1 - P_D(x)) moment-matched over fixed samples (Koch 2007)"""
import math
import numpy as np
from .rng import Mulberry32
from .hungarian import hungarian
from .occlusion import visibility, shadow_fraction, detection_prob, wrap
from .world import sig_range

DEFAULTS = dict(sigA=0.9, sigACoast=0.9, gate=11.83, rBirth=0.25, pSurvive=0.998, rConfirm=0.75, rDelete=0.08,
                maxSigma=7.0, pd=0.95, clutter=0.8,
                reid=False, reidSim=0.86, reidSpeed=1.6, reidSlack=3.0, reidMemory=30.0, appearanceWeight=8.0)
NAMES = dict(naive='Naive', aware='Occlusion-aware', negInfo='Aware + neg. info')


def _samples():
    r, z = Mulberry32(424242), []
    for _ in range(32):
        v = [r.randn(), r.randn(), r.randn(), r.randn()]
        z.append(v); z.append([-q for q in v])
    return np.array(z)


Z = _samples()


def cholj(P):
    """Lower Cholesky with the JS's growing diagonal jitter."""
    jit = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(P + jit * np.eye(len(P)))
        except np.linalg.LinAlgError:
            jit = jit * 10 if jit else 1e-10
    raise np.linalg.LinAlgError('covariance not positive definite')


def cos_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Tracker:
    def __init__(self, cams, boulders, variant='negInfo', sun=None, name=None, **params):
        self.cams = cams if isinstance(cams, list) else [cams]
        self.boulders, self.sun = boulders, sun
        self.aware, self.negInfo = variant != 'naive', variant == 'negInfo'
        self.p = {**DEFAULTS, **params}
        self.name = name or (NAMES[variant] + (' + re-ID' if self.p['reid'] else ''))
        self.tracks, self.nextId, self.dt, self.t = [], 1, 0.1, 0.0
        self.gallery = []          # recently deleted confirmed tracks, for re-identification
        self.reidEvents = []       # (time, id, similarity, gap seconds) for display

    # ---------------------------------------------------------------- detection probability
    def pd_at(self, cam, X, Y):
        v = visibility(cam, self.boulders, X, Y, cam.get('targetR', 0.4))
        sh = shadow_fraction(self.boulders, X, Y, self.sun) if self.sun is not None else 0.0
        return detection_prob(v['inFov'], v['visFrac'], self.p['pd'], sh)

    def expected_pd(self, t, cam):
        """P_D x visibility averaged over 64 samples of the predicted Gaussian (per camera)."""
        if not self.aware:
            return self.p['pd'], None, None
        L = cholj(t['P'])
        X = t['x'] + Z @ L.T
        pds = self.pd_at(cam, X[:, 0], X[:, 1])
        return float(pds.mean()), X, pds

    # ---------------------------------------------------------------- one frame
    def step(self, frames):
        """frames: list of (cam, dets[K,2] range/bearing, feats[K,F] or None). A bare array = one frame, first camera."""
        if isinstance(frames, np.ndarray):
            frames = [(self.cams[0], frames, None)]
        p, dt = self.p, self.dt
        self.t += dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q0 = np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        for t in self.tracks:
            q = (p['sigACoast'] if t['lastSeen'] > 0 else p['sigA']) ** 2
            t['x'] = F @ t['x']; t['P'] = F @ t['P'] @ F.T + Q0 * q; t['P'] = (t['P'] + t['P'].T) / 2
            t['r'] *= p['pSurvive']; t['age'] += 1; t['seenNow'] = False
        leftovers = []
        for ci, (cam, dets, feats) in enumerate(frames):
            used, gated = self._update(cam, dets, feats)
            leftovers.append((cam, dets, feats, used, gated))
        for t in self.tracks:
            if not t['seenNow']:
                t['lastSeen'] += 1
        keep = []
        for t in self.tracks:
            if t['r'] >= p['rDelete'] and math.sqrt(max(t['P'][0, 0], t['P'][1, 1])) <= p['maxSigma']:
                keep.append(t)
            elif t['confirmed'] and p['reid'] and t.get('feat') is not None:
                self.gallery.append(dict(id=t['id'], feat=t['feat'], x=t['x'][:2].copy(), t=self.t))
        self.tracks = keep
        self.gallery = [g for g in self.gallery if self.t - g['t'] <= p['reidMemory']]
        born = []
        for ci, (cam, dets, feats, used, gated) in enumerate(leftovers):   # births only outside every gate (no duplicates)
            for j in range(len(dets)):
                if used[j] or gated[j]:
                    continue
                pos = np.array([cam['x'] + dets[j, 0] * math.cos(dets[j, 1]), cam['y'] + dets[j, 0] * math.sin(dets[j, 1])])
                if any(cj != ci and np.hypot(*(pos - q)) < 2.0 for cj, q in born):
                    continue                              # ANOTHER camera already started this person this frame
                f = None if feats is None else feats[j]
                if p['reid'] and f is not None and self._reidentify(cam, dets[j], f):
                    continue
                self._birth(cam, dets[j], f); born.append((ci, pos))

    def _update(self, cam, dets, feats):
        p, D, T = self.p, len(dets), len(self.tracks)
        FORBID, PG = 1e9, 0.9973
        kappa = p['clutter'] / (cam['fov'] * (cam['range'] - 4))    # clutter density per (m x rad)
        cost = np.full((T, D + T), FORBID); g = np.zeros((T, D)); pre = []
        for i, t in enumerate(self.tracks):
            t['pdExp'], t['_X'], t['_pds'] = self.expected_pd(t, cam)
            t['hidden'] = t['pdExp'] < 0.3
            dx, dy = t['x'][0] - cam['x'], t['x'][1] - cam['y']; qq = dx * dx + dy * dy; rr = math.sqrt(qq)
            H = np.array([[dx / rr, dy / rr, 0, 0], [-dy / qq, dx / qq, 0, 0]])
            R = np.diag([sig_range(cam, rr) ** 2, cam['sigB'] ** 2 * 2])
            S = H @ t['P'] @ H.T + R; Si = np.linalg.inv(S); zh = np.array([rr, math.atan2(dy, dx)])
            pre.append((H, R, Si, zh))
            pdi = max(t['pdExp'], 0.02); dS = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            for j in range(D):
                y = np.array([dets[j, 0] - zh[0], float(wrap(dets[j, 1] - zh[1]))])
                nis = y[0] * (Si[0, 0] * y[0] + Si[0, 1] * y[1]) + y[1] * (Si[1, 0] * y[0] + Si[1, 1] * y[1])
                if nis > p['gate']:
                    continue
                g[i, j] = math.exp(-0.5 * nis) / (2 * math.pi * math.sqrt(dS))
                lr = pdi * g[i, j] / kappa
                if p['reid'] and feats is not None and t.get('feat') is not None:
                    lr *= math.exp(p['appearanceWeight'] * (cos_sim(t['feat'], feats[j]) - p['reidSim']))  # appearance LR
                cost[i, j] = -math.log(lr)
            cost[i, D + i] = -math.log(1 - pdi * PG)
        asg = hungarian(cost.tolist(), FORBID) if T else []
        used = np.zeros(D, bool)
        for i, t in enumerate(self.tracks):
            j = asg[i]
            if 0 <= j < D:
                used[j] = True
                H, R, Si, zh = pre[i]
                y = np.array([dets[j, 0] - zh[0], float(wrap(dets[j, 1] - zh[1]))])
                K = t['P'] @ H.T @ Si; t['x'] = t['x'] + K @ y
                IKH = np.eye(4) - K @ H; t['P'] = IKH @ t['P'] @ IKH.T + K @ R @ K.T; t['P'] = (t['P'] + t['P'].T) / 2
                pdH = max(t['pdExp'], 0.02); kap = kappa
                t['r'] = t['r'] * (pdH * g[i, j] + (1 - pdH) * kap) / (t['r'] * pdH * g[i, j] + (1 - t['r'] * pdH) * kap)
                t['hits'] += 1; t['lastSeen'] = 0; t['seenNow'] = True
                if not t['confirmed'] and t['r'] >= p['rConfirm'] and t['hits'] >= 3:
                    t['confirmed'] = True
                if feats is not None:
                    t['feat'] = feats[j].copy() if t.get('feat') is None else 0.85 * t['feat'] + 0.15 * feats[j]
            else:
                pd = t['pdExp']
                t['r'] = t['r'] * (1 - pd) / (1 - t['r'] * pd)
                if self.negInfo:
                    self._miss_update(t)
        gated = (cost[:, :D] < FORBID).any(0) if T else np.zeros(D, bool)
        return used, gated

    def _miss_update(self, t):
        X, pds = t.get('_X'), t.get('_pds')
        if X is None:
            return
        w = 1 - pds; W = w.sum(); N = len(w)
        if W < 1e-6 or W > N * (1 - 1e-6):
            return
        m = (w @ X) / W; Xc = X - m
        P = (Xc * (w / W)[:, None]).T @ Xc + np.diag([0.02, 0.02, 0.01, 0.01])
        t['x'], t['P'] = m, (P + P.T) / 2

    def _birth(self, cam, d, feat=None, tid=None, confirmed=False):
        cb, sb, r = math.cos(d[1]), math.sin(d[1]), d[0]
        J = np.array([[cb, -r * sb], [sb, r * cb]]); Rm = np.diag([sig_range(cam, r) ** 2, cam['sigB'] ** 2 * 2])
        P = np.diag([0, 0, 1.0, 1.0]); P[:2, :2] = J @ Rm @ J.T + 0.05 * np.eye(2)
        if tid is None:
            tid = self.nextId; self.nextId += 1
        t = dict(id=tid, x=np.array([cam['x'] + r * cb, cam['y'] + r * sb, 0, 0], float), P=P,
                 r=self.p['rConfirm'] if confirmed else self.p['rBirth'], hits=3 if confirmed else 1, age=0, lastSeen=0,
                 confirmed=confirmed, hidden=False, pdExp=1.0, seenNow=True, feat=None if feat is None else feat.copy())
        self.tracks.append(t)
        return t

    def _reidentify(self, cam, d, f):
        """A detection nobody's gate claimed: is it someone we lost (deleted) or are coasting (hidden, gate too small)?"""
        p = self.p
        pos = np.array([cam['x'] + d[0] * math.cos(d[1]), cam['y'] + d[0] * math.sin(d[1])])
        best, bestSim = None, p['reidSim']
        for t in self.tracks:
            if t['confirmed'] and t['lastSeen'] > 3 and t.get('feat') is not None:
                gap = t['lastSeen'] * self.dt
                if np.hypot(*(pos - t['x'][:2])) <= p['reidSpeed'] * gap + p['reidSlack']:
                    s = cos_sim(t['feat'], f)
                    if s > bestSim:
                        best, bestSim = ('live', t, gap), s
        for gI in self.gallery:
            gap = self.t - gI['t'] + 0.3
            if np.hypot(*(pos - gI['x'])) <= p['reidSpeed'] * gap + p['reidSlack'] and not any(t['id'] == gI['id'] for t in self.tracks):
                s = cos_sim(gI['feat'], f)
                if s > bestSim:
                    best, bestSim = ('gallery', gI, gap), s
        if best is None:
            return False
        kind, obj, gap = best
        if kind == 'live':                                # coasting track: jump it to the detection
            self.tracks.remove(obj)
        else:
            self.gallery.remove(obj)
        t = self._birth(cam, d, f, tid=obj['id'], confirmed=True)
        t['feat'] = 0.5 * obj['feat'] + 0.5 * f
        self.reidEvents.append((self.t, obj['id'], bestSim, gap))
        return True

    def reported(self):
        return [t for t in self.tracks if t['confirmed'] and t['lastSeen'] <= 3]

    def confirmed(self):
        return [t for t in self.tracks if t['confirmed']]
