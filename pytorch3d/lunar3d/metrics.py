"""Tracking scores against ground truth (port of src/mot/metrics.js): CLEAR-MOT (MOTA, ID switches), IDF1, GOSPA,
and occlusion events (tracked before hiding >= 1 s; kept = same ID within 2 s of reappearing)."""
import math
import numpy as np
from .hungarian import hungarian

TAU, GOSPA_C, MIN_OCC, REACQ, CHI2_95_2 = 2.0, 2.0, 1.0, 2.0, 5.991


class MotMetrics:
    def __init__(self):
        self.f = self.gtCount = self.tp = self.fp = self.fn = self.idsw = self.predCount = 0
        self.distSum = self.gospaSum = 0.0
        self.lastMatch, self.pairCounts, self.occ, self.events, self.switches = {}, {}, {}, [], []
        self.hiddenFrames = self.hiddenAlive = self.hiddenCovered = 0

    def update(self, world, tracker):
        T, F = world.t, world.frame
        hid = {g['id']: g['vis']['inFov'] and g['vis']['pd'] < 0.15 for g in world.targets}
        vis = [g for g in world.targets if g['vis']['inFov'] and not hid[g['id']]]
        rep = tracker.reported()
        self.f += 1; self.gtCount += len(vis); self.predCount += len(rep)
        d = lambda g, k: math.hypot(g['x'] - k['x'][0], g['y'] - k['x'][1])
        match, usedK = {}, set()
        for g in vis:
            kid = self.lastMatch.get(g['id'])
            k = next((x for x in rep if x['id'] == kid), None) if kid is not None else None
            if k is not None and k['id'] not in usedK and d(g, k) <= TAU:
                match[g['id']] = k; usedK.add(k['id'])
        G = [g for g in vis if g['id'] not in match]; K = [k for k in rep if k['id'] not in usedK]
        a = hungarian([[d(g, k) if d(g, k) <= TAU else 1e9 for k in K] for g in G], 1e9)
        for i, j in enumerate(a):
            if j >= 0:
                match[G[i]['id']] = K[j]; usedK.add(K[j]['id'])
        for gid, k in match.items():
            prev = self.lastMatch.get(gid)
            if prev is not None and prev != k['id']:
                self.idsw += 1; self.switches.append((T, gid, prev, k['id']))
            self.lastMatch[gid] = k['id']; self.tp += 1
            self.distSum += d(next(g for g in vis if g['id'] == gid), k)
        self.fn += len(vis) - len(match); self.fp += len(rep) - len(match)
        for g in vis:
            for k in rep:
                if d(g, k) <= TAU:
                    key = (g['id'], k['id']); self.pairCounts[key] = self.pairCounts.get(key, 0) + 1
        ga = hungarian([[d(g, k) ** 2 if d(g, k) < GOSPA_C else 1e9 for k in rep] for g in vis], 1e9)
        gs, nA = 0.0, 0
        for i, j in enumerate(ga):
            if j >= 0:
                gs += d(vis[i], rep[j]) ** 2; nA += 1
        gs += GOSPA_C ** 2 / 2 * (len(vis) - nA + len(rep) - nA)
        self.gospaSum += math.sqrt(gs)
        for g in world.targets:
            hiddenNow = hid[g['id']]; visibleNow = g['vis']['inFov'] and not hiddenNow
            o = self.occ.get(g['id'], {'state': 'idle'})
            if o['state'] == 'idle' and hiddenNow and g['id'] in self.lastMatch and o.get('lvm', -1e9) >= F - 5:
                o = {'state': 'hidden', 'since': T, 'before': self.lastMatch[g['id']]}
            elif o['state'] == 'hidden':
                if not g['vis']['inFov']:
                    o = {'state': 'idle'}
                elif visibleNow:
                    o = {'state': 'reacq', 'since': T, 'before': o['before'], 'dur': T - o['since']} if T - o['since'] >= MIN_OCC else {'state': 'idle'}
                else:
                    self.hiddenFrames += 1
                    k = next((x for x in tracker.tracks if x['id'] == o['before']), None)
                    if k is not None:
                        self.hiddenAlive += 1
                        e = np.array([g['x'] - k['x'][0], g['y'] - k['x'][1]]); Pp = k['P'][:2, :2]
                        if np.linalg.det(Pp) > 0 and e @ np.linalg.solve(Pp, e) <= CHI2_95_2:
                            self.hiddenCovered += 1
            elif o['state'] == 'reacq':
                m = match.get(g['id'])
                if m is not None:
                    self.events.append(dict(t=T, target=g['id'], dur=o['dur'], kept=m['id'] == o['before'])); o = {'state': 'idle'}
                elif T - o['since'] > REACQ or not g['vis']['inFov']:
                    self.events.append(dict(t=T, target=g['id'], dur=o['dur'], kept=False)); o = {'state': 'idle'}
            if visibleNow and g['id'] in match:
                o['lvm'] = F
            self.occ[g['id']] = o

    def summary(self):
        # IDF1: the JS builds row/column order from first appearance in the count map
        gI = list(dict.fromkeys(k[0] for k in self.pairCounts)); kI = list(dict.fromkeys(k[1] for k in self.pairCounts))
        maxC = max([1] + list(self.pairCounts.values()))
        a = hungarian([[maxC - self.pairCounts.get((g, k), 0) for k in kI] for g in gI], 1e12)
        idtp = sum(self.pairCounts.get((gI[i], kI[j]), 0) for i, j in enumerate(a) if j >= 0)
        ev = self.events
        by = []
        for lo, hi in [(1, 2), (2, 5), (5, 10), (10, math.inf)]:
            e = [x for x in ev if lo <= x['dur'] < hi]
            by.append(dict(range=(lo, hi), n=len(e), kept=sum(x['kept'] for x in e)))
        return dict(frames=self.f, mota=1 - (self.fn + self.fp + self.idsw) / max(self.gtCount, 1e-12),
                    idf1=2 * idtp / max(self.gtCount + self.predCount, 1e-12), idsw=self.idsw, fp=self.fp, fn=self.fn,
                    gospa=self.gospaSum / max(1, self.f), occEvents=len(ev), occKept=sum(x['kept'] for x in ev), occByDuration=by)
