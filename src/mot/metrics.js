// Tracking scores against ground truth.
//  * Reported tracks = confirmed AND seen within 0.3 s (what a system would draw as a box; identical rule for every
//    tracker). Coasting tracks are kept internally and judged separately by the occlusion metrics.
//  * CLEAR-MOT (Bernardin & Stiefelhagen 2008): MOTA, MOTP, ID switches, with previous matches kept when still valid.
//  * IDF1 (Ristani et al. 2016): global identity mapping, then identity precision/recall.
//  * GOSPA (Rahmathullah, Garcia-Fernandez & Svensson 2017), p=2, c=2 m, alpha=2.
//  * Occlusion events: a target tracked before going behind cover for >= 1 s; success = same track ID within 2 s
//    of reappearing. While hidden we also score whether the coasting track still exists and contains the truth.
import { hungarian } from './hungarian.js';
import { isHidden } from './occlusion.js';

const TAU = 2.0, GOSPA_C = 2.0, MIN_OCC = 1.0, REACQ = 2.0, CHI2_95_2 = 5.991;

export class MotMetrics {
  constructor() {
    this.f = 0; this.gtCount = 0; this.tp = 0; this.fp = 0; this.fn = 0; this.idsw = 0; this.distSum = 0;
    this.gospaSum = 0; this.lastMatch = new Map(); this.pairCounts = new Map(); this.predCount = 0;
    this.occ = new Map(); this.events = []; this.hiddenFrames = 0; this.hiddenAlive = 0; this.hiddenCovered = 0; this.hiddenErrSum = 0;
  }

  update(world, tracker) {
    const T = world.t;
    const vis = world.targets.filter((t) => t.vis.inFov && !isHidden(t.vis));
    const rep = tracker.reported(), F = world.frame;
    this.f++; this.gtCount += vis.length; this.predCount += rep.length;
    const d = (g, k) => Math.hypot(g.x - k.x[0], g.y - k.x[1]);

    // --- CLEAR-MOT matching: keep previous correspondences that are still valid, Hungarian for the rest
    const match = new Map(), usedK = new Set();
    for (const g of vis) {
      const kid = this.lastMatch.get(g.id), k = kid !== undefined && rep.find((x) => x.id === kid);
      if (k && !usedK.has(k.id) && d(g, k) <= TAU) { match.set(g.id, k); usedK.add(k.id); }
    }
    const G = vis.filter((g) => !match.has(g.id)), K = rep.filter((k) => !usedK.has(k.id));
    const a = hungarian(G.map((g) => K.map((k) => (d(g, k) <= TAU ? d(g, k) : 1e9))), 1e9);
    a.forEach((j, i) => { if (j >= 0) { match.set(G[i].id, K[j]); usedK.add(K[j].id); } });
    for (const [gid, k] of match) {
      const prev = this.lastMatch.get(gid);
      if (prev !== undefined && prev !== k.id) this.idsw++;
      this.lastMatch.set(gid, k.id); this.tp++; this.distSum += d(vis.find((g) => g.id === gid), k);
    }
    this.fn += vis.length - match.size; this.fp += rep.length - match.size;

    // --- IDF1 co-occurrence counts (pairs within TAU)
    for (const g of vis) for (const k of rep) if (d(g, k) <= TAU) { const key = g.id + ':' + k.id; this.pairCounts.set(key, (this.pairCounts.get(key) || 0) + 1); }

    // --- GOSPA
    const ga = hungarian(vis.map((g) => rep.map((k) => (d(g, k) < GOSPA_C ? d(g, k) ** 2 : 1e9))), 1e9);
    let gs = 0, nA = 0;
    ga.forEach((j, i) => { if (j >= 0) { gs += d(vis[i], rep[j]) ** 2; nA++; } });
    gs += (GOSPA_C ** 2 / 2) * (vis.length - nA + rep.length - nA);
    this.gospaSum += Math.sqrt(gs);

    // --- occlusion events
    for (const g of world.targets) {
      const hiddenNow = isHidden(g.vis), visibleNow = g.vis.inFov && !hiddenNow;
      let o = this.occ.get(g.id) || { state: 'idle' };
      if (o.state === 'idle' && hiddenNow && this.lastMatch.has(g.id) && o.lastVisibleMatched >= F - 5) o = { state: 'hidden', since: T, before: this.lastMatch.get(g.id) };
      else if (o.state === 'hidden') {
        if (!g.vis.inFov) o = { state: 'idle' };                       // walked out of view: not an occlusion
        else if (visibleNow) o = T - o.since >= MIN_OCC ? { state: 'reacq', since: T, before: o.before, dur: T - o.since } : { state: 'idle' };
        else { // still hidden: is the coasting track alive and honest?
          this.hiddenFrames++;
          const k = tracker.tracks.find((x) => x.id === o.before);
          if (k) {
            this.hiddenAlive++;
            const ex = g.x - k.x[0], ey = g.y - k.x[1], a0 = k.P.get(0, 0), b0 = k.P.get(0, 1), c0 = k.P.get(1, 1), det = a0 * c0 - b0 * b0;
            this.hiddenErrSum += Math.hypot(ex, ey);
            if (det > 0 && (c0 * ex * ex - 2 * b0 * ex * ey + a0 * ey * ey) / det <= CHI2_95_2) this.hiddenCovered++;
          }
        }
      } else if (o.state === 'reacq') {
        const m = match.get(g.id);
        if (m) { this.events.push({ t: T, target: g.id, dur: o.dur, kept: m.id === o.before }); o = { state: 'idle' }; }
        else if (T - o.since > REACQ || !g.vis.inFov) { this.events.push({ t: T, target: g.id, dur: o.dur, kept: false }); o = { state: 'idle' }; }
      }
      if (visibleNow && match.has(g.id)) o.lastVisibleMatched = F;
      this.occ.set(g.id, o);
    }
  }

  summary() {
    // IDF1: global identity mapping maximising co-occurrence
    const gIds = [...new Set([...this.pairCounts.keys()].map((k) => k.split(':')[0]))];
    const kIds = [...new Set([...this.pairCounts.keys()].map((k) => k.split(':')[1]))];
    const maxC = Math.max(1, ...this.pairCounts.values());
    const a = hungarian(gIds.map((g) => kIds.map((k) => maxC - (this.pairCounts.get(g + ':' + k) || 0))), 1e12);
    let idtp = 0; a.forEach((j, i) => { if (j >= 0) idtp += this.pairCounts.get(gIds[i] + ':' + kIds[j]) || 0; });
    const ev = this.events, kept = ev.filter((e) => e.kept).length;
    return {
      frames: this.f,
      mota: this.gtCount ? 1 - (this.fn + this.fp + this.idsw) / this.gtCount : NaN,
      motp: this.tp ? this.distSum / this.tp : NaN,
      idf1: this.gtCount + this.predCount ? (2 * idtp) / (this.gtCount + this.predCount) : NaN,
      idsw: this.idsw, fp: this.fp, fn: this.fn,
      gospa: this.gospaSum / Math.max(1, this.f),
      occEvents: ev.length, occKept: kept, occKeptFrac: ev.length ? kept / ev.length : NaN,
      occByDuration: [[1, 2], [2, 5], [5, 10], [10, Infinity]].map(([a, b]) => {
        const e = ev.filter((x) => x.dur >= a && x.dur < b); return { range: [a, b], n: e.length, kept: e.filter((x) => x.kept).length };
      }),
      hiddenAliveFrac: this.hiddenFrames ? this.hiddenAlive / this.hiddenFrames : NaN,
      hiddenCoverage: this.hiddenAlive ? this.hiddenCovered / this.hiddenAlive : NaN,
      hiddenErr: this.hiddenAlive ? this.hiddenErrSum / this.hiddenAlive : NaN,
    };
  }
}
