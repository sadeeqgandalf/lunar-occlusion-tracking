import test from 'node:test';
import assert from 'node:assert/strict';
import { hungarian } from '../src/mot/hungarian.js';
import { visibility } from '../src/mot/occlusion.js';
import { MotSession } from '../src/mot/session.js';
import { RNG } from '../src/core/rng.js';

test('hungarian: optimal on random dense square/rectangular matrices (vs brute force)', () => {
  const r = new RNG(3);
  for (let k = 0; k < 60; k++) {
    const nr = 1 + r.int(5), nc = 1 + r.int(5), C = Array.from({ length: nr }, () => Array.from({ length: nc }, () => r.uniform(0, 10)));
    const a = hungarian(C), cost = (x) => x.reduce((s, j, i) => s + (j >= 0 ? C[i][j] : 0), 0);
    assert.equal(a.filter((j) => j >= 0).length, Math.min(nr, nc), 'dense problem must assign min(nr,nc)');
    // brute-force optimum over all full matchings (dense matrix => a full matching is optimal)
    const fullBest = (() => { let best = Infinity; const rec = (i, used, c) => { if (i === nr || used.size === Math.min(nr, nc)) { if (used.size === Math.min(nr, nc)) best = Math.min(best, c); return; } for (let j = 0; j < nc; j++) if (!used.has(j)) { used.add(j); rec(i + 1, used, c + C[i][j]); used.delete(j); } if (nr > nc) rec(i + 1, used, c); }; rec(0, new Set(), 0); return best; })();
    assert.ok(Math.abs(cost(a) - fullBest) < 1e-9, `cost ${cost(a)} vs optimum ${fullBest}`);
  }
});

test('hungarian: forbidden pairs are never returned', () => {
  const C = [[1, 1e9], [1e9, 1e9], [2, 3]];
  const a = hungarian(C, 1e9);
  assert.equal(a[1], -1);
  assert.ok(a.every((j, i) => j < 0 || C[i][j] < 1e9));
});

test('occlusion geometry: fully behind, clear, and partial cases', () => {
  const cam = { x: 0, y: 0, th: Math.PI / 2, fov: Math.PI / 2, range: 50 }, rock = [{ x: 0, y: 10, r: 2 }];
  assert.equal(visibility(cam, rock, 0, 20, 0.4).visFrac, 0);                       // directly behind
  assert.equal(visibility(cam, rock, 10, 20, 0.4).visFrac, 1);                      // far to the side
  assert.equal(visibility(cam, rock, 0, 5, 0.4).visFrac, 1);                        // in front of the rock
  const edge = visibility(cam, rock, 4.08, 20, 0.4);                                // straddling the shadow edge
  assert.ok(edge.visFrac > 0.2 && edge.visFrac < 0.8, `partial ${edge.visFrac}`);
  assert.ok(edge.visCenter < edge.bearing, 'visible centroid shifts away from the rock (to the right here)');
  assert.equal(visibility(cam, rock, 0, -10, 0.4).inFov, false);                    // behind the camera
});

test('tracker never sees identities: only range and bearing are passed', () => {
  const s = new MotSession({ scenario: 'boulders', seed: 2 }), seen = [];
  const orig = s.runs[0].tracker.step.bind(s.runs[0].tracker);
  s.runs[0].tracker.step = (dets) => { seen.push(...dets); return orig(dets); };
  s.run(5);
  assert.ok(seen.length > 0 && seen.every((d) => Object.keys(d).sort().join() === 'bearing,range'));
});

test('MOT runs are deterministic for a seed', () => {
  const a = new MotSession({ scenario: 'crossing', seed: 9 }).run(60).summaries(), b = new MotSession({ scenario: 'crossing', seed: 9 }).run(60).summaries();
  assert.deepEqual(a, b);
});

test('headline: occlusion-aware trackers keep IDs through short occlusions; the naive tracker cannot', () => {
  const tot = {};
  for (let seed = 1; seed <= 8; seed++) for (const o of new MotSession({ scenario: 'boulders', seed }).run(300).summaries()) {
    const t = (tot[o.name] ||= { k: 0, n: 0, idsw: 0 });
    for (const b of o.occByDuration.slice(0, 2)) { t.k += b.kept; t.n += b.n; } t.idsw += o.idsw;
  }
  const f = (n) => tot[n].k / tot[n].n;
  assert.equal(tot['Naive'].k, 0, 'naive tracker should never survive a >=1 s occlusion');
  assert.ok(f('Occlusion-aware') >= 0.15, `aware short-occlusion retention ${f('Occlusion-aware')}`);
  assert.ok(f('Aware + neg. info') >= 0.15, `aware+NI short-occlusion retention ${f('Aware + neg. info')}`);
  assert.ok(tot['Aware + neg. info'].idsw < tot['Naive'].idsw, 'fewer ID switches than naive');
});

test('3-D visibility: a low rock hides legs but not the head; a tall rock hides everything', () => {
  const cam = { x: 0, y: 0, th: Math.PI / 2, fov: Math.PI / 2, range: 50 };
  const low = visibility(cam, [{ x: 0, y: 10, r: 2, h: 0.9 }], 0, 14, 0.4);
  assert.ok(low.visFrac > 0.25 && low.visFrac < 1, `low rock: partly visible, got ${low.visFrac}`);
  assert.equal(visibility(cam, [{ x: 0, y: 10, r: 2, h: 3.2 }], 0, 14, 0.4).visFrac, 0, 'tall rock: fully hidden');
  assert.equal(visibility(cam, [{ x: 0, y: 10, r: 2, h: 3.2 }], 0, 5, 0.4).visFrac, 1, 'in front of the rock: fully visible');
});

test('3-D visibility: a boulder without a height behaves like the old planar full occluder', () => {
  const cam = { x: 0, y: 0, th: Math.PI / 2, fov: Math.PI / 2, range: 50 };
  assert.equal(visibility(cam, [{ x: 0, y: 10, r: 2 }], 0, 20, 0.4).visFrac, 0);
});

test('shadows: a person down-sun of a tall rock is in shadow; in the open, or up-sun of it, they are not', async () => {
  const { shadowFraction } = await import('../src/mot/occlusion.js');
  const sun = { az: 0, el: (8 * Math.PI) / 180 };                // sun low in the +x direction
  const rock = [{ x: 10, y: 0, r: 2, h: 3 }];
  assert.ok(shadowFraction(rock, 4, 0, sun) > 0.5, 'behind the rock relative to the sun');
  assert.equal(shadowFraction(rock, 16, 0, sun), 0, 'on the sunny side');
  assert.equal(shadowFraction(rock, 4, 8, sun), 0, 'off to the side');
  assert.equal(shadowFraction([], 4, 0, sun), 0, 'no rocks');
});

test('polar scenario: people actually vanish into shadow while still in line of sight', () => {
  const s = new MotSession({ scenario: 'polar', seed: 3 });
  let shadowHidden = 0;
  for (let i = 0; i < 1500; i++) { s.step(); for (const t of s.world.targets) if (t.vis.inFov && t.vis.visFrac >= 0.25 && t.vis.pd < 0.15) shadowHidden++; }
  assert.ok(shadowHidden > 0, 'expected some shadow-only occlusion in 150 s');
});

test('hidden-cause wording: shadow only when shadow is measured, never in shadow-free scenes', async () => {
  const { hiddenCause } = await import('../src/mot/occlusion.js');
  assert.equal(hiddenCause({ visFrac: 0.3, shadow: 0 }), 'rock');          // a sliver visible behind a rock
  assert.equal(hiddenCause({ visFrac: 1, shadow: 1 }), 'shadow');
  const s = new MotSession({ scenario: 'boulders', seed: 4 });
  for (let i = 0; i < 800; i++) { s.step(); for (const t of s.world.targets) assert.notEqual(hiddenCause(t.vis), 'shadow'); }
});

// ---------------------------------------------------------------- re-identification
import { personSignature, clutterSignature, cosine } from '../src/mot/appearance.js';
import { RNG as RNG2 } from '../src/core/rng.js';

test('appearance signatures: same person similar, different people and rocks not', () => {
  const r = new RNG2(3);
  const a = personSignature(1, 0.9, 0, r), b = personSignature(1, 0.4, 0, r), c = personSignature(2, 0.9, 0, r), d = clutterSignature(r);
  assert.ok(cosine(a, b) > 0.86, `same person ${cosine(a, b)}`);
  assert.ok(cosine(a, c) < 0.5, `different people ${cosine(a, c)}`);
  assert.ok(cosine(a, d) < 0.5, `person vs rock ${cosine(a, d)}`);
});

test('re-ID tracker keeps more identities through long hides and switches IDs less (polar, 300 s)', async () => {
  const { MotSession } = await import('../src/mot/session.js');
  const s = new MotSession({ scenario: 'polar', seed: 3 }).run(300);
  const [ni, , , re] = s.summaries();
  const long = (x) => x.occByDuration[2].kept + x.occByDuration[3].kept;
  assert.ok(re.idsw < ni.idsw, `ID switches re-ID ${re.idsw} vs ${ni.idsw}`);
  assert.ok(long(re) > long(ni), `long hides kept re-ID ${long(re)} vs ${long(ni)}`);
  assert.ok(re.idf1 > ni.idf1, `IDF1 re-ID ${re.idf1} vs ${ni.idf1}`);
});

test('re-ID never changes the other three trackers (appearance has its own random stream)', async () => {
  const { MotSession } = await import('../src/mot/session.js');
  const a = new MotSession({ scenario: 'boulders', seed: 5 }).run(60).summaries();
  const { MotWorld } = await import('../src/mot/world.js');
  const { Tracker } = await import('../src/mot/tracker.js');
  const { MotMetrics } = await import('../src/mot/metrics.js');
  const W = new MotWorld('boulders', 5), T = new Tracker(W.cam, W.boulders, { occlusionAware: true, negInfo: true, pd: W.cfg.pd, clutter: W.cfg.clutter }), M = new MotMetrics();
  for (let i = 0; i < 600; i++) { W.step(); const { dets } = W.sense(); T.step(dets.map((d) => ({ range: d.range, bearing: d.bearing }))); M.update(W, T); }
  assert.equal(M.summary().idsw, a[0].idsw);
  assert.equal(M.summary().idf1, a[0].idf1);
});

test('PHD heat map: expected number of people tracks the truth, and it is fast', async () => {
  const { MotSession } = await import('../src/mot/session.js');
  const s = new MotSession({ scenario: 'boulders', seed: 8, phd: true });
  const c = []; const t0 = performance.now();
  for (let f = 0; f < 1200; f++) { s.step(); if (f > 100) c.push(s.phd.expectedCount); }
  c.sort((a, b) => a - b); const med = c[c.length >> 1];
  assert.ok(med > 3.5 && med < 7, `median expected people ${med} (truth 5)`);
  const h = s.phd.density(); assert.ok(h.D.some((v) => v > 0.02), 'density has mass');
});

test('lander camera: two viewpoints cut ID switches, and single-camera results are unchanged', async () => {
  const { MotSession } = await import('../src/mot/session.js');
  const one = new MotSession({ scenario: 'boulders', seed: 8 }).run(240).summaries();
  const two = new MotSession({ scenario: 'boulders', seed: 8, lander: true, phd: true }).run(240);
  const s2 = two.summaries();
  assert.ok(s2[0].idsw < one[0].idsw, `NI with lander ${s2[0].idsw} vs rover only ${one[0].idsw}`);
  assert.ok(two.phd.expectedCount > 2 && two.phd.expectedCount < 9, `PHD expects ${two.phd.expectedCount}`);
  const again = new MotSession({ scenario: 'boulders', seed: 8 }).run(240).summaries();
  assert.deepEqual(again.map((r) => r.idsw), one.map((r) => r.idsw));
});

test('lidar: sees in polar shadow, so adding it to the camera cuts ID switches; camera-only results unchanged', async () => {
  const { MotSession } = await import('../src/mot/session.js');
  let cam = 0, fused = 0;
  for (const seed of [3, 5]) {
    cam += new MotSession({ scenario: 'polar', seed }).run(240).summaries()[0].idsw;
    fused += new MotSession({ scenario: 'polar', seed, lidar: true }).run(240).summaries()[0].idsw;
  }
  assert.ok(fused < cam, `Aware+NI ID switches: camera ${cam}, camera + lidar ${fused}`);
  const { sensorPd } = await import('../src/mot/occlusion.js');
  const { LIDAR } = await import('../src/mot/world.js');
  const v = { inFov: true, visFrac: 1, range: 20 };
  assert.equal(sensorPd(LIDAR, v, 0.95, 1), LIDAR.pd);                 // full shadow: lidar unaffected
  assert.ok(sensorPd(null, v, 0.95, 1) < 0.25);                        // ... the camera is not
});

test('lidar scan image: suits return more light than regolith, far grazing ground drops out, a scan replays identically', async () => {
  const { lidarScan } = await import('../src/mot/lidarscan.js');
  const sess = new MotSession({ scenario: 'boulders', seed: 8, lidar: true }); sess.run(5);
  const w = sess.world, a = lidarScan(w), mean = (f) => { let s = 0, n = 0; for (let k = 0; k < a.img.length; k++) if (a.img[k] && f(k)) { s += a.inten[k]; n++; } return n ? s / n : NaN; };
  const suit = mean((k) => a.who[k] > 0), ground = mean((k) => a.who[k] === 0);
  assert.ok(suit > 3 * ground, `suit ${suit} vs ground ${ground}`);
  let far = 0; for (let k = 0; k < a.img.length; k++) if (a.who[k] === 0 && a.img[k] > 35) far++;
  assert.equal(far, 0, 'dark ground at a grazing angle beyond 35 m should give no return');
  assert.ok(a.ret.length / 6 > 5000, 'still a dense scan');
  const again = lidarScan({ ...w, lidar: w.lidar, boulders: w.boulders, targets: w.targets, frame: w.frame });   // new object: bypasses the cache
  assert.deepEqual(again.img, a.img);
});

test('lidar instance segmentation: masks found from the ranges alone overlap the true person masks (mean IoU > 0.85)', async () => {
  const { lidarScan, segmentScan } = await import('../src/mot/lidarscan.js');
  const sess = new MotSession({ scenario: 'boulders', seed: 2, lidar: true }); let sum = 0, n = 0, flagged = 0;
  for (let f = 0; f < 30; f++) {
    sess.run(1); const w = sess.world, sc = lidarScan(w), sg = segmentScan(sc, w.lidar), truth = new Map(), inter = new Map();
    for (let k = 0; k < sc.img.length; k++) { const id = sc.who[k]; if (id <= 0) continue; truth.set(id, (truth.get(id) || 0) + 1); const l = sg.lab[k]; if (l) inter.set(`${id}:${l}`, (inter.get(`${id}:${l}`) || 0) + 1); }
    for (const [id, tn] of truth) {
      if (tn < 8) continue; let best = 0, bi = null;
      for (const i of sg.inst) { const it = inter.get(`${id}:${i.id}`) || 0, iou = it / (tn + i.n - it); if (iou > best) { best = iou; bi = i; } }
      sum += best; n++; if (bi?.personLike) flagged++;
    }
  }
  assert.ok(n > 50, `enough visible people (${n})`);
  assert.ok(sum / n > 0.85, `mean IoU ${(sum / n).toFixed(3)}`);
  assert.ok(flagged / n > 0.9, `person-shaped flag recall ${(flagged / n).toFixed(2)}`);
});

test('event camera: sees motion, not presence; default camera results are unchanged', async () => {
  const { sensorPd, motionSignal } = await import('../src/mot/occlusion.js');
  const { EVENT, CAMERA } = await import('../src/mot/world.js');
  const cam = { ...CAMERA, ...EVENT }, v = { inFov: true, visFrac: 1, range: 20, bearing: Math.PI / 2 };
  assert.equal(sensorPd(cam, v, 0.95, 0, [0, 0]), 0);                               // standing still in plain view: nothing to report
  assert.ok(sensorPd(cam, v, 0.95, 0, [1, 0]) > 0.94);                              // walking across the view at 1 m/s
  assert.ok(motionSignal(cam, v, [0, 1]) < motionSignal(cam, v, [1, 0]));           // straight toward the camera is weaker than across
  assert.ok(sensorPd(cam, v, 0.95, 1, [1, 0]) > 0.8);                               // full shadow costs an event camera little
  assert.ok(sensorPd(null, v, 0.95, 1) < 0.25);                                     // ... and an ordinary camera a lot
  assert.equal(sensorPd(cam, { ...v, visFrac: 0.1 }, 0.95, 0, [1, 0]), 0);          // still needs line of sight
  // people who stop to work: the event camera loses sight of them, the ordinary camera does not
  const hidden = (eyes) => { const s = new MotSession({ scenario: 'work', seed: 3, eyes }); let n = 0, k = 0; for (let i = 0; i < 600; i++) { s.step(); for (const t of s.world.targets) if (t.pause > 0 && t.vis.inFov && t.vis.visFrac >= 0.7) { n++; if (t.vis.pd < 0.15) k++; } } return [k, n]; };
  const [ke, ne] = hidden('event'), [kc, nc] = hidden('camera');
  assert.ok(ne > 100 && ke === ne, `event camera: ${ke} of ${ne} stopped, fully visible person-frames undetectable`);
  assert.equal(kc, 0, 'ordinary camera still sees them');
  // the default lab is untouched: same switches as before the event camera existed (boulders, seed 8, 60 s)
  const base = new MotSession({ scenario: 'boulders', seed: 8 }).run(60).summaries().map((r) => r.idsw);
  assert.deepEqual(base, new MotSession({ scenario: 'boulders', seed: 8, eyes: 'camera' }).run(60).summaries().map((r) => r.idsw));
});
