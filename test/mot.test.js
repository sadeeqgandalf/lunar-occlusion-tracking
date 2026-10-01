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
