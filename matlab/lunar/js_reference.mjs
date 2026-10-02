// Prints the web lab's results for one run as JSON, for lunar_crosscheck.m to compare against MATLAB.
// usage: node js_reference.mjs <scenario> <seed> <seconds>
import { MotSession } from '../../src/mot/session.js';
const [scenario = 'boulders', seed = '7', seconds = '120'] = process.argv.slice(2);
const s = new MotSession({ scenario, seed: +seed });
let nDets = 0;
const n = Math.round(+seconds / s.world.dt);
for (let i = 0; i < n; i++) { s.step(); nDets += s.last.dets.length; }
console.log(JSON.stringify({
  nDets, truth: s.world.targets.map((t) => [t.x, t.y]),
  boulders: s.world.boulders.map((b) => [b.x, b.y, b.r, b.h]),
  runs: s.summaries().map((r) => ({ name: r.name, idsw: r.idsw, idf1: r.idf1, mota: r.mota, gospa: r.gospa, occEvents: r.occEvents, occKept: r.occKept, fp: r.fp, fn: r.fn })),
}));
