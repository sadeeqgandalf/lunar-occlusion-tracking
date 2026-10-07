// The event camera as the rover's only eyes, against the ordinary camera, on identical people and paths.
//   node experiments/event_camera.mjs 20 300   ->  experiments/event_camera.json + printed table
// The event camera's response to motion is fitted to a real recording (EVOS; see src/mot/occlusion.js).
import { writeFileSync } from 'node:fs';
import { MotSession } from '../src/mot/session.js';

const seeds = +(process.argv[2] || 20), seconds = +(process.argv[3] || 300), rows = [];
for (const scenario of ['boulders', 'polar', 'work']) for (const eyes of ['camera', 'event']) {
  const acc = {};
  for (let seed = 1; seed <= seeds; seed++) {
    const s = new MotSession({ scenario, seed, eyes }).run(seconds);
    for (const r of s.summaries()) {
      const a = (acc[r.name] ||= { sk: 0, sn: 0, lk: 0, ln: 0, idsw: [], idf1: [], mota: [] }), b = r.occByDuration;
      a.sk += b[0].kept + b[1].kept; a.sn += b[0].n + b[1].n; a.lk += b[2].kept + b[3].kept; a.ln += b[2].n + b[3].n;
      a.idsw.push(r.idsw); a.idf1.push(r.idf1); a.mota.push(r.mota);
    }
    process.stderr.write(`${scenario} ${eyes} seed ${seed}\n`);
  }
  for (const [name, a] of Object.entries(acc)) rows.push({ scenario, eyes, name, ...a });
}
writeFileSync(new URL('./event_camera.json', import.meta.url), JSON.stringify({ seeds, seconds, rows }, null, 1));
const wil = (k, n) => { if (!n) return '–'; const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d; return `${Math.round(100 * p)}% [${Math.round(100 * (c - h))}-${Math.round(100 * (c + h))}] n=${n}`; };
const ms = (x) => { const m = x.reduce((a, b) => a + b, 0) / x.length, sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1 || 1)); return [m, 1.96 * sd / Math.sqrt(x.length)]; };
console.log('scenario | eyes | tracker | same ID, hide 1-5 s | same ID, hide > 5 s | ID switches / run | IDF1 | MOTA');
for (const r of rows) { const [i, ie] = ms(r.idsw), [f] = ms(r.idf1), [m] = ms(r.mota); console.log([r.scenario, r.eyes, r.name, wil(r.sk, r.sn), wil(r.lk, r.ln), `${i.toFixed(1)} ± ${ie.toFixed(1)}`, f.toFixed(3), m.toFixed(2)].join(' | ')); }
