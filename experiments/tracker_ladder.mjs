// The tracker ladder: every web-lab tracker, rover-only and rover + lander, many seeds, one table.
//   node experiments/tracker_ladder.mjs 20 300   ->  experiments/tracker_ladder.json + printed table
import { writeFileSync } from 'node:fs';
import { MotSession } from '../src/mot/session.js';

const seeds = +(process.argv[2] || 20), seconds = +(process.argv[3] || 300);
const rows = [];
const SENSORS = [{}, { lidar: true }, { lander: true }, { lidar: true, lander: true }];
for (const scenario of ['boulders', 'polar']) for (const sensors of SENSORS) {
  const acc = {};
  for (let seed = 1; seed <= seeds; seed++) {
    const s = new MotSession({ scenario, seed, ...sensors }).run(seconds);
    for (const r of s.summaries()) {
      const a = (acc[r.name] ||= { sk: 0, sn: 0, lk: 0, ln: 0, idsw: 0, idf1: 0, mota: 0, n: 0 }), b = r.occByDuration;
      a.sk += b[0].kept + b[1].kept; a.sn += b[0].n + b[1].n; a.lk += b[2].kept + b[3].kept; a.ln += b[2].n + b[3].n;
      a.idsw += r.idsw; a.idf1 += r.idf1; a.mota += r.mota; a.n++;
    }
    process.stderr.write(`${scenario} ${JSON.stringify(sensors)} seed ${seed}\n`);
  }
  const label = ['camera', sensors.lidar && 'lidar', sensors.lander && 'lander'].filter(Boolean).join(' + ');
  for (const [name, a] of Object.entries(acc)) rows.push({ scenario, sensors: label, name, ...a });
}
writeFileSync(new URL('./tracker_ladder.json', import.meta.url), JSON.stringify({ seeds, seconds, rows }, null, 1));
const w = (k, n) => { if (!n) return '–'; const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d; return `${Math.round(100 * p)}% (${Math.round(100 * (c - h))}–${Math.round(100 * (c + h))})`; };
for (const r of rows) console.log([r.scenario, r.sensors, r.name, w(r.sk, r.sn), w(r.lk, r.ln), (r.idsw / r.n).toFixed(1), (r.idf1 / r.n).toFixed(3)].join(' | '));
