// Reproducible benchmark: every filter on identical sensor streams, 5 seeds x every non-sandbox scenario x 3 platforms.
// The oracle autopilot flies the planned route from ground truth, so all filters see the exact same trajectory.
import { writeFileSync } from 'node:fs';
import { Mission } from '../src/sim/mission.js';
import { listPlatforms } from '../src/platforms/index.js';

const SEEDS = [1, 2, 3, 4, 5], IDS = ['dr', 'ekf', 'ukf', 'pf'], rows = [];
for (const P of listPlatforms()) for (const [sc, cfg] of Object.entries(P.scenarios)) {
  if (!cfg.goal) continue;
  const acc = Object.fromEntries(IDS.map((i) => [i, { rmse: [], anees: [], us: [], inb: [] }]));
  let dof = 0, simT = 0;
  for (const seed of SEEDS) {
    const m = new Mission({ platform: P.id, scenario: sc, seed, oracle: true, filters: IDS });
    m.issue({ type: 'plan' });
    for (let i = 0; i < 20 * 900 && m.status === 'running'; i++) m.step();
    dof = m.model.neesIdx.length; simT += m.t;
    for (const id of IDS) { const s = m.summary(id); acc[id].rmse.push(s.rmse); acc[id].anees.push(s.anees); acc[id].us.push(s.usPerStep); acc[id].inb.push(s.neesFrac); }
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  for (const id of IDS) rows.push({ platform: P.id, scenario: sc, filter: id, rmse: mean(acc[id].rmse), anees: mean(acc[id].anees), dof, inband: mean(acc[id].inb), us: mean(acc[id].us), simT: simT / SEEDS.length });
}
const fmt = (v) => (v >= 1000 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
let md = '| platform | scenario | filter | RMSE [m] | ANEES (ideal = dof) | NEES in 95% band | µs/step |\n|---|---|---|---:|---:|---:|---:|\n';
for (const r of rows) md += `| ${r.platform} | ${r.scenario} | ${r.filter} | ${fmt(r.rmse)} | ${fmt(r.anees)} (dof ${r.dof}) | ${(100 * r.inband).toFixed(0)}% | ${r.us.toFixed(1)} |\n`;
console.log(md);
writeFileSync(new URL('./results.md', import.meta.url), `# Benchmark results\n\nMean over seeds ${SEEDS.join(',')}; oracle autopilot, identical sensor streams for every filter.\n\n${md}`);
