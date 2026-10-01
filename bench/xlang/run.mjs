// Cross-language equivalence + speed benchmark: JS vs C++ vs Python/numpy on one recorded rover trace.
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import os from 'node:os';
import { Mission } from '../../src/sim/mission.js';
import { EKF } from '../../src/filters/ekf.js';
import { roverModel } from '../../src/models/rover.js';
import { ROVER_SENSORS } from '../../src/sim/sensors.js';

const dir = new URL('.', import.meta.url).pathname, tmp = `${dir}.build`;
mkdirSync(tmp, { recursive: true });

// 1. record a deterministic sensor trace from the simulator
const m = new Mission({ platform: 'rover', scenario: 'dust', seed: 7, oracle: true, filters: ['ekf'] });
const P = m.platform, lines = [`X ${m.nominal.join(' ')}`], trace = [];
const step = P.step.bind(P), sense = P.sense.bind(P);
P.step = (mm, c, dt) => { const u = step(mm, c, dt); lines.push(`U ${u.v} ${u.w}`); trace.push({ t: 'U', u }); return u; };
P.sense = (mm, r) => { const out = sense(mm, r); for (const z of out) { lines.push(z.kind === 'landmark' ? `L ${z.lx} ${z.ly} ${z.range} ${z.bearing} ${z.sigR} ${z.sigB}` : `F ${z.x} ${z.y} ${z.sigma}`); trace.push({ t: 'Z', z }); } return out; };
m.issue({ type: 'plan' });
for (let i = 0; i < 20 * 150 && m.status === 'running'; i++) m.step();
const steps = trace.filter((r) => r.t === 'U').length;
writeFileSync(`${tmp}/trace.txt`, lines.join('\n'));

// 2. JS replay (the shipped filter class)
const REPS = 100;
const t0 = performance.now(); let f;
for (let r = 0; r < REPS; r++) {
  f = new EKF(roverModel, ROVER_SENSORS); f.init(m.nominal, roverModel.P0());
  for (const rec of trace) rec.t === 'U' ? f.predict(rec.u, 0.05) : f.update(rec.z);
}
const jsUs = ((performance.now() - t0) * 1000) / (steps * REPS), jsX = f.getState();

// 3. C++ and Python
execFileSync('clang++', ['-O2', '-std=c++17', '-o', `${tmp}/ekf_cpp`, `${dir}ekf.cpp`]);
const run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8' }).trim().split(/\s+/).map(Number);
const cpp = run(`${tmp}/ekf_cpp`, [`${tmp}/trace.txt`, `${REPS}`]);
const py = run('python3', [`${dir}ekf.py`, `${tmp}/trace.txt`, `${REPS / 5}`]);

const dev = (a) => Math.max(...[0, 1, 2, 3, 4].map((i) => Math.abs(a[i] - jsX[i])));
console.log(`trace: ${steps} predict steps, ${trace.filter((r) => r.t === 'Z').length} measurements\n`);
console.log('impl            final-state max|Δ| vs JS   µs / filter-step');
console.log(`JS (V8)         ${'—'.padEnd(26)} ${jsUs.toFixed(2)}`);
console.log(`C++17 -O2       ${dev(cpp).toExponential(2).padEnd(26)} ${cpp[5].toFixed(2)}`);
console.log(`Python + numpy  ${dev(py).toExponential(2).padEnd(26)} ${py[5].toFixed(2)}`);
console.log(`\nspeedups vs Python: C++ ${(py[5] / cpp[5]).toFixed(0)}×, JS ${(py[5] / jsUs).toFixed(0)}×;  C++ vs JS: ${(jsUs / cpp[5]).toFixed(1)}×`);
console.log(`\n20 Hz budget per step = 50000 µs. Headroom: C++ ${(50000 / cpp[5]).toFixed(0)}×, JS ${(50000 / jsUs).toFixed(0)}×, Python ${(50000 / py[5]).toFixed(0)}×`);

// 4. record provenance so the numbers can be cited
const ver = (cmd, args) => { try { return execFileSync(cmd, args, { encoding: 'utf8' }).split('\n')[0].trim(); } catch { return 'unknown'; } };
const nump = ver('python3', ['-c', 'import numpy; print(numpy.__version__)']);
const md = `# Cross-language EKF benchmark

One recorded sensor trace (rover, scenario \`dust\`, seed 7, oracle autopilot): ${steps} predict steps + ${trace.filter((r) => r.t === 'Z').length} measurements.
All three implementations replay the **same file** (\`.build/trace.txt\`), re-initialising the filter for each repetition, so the final state is directly comparable.

| impl | final-state max abs diff vs JS | µs per predict step (incl. amortised updates) | repetitions timed |
|---|---|---:|---:|
| JS (shipped \`src/filters/ekf.js\`) | – | ${jsUs.toFixed(2)} | ${REPS} |
| C++17 (\`ekf.cpp\`) | ${dev(cpp).toExponential(2)} | ${cpp[5].toFixed(3)} | ${REPS} |
| Python + numpy (\`ekf.py\`) | ${dev(py).toExponential(2)} | ${py[5].toFixed(2)} | ${REPS / 5} |

Machine: ${os.cpus()[0].model}, ${os.cpus().length} cores, ${os.platform()} ${os.release()}
Toolchain: node ${process.version}; ${ver('clang++', ['--version'])}; flags \`-O2 -std=c++17\`; ${ver('python3', ['--version'])}, numpy ${nump}

Caveats: single trace, single machine, single run (no variance reported); C++ uses fixed-size stack matrices, Python uses numpy on 5x5 arrays where call overhead dominates (a plain-Python or compiled-numpy comparison would differ); JS time includes the generic Filter/Mat abstraction of the shipped code. Python is timed over fewer repetitions for run time only. Timings are not pinned to a core and vary a few percent between runs.
`;
writeFileSync(`${dir}results.md`, md);
console.log('\nwrote bench/xlang/results.md');
