// Closed-loop driving: the rover plans and steers from the pose its "brain" reports, not from the truth.
// The question: can a published SLAM method drive a rover on the Moon? It drives a fixed mission (4 worksites,
// then back to the lander) through the 3-D lab, and we measure how far off it really was at each stop.
//   node slam/drive_closed_loop.mjs --brain orb|wheels|truth [--seed 1] [--sun 7] [--slip 0.05] [--out slam/runs/closed/...]
// Brains:
//   truth  - steers by the true pose (best possible; shows what the planner alone achieves)
//   wheels - wheel odometry only (dead reckoning); slip is invisible to it
//   orb    - stereo ORB-SLAM3 (Campos et al., T-RO 2021), live in Docker (image: slam/orbslam3), one pose per frame;
//            between frames and while it is not tracking, the wheels carry the pose forward
// Needs the lab server (npm start). The rover knows where it starts (the lander), as every brain here does.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { homedir } from 'node:os';
import { readdirSync } from 'node:fs';
import { chromium } from 'playwright-core';
import { AE, SENSOR, shootInPage } from './lab_camera.mjs';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const brain = arg('brain', 'orb'), seed = +arg('seed', 1), sun = +arg('sun', 7), slip = +arg('slip', 0.05);
const fps = 5, W = 640, H = 360, baseline = 0.42, maxSeconds = +arg('seconds', 400);
const out = arg('out', `slam/runs/closed/${brain}_seed${seed}_sun${sun}`);
const render = brain !== 'wheels' && brain !== 'truth';
mkdirSync(join(out, 'frames'), { recursive: true });
mkdirSync(join(out, 'frames_right'), { recursive: true });

// ---- small 3-D pose maths (quaternions [x, y, z, w])
const qmul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const qrot = (q, v) => { const p = qmul(qmul(q, [...v, 0]), [-q[0], -q[1], -q[2], q[3]]); return [p[0], p[1], p[2]]; };
// ORB-SLAM3's camera axes (x right, y down, z forward) vs three.js's (x right, y up, looks down -z): a 180° turn about x.
const FLIP = [1, 0, 0, 0];
/** ORB-SLAM3 camera-to-world pose -> rover pose in the lab (x, y, heading), anchored at the true start camera. */
function orbToLab(anchor, p, q) {
  const qw = qmul(qmul(qmul(anchor.q, FLIP), q), FLIP);
  const t = qrot(anchor.q, qrot(FLIP, p));
  const pos = [anchor.p[0] + t[0], anchor.p[1] + t[1], anchor.p[2] + t[2]];
  const f = qrot(qw, [0, 0, -1]);   // where the three.js camera looks
  return { x: pos[0], y: -pos[2], h: Math.atan2(-f[2], f[0]) };   // three.js (x, up, -y) -> lab (x, y)
}

// ---- ORB-SLAM3, live
let orb = null;
const orbLines = [];
let orbWaiter = null;
if (brain === 'orb') {
  const abs = resolve(out);
  orb = spawn('docker', ['run', '--rm', '-i', '-v', `${abs}:/drive`, 'orbslam3', 'bash', '-c',
    `/live_settings.sh /tmp/s.yaml ${W} ${H} 100 ${baseline} ${fps} && cd /tmp && exec /orb/Examples_old/Stereo-Inertial/orb_live /orb/Vocabulary/ORBvoc.txt /tmp/s.yaml`]);
  const log = [];
  createInterface({ input: orb.stdout }).on('line', (l) => {
    if (l.startsWith('POSE ')) { if (orbWaiter) { const w = orbWaiter; orbWaiter = null; w(l); } else orbLines.push(l); } else log.push(l);
  });
  orb.stderr.on('data', (d) => log.push(String(d)));
  orb.on('exit', () => writeFileSync(join(out, 'orb_log.txt'), log.join('\n')));
}
const orbPose = (line) => new Promise((res) => { orb.stdin.write(line + '\n'); orbWaiter = res; });

// ---- the lab, headless
const cache = join(homedir(), 'Library/Caches/ms-playwright'), dir = readdirSync(cache).find((d) => d.startsWith('chromium_headless_shell'));
const executablePath = join(cache, dir, readdirSync(join(cache, dir)).find((d) => d.startsWith('chrome-')), 'chrome-headless-shell');
const browser = await chromium.launch({ executablePath, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto('http://localhost:8080/track.html?scenario=drive&record');
await page.waitForFunction(() => window.__lab && window.__lab.w3d, null, { timeout: 120000 });
await page.waitForTimeout(1500);

// Same world as the open-loop recordings (seed, sun), plus the mission and regolith slip.
const { plan: mission, boulders } = await page.evaluate(({ seed, sun, slip }) => {
  const L = window.__lab; L.S.seed = seed;
  document.getElementById('seed').value = seed; document.getElementById('seed').dispatchEvent(new Event('change'));
  const w = L.sess.world;
  w.sun.el = (sun * Math.PI) / 180; L.w3d.setWorld(w);
  const R = w.rover, home = { x: R.x, y: R.y };
  // worksites: fixed bearings and ranges from the lander, nudged off any rock; then home (a loop)
  const free = (p) => !w.boulders.some((b) => Math.hypot(b.x - p.x, b.y - p.y) < b.r + 3.5);
  const sites = [[16, 25], [30, 10], [28, -25], [14, -30]].map(([r, deg]) => {
    for (let k = 0; k < 60; k++) {
      const rr = r + (k % 6) * 1.5, a = ((deg + 90) * Math.PI) / 180 + (Math.floor(k / 6) - 5) * 0.06;
      const p = { x: home.x + rr * Math.cos(a), y: home.y + rr * Math.sin(a) };
      if (free(p)) return p;
    }
    return null;
  }).filter(Boolean);
  const plan = [...sites, home];
  // Yaw creep from slip: one sign per run (the same for every brain, so runs compare)
  w.roverNav = { belief: { x: R.x, y: R.y, h: R.h }, mission: plan, idx: 0, slip, yawBias: 0.02 * slip * (seed % 2 ? 1 : -1), arrivals: [], bumps: 0 };
  L.ready = true;
  return { plan, boulders: w.boulders.map((o) => ({ x: o.x, y: o.y, r: o.r, h: o.h })) };
}, { seed, sun, slip });
console.log(`brain=${brain} seed=${seed} sun=${sun}° slip=${slip}: ${mission.length} stops`, mission.map((p) => `(${p.x.toFixed(1)}, ${p.y.toFixed(1)})`).join(' '));

const steps = Math.round(1 / (fps * 0.1));
const shoot = (i, settleOnly) => page.evaluate(shootInPage, { steps, W, H, AE, SENSOR, seed, i, settleOnly, baseline, render });
const setBelief = (b) => page.evaluate((b) => Object.assign(window.__lab.sess.world.roverNav.belief, b), b);
const getBelief = () => page.evaluate(() => ({ ...window.__lab.sess.world.roverNav.belief }));

if (render) for (let k = 0; k < 15; k++) await shoot(-1, true);   // auto-exposure settles while parked
const truthRows = [], beliefRows = [], log = [];
let anchor = null, tracked = 0, lostRun = 0, offset = null, last = null, frames = 0;
const n = Math.round(maxSeconds * fps);
for (let i = 0; i < n; i++) {
  const r = await shoot(i, false);
  frames++;
  const belief = await getBelief();   // wheels have carried it forward since the last frame
  let state = brain;
  if (brain === 'truth') await setBelief({ x: r.rover.x, y: r.rover.y, h: r.rover.h });
  else if (brain === 'orb') {
    const name = String(i).padStart(6, '0');
    writeFileSync(join(out, 'frames', `${name}.png`), Buffer.from(r.png, 'base64'));
    writeFileSync(join(out, 'frames_right', `${name}.png`), Buffer.from(r.right, 'base64'));
    const v = (await orbPose(`${(i / fps).toFixed(6)} /drive/frames/${name}.png /drive/frames_right/${name}.png`)).split(' ').map(Number);
    const ok = v[2] === 2;
    if (!anchor) anchor = { p: r.pos, q: r.quat };   // the start pose: where the lander put the rover
    if (ok) {
      let pose = orbToLab(anchor, v.slice(3, 6), v.slice(6, 10));
      // After a long loss ORB-SLAM3 may start a new map with its own origin: re-anchor that map to where the wheels say we are.
      if (lostRun > 0 && Math.hypot(pose.x - belief.x, pose.y - belief.y) > 3) {
        offset = { dx: belief.x - pose.x, dy: belief.y - pose.y };
        log.push(`${(i / fps).toFixed(1)}s: ORB-SLAM3 map restarted, re-anchored to the wheel estimate`);
      }
      if (offset) pose = { ...pose, x: pose.x + offset.dx, y: pose.y + offset.dy };
      await setBelief(pose);
      tracked++;
      lostRun = 0;
      state = 'ok';
    } else { lostRun++; state = `lost(${v[2]})`; }
  }
  const b = await getBelief();
  truthRows.push([r.t.toFixed(2), r.rover.x.toFixed(3), r.rover.y.toFixed(3), r.rover.h.toFixed(4)].join(' '));
  beliefRows.push([r.t.toFixed(2), b.x.toFixed(3), b.y.toFixed(3), b.h.toFixed(4), state].join(' '));
  last = r;
  if (i % 50 === 0) process.stdout.write(`${(i / fps).toFixed(0)}s stop ${r.nav.idx}/${mission.length} err ${Math.hypot(b.x - r.rover.x, b.y - r.rover.y).toFixed(2)} m | `);
  if (r.nav.idx >= mission.length) break;
}
if (orb) orb.stdin.end();
await browser.close();

const err = (k) => { const t = truthRows[k].split(' ').map(Number), b = beliefRows[k].split(' ').map(Number); return Math.hypot(t[1] - b[1], t[2] - b[2]); };
const errs = truthRows.map((_, k) => err(k));
let dist = 0;
for (let k = 1; k < truthRows.length; k++) { const a = truthRows[k - 1].split(' ').map(Number), c = truthRows[k].split(' ').map(Number); dist += Math.hypot(c[1] - a[1], c[2] - a[2]); }
const result = {
  brain, seed, sunDeg: sun, slip, frames, seconds: +(frames / fps).toFixed(1), drivenM: +dist.toFixed(1),
  missionDone: last.nav.idx >= mission.length,
  stops: last.nav.arrivals.map((a) => ({ stop: a.idx + 1, atS: +a.t.toFixed(1), missM: +a.missM.toFixed(2) })),
  homeMissM: last.nav.idx >= mission.length ? +last.nav.arrivals.at(-1).missM.toFixed(2) : null,
  poseErrRmsM: +Math.sqrt(errs.reduce((s, e) => s + e * e, 0) / errs.length).toFixed(2),
  poseErrMaxM: +Math.max(...errs).toFixed(2),
  rockBumps: last.nav.bumps,   // 0.1 s steps the world's safety net pushed the rover back (it came within 0.4 m of a rock)
  ...(brain === 'orb' && { trackedFrames: tracked, trackedPct: +((100 * tracked) / frames).toFixed(1) }),
  notes: log,
};
writeFileSync(join(out, 'poses_truth_rover.txt'), '# t x y heading (lab frame, metres / rad)\n' + truthRows.join('\n') + '\n');
writeFileSync(join(out, 'poses_belief_rover.txt'), '# t x y heading state\n' + beliefRows.join('\n') + '\n');
writeFileSync(join(out, 'mission.json'), JSON.stringify(mission));
writeFileSync(join(out, 'boulders.json'), JSON.stringify(boulders));
writeFileSync(join(out, 'result.json'), JSON.stringify(result, null, 1));
console.log('\n' + JSON.stringify(result, null, 1));
