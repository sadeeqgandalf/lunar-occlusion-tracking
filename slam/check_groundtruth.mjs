// Ground-truth check: project every boulder with ONLY the recorded pose (poses_truth.txt) and a pinhole camera built
// from meta.json (independent of the renderer's camera code), draw crosses on the recorded frames, and report how
// far each projected rock centre falls from the rock's dark/bright blob is left to the eye: if the crosses sit on the
// rocks in every frame, the recorded poses match the images.
//   node slam/check_groundtruth.mjs RUN_DIR [frame frame ...]     -> RUN_DIR/check_<frame>.png
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { homedir } from 'node:os';
import { readdirSync } from 'node:fs';
import { MotWorld } from '../src/mot/world.js';

const run = process.argv[2], frames = process.argv.slice(3).map(Number);
const meta = JSON.parse(readFileSync(join(run, 'meta.json'), 'utf8'));
const poses = readFileSync(join(run, 'poses_truth.txt'), 'utf8').trim().split('\n').map((l) => l.split(' ').map(Number));
const W = meta.width, H = meta.height, f = W / 2 / Math.tan((meta.hfovDeg * Math.PI) / 360);   // square pixels: fx = fy
const world = new MotWorld(meta.scenario, meta.seed);
// three.js coordinates: sim (x, y) -> (x, up, -y); rock = half-buried ellipsoid of height b.h, so mark 40% and 90% up
const pts = world.boulders.flatMap((b) => [[b.x, 0.4 * b.h, -b.y], [b.x, 0.9 * b.h, -b.y]]);

function project([, px, py, pz, qx, qy, qz, qw], [X, Y, Z]) {
  // rotate the world point into the camera frame with the inverse (conjugate) quaternion
  const vx = X - px, vy = Y - py, vz = Z - pz, x = -qx, y = -qy, z = -qz, w = qw;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  const cx = vx + w * tx + (y * tz - z * ty), cy = vy + w * ty + (z * tx - x * tz), cz = vz + w * tz + (x * ty - y * tx);
  if (cz > -0.5) return null;                                   // behind the camera (three.js cameras look down -z)
  return [W / 2 + (f * cx) / -cz, H / 2 - (f * cy) / -cz];
}

let executablePath;
if (process.platform === 'darwin') {
  const cache = join(homedir(), 'Library/Caches/ms-playwright'), dir = readdirSync(cache).find((d) => d.startsWith('chromium_headless_shell'));
  executablePath = join(cache, dir, readdirSync(join(cache, dir)).find((d) => d.startsWith('chrome-')), 'chrome-headless-shell');
}
const browser = await chromium.launch({ executablePath }), page = await browser.newPage();
for (const i of frames) {
  const uv = pts.map((p) => project(poses[i], p)).filter((q) => q && q[0] > -20 && q[0] < W + 20 && q[1] > -20 && q[1] < H + 20);
  const img = readFileSync(join(run, 'frames', `${String(i).padStart(6, '0')}.png`)).toString('base64');
  const png = await page.evaluate(async ({ img, uv, W, H }) => {
    const im = new Image(); im.src = 'data:image/png;base64,' + img; await im.decode();
    const c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d'); g.drawImage(im, 0, 0);
    g.strokeStyle = '#ff2d55'; g.lineWidth = 2;
    for (const [u, v] of uv) { g.beginPath(); g.moveTo(u - 7, v); g.lineTo(u + 7, v); g.moveTo(u, v - 7); g.lineTo(u, v + 7); g.stroke(); }
    return c.toDataURL('image/png').split(',')[1];
  }, { img, uv, W, H });
  writeFileSync(join(run, `check_${i}.png`), Buffer.from(png, 'base64'));
  console.log(`frame ${i}: ${uv.length / 1} rock marks drawn`);
}
await browser.close();
