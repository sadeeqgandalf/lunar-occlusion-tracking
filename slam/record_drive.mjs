// Record the driving rover's camera for SLAM: PNG frames + the camera's TRUE pose per frame (TUM format), from the
// 3-D lab in a headless browser. Needs the lab server running (npm start in the repo root).
//   node slam/record_drive.mjs [--seed 3] [--seconds 120] [--fps 5] [--sun 7] [--size 640x360] [--baseline 0] [--out slam/runs/sun7]
// --baseline B > 0 also records a right camera B metres to the right (rectified stereo) into frames_right/.
// Poses: "t x y z qx qy qz qw", three.js world frame (y up), three.js camera convention (looks down -z, y up).
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { homedir } from 'node:os';
import { readdirSync } from 'node:fs';
import { AE, SENSOR, shootInPage } from './lab_camera.mjs';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const seed = +arg('seed', 3), seconds = +arg('seconds', 120), fps = +arg('fps', 5), sun = +arg('sun', 7);
const baseline = +arg('baseline', 0), [W, H] = arg('size', '640x360').split('x').map(Number), out = arg('out', `slam/runs/seed${seed}_sun${sun}`);
mkdirSync(join(out, 'frames'), { recursive: true }); if (baseline) mkdirSync(join(out, 'frames_right'), { recursive: true });

// macOS: reuse the headless Chromium Playwright already cached; Linux: `npx playwright-core install chromium` first
let executablePath;
if (process.platform === 'darwin') {
  const cache = join(homedir(), 'Library/Caches/ms-playwright'), dir = readdirSync(cache).find((d) => d.startsWith('chromium_headless_shell'));
  executablePath = join(cache, dir, readdirSync(join(cache, dir)).find((d) => d.startsWith('chrome-')), 'chrome-headless-shell');
}
const browser = await chromium.launch({ executablePath, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(`http://localhost:8080/track.html?scenario=drive&record`);
await page.waitForFunction(() => window.__lab && window.__lab.w3d, null, { timeout: 120000 });
await page.waitForTimeout(1500);   // let the lab lay out and size its 3-D canvas, then stop its own redraws

// fresh session with the chosen seed and sun elevation (the 3-D view lights the scene from world.sun)
await page.evaluate(({ seed, sun }) => {
  const L = window.__lab; L.S.seed = seed;
  document.getElementById('seed').value = seed; document.getElementById('seed').dispatchEvent(new Event('change'));
  L.sess.world.sun.el = (sun * Math.PI) / 180; L.w3d.setWorld(L.sess.world);
  L.ready = true;
}, { seed, sun });

const steps = Math.round(1 / (fps * 0.1)), n = Math.round(seconds * fps), poses = [], exposures = [];
const shoot = (i, settleOnly) => page.evaluate(shootInPage, { steps, W, H, AE, SENSOR, seed, i, settleOnly, baseline });
for (let k = 0; k < 15; k++) await shoot(-1, true);   // let auto-exposure settle before the first recorded frame
for (let i = 0; i < n; i++) {
  const r = await shoot(i, false);
  if (!r) throw new Error(`canvas smaller than ${W}x${H}`);
  writeFileSync(join(out, 'frames', `${String(i).padStart(6, '0')}.png`), Buffer.from(r.png, 'base64'));
  if (r.right) writeFileSync(join(out, 'frames_right', `${String(i).padStart(6, '0')}.png`), Buffer.from(r.right, 'base64'));
  poses.push([r.t.toFixed(3), ...r.pos.map((v) => v.toFixed(5)), ...r.quat.map((v) => v.toFixed(6))].join(' '));
  exposures.push(r.exposure / AE.base);
  if (i % 50 === 0) process.stdout.write(`${i}/${n} `);
}
writeFileSync(join(out, 'poses_truth.txt'), poses.join('\n') + '\n');
const ex = exposures.slice().sort((a, b) => a - b);
writeFileSync(join(out, 'meta.json'), JSON.stringify({ seed, seconds, fps, sunElevationDeg: sun, width: W, height: H, hfovDeg: 100, ...(baseline && { stereoBaselineM: baseline }), scenario: 'drive',
  poseFrame: 'three.js world (y up), camera looks down -z', photometry: 'Lunar-Lambert (McEwen) on ground and rocks',
  camera: { autoExposure: AE, sensor: SENSOR, exposureGainMedian: ex[ex.length >> 1], exposureGainMax: ex[ex.length - 1] } }, null, 1));
await browser.close();
console.log(`\nwrote ${n} frames and poses to ${out}`);
