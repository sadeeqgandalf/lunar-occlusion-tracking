// Occlusion Tracking Lab: bird's-eye scene with blind-spot wedges, a rendered camera view, three trackers
// compared live on the same detections, and a plain-language event feed.
import { MotSession } from '../mot/session.js';
import { MOT_SCENARIOS } from '../mot/world.js';
import { wrapAngle, TAU } from '../core/linalg.js';
import { isHidden, hiddenCause } from '../mot/occlusion.js';
import { runTour } from './tour.js';
import { lineChart } from './charts.js';

const $ = (id) => document.getElementById(id);
const S = { scenario: 'boulders', seed: 7, speed: 2, paused: false, sel: 0, showTruth: true, view: '3d', overrides: {}, sw: [] };
let w3d = null, w3dLost = null; // w3dLost: a 3-D view waiting for the browser to restore its GPU context
const COLORS = ['#4ade80', '#4cc9f0', '#ff6b6b'];
const PLAIN = {
  'Aware + neg. info': 'Knows the blind spots AND reasons "I can\'t see them, so they must be behind that rock".',
  'Occlusion-aware': 'Knows where the camera is blind, so it waits for people who walk behind rocks.',
  Naive: 'Thinks it can always see everyone. If someone vanishes for a moment, it forgets them.',
};
const BLURB = {
  'Aware + neg. info': 'Uses the missed detection as information about where the person is (inside the shadow).',
  'Occlusion-aware': 'Expected detection probability = P_D × visibility, so a miss behind a rock is not evidence they left.',
  Naive: 'Every miss counts as evidence the person is gone. After a few frames the track is deleted.',
};
let sess, feed = [], pending = new Map(), seenEv = [], prevHidden = new Map(), marks = [];

function newSession() {
  sess = new MotSession({ scenario: S.scenario, seed: S.seed, overrides: S.overrides });
  S.sw = sess.runs.map(() => []);
  feed = []; pending = new Map(); seenEv = sess.runs.map(() => 0); prevHidden = new Map(); marks = [];
  pushFeed(`${MOT_SCENARIOS[S.scenario].label}: ${MOT_SCENARIOS[S.scenario].blurb}`, '');
  buildCards();
}

// ------------------------------------------------------------------ feed
function pushFeed(text, cls) { feed.unshift({ t: sess.world.t, text, cls }); if (feed.length > 60) feed.pop(); }
function updateFeed() {
  const W = sess.world;
  for (const g of W.targets) {
    const hid = isHidden(g.vis), was = prevHidden.get(g.id);
    if (hid && was === false) pushFeed(hiddenCause(g.vis) === 'shadow' ? `Crew ${g.id} walked into deep shadow (in line of sight, but too dark to detect)` : `Crew ${g.id} went behind a boulder`, 'hide');
    prevHidden.set(g.id, hid);
  }
  sess.runs.forEach((r, i) => {
    const ev = r.metrics.events;
    for (; seenEv[i] < ev.length; seenEv[i]++) {
      const e = ev[seenEv[i]], key = e.target;
      const g = W.targets.find((x) => x.id === e.target);
      if (g) marks.push({ run: i, x: g.x, y: g.y, t: W.t, kept: e.kept });
      const p = pending.get(key) || { t: W.t, dur: e.dur, res: {} };
      p.res[r.tracker.name] = e.kept; pending.set(key, p);
    }
  });
  for (const [key, p] of pending) {
    if (Object.keys(p.res).length < sess.runs.length && W.t - p.t < 2.5) continue;
    const parts = sess.runs.map((r) => { const k = p.res[r.tracker.name]; return `${short(r.tracker.name)} ${k === undefined ? '…' : k ? '✔ same ID' : '✘ new ID'}`; });
    const anyKept = Object.values(p.res).some(Boolean);
    pushFeed(`Crew ${key} back after ${p.dur.toFixed(1)} s · ${parts.join(' · ')}`, anyKept ? 'ok' : 'bad');
    pending.delete(key);
  }
}
const short = (n) => ({ 'Aware + neg. info': 'Aware+NI', 'Occlusion-aware': 'Aware', Naive: 'Naive' }[n] || n);

// ------------------------------------------------------------------ scoreboard
function buildCards() {
  $('cards').innerHTML = sess.runs.map((r, i) => `<div class="tcard ${i === S.sel ? 'sel' : ''}" data-i="${i}">
    <div class="hd"><span class="nm" style="color:${COLORS[i]}"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:${COLORS[i]};margin-right:7px"></span>${r.tracker.name}</span><span class="sub">${i === S.sel ? '◉ shown on map' : 'click to show'}</span></div>
    <div class="plain">${PLAIN[r.tracker.name] || ''}</div>
    <p>${BLURB[r.tracker.name] || ''}</p>
    <div class="big" id="kept${i}">–</div><div class="sub" id="keptn${i}">ID kept through 1–5 s occlusions</div>
    <div class="bar"><i id="bar${i}" style="width:0;background:${COLORS[i]}"></i></div>
    <div class="kv"><span>ID switches<b id="idsw${i}">–</b></span><span>IDF1<b id="idf1${i}">–</b></span><span>MOTA<b id="mota${i}">–</b></span><span>GOSPA<b id="gospa${i}">–</b></span></div>
  </div>`).join('');
}
function updateCards() {
  sess.runs.forEach((r, i) => {
    const s = r.metrics.summary(), b = s.occByDuration.slice(0, 2), k = b.reduce((a, x) => a + x.kept, 0), n = b.reduce((a, x) => a + x.n, 0);
    $('kept' + i).textContent = n ? Math.round((100 * k) / n) + '%' : '–';
    $('keptn' + i).textContent = `ID kept through 1–5 s occlusions (${k}/${n})`;
    $('bar' + i).style.width = (n ? (100 * k) / n : 0) + '%';
    $('idsw' + i).textContent = s.idsw;
    $('idf1' + i).textContent = Number.isFinite(s.idf1) ? s.idf1.toFixed(2) : '–';
    $('mota' + i).textContent = Number.isFinite(s.mota) ? s.mota.toFixed(2) : '–';
    $('gospa' + i).textContent = s.gospa.toFixed(2);
  });
  $('feed').innerHTML = feed.map((f) => `<div class="${f.cls}"><b>${f.t.toFixed(0).padStart(4)}s</b> ${f.text}</div>`).join('');
  $('clock').textContent = `T+${sess.world.t.toFixed(1).padStart(5, '0')} s`;
  lineChart($('chSw'), { title: 'cumulative ID switches (green / blue / red trackers)', tMin: 0, tMax: Math.max(10, sess.world.t), series: S.sw.map((pts, i) => ({ color: COLORS[i], width: i === S.sel ? 2.2 : 1.4, pts })) });
  const hiddenNow = sess.world.targets.filter((t) => isHidden(t.vis)).length, col = COLORS[S.sel];
  $('now').innerHTML = `Watching: <span class="sw" style="background:${col}"></span><b style="color:${col}">${sess.runs[S.sel].tracker.name}</b><br>${hiddenNow} of ${sess.world.targets.length} people hidden right now (behind rocks${sess.world.cfg.shadows ? ' or in shadow' : ''})`;
  document.documentElement.style.setProperty('--trk', col);
  document.querySelectorAll('#key .kname').forEach((el) => { el.style.color = col; });
  $('camTrk').style.color = col;
}

// ------------------------------------------------------------------ canvases
function fit(cv) {
  const dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, W: r.width, H: r.height };
}

// ---- glyphs: draw things as what they are
const frac = (v) => v - Math.floor(v);
function rockShape(b, i) { // deterministic irregular outline, 14 vertices
  if (!b.shape) b.shape = Array.from({ length: 14 }, (_, k) => 0.82 + 0.3 * frac(Math.sin(i * 12.9898 + k * 78.233) * 43758.5453));
  return b.shape;
}
function drawRock(c, px, py, r, shape) {
  c.beginPath();
  shape.forEach((f, k) => { const a = (k / shape.length) * TAU, x = px + Math.cos(a) * r * f, y = py + Math.sin(a) * r * f; k ? c.lineTo(x, y) : c.moveTo(x, y); });
  c.closePath();
  const g = c.createRadialGradient(px - r * 0.35, py - r * 0.35, r * 0.1, px, py, r * 1.1); g.addColorStop(0, '#8c9099'); g.addColorStop(1, '#3a3d44');
  c.fillStyle = g; c.fill(); c.strokeStyle = '#25272c'; c.lineWidth = 1.5; c.stroke();
}
function drawAstronaut(c, px, py, r, heading, alpha) {
  c.save(); c.globalAlpha = alpha; c.translate(px, py); c.rotate(-heading);
  c.fillStyle = '#cfd2d6'; c.beginPath(); c.roundRect(-r * 1.25, -r * 0.7, r * 0.7, r * 1.4, r * 0.25); c.fill(); // backpack
  c.fillStyle = '#ffffff'; c.beginPath(); c.arc(0, 0, r, 0, TAU); c.fill();                                      // suit/helmet
  c.strokeStyle = '#d4a017'; c.lineWidth = r * 0.55; c.beginPath(); c.arc(0, 0, r * 0.62, -0.9, 0.9); c.stroke(); // gold visor = facing
  c.restore();
}
function tag(c, x, y, text, col, dashed) {
  c.font = 'bold 12px ui-monospace,monospace'; const w = c.measureText(text).width + 12;
  c.beginPath(); c.roundRect(x, y - 15, w, 18, 9);
  if (dashed) { c.fillStyle = '#0a0e16dd'; c.fill(); c.setLineDash([3, 3]); c.strokeStyle = col; c.lineWidth = 1.5; c.stroke(); c.setLineDash([]); c.fillStyle = col; }
  else { c.fillStyle = col; c.fill(); c.fillStyle = '#06101a'; }
  c.fillText(text, x + 6, y - 2);
}

function drawMap() {
  const { c, W, H } = fit($('map')), w = sess.world, cam = w.cam, col = COLORS[S.sel];
  if (W < 120 || H < 120) return; // hidden or too small to draw into (would give negative scales)
  const xmin = -cam.range * Math.sin(cam.fov / 2) - 2, xmax = -xmin, ymin = cam.y - 5, ymax = cam.y + cam.range + 1, pad = 18, top = 54;
  const s = Math.min((W - 2 * pad) / (xmax - xmin), (H - top - pad) / (ymax - ymin));
  const ox = W / 2 - ((xmin + xmax) / 2) * s, oy = top + (H - top - pad) / 2 + ((ymin + ymax) / 2) * s;
  const P = (x, y) => [ox + x * s, oy - y * s];
  c.fillStyle = '#0d0f13'; c.fillRect(0, 0, W, H);
  // what the camera can see: lit lunar ground inside the field of view
  const [cx, cy] = P(cam.x, cam.y), a0 = cam.th - cam.fov / 2, a1 = cam.th + cam.fov / 2;
  c.save(); c.beginPath(); c.moveTo(cx, cy); c.arc(cx, cy, cam.range * s, -a1, -a0); c.closePath();
  const g = c.createRadialGradient(cx, cy, 0, cx, cy, cam.range * s); g.addColorStop(0, '#5a5d63'); g.addColorStop(1, '#3a3d43');
  c.fillStyle = g; c.fill(); c.clip();
  for (let k = 0; k < 140; k++) { const x = xmin + frac(Math.sin(k * 3.17) * 999) * (xmax - xmin), y = ymin + frac(Math.sin(k * 7.31) * 999) * (ymax - ymin), [px, py] = P(x, y); c.fillStyle = '#ffffff0d'; c.beginPath(); c.arc(px, py, 1 + frac(k * 0.37) * 3, 0, TAU); c.fill(); } // regolith speckle
  // blind zones behind each boulder
  let biggest = null;
  for (const b of w.boulders) {
    const dx = b.x - cam.x, dy = b.y - cam.y, d = Math.hypot(dx, dy), half = Math.asin(Math.min(1, b.r / d)), cb = Math.atan2(dy, dx), tl = d * Math.cos(half);
    c.beginPath();
    const p0 = P(cam.x + tl * Math.cos(cb - half), cam.y + tl * Math.sin(cb - half)), p1 = P(cam.x + tl * Math.cos(cb + half), cam.y + tl * Math.sin(cb + half));
    c.moveTo(...p0); c.arc(cx, cy, cam.range * 1.2 * s, -(cb - half), -(cb + half), true); c.lineTo(...p1); c.closePath();
    c.fillStyle = b.h !== undefined && b.h < 2.2 ? 'rgba(5,6,10,0.32)' : 'rgba(5,6,10,0.72)'; c.fill(); // low rocks only hide legs
    if (!biggest || half > biggest.half) biggest = { half, cb, d };
  }
  if (biggest) { const rr = Math.min(cam.range - 4, biggest.d + 9), [lx, ly] = P(cam.x + rr * Math.cos(biggest.cb), cam.y + rr * Math.sin(biggest.cb)); c.fillStyle = '#9aa3b2'; c.font = 'italic 11px system-ui'; c.textAlign = 'center'; c.fillText('blind zone', lx, ly); c.fillText('(camera can\'t see here)', lx, ly + 13); c.textAlign = 'left'; }
  c.restore();
  // boulders
  w.boulders.forEach((b, i) => { const [px, py] = P(b.x, b.y); drawRock(c, px, py, b.r * s, rockShape(b, i)); });
  // rover with camera mast
  c.save(); c.translate(cx, cy + 10); c.fillStyle = '#c8ccd2'; c.fillRect(-14, -7, 28, 14); c.fillStyle = '#2b2d33'; for (const wx of [-12, 0, 12]) for (const wy of [-9, 9]) { c.beginPath(); c.arc(wx, wy, 3, 0, TAU); c.fill(); } c.restore();
  c.fillStyle = '#4cc9f0'; c.beginPath(); c.arc(cx, cy, 4, 0, TAU); c.fill();
  c.fillStyle = '#e6edf7'; c.font = '12px system-ui'; c.fillText('Rover camera (you)', cx + 20, cy + 14);
  // true people
  const r = Math.max(7, 0.5 * s);
  if (S.showTruth) for (const t of w.targets) {
    const [px, py] = P(t.x, t.y), hid = !t.vis.inFov || isHidden(t.vis);
    if (t.trail.length > 1) { c.strokeStyle = '#ffffff26'; c.lineWidth = 1.5; c.setLineDash([2, 4]); c.beginPath(); t.trail.forEach(([x, y], i) => { const q = P(x, y); i ? c.lineTo(...q) : c.moveTo(...q); }); c.stroke(); c.setLineDash([]); }
    drawAstronaut(c, px, py, r, t.h, hid ? 0.45 : 1);
    c.fillStyle = hid ? '#ffffff99' : '#ffffffdd'; c.font = '11px system-ui'; c.fillText(hid ? `person ${t.id} · ${hiddenCause(t.vis) === 'shadow' ? 'in shadow' : 'behind rock'}` : `person ${t.id}`, px - r, py + r + 13);
  }
  // camera pings this frame (orange); in training view, false alarms are tagged
  sess.last.dets.forEach((d, i) => {
    const [px, py] = P(cam.x + d.range * Math.cos(d.bearing), cam.y + d.range * Math.sin(d.bearing)), fa = sess.last.gt[i] === 0;
    c.strokeStyle = fa ? '#ffb45499' : '#ffb454'; c.lineWidth = 2;
    c.beginPath(); c.arc(px, py, 5, 0, TAU); c.moveTo(px - 9, py); c.lineTo(px - 4, py); c.moveTo(px + 4, py); c.lineTo(px + 9, py); c.moveTo(px, py - 9); c.lineTo(px, py - 4); c.moveTo(px, py + 4); c.lineTo(px, py + 9); c.stroke();
    if (fa && S.showTruth) { c.fillStyle = '#9aa3b2'; c.font = '10px system-ui'; c.fillText('false alarm', px + 10, py + 4); }
  });
  // the tracker's beliefs: name tag when seen, dashed search area when hidden
  const T = sess.runs[S.sel].tracker;
  for (const t of T.tracks) {
    if (!t.confirmed) continue;
    const [px, py] = P(t.x[0], t.x[1]), lost = t.lastSeen > 3;
    if (lost) {
      const a = t.P.get(0, 0), b = t.P.get(0, 1), d2 = t.P.get(1, 1), m = (a + d2) / 2, q = Math.sqrt(((a - d2) / 2) ** 2 + b * b), k = Math.sqrt(5.991);
      c.save(); c.translate(px, py); c.rotate(-0.5 * Math.atan2(2 * b, a - d2));
      c.beginPath(); c.ellipse(0, 0, Math.max(r + 4, k * Math.sqrt(m + q) * s), Math.max(r + 4, k * Math.sqrt(Math.max(m - q, 0)) * s), 0, 0, TAU);
      c.fillStyle = col + '2e'; c.fill(); c.setLineDash([6, 4]); c.strokeStyle = col; c.lineWidth = 2; c.stroke(); c.setLineDash([]); c.restore();
      c.fillStyle = col; c.font = 'bold 14px system-ui'; c.textAlign = 'center'; c.fillText('?', px, py + 5); c.textAlign = 'left';
      tag(c, px + r + 4, py - r, `#${t.id} lost sight · searching`, col, true);
    } else {
      c.strokeStyle = col; c.lineWidth = 3; c.beginPath(); c.arc(px, py, r + 5, 0, TAU); c.stroke();
      tag(c, px + r + 6, py - r - 2, `#${t.id}`, col, false);
    }
  }
  // reappearance verdicts (fade after 4 s)
  marks = marks.filter((m) => w.t - m.t < 4);
  for (const m of marks) {
    if (m.run !== S.sel) continue;
    const [px, py] = P(m.x, m.y), al = Math.max(1 - (w.t - m.t) / 4, 0.2);
    c.globalAlpha = al; c.font = 'bold 13px system-ui'; const txt = m.kept ? '✔ same ID kept' : '✘ lost them: new ID', tw = c.measureText(txt).width + 14;
    c.fillStyle = m.kept ? '#14532d' : '#4c1219'; c.beginPath(); c.roundRect(px - tw / 2, py - r - 36, tw, 22, 11); c.fill();
    c.fillStyle = m.kept ? '#4ade80' : '#ff8a8a'; c.textAlign = 'center'; c.fillText(txt, px, py - r - 20); c.textAlign = 'left'; c.globalAlpha = 1;
  }
  const bar = 10 * s; c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(W - 20 - bar, H - 12); c.lineTo(W - 20, H - 12); c.stroke(); c.fillStyle = '#fff'; c.font = '11px system-ui'; c.fillText('10 m', W - 20 - bar, H - 17);
}

function drawCam() {
  const { c, W, H } = fit($('cam')), w = sess.world, cam = w.cam, col = COLORS[S.sel];
  if (W < 60 || H < 40) return;
  const f = W / 2 / Math.tan(cam.fov / 2), fv = H * 1.7, y0 = H * 0.24, camH = 2.2;
  const U = (brg) => W / 2 - f * Math.tan(wrapAngle(brg - cam.th));
  const ground = (d) => y0 + (fv * camH) / Math.max(d, 1);
  c.fillStyle = '#000'; c.fillRect(0, 0, W, y0);
  for (let k = 0; k < 90; k++) { c.fillStyle = `rgba(255,255,255,${0.3 + frac(k * 0.71) * 0.6})`; c.fillRect(frac(Math.sin(k * 5.1) * 999) * W, frac(Math.sin(k * 9.7) * 999) * y0, 1.5, 1.5); }
  const gr = c.createLinearGradient(0, y0, 0, H); gr.addColorStop(0, '#45474d'); gr.addColorStop(1, '#8a8c92'); c.fillStyle = gr; c.fillRect(0, y0, W, H - y0);
  const items = [];
  w.boulders.forEach((b, i) => items.push({ d: Math.hypot(b.x - cam.x, b.y - cam.y), kind: 'rock', b, i }));
  for (const t of w.targets) if (t.vis.inFov) items.push({ d: t.vis.range, kind: 'person', t });
  items.sort((a, b) => b.d - a.d); // far to near: near things hide far things, exactly like a real camera
  for (const it of items) {
    if (it.kind === 'rock') {
      const b = it.b, half = Math.asin(Math.min(1, b.r / it.d)), cb = Math.atan2(b.y - cam.y, b.x - cam.x), ul = U(cb + half), ur = U(cb - half), yb = ground(it.d - b.r * 0.5), h = (fv * b.r * 1.15) / it.d;
      const g = c.createLinearGradient(0, yb - h, 0, yb); g.addColorStop(0, '#9a9da5'); g.addColorStop(1, '#3b3e45');
      c.fillStyle = g; c.beginPath(); c.moveTo(ul, yb); c.bezierCurveTo(ul, yb - h * 0.8, ul + (ur - ul) * 0.3, yb - h * 1.05, (ul + ur) / 2, yb - h); c.bezierCurveTo(ur - (ur - ul) * 0.25, yb - h * 0.95, ur, yb - h * 0.6, ur, yb); c.closePath(); c.fill();
    } else {
      const t = it.t, u = U(t.vis.bearing), yb = ground(it.d), h = (fv * 1.8) / it.d, wpx = Math.max(4, (f * 0.75) / it.d);
      c.fillStyle = '#d0d3d8'; c.fillRect(u - wpx * 0.6, yb - h * 0.8, wpx * 0.25, h * 0.4);       // backpack
      c.fillStyle = '#f2f2f2'; c.fillRect(u - wpx / 2, yb - h * 0.82, wpx, h * 0.82);              // suit
      c.beginPath(); c.arc(u, yb - h * 0.88, wpx * 0.55, 0, TAU); c.fill();                          // helmet
      c.fillStyle = '#d4a017'; c.beginPath(); c.arc(u, yb - h * 0.88, wpx * 0.38, 0, TAU); c.fill(); // gold visor
    }
  }
  sess.last.dets.forEach((d, i) => {
    const u = U(d.bearing), yb = ground(d.range), h = (fv * 1.8) / d.range, wpx = Math.max(8, (f * 1.0) / d.range);
    c.strokeStyle = sess.last.gt[i] === 0 ? '#ffb45488' : '#ffb454'; c.lineWidth = 1.5; c.strokeRect(u - wpx / 2, yb - h * 1.05, wpx, h * 1.1);
    if (sess.last.gt[i] === 0 && S.showTruth) { c.fillStyle = '#c9ced8'; c.font = '10px system-ui'; c.fillText('false alarm', u - wpx / 2, yb + 12); }
  });
  const T = sess.runs[S.sel].tracker;
  for (const t of T.tracks) {
    if (!t.confirmed) continue;
    const dx = t.x[0] - cam.x, dy = t.x[1] - cam.y, d = Math.hypot(dx, dy), brg = Math.atan2(dy, dx);
    if (Math.abs(wrapAngle(brg - cam.th)) > cam.fov / 2 || d < 1) continue;
    const u = U(brg), yb = ground(d), h = (fv * 1.8) / d, wpx = Math.max(12, (f * 1.3) / d), lost = t.lastSeen > 3;
    c.strokeStyle = col; c.lineWidth = 2.5; c.setLineDash(lost ? [5, 4] : []); c.strokeRect(u - wpx / 2, yb - h * 1.1 - 2, wpx, h * 1.2 + 4); c.setLineDash([]);
    tag(c, u - wpx / 2, yb - h * 1.1 - 6, lost ? `#${t.id} behind rock?` : `#${t.id}`, col, lost);
  }
}

// ------------------------------------------------------------------ loop & wiring
let acc = 0, last = performance.now(), uiT = 0;

/** Show a visible, copyable message instead of failing silently. */
function showError(where, err) {
  console.error(`[${where}]`, err);
  let el = document.getElementById('err3d');
  if (!el) {
    el = document.createElement('div'); el.id = 'err3d';
    el.style.cssText = 'position:absolute;left:50%;top:60px;transform:translateX(-50%);z-index:20;max-width:80%;pointer-events:auto;background:#4c1219;color:#ffd7db;border:1px solid #ff5d6c;border-radius:8px;padding:8px 12px;font:12px/1.4 ui-monospace,monospace;user-select:text';
    (document.getElementById('mapwrap') || document.body).appendChild(el);
  }
  const msg = err?.message || String(err), lost = /context lost/i.test(msg), low = /low-power 3D/.test(msg);
  el.innerHTML = '';
  el.append(`3-D view paused, showing the 2-D map. ${lost && !low ? 'Your browser ran short of graphics memory. ' : ''}Details: ${msg}`);
  if (lost && !low) {
    const b = document.createElement('button'); b.textContent = '↻ Retry 3D in low-power mode';
    b.style.cssText = 'margin-left:10px;padding:3px 10px;background:#0d3a52;color:#cfe;border:1px solid #4cc9f0;border-radius:6px;cursor:pointer';
    b.onclick = () => { try { localStorage.setItem('gfx', 'low'); } catch {} location.reload(); }; el.append(b);
  } else if (lost) el.append(' Please send this line to get it fixed.');
}
function frame(now) {
  requestAnimationFrame(frame); // schedule first: one bad frame must never stop the loop
  try { frameBody(now); } catch (err) { showError('frame', err); }
}
function frameBody(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (!S.paused) { acc += dt * S.speed; let n = 0; while (acc >= sess.world.dt && n < 200) {
    sess.step(); updateFeed(); acc -= sess.world.dt; n++;
    if (sess.world.frame % 10 === 0) sess.runs.forEach((r, i) => S.sw[i].push([sess.world.t, r.metrics.idsw])); // 1 Hz series
  } if (n === 200) acc = 0; }
  const use3d = S.view === '3d' && w3d;
  for (const id of ['map', 'cam']) $(id).style.display = use3d ? 'none' : 'block';
  for (const id of ['glAll', 'ov3d', 'ovcam', 'hint3d']) $(id).style.display = use3d ? 'block' : 'none';
  if (use3d) { const mw = $('mapwrap'), cw = $('camwrap'), g = $('glAll'); g.style.top = mw.offsetTop + 'px'; g.style.height = (cw.offsetTop + cw.offsetHeight - mw.offsetTop) + 'px'; }
  if (w3dLost && !w3dLost.isLost()) { // the browser gave the GPU context back: return to 3-D
    w3d = w3dLost; w3dLost = null; S.view = '3d'; [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b.dataset.v === '3d'));
    $('err3d')?.remove(); pushFeed('3-D view restored after a GPU reset', '');
  }
  if (use3d) {
    marks = marks.filter((m) => sess.world.t - m.t < 4);
    try {
      w3d.render(sess, { sel: S.sel, color: COLORS[S.sel], showTruth: S.showTruth, marks: marks.filter((m) => m.run === S.sel).map((m) => ({ ...m, alpha: Math.max(1 - (sess.world.t - m.t) / 4, 0.2) })) });
    } catch (err) { // fall back to the 2-D views and say why
      showError('3-D view', err); if (w3d.isLost?.()) w3dLost = w3d; w3d = null; S.view = '2d';
      [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b.dataset.v === '2d'));
      // the 2-D canvases become visible on the next frame and are drawn then
    }
  } else { drawMap(); drawCam(); }
  if (now - uiT > 250) { uiT = now; updateCards(); }
}

function wire() {
  $('scenario').innerHTML = Object.entries(MOT_SCENARIOS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  $('scenario').value = S.scenario;
  $('speed').innerHTML = [1, 2, 4, 8].map((v) => `<button data-v="${v}" class="${v === S.speed ? 'on' : ''}">${v}×</button>`).join('');
  $('scenario').onchange = (e) => { S.scenario = e.target.value; newSession(); };
  $('seed').onchange = (e) => { S.seed = Math.max(1, +e.target.value || 1); newSession(); };
  $('reset').onclick = newSession;
  $('play').onclick = () => { S.paused = !S.paused; $('play').textContent = S.paused ? '▶ Play' : '⏸ Pause'; };
  $('speed').onclick = (e) => { const v = e.target.dataset.v; if (!v) return; S.speed = +v; [...$('speed').children].forEach((b) => b.classList.toggle('on', b === e.target)); };
  $('cards').onclick = (e) => { const el = e.target.closest('.tcard'); if (!el) return; S.sel = +el.dataset.i; buildCards(); updateCards(); };
  $('optTruth').onchange = (e) => { S.showTruth = e.target.checked; };
  $('viewsw').onclick = (e) => { const v = e.target.dataset.v; if (!v) return; S.view = v; [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b === e.target)); };
  // ---- what-if panel: sliders show live values; Apply restarts the same scenario/seed with the overrides
  const W_KEYS = ['pd', 'clutter', 'targets', 'lowFrac'];
  const syncWhatIf = () => {
    const c = { ...MOT_SCENARIOS[S.scenario], ...S.overrides };
    for (const k of W_KEYS) { $('w-' + k).value = c[k] ?? (k === 'lowFrac' ? 0.35 : 0); $('wv-' + k).textContent = (+$('w-' + k).value).toString(); }
    $('w-shadows').checked = !!c.shadows;
  };
  for (const k of W_KEYS) $('w-' + k).oninput = (e) => { $('wv-' + k).textContent = e.target.value; };
  $('w-apply').onclick = () => {
    S.overrides = Object.fromEntries(W_KEYS.map((k) => [k, +$('w-' + k).value]));
    S.overrides.shadows = $('w-shadows').checked;
    newSession(); pushFeed(`What-if applied: detection ${S.overrides.pd}, false alarms ${S.overrides.clutter}/frame, ${S.overrides.targets} people, ${Math.round(S.overrides.lowFrac * 100)}% low rocks, shadows ${S.overrides.shadows ? 'on' : 'off'}`, '');
  };
  $('w-reset').onclick = () => { S.overrides = {}; syncWhatIf(); newSession(); };
  const oldScenario = $('scenario').onchange;
  $('scenario').onchange = (e) => { S.overrides = {}; oldScenario(e); syncWhatIf(); };
  syncWhatIf();
  $('tourBtn').onclick = () => runTour({
    prepare() { S.view = '3d'; [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b.dataset.v === '3d')); S.sel = 0; S.paused = false; newSession(); },
    showNaive() { S.sel = 2; buildCards(); },
    polar() { S.scenario = 'polar'; $('scenario').value = 'polar'; S.overrides = {}; syncWhatIf(); newSession(); },
  }, TRACK_TOUR);
  window.onkeydown = (e) => { if (e.key === ' ' && e.target.tagName !== 'INPUT') { e.preventDefault(); $('play').click(); } };
}

const TRACK_TOUR = (api) => [
  { at: '#goal', title: '1 · The question', text: 'When an astronaut walks behind a boulder, does the tracker give them back the <b>same ID</b> when they reappear? A new ID means the system thinks there are now two people: an <b>ID switch</b>.' },
  { at: '#mapwrap', title: '2 · The world (real NASA models)', text: 'The white <b>astronauts</b> are where people really are; only training views show this. The coloured <b>rings and #tags</b> are what the selected tracker believes. A shaded <b>search area</b> means "lost sight, probably in here". Drag to orbit.' },
  { at: '#camwrap', title: '3 · What the robot actually sees', text: 'Rendered from the 2.2 m camera mast with real depth. Tall rocks hide people completely; low rocks only hide legs. A <b>dashed box</b> is the tracker saying "I think someone is behind this rock".' },
  { at: '#cards', title: '4 · Three trackers, one difference', text: 'All three get identical detections. They differ only in how they treat a missed detection. The big number is <b>% of IDs kept</b> through 1–5 s occlusions. The naive tracker deletes anyone it cannot see.', action: { label: 'Show the naive tracker on the map', run: () => api.showNaive() } },
  { at: '#feed', title: '5 · The story, live', text: 'Every disappearance and reappearance is narrated, with a ✔ (same ID) or ✘ (new ID) verdict for each tracker.' },
  { at: '#chSw', title: '6 · Watch the gap open', text: 'Cumulative ID switches per tracker. Over a few minutes the red (naive) line usually climbs fastest, but single runs are noisy: the README reports 20-run averages with confidence intervals.' },
  { at: '#scenario', title: '7 · A second kind of occlusion', text: 'At the lunar south pole the sun hugs the horizon, so boulders cast very long shadows. People in shadow are in line of sight but too dark to detect.', action: { label: '🌑 Switch to South pole · Long shadows', run: () => api.polar() } },
  { at: '#whatif', title: '8 · Your experiments', text: 'Change the world: detection rate, false alarms, crowd size, low rocks, shadows. Press Apply and see which tracker suffers. Every exercise in the guide uses these controls.' },
  { at: '#viewsw', title: '9 · The map view', text: 'Top-down map: dark wedges show exactly where the camera is blind. Useful to understand <i>why</i> someone disappeared.' },
  { at: null, title: 'Go deeper', text: 'The <b>📘 Guide</b> button explains each idea in plain words, then the maths, then the exact code, then the paper behind it (all references verified). Every number in the README can be regenerated with one command.' },
];

wire(); newSession(); requestAnimationFrame(frame);
// 3-D view loads on its own: if WebGL or the models are unavailable, the 2-D views keep working.
import('./world3d.js')
  .then((m) => m.createWorld3D({ canvas: $('glAll'), mainEl: $('mapwrap'), camEl: $('camwrap'), mainOverlay: $('ov3d'), camOverlay: $('ovcam') }))
  .then((w) => { w3d = w; $('load3d').remove(); import('./three-common.js').then((c) => { if (c.GFX.low) $('gfxNote').innerHTML = ' · <b>low-power 3D</b> (no shadows) · <a href="?gfx=high" style="color:#7cf">try full quality</a>'; }); })
  .catch((err) => { showError('3-D load', err); $('load3d')?.remove(); S.view = '2d'; });
window.__track = () => sess;
