// Occlusion Tracking Lab: bird's-eye scene with blind-spot wedges, a rendered camera view, three trackers
// compared live on the same detections, and a plain-language event feed.
import { MotSession } from '../mot/session.js';
import { MOT_SCENARIOS } from '../mot/world.js';
import { wrapAngle, TAU } from '../core/linalg.js';

const $ = (id) => document.getElementById(id);
const S = { scenario: 'boulders', seed: 7, speed: 2, paused: false, sel: 0, showTruth: true };
const COLORS = ['#4ade80', '#4cc9f0', '#ff6b6b'];
const BLURB = {
  'Aware + neg. info': 'Knows where the camera is blind, and uses the miss itself: "I can\'t see them, so they must be in a shadow".',
  'Occlusion-aware': 'Knows where the camera is blind. Missing someone behind a rock is expected, not evidence they left.',
  Naive: 'Assumes everyone is always visible. A few missed frames and it deletes the track.',
};
let sess, feed = [], pending = new Map(), seenEv = [], prevHidden = new Map();
const trackColor = (id) => `hsl(${(id * 67) % 360} 85% 62%)`;

function newSession() {
  sess = new MotSession({ scenario: S.scenario, seed: S.seed });
  feed = []; pending = new Map(); seenEv = sess.runs.map(() => 0); prevHidden = new Map();
  pushFeed(`${MOT_SCENARIOS[S.scenario].label}: ${MOT_SCENARIOS[S.scenario].blurb}`, '');
  buildCards();
}

// ------------------------------------------------------------------ feed
function pushFeed(text, cls) { feed.unshift({ t: sess.world.t, text, cls }); if (feed.length > 60) feed.pop(); }
function updateFeed() {
  const W = sess.world;
  for (const g of W.targets) {
    const hid = g.vis.inFov && g.vis.visFrac < 0.25, was = prevHidden.get(g.id);
    if (hid && was === false) pushFeed(`Crew ${g.id} went behind a boulder`, 'hide');
    prevHidden.set(g.id, hid);
  }
  sess.runs.forEach((r, i) => {
    const ev = r.metrics.events;
    for (; seenEv[i] < ev.length; seenEv[i]++) {
      const e = ev[seenEv[i]], key = e.target;
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
    <div class="hd"><span class="nm" style="color:${COLORS[i]}">${r.tracker.name}</span><span class="sub">${i === S.sel ? 'shown on map' : ''}</span></div>
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
}

// ------------------------------------------------------------------ canvases
function fit(cv) {
  const dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, W: r.width, H: r.height };
}

function drawMap() {
  const { c, W, H } = fit($('map')), w = sess.world, cam = w.cam;
  const xmin = -cam.range * Math.sin(cam.fov / 2) - 2, xmax = -xmin, ymin = cam.y - 3, ymax = cam.y + cam.range + 1, pad = 18, top = 32;
  const s = Math.min((W - 2 * pad) / (xmax - xmin), (H - top - pad) / (ymax - ymin));
  const ox = W / 2 - ((xmin + xmax) / 2) * s, oy = top + (H - top - pad) / 2 + ((ymin + ymax) / 2) * s;
  const P = (x, y) => [ox + x * s, oy - y * s];
  c.fillStyle = '#121418'; c.fillRect(0, 0, W, H);
  // field of view wedge
  const [cx, cy] = P(cam.x, cam.y), a0 = cam.th - cam.fov / 2, a1 = cam.th + cam.fov / 2;
  c.save(); c.beginPath(); c.moveTo(cx, cy); c.arc(cx, cy, cam.range * s, -a1, -a0); c.closePath();
  const g = c.createRadialGradient(cx, cy, 0, cx, cy, cam.range * s); g.addColorStop(0, '#3a3d44'); g.addColorStop(1, '#262930');
  c.fillStyle = g; c.fill(); c.clip();
  c.strokeStyle = '#ffffff10'; for (let r = 10; r < cam.range; r += 10) { c.beginPath(); c.arc(cx, cy, r * s, 0, TAU); c.stroke(); }
  // blind-spot wedges behind each boulder
  for (const b of w.boulders) {
    const dx = b.x - cam.x, dy = b.y - cam.y, d = Math.hypot(dx, dy), half = Math.asin(Math.min(1, b.r / d)), cb = Math.atan2(dy, dx), tl = d * Math.cos(half);
    c.beginPath();
    const p0 = P(cam.x + tl * Math.cos(cb - half), cam.y + tl * Math.sin(cb - half)), p1 = P(cam.x + tl * Math.cos(cb + half), cam.y + tl * Math.sin(cb + half));
    c.moveTo(...p0); c.arc(cx, cy, cam.range * 1.2 * s, -(cb - half), -(cb + half), true); c.lineTo(...p1); c.closePath();
    c.fillStyle = 'rgba(0,0,0,0.55)'; c.fill();
  }
  c.restore();
  c.strokeStyle = '#4cc9f055'; c.setLineDash([4, 5]); c.beginPath(); c.moveTo(cx, cy); c.lineTo(...P(cam.x + cam.range * Math.cos(a0), cam.y + cam.range * Math.sin(a0))); c.moveTo(cx, cy); c.lineTo(...P(cam.x + cam.range * Math.cos(a1), cam.y + cam.range * Math.sin(a1))); c.stroke(); c.setLineDash([]);
  // boulders
  for (const b of w.boulders) { const [px, py] = P(b.x, b.y); c.beginPath(); c.arc(px, py, b.r * s, 0, TAU); c.fillStyle = '#5b5e66'; c.fill(); c.strokeStyle = '#8a8e98'; c.lineWidth = 1.5; c.stroke(); }
  // camera
  c.save(); c.translate(cx, cy); c.rotate(-cam.th); c.fillStyle = '#4cc9f0'; c.beginPath(); c.moveTo(12, 0); c.lineTo(-7, 7); c.lineTo(-7, -7); c.closePath(); c.fill(); c.restore();
  c.fillStyle = '#4cc9f0'; c.font = '11px ui-monospace,monospace'; c.fillText('rover mast camera', cx + 12, cy + 4);
  // truth
  if (S.showTruth) for (const t of w.targets) {
    const [px, py] = P(t.x, t.y), hid = !t.vis.inFov || t.vis.visFrac < 0.25;
    if (t.trail.length > 1) { c.strokeStyle = '#ffffff22'; c.lineWidth = 1; c.beginPath(); t.trail.forEach(([x, y], i) => { const q = P(x, y); i ? c.lineTo(...q) : c.moveTo(...q); }); c.stroke(); }
    c.setLineDash(hid ? [3, 3] : []); c.strokeStyle = hid ? '#ffffff88' : '#fff'; c.lineWidth = 1.5; c.beginPath(); c.arc(px, py, Math.max(4, 0.45 * s), 0, TAU); c.stroke(); c.setLineDash([]);
    c.fillStyle = hid ? '#ffffff88' : '#fff'; c.font = '10px ui-monospace,monospace'; c.fillText('crew ' + t.id + (hid ? ' (hidden)' : ''), px + 7, py + 14);
  }
  // detections
  c.strokeStyle = '#ffb454'; c.lineWidth = 1.5;
  for (const d of sess.last.dets) { const [px, py] = P(cam.x + d.range * Math.cos(d.bearing), cam.y + d.range * Math.sin(d.bearing)); c.beginPath(); c.moveTo(px - 4, py - 4); c.lineTo(px + 4, py + 4); c.moveTo(px + 4, py - 4); c.lineTo(px - 4, py + 4); c.stroke(); }
  // tracks of the selected tracker
  const T = sess.runs[S.sel].tracker;
  for (const t of T.tracks) {
    if (!t.confirmed) continue;
    const [px, py] = P(t.x[0], t.x[1]), col = trackColor(t.id), coasting = t.lastSeen > 3;
    const a = t.P.get(0, 0), b = t.P.get(0, 1), d2 = t.P.get(1, 1), m = (a + d2) / 2, q = Math.sqrt(((a - d2) / 2) ** 2 + b * b), k = Math.sqrt(5.991);
    c.save(); c.translate(px, py); c.rotate(-0.5 * Math.atan2(2 * b, a - d2));
    c.setLineDash(coasting ? [5, 4] : []); c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); c.ellipse(0, 0, Math.max(3, k * Math.sqrt(m + q) * s), Math.max(3, k * Math.sqrt(Math.max(m - q, 0)) * s), 0, 0, TAU); c.stroke(); c.restore(); c.setLineDash([]);
    c.fillStyle = col; c.beginPath(); c.arc(px, py, 3, 0, TAU); c.fill();
    c.font = 'bold 11px ui-monospace,monospace'; c.fillText(`#${t.id}${coasting ? (t.hidden ? ' hidden' : ' coasting') : ''}`, px + 7, py - 7);
  }
  // scale
  const bar = 10 * s; c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(W - 20 - bar, H - 12); c.lineTo(W - 20, H - 12); c.stroke(); c.fillStyle = '#fff'; c.fillText('10 m', W - 20 - bar, H - 17);
}

function drawCam() {
  const { c, W, H } = fit($('cam')), w = sess.world, cam = w.cam;
  const f = W / 2 / Math.tan(cam.fov / 2), fv = H * 1.15, y0 = H * 0.3, camH = 2.2;
  const U = (brg) => W / 2 - f * Math.tan(wrapAngle(brg - cam.th));
  const ground = (d) => y0 + (fv * camH) / Math.max(d, 1);
  const sky = c.createLinearGradient(0, 0, 0, y0); sky.addColorStop(0, '#000'); sky.addColorStop(1, '#0b0d12'); c.fillStyle = sky; c.fillRect(0, 0, W, y0);
  const gr = c.createLinearGradient(0, y0, 0, H); gr.addColorStop(0, '#3a3c42'); gr.addColorStop(1, '#6a6c72'); c.fillStyle = gr; c.fillRect(0, y0, W, H - y0);
  // painter's algorithm: far to near, so near objects occlude far ones exactly as the camera sees it
  const items = [];
  for (const b of w.boulders) { const d = Math.hypot(b.x - cam.x, b.y - cam.y); items.push({ d, kind: 'rock', b }); }
  for (const t of w.targets) if (t.vis.inFov) items.push({ d: t.vis.range, kind: 'crew', t });
  items.sort((a, b) => b.d - a.d);
  for (const it of items) {
    if (it.kind === 'rock') {
      const b = it.b, half = Math.asin(Math.min(1, b.r / it.d)), cb = Math.atan2(b.y - cam.y, b.x - cam.x);
      const ul = U(cb + half), ur = U(cb - half), yb = ground(it.d - b.r * 0.5), h = (fv * b.r * 1.1) / it.d;
      c.fillStyle = '#2b2d33'; c.strokeStyle = '#55585f'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(ul, yb); c.quadraticCurveTo(ul + (ur - ul) * 0.1, yb - h, (ul + ur) / 2, yb - h * 1.05); c.quadraticCurveTo(ur - (ur - ul) * 0.1, yb - h, ur, yb); c.closePath(); c.fill(); c.stroke();
    } else {
      const t = it.t, u = U(t.vis.bearing), yb = ground(it.d), h = (fv * 1.8) / it.d, wpx = Math.max(3, (f * 0.7) / it.d);
      c.fillStyle = '#e8e8e8'; c.fillRect(u - wpx / 2, yb - h, wpx, h);
      c.fillStyle = '#d4a017'; c.fillRect(u - wpx * 0.3, yb - h * 0.95, wpx * 0.6, h * 0.16); // gold visor
    }
  }
  // detections (raw camera output, no IDs)
  c.strokeStyle = '#ffb454'; c.lineWidth = 1.2;
  for (const d of sess.last.dets) { const u = U(d.bearing), yb = ground(d.range), h = (fv * 1.8) / d.range, wpx = Math.max(6, (f * 0.9) / d.range); c.strokeRect(u - wpx / 2, yb - h - 2, wpx, h + 4); }
  // tracks of the selected tracker
  const T = sess.runs[S.sel].tracker;
  for (const t of T.tracks) {
    if (!t.confirmed) continue;
    const dx = t.x[0] - cam.x, dy = t.x[1] - cam.y, d = Math.hypot(dx, dy), brg = Math.atan2(dy, dx);
    if (Math.abs(wrapAngle(brg - cam.th)) > cam.fov / 2 || d < 1) continue;
    const u = U(brg), yb = ground(d), h = (fv * 1.8) / d, wpx = Math.max(8, (f * 1.1) / d), coasting = t.lastSeen > 3, col = trackColor(t.id);
    c.strokeStyle = col; c.lineWidth = 2; c.setLineDash(coasting ? [5, 4] : []); c.strokeRect(u - wpx / 2, yb - h - 4, wpx, h + 8); c.setLineDash([]);
    c.fillStyle = col; c.font = 'bold 11px ui-monospace,monospace'; c.fillText(`#${t.id}${coasting && t.hidden ? ' hidden' : ''}`, u - wpx / 2, yb - h - 8);
  }
}

// ------------------------------------------------------------------ loop & wiring
let acc = 0, last = performance.now(), uiT = 0;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (!S.paused) { acc += dt * S.speed; let n = 0; while (acc >= sess.world.dt && n < 200) { sess.step(); updateFeed(); acc -= sess.world.dt; n++; } if (n === 200) acc = 0; }
  drawMap(); drawCam();
  if (now - uiT > 250) { uiT = now; updateCards(); }
  requestAnimationFrame(frame);
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
  window.onkeydown = (e) => { if (e.key === ' ' && e.target.tagName !== 'INPUT') { e.preventDefault(); $('play').click(); } };
}

wire(); newSession(); requestAnimationFrame(frame);
window.__track = () => sess;
