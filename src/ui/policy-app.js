// Multimodal Policy Lab page: four copies of one world, one per policy, driven live by the exported networks.
import { THREE, toV, fbm, makeRenderer, fitRenderer, loadModel, makeEnvironment, starfield, drawLabels } from './three-common.js';
import { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
import { DT, H, EXEC, GOAL_X, GOAL_TOL, ROVER_R, W_MAX, Episode, makeExpert, makeRng, chunkPath } from '../policy/sim.js';
import { unpack, bcAct, arSample, dpSample, cosineSchedule, tokenValue, VOCAB } from '../policy/policies.js';

const $ = (id) => document.getElementById(id);
const POLICIES = [
  { key: 'expert', name: 'Expert', color: '#e5e7eb' },
  { key: 'bc', name: 'BC (regression)', color: '#ff5d6c' },
  { key: 'ar', name: 'Autoregressive', color: '#ffb454' },
  { key: 'dp', name: 'Diffusion', color: '#b794f6' },
];
const LANE_GAP = 8;                      // m between the four copies of the world
const FAN = 12;                          // plans sampled per decision by the two sampling policies
const SNAPS = [49, 25, 8, 0];            // diffusion steps shown in the noise-to-plan strip
const ROCK_R = 1.2;
const laneX = (i) => (i - 1.5) * LANE_GAP;
/** Lane i, simulator point (x forward, y left) -> ground (gx, gy) used by toV. */
const ground = (i, x, y) => [laneX(i) - y, x];

const S = {
  off: 0, dist: 7, sigma: 0.05, temp: 1, ckpt: null, speed: 1,
  show: Object.fromEntries(POLICIES.map((p) => [p.key, true])),
  paused: false, run: 0, steps: 0, acc: 0, restartAt: null,
  manifest: null, results: null, weights: new Map(), schedule: null, W: null, lanes: [],
};

// ---------------------------------------------------------------- run control

function world() { return { rockX: S.dist, rockY: S.off, rockR: ROCK_R, goalY: 0, sigma: S.sigma }; }

function newRun() {
  S.run++; S.steps = 0; S.acc = 0; S.restartAt = null;
  const seed = 1000 + S.run, w = world();
  S.lanes = POLICIES.map((p, i) => {
    const lane = { ...p, i, ep: new Episode(w, seed), rng: makeRng(seed * 7 + i), fan: [], plan: [], view: {} };
    lane.decide = deciders[p.key](lane);
    return lane;
  });
  $('runSeed').innerHTML = `run ${S.run} · <a href="?run=${S.run}&off=${S.off}&dist=${S.dist}&sigma=${S.sigma}&temp=${S.temp}&epoch=${S.ckpt}&scene=${$('scene').value}" title="Link that replays this exact run">link</a>`;
  scene3d?.rebuildWorld(w);
  stepAll(); // first decision, so every lane shows its plan before moving
}

const meanTurn = (chunk) => chunk.reduce((a, [, w]) => a + w, 0) / chunk.length;

const deciders = {
  expert: (lane) => { const expert = makeExpert(lane.rng); return (o) => keep(lane, expert(o), []); },
  bc: (lane) => (o) => { const c = bcAct(S.W.bc, o); S.bcTurn = meanTurn(c); return keep(lane, c, []); },
  ar: (lane) => (o) => {
    const s = Array.from({ length: FAN }, () => arSample(S.W.ar, o, S.temp, lane.rng));
    lane.view.turnProbs = s[0].probs[1];   // token 2 (average turn), given the speed token it drew
    lane.view.firstProbs ??= s[0].probs[1]; // the moment the side is chosen
    lane.view.turnToken = s[0].tokens[1];
    return keep(lane, s[0].chunk, s.slice(1).map((x) => x.chunk));
  },
  dp: (lane) => (o) => {
    const first = dpSample(S.W.dp, o, S.schedule, lane.rng, SNAPS);
    const rest = Array.from({ length: FAN - 1 }, () => dpSample(S.W.dp, o, S.schedule, lane.rng).chunk);
    lane.view = { snaps: first.snaps, finals: rest, obs: o };
    return keep(lane, first.chunk, rest);
  },
};

/** Remember what a policy decided (world-frame paths for drawing) and return the chunk to execute. */
function keep(lane, chunk, alternatives) {
  const pose = lane.ep.pose;
  lane.plan = chunkPath(pose, chunk);
  lane.fan = alternatives.map((c) => chunkPath(pose, c));
  return chunk;
}

function stepAll() {
  for (const l of S.lanes) l.ep.step(l.decide);
  S.steps++;
  if (S.lanes.every((l) => l.ep.done) && S.restartAt == null) S.restartAt = performance.now() + 3000;
  updateLive();
}

function nextDecision() {
  pause(true);
  do stepAll(); while ((S.steps - 1) % EXEC !== 0 && !S.lanes.every((l) => l.ep.done));
}

function pause(on) { S.paused = on; $('play').textContent = on ? '▶ Play' : '⏸ Pause'; }

// ---------------------------------------------------------------- weights

async function loadCheckpoint(epoch) {
  if (!S.weights.has(epoch)) {
    $('runMsg').textContent = `Loading epoch ${epoch} weights…`;
    const buf = await (await fetch(`policy/weights/epoch_${epoch}.bin`)).arrayBuffer();
    S.weights.set(epoch, unpack(S.manifest, buf));
  }
  S.ckpt = epoch; S.W = S.weights.get(epoch);
  document.querySelectorAll('#ckpt button').forEach((b) => b.classList.toggle('on', +b.dataset.ep === epoch));
  renderScorecard(); drawLoss();
  newRun();
}

// ---------------------------------------------------------------- panels

const STATUS_TEXT = { driving: 'driving', goal: 'reached goal', crash: 'crashed', stuck: 'stuck' };

function updateLive() {
  $('clock').textContent = `t = ${(S.steps * DT).toFixed(1)} s`;
  $('liveRows').innerHTML = S.lanes.map((l) => {
    const e = l.ep, w = e.chunk ? e.chunk[Math.max(0, e.k - 1)][1] : 0;
    const side = e.passedY == null ? '—' : e.passedY > 0 ? 'left' : 'right';
    const res = e.status === 'goal' ? `✓ ${e.t.toFixed(1)} s` : e.status === 'crash' ? '✗ crashed' : e.status === 'stuck' ? 'stuck' : '…';
    return `<tr style="${S.show[l.key] ? '' : 'opacity:.35'}"><td><i class="sw" style="background:${l.color}"></i>${l.name}</td>`
      + `<td>${e.done ? '—' : `${w >= 0 ? '+' : ''}${w.toFixed(2)} rad/s`}</td><td>${side}</td><td class="st-${e.status}">${res}</td></tr>`;
  }).join('');
  const done = S.lanes.every((l) => l.ep.done);
  $('runMsg').textContent = done ? `Run ${S.run} finished · next run in a moment (R: now, Space: hold)` : '';
  $('runMsg').hidden = !done;
}

/** Final dead-ahead crash rate across training seeds: mean and range. */
function seedStat(key) {
  const x = S.results.seeds.final[key].dead_ahead.crash, mean = x.reduce((a, b) => a + b, 0) / x.length;
  return { mean, text: `${mean.toFixed(1)}% [${Math.min(...x)}–${Math.max(...x)}]` };
}

function renderScorecard() {
  const r = S.results, ck = r.checkpoints[S.ckpt];
  $('scoreCk').textContent = `epoch ${S.ckpt} · 400 runs per test`;
  const crash = (v) => `<td class="${v >= 10 ? 'hi' : v <= 1 ? 'lo' : ''}">${v}%</td>`;
  const goal = (v) => `<td class="${v >= 99 ? 'lo' : v < 90 ? 'hi' : ''}">${v}%</td>`;
  $('scoreRows').innerHTML = POLICIES.map((p) => {
    const s = p.key === 'expert' ? r.expert : ck[p.key], a = s.dead_ahead;
    return `<tr><td><i class="sw" style="background:${p.color}"></i>${p.name.split(' ')[0]}</td>${goal(s.random.goal)}${crash(s.random.crash)}`
      + `${goal(a.goal)}${crash(a.crash)}<td>${a.left == null ? '—' : `${a.left}%`}</td></tr>`;
  }).join('');
  $('seedNote').innerHTML = `<b>Across ${r.seeds.n} training seeds</b> (epoch ${r.config.epochs}), rock dead ahead, crash mean [range]: `
    + POLICIES.slice(1).map((p) => `${p.name.split(' ')[0]} ${seedStat(p.key).text}`).join(' · ')
    + `. The table shows seed 0, the one this page runs.`;
}

function canvas2d(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  g.font = '11px ui-monospace, Menlo, monospace';
  return { g, w, h };
}

function drawLoss() {
  const { g, w, h } = canvas2d($('cvLoss')), L = S.results.loss, last = L.at(-1).epoch;
  const pad = { l: 34, r: 100, t: 8, b: 20 }, lo = -2.5;
  const X = (e) => pad.l + (e / last) * (w - pad.l - pad.r), Y = (v) => pad.t + (Math.log10(v) / lo) * (h - pad.t - pad.b);
  g.strokeStyle = '#1b2538'; g.fillStyle = '#7d8ba3';
  for (const v of [1, 0.1, 0.01]) { g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(w - pad.r, Y(v)); g.stroke(); g.fillText(String(v), 4, Y(v) + 4); }
  for (const e of [0, 20, 40, 60].filter((e) => e <= last)) g.fillText(String(e), X(e) - 6, h - 6);
  g.fillText('epoch', w - pad.r + 22, h - 6);
  g.strokeStyle = '#4cc9f0'; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(S.ckpt), pad.t); g.lineTo(X(S.ckpt), h - pad.b); g.stroke(); g.setLineDash([]);
  const ends = [];
  for (const p of POLICIES.slice(1)) {
    const v0 = L[0][p.key];
    g.strokeStyle = p.color; g.lineWidth = 2; g.beginPath();
    L.forEach((d, i) => (i ? g.lineTo : g.moveTo).call(g, X(d.epoch), Y(d[p.key] / v0)));
    g.stroke(); g.lineWidth = 1;
    ends.push({ p, y: Y(L.at(-1)[p.key] / v0) + 4 });
  }
  ends.sort((a, b) => a.y - b.y).forEach((e, i, a) => { if (i) e.y = Math.max(e.y, a[i - 1].y + 12); g.fillStyle = e.p.color; g.fillText(e.p.name.split(' ')[0], w - pad.r + 4, e.y); });
}

/** Average turn rate a turn token stands for: DCT row 0 is 1/sqrt(H), so the mean is coefficient / sqrt(H). */
const tokenTurn = (t) => Math.max(-1, Math.min(1, tokenValue(t) / Math.sqrt(H))) * W_MAX;

function drawTokens() {
  const { g, w, h } = canvas2d($('cvTok')), ar = S.lanes.find((l) => l.key === 'ar'), p = ar?.view.turnProbs;
  if (!p || !S.show.ar) { g.fillStyle = '#7d8ba3'; g.fillText(S.show.ar ? 'waiting for the first decision' : 'Autoregressive policy hidden', 8, 20); return; }
  const pad = { l: 8, r: 8, t: 10, b: 30 }, X = (wr) => pad.l + ((W_MAX - wr) / (2 * W_MAX)) * (w - pad.l - pad.r);   // left turns on the left
  const first = ar.view.firstProbs, maxP = Math.max(...p, ...first), bw = Math.max(2, (w - pad.l - pad.r) / VOCAB * 1.6 - 1);
  for (let t = 0; t < VOCAB; t++) {
    const turn = tokenTurn(t), bh = (p[t] / maxP) * (h - pad.t - pad.b);
    if (bh < 0.5) continue;
    g.fillStyle = t === ar.view.turnToken ? '#ffffff' : Math.abs(turn) < 0.03 ? '#94a3b8' : turn > 0 ? '#4cc9f0' : '#4ade80';
    g.fillRect(X(turn) - bw / 2, h - pad.b - bh, bw, bh);
  }
  g.strokeStyle = '#e5e7eb99'; g.setLineDash([2, 2]);
  for (let t = 0; t < VOCAB; t++) if (first[t] / maxP > 0.004) g.strokeRect(X(tokenTurn(t)) - bw / 2, h - pad.b - (first[t] / maxP) * (h - pad.t - pad.b), bw, (first[t] / maxP) * (h - pad.t - pad.b));
  g.setLineDash([]);
  g.strokeStyle = '#334155'; g.beginPath(); g.moveTo(pad.l, h - pad.b); g.lineTo(w - pad.r, h - pad.b); g.stroke();
  g.fillStyle = '#7d8ba3'; g.textAlign = 'center';
  g.fillText('straight', X(0), h - pad.b + 13);
  g.textAlign = 'left'; g.fillText(`← left  +${W_MAX}`, pad.l, h - pad.b + 13);
  g.textAlign = 'right'; g.fillText(`−${W_MAX}  right →`, w - pad.r, h - pad.b + 13); g.textAlign = 'left';
  g.fillText('average turn, rad/s · bars: now · dotted: first decision · white: drawn', pad.l, h - 3);
  if (S.bcTurn != null && S.show.bc) {
    const x = X(S.bcTurn); g.strokeStyle = '#ff5d6c'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, h - pad.b); g.stroke(); g.lineWidth = 1;
    g.fillStyle = '#ff5d6c'; g.fillText('BC', x + 4, pad.t + 9);
  }
}

function drawDiffusion() {
  const { g, w, h } = canvas2d($('cvDiff')), dp = S.lanes.find((l) => l.key === 'dp'), v = dp?.view;
  if (!v?.snaps || !S.show.dp) { g.fillStyle = '#7d8ba3'; g.fillText(S.show.dp ? 'waiting for the first decision' : 'Diffusion policy hidden', 8, 20); return; }
  const n = SNAPS.length, pw = w / n;                             // each panel frames the plan and the rock, true to scale
  const fwd = Math.max(4, Math.min(10, v.obs[0] + v.obs[2] + ROVER_R + 0.5));
  const side = Math.max(1.5, Math.min(3.5, Math.abs(v.obs[1]) + v.obs[2] + ROVER_R));
  const scale = Math.min((pw - 12) / (2 * side), (h - 34) / fwd);
  SNAPS.forEach((k, j) => {
    const cx = j * pw + pw / 2, base = h - 12;
    const P = (x, y) => [cx - y * scale, base - x * scale];
    const [rx, ry] = P(v.obs[0], v.obs[1]);
    g.fillStyle = '#ff5d6c33'; g.beginPath(); g.arc(rx, ry, v.obs[2] * scale, 0, 7); g.fill();
    g.strokeStyle = '#ff5d6c'; g.setLineDash([3, 3]); g.beginPath(); g.arc(rx, ry, (v.obs[2] + ROVER_R) * scale, 0, 7); g.stroke(); g.setLineDash([]);
    const path = (c, col, lw) => { g.strokeStyle = col; g.lineWidth = lw; g.beginPath(); chunkPath({ x: 0, y: 0, th: 0 }, c).forEach(([x, y], i) => (i ? g.lineTo : g.moveTo).call(g, ...P(x, y))); g.stroke(); g.lineWidth = 1; };
    if (k === 0) v.finals.forEach((c) => path(c, '#b794f655', 1));
    path(v.snaps[k], '#b794f6', 2.6);
    g.fillStyle = '#e5e7eb'; g.beginPath(); g.moveTo(cx, base - 6); g.lineTo(cx - 4, base + 3); g.lineTo(cx + 4, base + 3); g.fill();
    g.fillStyle = '#94a3b8'; g.textAlign = 'center';
    g.fillText(k === SNAPS[0] ? `step ${k}: noise` : k === 0 ? 'step 0: plan' : `step ${k}`, cx, 12); g.textAlign = 'left';
    if (j) { g.strokeStyle = '#1b2538'; g.beginPath(); g.moveTo(j * pw, 18); g.lineTo(j * pw, h); g.stroke(); }
  });
}

// ---------------------------------------------------------------- 3-D

/** A flat strip along a polyline, lying on the ground: wheel tracks and the executed plan. */
class Ribbon {
  constructor(max, width, color, opacity = 1, lift = 0.03) {
    Object.assign(this, { max, width, lift });
    this.pos = new Float32Array(max * 6);
    const idx = [];
    for (let i = 0; i < max - 1; i++) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setIndex(idx);
    this.mesh = new THREE.Mesh(this.geo, new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, side: THREE.DoubleSide, depthWrite: false }));
    this.mesh.frustumCulled = false;
  }

  set(pts) {                                // pts: ground [gx, gy]
    const n = Math.min(pts.length, this.max);
    for (let i = 0; i < n; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)], dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
      const nx = (-dy / L) * this.width / 2, ny = (dx / L) * this.width / 2, [x, y] = pts[i];
      this.pos.set([x + nx, this.lift, -(y + ny), x - nx, this.lift, -(y - ny)], i * 6);
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(0, n - 1) * 6);
  }
}

function makeLine(n, color, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  l.frustumCulled = false;
  l.setPoints = (pts) => { pts.forEach(([x, y], i) => g.attributes.position.setXYZ(i, x, 0.05, -y)); g.attributes.position.needsUpdate = true; g.setDrawRange(0, pts.length); };
  return l;
}

function rockGeometry(seed) {
  const geo = new THREE.IcosahedronGeometry(1, 4), p = geo.attributes.position, v = new THREE.Vector3();
  for (let k = 0; k < p.count; k++) {
    v.fromBufferAttribute(p, k);
    const s = 0.86 + 0.28 * fbm(v.x * 1.6 + seed, v.z * 1.6 + v.y);
    p.setXYZ(k, v.x * s, Math.max(v.y * s * 0.8, -0.15), v.z * s);
  }
  geo.computeVertexNormals();
  return geo;
}

/** A simple car, forward = -Z like the rover model. hazard: stalled car with orange flashers. */
function carMesh(L, W, body, accent, hazard = false) {
  const g = new THREE.Group(), std = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.55, ...o });
  const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; g.add(m); return m; };
  const wr = 0.17 * W + 0.02, h = 0.42 * W;
  add(new THREE.BoxGeometry(W, h, L), std(body), 0, wr + h / 2, 0);
  add(new THREE.BoxGeometry(W * 0.86, h * 0.8, L * 0.48), std(0x0b1020, { roughness: 0.1, metalness: 0.9 }), 0, wr + h * 1.4, L * 0.04);
  const lamp = (color, z) => { for (const sx of [-1, 1]) add(new THREE.BoxGeometry(W * 0.22, h * 0.22, 0.02), new THREE.MeshBasicMaterial({ color }), sx * W * 0.32, wr + h * 0.65, z); };
  lamp(0xe0f2fe, -L / 2 - 0.01); lamp(hazard ? 0xf97316 : 0xef4444, L / 2 + 0.01);
  if (!hazard) add(new THREE.BoxGeometry(W * 0.5, 0.05, L * 0.2), new THREE.MeshBasicMaterial({ color: accent }), 0, wr + h * 1.82, L * 0.04);   // policy colour on the roof
  const wheel = new THREE.CylinderGeometry(wr, wr, W * 0.16, 18).rotateZ(Math.PI / 2), tyre = std(0x111111, { metalness: 0, roughness: 0.9 });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) add(wheel, tyre, sx * W * 0.5, wr, sz * L * 0.32);
  return g;
}

let scene3d = null;

async function initScene() {
  const canvas = $('gl'), R = makeRenderer(canvas, 0.9), scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020308);
  scene.environment = makeEnvironment(R, 0x9aa6bb, 0x111111); scene.environmentIntensity = 0.35;
  const stars = starfield(1800);
  scene.add(stars, new THREE.HemisphereLight(0xc9ced8, 0x2a2a2a, 0.12));
  const sun = new THREE.DirectionalLight(0xfff3e0, 3.0);
  sun.position.set(-30, 14, 6); sun.target.position.set(0, 0, -8); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 120 });
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0005;
  scene.add(sun, sun.target);

  // regolith: flat where the rovers drive, rolling outside
  const G = new THREE.PlaneGeometry(160, 110, 160, 110); G.rotateX(-Math.PI / 2);
  const pos = G.attributes.position, col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const gx = pos.getX(i), gy = -pos.getZ(i) + 8;
    pos.setZ(i, -gy);
    const out = Math.max(0, Math.abs(gx) - 24, -6 - gy, gy - 24) / 10;
    pos.setY(i, (fbm(gx * 0.07, gy * 0.07) - 0.5) * 3 * Math.min(1, out) - 0.02);
    const c = 0.19 + (fbm(gx * 0.5, gy * 0.5) - 0.5) * 0.08 + (fbm(gx * 3, gy * 3) - 0.5) * 0.04;
    col.set([c, c, c * 0.97], i * 3);
  }
  G.setAttribute('color', new THREE.BufferAttribute(col, 3)); G.computeVertexNormals();
  const groundMesh = new THREE.Mesh(G, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.97 }));
  groundMesh.receiveShadow = true; scene.add(groundMesh);

  // road view: one straight road per lane, wide enough for every swerve the policies make
  const road = new THREE.Group(); road.visible = false; scene.add(road);
  const flat = (w, h, color, x, y, lift) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color, roughness: 0.9 }));
    m.position.copy(toV(x, y, lift)); m.receiveShadow = true; road.add(m); return m;
  };
  flat(160, 110, 0x2b3a2a, 0, 8, -0.03);                                  // verge
  POLICIES.forEach((_, i) => {
    flat(LANE_GAP - 1, 40, 0x1c1f25, laneX(i), 8, -0.01);                 // asphalt
    for (const e of [-1, 1]) flat(0.12, 40, 0xe5e7eb, laneX(i) + e * (LANE_GAP / 2 - 0.8), 8, 0.0);
    for (let y = -10; y < 28; y += 3) flat(0.1, 1.5, 0xfbbf24, laneX(i), y, 0.0);
  });

  // lane furniture: dividers, start lines, goals
  const furn = new THREE.Group(); scene.add(furn);
  const dash = new THREE.LineDashedMaterial({ color: 0x3b4660, dashSize: 0.6, gapSize: 0.6 });
  for (let i = 0; i <= 4; i++) {
    const x = laneX(i) - LANE_GAP / 2, l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([toV(x, -3, 0.02), toV(x, GOAL_X + 3, 0.02)]), dash);
    l.computeLineDistances(); furn.add(l);
  }
  POLICIES.forEach((p, i) => {
    const start = new THREE.Mesh(new THREE.PlaneGeometry(LANE_GAP * 0.7, 0.08).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x475569 }));
    start.position.copy(toV(laneX(i), -0.6, 0.02)); furn.add(start);
  });

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1500);
  camera.position.set(0, 20, 14);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0, -7.5); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.47;

  const rover = await loadModel('assets/nasa/rassor.glb', 2 * ROVER_R, 'xz');
  const perLane = POLICIES.map((p, i) => {
    const g = new THREE.Group();
    const roverMesh = rover ? rover.clone() : new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.5, 1), new THREE.MeshStandardMaterial({ color: 0x9ca3af }));
    const egoCar = carMesh(0.88, 0.52, 0xd1d5db, p.color);                // fits the 0.5 m body circle
    egoCar.visible = false; g.add(roverMesh, egoCar);
    const ring = new THREE.Mesh(new THREE.RingGeometry(ROVER_R - 0.06, ROVER_R, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: p.color }));
    ring.position.y = 0.04; g.add(ring);
    const trail = new Ribbon(4 * 60 / DT, 0.16, p.color, 0.85), plan = new Ribbon(H + 1, 0.09, p.color, 1, 0.05);
    const fan = Array.from({ length: FAN }, () => makeLine(H + 1, p.color, 0.38));
    const rock = new THREE.Mesh(rockGeometry(i * 3.1), new THREE.MeshStandardMaterial({ color: 0x8d9099, roughness: 0.95, flatShading: true }));
    rock.castShadow = rock.receiveShadow = true;
    const keepOut = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xff5d6c, dashSize: 0.25, gapSize: 0.18 }));
    const goal = new THREE.Group();
    goal.add(new THREE.Mesh(new THREE.RingGeometry(GOAL_TOL - 0.08, GOAL_TOL, 48).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x4ade80, transparent: true, opacity: 0.7 })));
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6), new THREE.MeshStandardMaterial({ color: 0xd1d5db }));
    pole.position.y = 0.8; goal.add(pole);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.35), new THREE.MeshBasicMaterial({ color: p.color, side: THREE.DoubleSide }));
    flag.position.set(0.3, 1.42, 0); goal.add(flag);
    const stalled = carMesh(1.75, 0.85, 0xa3aab5, '#f97316', true);      // fits inside a unit circle; scaled to the obstacle radius
    stalled.visible = false;
    scene.add(g, trail.mesh, plan.mesh, rock, stalled, keepOut, goal, ...fan);
    return { g, trail, plan, fan, rock, stalled, keepOut, goal, roverMesh, egoCar };
  });

  function rebuildWorld(w) {
    perLane.forEach((o, i) => {
      const [gx, gy] = ground(i, w.rockX, w.rockY);
      o.rock.scale.set(w.rockR, w.rockR, w.rockR); o.rock.position.copy(toV(gx, gy, 0));
      o.stalled.scale.setScalar(w.rockR); o.stalled.position.copy(toV(gx, gy, 0));
      o.keepOut.geometry.setFromPoints(Array.from({ length: 72 }, (_, k) => { const a = (k / 72) * Math.PI * 2; return toV(gx + (w.rockR + ROVER_R) * Math.cos(a), gy + (w.rockR + ROVER_R) * Math.sin(a), 0.04); }));
      o.keepOut.computeLineDistances();
      o.goal.position.copy(toV(...ground(i, GOAL_X, w.goalY), 0));
    });
  }

  // click the ground: move the rock there (in whichever lane was clicked)
  const ray = new THREE.Raycaster(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit = new THREE.Vector3();
  let down = null;
  canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    if (!ray.ray.intersectPlane(plane, hit)) return;
    const i = Math.max(0, Math.min(3, Math.round(hit.x / LANE_GAP + 1.5)));
    setWorld({ off: laneX(i) - hit.x, dist: -hit.z });
  });

  const lerpPose = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, th: a.th + Math.atan2(Math.sin(b.th - a.th), Math.cos(b.th - a.th)) * t });

  function render(alpha) {
    if (!fitRenderer(R, canvas, camera)) return;
    controls.update();
    const labels = [];
    S.lanes.forEach((l, i) => {
      const o = perLane[i], vis = S.show[l.key], e = l.ep, p = e.done ? e.pose : lerpPose(e.prev, e.pose, alpha);
      for (const m of [o.g, o.trail.mesh, o.plan.mesh, ...o.fan]) m.visible = vis;
      if (!vis) return;
      o.g.position.copy(toV(...ground(i, p.x, p.y), 0)); o.g.rotation.y = p.th;
      o.trail.set([...e.path.slice(0, -1), [p.x, p.y]].map(([x, y]) => ground(i, x, y)));
      o.plan.set(e.done ? [] : l.plan.map(([x, y]) => ground(i, x, y)));
      o.fan.forEach((f, k) => { const pts = e.done ? null : l.fan[k]; f.visible = !!pts; if (pts) f.setPoints(pts.map(([x, y]) => ground(i, x, y))); });
      const tag = e.status === 'goal' ? ' ✓' : e.status === 'crash' ? ' ✗ crashed' : e.status === 'stuck' ? ' · stuck' : '';
      labels.push({ p: toV(...ground(i, p.x, p.y), 1.5), text: l.name + tag, color: l.color, bg: e.status === 'crash' ? '#7f1d1d' : undefined, fg: e.status === 'crash' ? '#fecaca' : undefined });
    });
    R.render(scene, camera);
    drawLabels($('ov'), camera, labels);
  }

  function setScene(mode) {
    const onRoad = mode === 'road';
    road.visible = onRoad; groundMesh.visible = stars.visible = !onRoad;
    scene.background.set(onRoad ? 0x0b1220 : 0x020308);
    perLane.forEach((o) => { o.rock.visible = o.roverMesh.visible = !onRoad; o.stalled.visible = o.egoCar.visible = onRoad; });
  }

  scene3d = { rebuildWorld, render, setScene };
}

// ---------------------------------------------------------------- controls

const TRY = {
  ahead: { set: { off: 0, dist: 7, sigma: 0.05, ckpt: 'last' }, note: () => `The rock sits exactly on the line to the goal. Averaged over 5 training seeds × 400 runs, BC crashes in ${seedStat('bc').mean.toFixed(0)}% of runs, autoregressive ${seedStat('ar').mean.toFixed(1)}%, diffusion ${seedStat('dp').mean.toFixed(1)}%: BC heads for the middle of the two expert choices and turns late.` },
  aside: { set: { off: 1.2, dist: 7, sigma: 0.05, ckpt: 'last' }, note: () => 'Only one side makes sense now, so every policy agrees. Averaging is harmless when the expert has one answer.' },
  blur: { set: { off: 0, dist: 7, sigma: 0.4, ckpt: 'last' }, note: () => 'Noise moves the rock off the line at every decision, which breaks the tie for BC too, and every policy keeps a wider gap. Averaging hurts most when perception is sharp.' },
  untrained: { set: { off: 0, dist: 7, sigma: 0.05, ckpt: 0 }, note: () => 'Random weights: the plans are noise. Step through the checkpoints below to watch each policy learn.' },
};

function syncSliders() {
  $('off').value = S.off; $('vOff').textContent = `${S.off >= 0 ? '+' : ''}${S.off.toFixed(2)} m`;
  $('dist').value = S.dist; $('vDist').textContent = `${S.dist.toFixed(1)} m`;
  $('sig').value = S.sigma; $('vSig').textContent = `${S.sigma.toFixed(2)} m`;
  $('temp').value = S.temp; $('vTemp').textContent = S.temp.toFixed(2);
}

function setWorld(change) {
  Object.assign(S, change);
  S.off = Math.max(-2, Math.min(2, S.off)); S.dist = Math.max(5, Math.min(9, S.dist));
  syncSliders(); newRun();
}

function initControls() {
  const speeds = [0.5, 1, 2, 4];
  $('speed').innerHTML = speeds.map((s) => `<button data-s="${s}" class="${s === 1 ? 'on' : ''}">${s}×</button>`).join('');
  $('speed').onclick = (e) => { const s = +e.target.dataset.s; if (!s) return; S.speed = s; [...$('speed').children].forEach((b) => b.classList.toggle('on', +b.dataset.s === s)); };
  $('play').onclick = () => pause(!S.paused);
  $('next').onclick = nextDecision;
  $('again').onclick = () => newRun();
  $('scene').onchange = (e) => { scene3d.setScene(e.target.value); $('roadNote').hidden = e.target.value !== 'road'; };
  for (const [id, key] of [['off', 'off'], ['dist', 'dist'], ['sig', 'sigma']]) $(id).oninput = (e) => setWorld({ [key]: +e.target.value });
  $('temp').oninput = (e) => { S.temp = +e.target.value; syncSliders(); };
  document.querySelectorAll('[data-show]').forEach((c) => (c.onchange = () => { S.show[c.dataset.show] = c.checked; updateLive(); }));
  $('ckpt').innerHTML = S.manifest.checkpoints.map((e) => `<button data-ep="${e}">${e}</button>`).join('');
  $('ckpt').onclick = (e) => { if (e.target.dataset.ep) loadCheckpoint(+e.target.dataset.ep); };
  $('ckNote').textContent = 'Each button loads the weights saved after that many passes over the demonstrations.';
  document.querySelectorAll('[data-try]').forEach((b) => (b.onclick = async () => {
    const t = TRY[b.dataset.try], { ckpt, ...world } = t.set;
    document.querySelectorAll('[data-try]').forEach((x) => x.classList.toggle('on', x === b));
    $('tryNote').textContent = t.note();
    Object.assign(S, world); syncSliders();
    const ep = ckpt === 'last' ? S.manifest.checkpoints.at(-1) : ckpt;
    if (ep !== S.ckpt) await loadCheckpoint(ep); else newRun();
    pause(false);
  }));
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.code === 'Space') { e.preventDefault(); pause(!S.paused); }
    else if (e.key === 'n' || e.key === 'N') nextDecision();
    else if (e.key === 'r' || e.key === 'R') newRun();
  });
}

// ---------------------------------------------------------------- main loop

async function boot() {
  [S.manifest, S.results] = await Promise.all(['policy/weights/manifest.json', 'policy/results.json'].map((u) => fetch(u).then((r) => r.json())));
  S.schedule = cosineSchedule(S.results.config.diffusion_steps);
  // shareable runs: ?run=37&off=0&dist=7&sigma=0.05&epoch=60&scene=road reproduces exactly what was on screen
  const q = new URLSearchParams(location.search), num = (k) => (q.has(k) && Number.isFinite(+q.get(k)) ? +q.get(k) : null);
  if (num('run') != null) S.run = Math.max(0, Math.round(num('run')) - 1);
  for (const [k, key] of [['off', 'off'], ['dist', 'dist'], ['sigma', 'sigma'], ['temp', 'temp']]) if (num(k) != null) S[key] = num(k);
  $('nPairs').textContent = S.results.dataset.pairs.toLocaleString('en-US');
  syncSliders(); initControls();
  await initScene();
  if (q.get('scene') === 'road') { $('scene').value = 'road'; $('scene').onchange({ target: $('scene') }); }
  await loadCheckpoint(S.manifest.checkpoints.includes(num('epoch')) ? num('epoch') : S.manifest.checkpoints.at(-1));
  let last = performance.now();
  const frame = (now) => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (!S.paused) {
      S.acc += dt * S.speed;
      while (S.acc >= DT && !S.lanes.every((l) => l.ep.done)) { stepAll(); S.acc -= DT; }
      if (S.restartAt != null && now > S.restartAt) newRun();
    }
    scene3d.render(S.lanes.every((l) => l.ep.done) ? 1 : Math.min(1, S.acc / DT));
    drawTokens(); drawDiffusion();
  };
  requestAnimationFrame(frame);
}

boot().catch((e) => { $('runMsg').hidden = false; $('runMsg').textContent = `Could not start: ${e.message}`; console.error(e); });
