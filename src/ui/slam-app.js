// Page controller for the SLAM lab (slam.html): runs LunarSlamSession (src/slam/engine.js),
// draws the 3-D reality view (src/ui/slam3d.js), the rover's own 2-D map, and the
// "SLAM loop" explanations from the numbers the filter just computed.

import { LunarSlamSession, SLAM_SCENARIOS, covEllipse } from '../slam/engine.js';
import { createSlam3D } from './slam3d.js';
import { lineChart } from './charts.js';

const $ = (id) => document.getElementById(id);
const deg = (r) => (r * 180) / Math.PI;

const S = {
  scenario: SLAM_SCENARIOS[new URLSearchParams(location.search).get('scenario')] ? new URLSearchParams(location.search).get('scenario') : 'loop_home',
  seed: 3,
  speed: 2,
  paused: false,
  mapBy: 0, // 0 = SLAM's map, 1 = wheels-only map
  showTruth: true,
  showAllPaths: true,
  showFrustums: false,
  showRays: true,
  showErrorStruts: false,
  followRover: false,
  pauseAtLoop: false,
};

let sess = null;
let s3d = null;
let seenLoops = 0;
let lastCameraT = -1;
const keysDown = new Set();

function initSession() {
  const sc = SLAM_SCENARIOS[S.scenario];
  sess = new LunarSlamSession({ scenario: S.scenario, seed: S.seed });
  seenLoops = 0;
  lastCameraT = -1;
  $('scenario-desc').textContent = sc.desc;
  syncKnobs();
  buildCards();
  updatePanels();
}

function syncKnobs() {
  $('sl-sun').value = sess.sunElDeg;
  $('val-sun').textContent = `${sess.sunElDeg}°`;
  $('sl-slip').value = sess.wheelSlip;
  $('val-slip').textContent = `${Math.round(sess.wheelSlip * 100)}%`;
  $('sl-noise').value = sess.cameraNoise;
  $('val-noise').textContent = `${sess.cameraNoise.toFixed(1)}×`;
  const b = $('btn-cam');
  b.textContent = sess.useCamera ? 'ON' : 'OFF';
  b.className = sess.useCamera ? 'on' : 'off';
}

function buildCards() {
  $('vs').innerHTML = sess.estimators
    .map((est, i) => `
      <div class="tcard">
        <div class="nm" style="color:${est.color}">${est.name}</div>
        <div class="plain">${est.plain}</div>
        <div class="big" id="err-${i}" style="color:${est.color}">0.00 m</div>
        <div style="font-size:10.5px;color:#64748b">off from the truth right now</div>
        <div class="row"><span>it thinks ±</span><b id="sig-${i}">0.00 m</b></div>
        <div class="row"><span>heading off</span><b id="hdg-${i}">0.0°</b></div>
        <div class="row"><span>truth inside its ellipse</span><b id="in-${i}">100%</b></div>
        <div class="row"><span>landmarks on its map</span><b id="lm-${i}">0</b></div>
        <div class="row"><span>landmark error (avg)</span><b id="me-${i}">–</b></div>
      </div>`)
    .join('');
}

const f2 = (v) => v.toFixed(2);
const n = (k, word) => `<b>${k}</b> ${word}${k === 1 ? '' : 's'}`;

function explainLoop() {
  const p = sess.lastPredict, c = sess.lastCamera;
  if (p) {
    $('txt-predict').innerHTML =
      `The wheels say it moved <b>${(p.dist * 100).toFixed(1)} cm</b> and turned <b>${(p.turnDeg).toFixed(2)}°</b>. ` +
      `SLAM moves its guess by that much. Wheels slip, so the uncertainty can only grow here: ` +
      `<b>±${f2(2 * p.sigmaBefore)}</b> → <b>±${f2(2 * p.sigmaAfter)} m</b>.`;
  } else {
    $('txt-predict').innerHTML = 'Waiting for the rover to move.';
  }
  if (!c) return;
  $('txt-observe').innerHTML =
    `${n(c.inView, 'rock feature')} in view. The camera found <b>${c.used}</b> it can use` +
    (c.missed ? ` and missed <b>${c.missed}</b> (shadow or noise)` : '') + '. ' +
    (c.cameraOff
      ? 'Camera corrections are off, so they are ignored.'
      : `<b>${c.added}</b> new → put on the map where the rover thinks they are. <b>${c.corrected}</b> already on the map → each one is a check.`);
  let corr;
  if (c.cameraOff) corr = 'Nothing to compare: the rover trusts its wheels only, and the error grows without anyone noticing.';
  else if (!c.corrected) corr = 'No known landmark in view, so there is nothing to check against. The guess keeps drifting until one comes back into view.';
  else {
    const b = c.biggest;
    corr =
      `Landmark <b>#${b.id}</b> should be at <b>${f2(b.expectedRange)} m</b>, <b>${deg(b.expectedBearing).toFixed(1)}°</b>; ` +
      `the camera measured <b>${f2(b.measRange)} m</b>, <b>${deg(b.measBearing).toFixed(1)}°</b>. ` +
      `The difference (innovation) is <b>${b.innovRange >= 0 ? '+' : ''}${(b.innovRange * 100).toFixed(0)} cm</b>, <b>${deg(b.innovBearing) >= 0 ? '+' : ''}${deg(b.innovBearing).toFixed(2)}°</b>. ` +
      `Using all ${c.corrected} checks, SLAM nudged the rover <b>${(c.moved * 100).toFixed(1)} cm</b> and the landmarks a little, and its uncertainty shrank ` +
      `<b>±${f2(2 * c.sigmaBefore)}</b> → <b>±${f2(2 * c.sigmaAfter)} m</b>.`;
  }
  $('txt-correct').innerHTML = corr;
  if (c.t !== lastCameraT) {
    lastCameraT = c.t;
    for (const id of ['loop-observe', 'loop-correct']) {
      const el = $(id);
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 180);
    }
  }
}

function updatePanels() {
  if (!sess) return;
  $('clock').textContent = `T+${sess.t.toFixed(1).padStart(5, '0')} s`;
  sess.estimators.forEach((est, i) => {
    const m = est.metrics;
    $(`err-${i}`).textContent = `${f2(m.rawPoseErr)} m`;
    $(`sig-${i}`).textContent = `${f2(2 * m.poseSigmaM)} m`;
    $(`hdg-${i}`).textContent = `${(m.headingErrDeg || 0).toFixed(1)}°`;
    $(`in-${i}`).textContent = `${Math.round(m.insideRate * 100)}%`;
    $(`lm-${i}`).textContent = est.id === 'wheels_only' ? '0 (just a picture)' : `${m.mappedLandmarks}`;
    $(`me-${i}`).textContent = m.mappedLandmarks ? `${f2(m.mapRmse)} m` : '–';
  });
  const slam = sess.estimators[0];
  $('hud-active').innerHTML = `· SLAM is <b style="color:#38bdf8">${f2(slam.metrics.rawPoseErr)} m</b> off, thinks ±${f2(2 * slam.metrics.poseSigmaM)} m · driven ${sess.distanceDriven.toFixed(0)} m · lap ${sess.lap || 1}`;
  explainLoop();

  $('slam-feed').innerHTML = sess.events
    .slice(0, 30)
    .map((e) => `<div class="ev ${e.level}"><b>${e.t.toFixed(1).padStart(5)}s</b> ${e.msg}</div>`)
    .join('');

  const [a, b] = sess.estimators;
  lineChart($('chAte'), {
    title: '',
    tMin: 0,
    tMax: Math.max(20, sess.t),
    series: [
      { color: '#ef4444', width: 1.4, pts: b.ateHistory },
      { color: '#38bdf8', width: 1.2, dash: [4, 3], pts: a.sigmaHistory },
      { color: '#38bdf8', width: 2.2, pts: a.ateHistory },
    ],
  });
}

/** The rover's own map: its occupancy grid, its landmarks with ellipses, and where it thinks it is. */
function drawMind(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#050811';
  g.fillRect(0, 0, w, h);

  const est = sess.estimators[S.mapBy];
  const top = 28, bottom = 24;
  const scale = Math.min(w / 62, (h - top - bottom) / 61);
  const cx = w * 0.5, cy = top + 51.5 * scale;
  const P = (x, y) => [cx + x * scale, cy - y * scale];
  const ellipse = (x, y, cov, color, fill) => {
    const e = covEllipse(cov.xx, cov.xy, cov.yy, 2);
    const [sx, sy] = P(x, y);
    g.beginPath();
    g.ellipse(sx, sy, Math.max(1.5, e.a * scale), Math.max(1.5, e.b * scale), -e.angle, 0, Math.PI * 2);
    if (fill) { g.fillStyle = fill; g.fill(); }
    g.strokeStyle = color;
    g.stroke();
  };

  // Occupancy grid built from this estimator's believed pose.
  const grid = est.grid, cell = grid.cellSize * scale;
  for (let r = 0; r < grid.rows; r++) {
    for (let c = 0; c < grid.cols; c++) {
      const lo = grid.logOdds[r * grid.cols + c];
      if (Math.abs(lo) < 0.15) continue;
      const [sx, sy] = P(grid.xMin + c * grid.cellSize, grid.yMin + (r + 1) * grid.cellSize);
      g.fillStyle = lo < 0 ? 'rgba(16,185,129,0.13)' : 'rgba(245,158,11,0.75)';
      g.fillRect(sx, sy, cell + 0.5, cell + 0.5);
    }
  }
  // 10 m grid
  g.strokeStyle = '#111a2e';
  g.lineWidth = 1;
  for (let x = -30; x <= 30; x += 10) { const [sx] = P(x, 0); g.beginPath(); g.moveTo(sx, top); g.lineTo(sx, h - bottom); g.stroke(); }
  for (let y = -10; y <= 50; y += 10) { const [, sy] = P(0, y); g.beginPath(); g.moveTo(0, sy); g.lineTo(w, sy); g.stroke(); }

  // Truth overlay (the rover cannot see this).
  if (S.showTruth) {
    g.setLineDash([2, 3]);
    g.strokeStyle = 'rgba(226,232,240,0.35)';
    for (const b of sess.boulders) {
      const [bx, by] = P(b.x, b.y);
      g.beginPath(); g.arc(bx, by, b.r * scale, 0, Math.PI * 2); g.stroke();
    }
    g.setLineDash([]);
    if (sess.truthKeyframes.length > 1) {
      g.strokeStyle = 'rgba(226,232,240,0.8)';
      g.lineWidth = 1.4;
      g.beginPath();
      sess.truthKeyframes.forEach((k, i) => { const [x, y] = P(k.pos[0], -k.pos[2]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      const [rx, ry] = P(sess.rover.x, sess.rover.y);
      g.lineTo(rx, ry);
      g.stroke();
    }
  }

  // Loop closures (green): from where a landmark was first mapped to where it was recognised again.
  g.setLineDash([5, 4]);
  g.strokeStyle = '#34d399';
  g.lineWidth = 1.6;
  for (const lc of sess.loopClosures) {
    const [x0, y0] = P(lc.fromPos[0], -lc.fromPos[2]);
    const [x1, y1] = P(lc.toPos[0], -lc.toPos[2]);
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
  }
  g.setLineDash([]);

  // Landmarks on SLAM's map, each with its 2-sigma ellipse.
  if (S.mapBy === 0) {
    const now = new Map(sess.activeObservations.map((o) => [o.lm.id, o.status]));
    g.lineWidth = 1;
    for (const lm of est.mapLandmarks.values()) {
      if (lm.x === undefined) continue;
      const st = now.get(lm.id);
      const col = st === 'known' ? '#34d399' : st === 'new' ? '#f59e0b' : '#38bdf8';
      ellipse(lm.x, lm.y, { xx: lm.sigmaXX, xy: lm.sigmaXY, yy: lm.sigmaYY }, col + 'aa');
      const [lx, ly] = P(lm.x, lm.y);
      g.fillStyle = col;
      g.fillRect(lx - 1.8, ly - 1.8, 3.6, 3.6);
      if (S.showTruth) {
        const t = sess.landmarks[lm.id - 1];
        const [tx, ty] = P(t.x, t.y);
        g.strokeStyle = 'rgba(226,232,240,0.35)';
        g.beginPath(); g.moveTo(lx, ly); g.lineTo(tx, ty); g.stroke();
      }
      // What the camera is checking right now: a line from the believed pose to the landmark.
      if (st === 'known' || st === 'new') {
        const [px, py] = P(est.pose.x, est.pose.y);
        g.strokeStyle = st === 'known' ? 'rgba(52,211,153,0.55)' : 'rgba(245,158,11,0.55)';
        g.beginPath(); g.moveTo(px, py); g.lineTo(lx, ly); g.stroke();
      }
    }
  }

  // Believed paths.
  sess.estimators.forEach((e, i) => {
    if (i !== S.mapBy && !S.showAllPaths) return;
    if (e.keyframes.length < 2) return;
    g.strokeStyle = e.color;
    g.globalAlpha = i === S.mapBy ? 1 : 0.5;
    g.lineWidth = i === S.mapBy ? 2 : 1.2;
    g.beginPath();
    e.keyframes.forEach((k, j) => { const [x, y] = P(k.pos[0], -k.pos[2]); j ? g.lineTo(x, y) : g.moveTo(x, y); });
    const [px, py] = P(e.pose.x, e.pose.y);
    g.lineTo(px, py);
    g.stroke();
    g.globalAlpha = 1;
  });

  // Next target
  const tgt = sess.waypoint || sess.frontierTarget;
  if (tgt) {
    const [fx, fy] = P(tgt.x, tgt.y);
    g.strokeStyle = '#a78bfa';
    g.lineWidth = 1.5;
    g.beginPath(); g.arc(fx, fy, 5, 0, Math.PI * 2); g.stroke();
  }

  const tri = (x, y, th, color, outline) => {
    const [sx, sy] = P(x, y);
    g.save();
    g.translate(sx, sy);
    g.rotate(-th);
    g.beginPath(); g.moveTo(8, 0); g.lineTo(-5, -5); g.lineTo(-5, 5); g.closePath();
    if (outline) { g.strokeStyle = color; g.lineWidth = 1.5; g.stroke(); } else { g.fillStyle = color; g.fill(); }
    g.restore();
  };
  if (S.showTruth) tri(sess.rover.x, sess.rover.y, sess.rover.th, '#e2e8f0', true);
  g.lineWidth = 1.8;
  ellipse(est.pose.x, est.pose.y, est.poseCov, est.color, est.color + '26');
  tri(est.pose.x, est.pose.y, est.pose.th, est.color, false);

  // Legend + scale bar
  g.fillStyle = 'rgba(8,12,22,0.88)';
  g.fillRect(0, h - bottom, w, bottom);
  g.font = '10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  g.fillStyle = '#cbd5e1';
  const who = S.mapBy === 0 ? 'SLAM' : 'wheels only';
  g.fillText(
    `map by ${who}: ${est.grid.coveragePct.toFixed(0)}% of the area seen · ${S.mapBy === 0 ? `${est.mapLandmarks.size} landmarks · ` : ''}` +
    `${S.showTruth ? 'white = truth' : 'truth hidden'} · ▲ belief, ellipse = ±2σ`,
    8, h - 8
  );
  const [s0] = P(20, 0), [s1] = P(30, 0);
  g.strokeStyle = '#cbd5e1';
  g.lineWidth = 2;
  g.beginPath(); g.moveTo(s0, top + 6); g.lineTo(s1, top + 6); g.stroke();
  g.fillText('10 m', s0 + 2, top + 18);
}

function syncManualDrive() {
  const up = keysDown.has('KeyW') || keysDown.has('ArrowUp');
  const dn = keysDown.has('KeyS') || keysDown.has('ArrowDown');
  const lt = keysDown.has('KeyA') || keysDown.has('ArrowLeft');
  const rt = keysDown.has('KeyD') || keysDown.has('ArrowRight');
  sess.manualControl.active = up || dn || lt || rt;
  sess.manualControl.v = (up ? 1.1 : 0) + (dn ? -0.7 : 0);
  sess.manualControl.w = (lt ? 0.58 : 0) + (rt ? -0.58 : 0);
}

function setPaused(p) {
  S.paused = p;
  $('play').dataset.state = p ? 'paused' : 'running';
  $('play').textContent = p ? '▶ Resume' : '⏸ Pause';
}

function stepSim() {
  sess.step();
  if (sess.loopClosures.length > seenLoops) {
    seenLoops = sess.loopClosures.length;
    if (S.pauseAtLoop) {
      setPaused(true);
      $('chk-pause-loop').checked = S.pauseAtLoop = false;
    }
  }
}

async function boot() {
  const sel = $('scenario');
  sel.innerHTML = Object.entries(SLAM_SCENARIOS)
    .map(([k, v]) => `<option value="${k}" ${k === S.scenario ? 'selected' : ''}>${v.label}</option>`)
    .join('');
  sel.addEventListener('change', () => { S.scenario = sel.value; initSession(); });
  $('seed').addEventListener('change', () => { S.seed = Math.max(1, +$('seed').value || 3); initSession(); });

  const spd = $('speed');
  [1, 2, 4, 8].forEach((m) => {
    const b = document.createElement('button');
    b.textContent = `${m}×`;
    if (m === S.speed) b.className = 'on';
    b.addEventListener('click', () => {
      S.speed = m;
      spd.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
    });
    spd.appendChild(b);
  });

  $('play').addEventListener('click', () => setPaused(!S.paused));
  $('stepBtn').addEventListener('click', () => {
    setPaused(true);
    // Run until the next camera frame so all three steps of the loop are fresh.
    const t0 = sess.lastCamera ? sess.lastCamera.t : -1;
    for (let i = 0; i < 5 && (!sess.lastCamera || sess.lastCamera.t === t0); i++) stepSim();
    updatePanels();
  });
  $('reset').addEventListener('click', initSession);
  $('togR').addEventListener('click', () => document.body.classList.toggle('hide-right'));

  $('sl-sun').addEventListener('input', (e) => { sess.sunElDeg = +e.target.value; $('val-sun').textContent = `${sess.sunElDeg}°`; });
  $('sl-slip').addEventListener('input', (e) => { sess.wheelSlip = +e.target.value; $('val-slip').textContent = `${Math.round(sess.wheelSlip * 100)}%`; });
  $('sl-noise').addEventListener('input', (e) => { sess.cameraNoise = +e.target.value; $('val-noise').textContent = `${sess.cameraNoise.toFixed(1)}×`; });
  $('btn-cam').addEventListener('click', () => {
    sess.useCamera = !sess.useCamera;
    sess.pushEvent(sess.useCamera ? 'Camera corrections back on.' : 'Camera corrections off: SLAM now runs on wheels only.', sess.useCamera ? 'good' : 'warn');
    syncKnobs();
  });
  $('chk-pause-loop').addEventListener('change', (e) => { S.pauseAtLoop = e.target.checked; });

  // "Try this" buttons
  $('try-loop').addEventListener('click', () => {
    S.scenario = 'loop_home';
    sel.value = S.scenario;
    initSession();
    S.pauseAtLoop = $('chk-pause-loop').checked = true;
    $('speed').querySelectorAll('button')[2].click();
    setPaused(false);
  });
  $('try-cam').addEventListener('click', () => { if (sess.useCamera) $('btn-cam').click(); });
  $('try-slip').addEventListener('click', () => { $('sl-slip').value = 0.2; $('sl-slip').dispatchEvent(new Event('input')); });
  $('try-sun').addEventListener('click', () => { $('sl-sun').value = 3; $('sl-sun').dispatchEvent(new Event('input')); });

  $('btnMapBy').addEventListener('click', (e) => {
    S.mapBy = 1 - S.mapBy;
    e.target.textContent = `Map by: ${S.mapBy ? 'Wheels only' : 'SLAM'}`;
    e.target.classList.toggle('on', S.mapBy === 0);
  });
  $('btnTruth').addEventListener('click', (e) => {
    S.showTruth = !S.showTruth;
    e.target.textContent = `Truth overlay: ${S.showTruth ? 'on' : 'off'}`;
    e.target.classList.toggle('on', S.showTruth);
  });
  const toggle = (id, prop, label) => {
    $(id).addEventListener('click', (e) => {
      S[prop] = !S[prop];
      e.target.classList.toggle('on', S[prop]);
      e.target.textContent = `${label}: ${S[prop] ? 'on' : 'off'}`;
    });
  };
  toggle('btnRays', 'showRays', '⚡ Camera rays');
  toggle('btnFrustums', 'showFrustums', '📐 Camera poses');
  toggle('btnStruts', 'showErrorStruts', '📏 Error lines');
  toggle('btnAllPaths', 'showAllPaths', '🧭 Wheels path');
  toggle('btnFollow', 'followRover', '🎥 Follow');
  $('btn-reset-cam').addEventListener('click', () => s3d && s3d.resetCamera());

  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); setPaused(!S.paused); }
    else if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
      e.preventDefault();
      keysDown.add(e.code);
      syncManualDrive();
    }
  });
  window.addEventListener('keyup', (e) => { if (keysDown.delete(e.code)) syncManualDrive(); });

  initSession();

  s3d = await createSlam3D({
    canvas: $('glAll'),
    mainEl: $('mapwrap'),
    camEl: $('camwrap'),
    mainOverlay: $('ov3d'),
    camOverlay: $('ovcam'),
    onGroundClick: (x, y) => {
      sess.waypoint = { x, y };
      sess.pushEvent(`Waypoint set at (${x.toFixed(1)}, ${y.toFixed(1)}) m. The rover will drive to where it believes that is.`, 'info');
    },
  });
  $('load-badge').hidden = true;

  let lastUi = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (!S.paused) for (let k = 0; k < S.speed && !S.paused; k++) stepSim();
    s3d.render(sess, {
      selectedIdx: 0,
      showAllPaths: S.showAllPaths,
      showFrustums: S.showFrustums,
      showRays: S.showRays,
      showLandmarks: true,
      showErrorStruts: S.showErrorStruts,
      robotMapOnly: false,
      followRover: S.followRover,
    });
    drawMind($('poseMap'));
    if (now - lastUi > 150) {
      updatePanels();
      lastUi = now;
    }
  }
  requestAnimationFrame(frame);
}

boot();
