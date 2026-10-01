import { Mission } from '../sim/mission.js';
import { listPlatforms, getPlatform } from '../platforms/index.js';
import { listFilters, getFilterClass, unregisterFilter } from '../filters/index.js';
import { neesBounds } from '../sim/metrics.js';
import { MapView } from './render.js';
import { lineChart } from './charts.js';
import { TEMPLATE, compileFilter } from './lab.js';
import { LESSONS } from './lessons.js';
import { coach, healthHtml, BRIEF } from './coach.js';
import { runTour } from './tour.js';

const $ = (id) => document.getElementById(id);
const S = {
  platform: 'rover', scenario: 'nominal', seed: 7, speed: 2, paused: false,
  enabled: new Set(['dr', 'ekf', 'ukf', 'pf']), primary: 'ekf', custom: [],
  ui: { showTruth: true, showCov: true, showParticles: true, colors: {} }, keys: {},
  safing: true, delay: 0, view: '3d',
};
const views3d = {}; // platform id -> 3-D view (each loads on its own; the 2-D map always works)
const VIEW3D_LABEL = { rover: '3D · NASA Perseverance', spacecraft: '3D · NASA Gateway' };
// one WebGL canvas per 3-D view: two renderers sharing a canvas share one GL context and corrupt each other's state
const VIEW3D_CANVAS = { rover: 'gl3d', spacecraft: 'gl3dS' };
let m, view;

const colorsNow = () => { for (const c of listFilters()) S.ui.colors[c.id] = c.color; };

function newMission() {
  colorsNow();
  m = new Mission({ platform: S.platform, scenario: S.scenario, seed: S.seed, filters: [...S.enabled], primary: S.primary });
  m.safing = S.safing; m.uplinkDelay = S.delay;
  S.primary = m.primary;
  view.setWorld(m.world);
  buildFilters(); buildFaults(); buildHud(); $('banner').hidden = true;
}

// ------------------------------------------------------------------ header / selectors
function buildSelectors() {
  $('platform').innerHTML = listPlatforms().map((p) => `<option value="${p.id}">${p.label}</option>`).join('');
  $('platform').value = S.platform;
  fillScenarios();
  $('speed').innerHTML = [0.5, 1, 2, 4, 8, 16].map((v) => `<button data-v="${v}" class="${v === S.speed ? 'on' : ''}">${v}×</button>`).join('');
}
function fillScenarios() {
  const sc = getPlatform(S.platform).scenarios;
  if (!sc[S.scenario]) S.scenario = Object.keys(sc)[0];
  $('scenario').innerHTML = Object.entries(sc).map(([id, s]) => `<option value="${id}" title="${s.blurb}">${s.label}</option>`).join('');
  $('scenario').value = S.scenario;
}

// ------------------------------------------------------------------ panels
function buildFaults() {
  const P = m.platform;
  $('faults').innerHTML = Object.entries(P.faults).map(([k, label]) => `<button data-f="${k}">${label}</button>`).join('');
}

function buildFilters() {
  $('filters').innerHTML = listFilters().map((c) => `
    <div class="frow">
      <input type="checkbox" data-en="${c.id}" ${S.enabled.has(c.id) ? 'checked' : ''}>
      <span class="dot" style="background:${c.color}"></span>
      <label for="">${c.label}</label>
      <span class="pri ${S.primary === c.id ? 'on' : ''}" data-pri="${c.id}" title="Drive the autopilot with this estimator">${S.primary === c.id ? '★ PRIMARY' : 'make primary'}</span>
      <small>${c.blurb}</small>
    </div>`).join('');
  buildTuning();
  $('legend').innerHTML = `<div><i style="background:#fff"></i>ground truth</div>` + [...S.enabled].filter((id) => getFilterClass(id)).map((id) => `<div><i style="background:${S.ui.colors[id]}"></i>${getFilterClass(id).label}</div>`).join('');
}

function buildTuning() {
  const f = m.fs.get(m.primary)?.filter, spec = f?.constructor.paramSpec || {};
  $('tuning').innerHTML = `<h3 style="margin-top:8px">Tune ${f?.constructor.label ?? ''}</h3>` + (Object.keys(spec).length ? Object.entries(spec).map(([k, p]) => `
    <label class="slider">${p.label} <b id="tv-${k}">${(+f.params[k]).toFixed(1)}</b><input type="range" data-tune="${k}" min="${p.min}" max="${p.max}" step="${p.step}" value="${f.params[k]}"></label>`).join('') : '<p class="hint">No tunable parameters.</p>');
}

function buildHud() {
  $('resName').textContent = m.platform.resource.name;
  $('objectives').innerHTML = m.targets.map((t) => `<div id="obj${t.id}"><span>◯</span>${m.platform.objective.noun} ${t.id}</div>`).join('') || '<p class="hint">Free play: no objectives.</p>';
  $('neesHint').innerHTML = `ANEES should sit near the filter's degrees of freedom (<b>${m.model.neesIdx.length}</b>). <span class="over">red</span> = overconfident, <span class="under">amber</span> = conservative.`;
  $('events').innerHTML = '';
  S.evCount = 0;
}

let lastEvents = 0;
function renderCoach() {
  const c = coach(m), el = $('coach');
  const html = `<b class="t">${c.title}</b><p>${c.body}</p>` + (c.cta ? `<div class="cta">${c.cta.map(([l, a]) => `<button data-act="${a}">${l}</button>`).join('')}</div>` : '');
  if (el.dataset.k !== html) { el.dataset.k = html; el.className = 'coach ' + c.tone; el.innerHTML = html; }
}
function tour() {
  $('brief').hidden = true; $('modal').hidden = true;
  runTour({
    prepare() { S.platform = 'rover'; $('platform').value = 'rover'; fillScenarios(); S.scenario = 'nominal'; $('scenario').value = 'nominal'; S.primary = 'ekf'; S.enabled = new Set(['dr', 'ekf', 'ukf', 'pf']); S.ui.showTruth = true; $('optTruth').checked = true; newMission(); S.paused = true; $('play').textContent = '▶ Play'; setTimeout(() => view.resize(), 0); },
    start() { startMission(); },
    fault(k) { m.injectFault(k); },
    stop() { m.issue({ type: 'stop' }); },
    truth(v) { S.ui.showTruth = v; $('optTruth').checked = v; },
    dr() { m.setPrimary('dr'); S.primary = 'dr'; buildFilters(); m.issue({ type: 'plan' }); S.paused = false; $('play').textContent = '⏸ Pause'; },
  });
}
function startMission() {
  m.issue({ type: 'plan' }); S.paused = false; $('play').textContent = '⏸ Pause';
  S.speed = Math.max(S.speed, 4); [...$('speed').children].forEach((b) => b.classList.toggle('on', +b.dataset.v === S.speed));
}
function updateHud() {
  const st = $('status'); st.textContent = m.status.toUpperCase(); st.className = 'badge ' + m.status;
  $('clock').textContent = `T+${m.t.toFixed(1).padStart(5, '0')} s`;
  const bar = (id, v, val) => { $(id + 'Bar').style.width = v + '%'; $(id + 'Bar').style.background = v > 50 ? 'var(--good)' : v > 25 ? 'var(--warn)' : 'var(--bad)'; $(id + 'Val').textContent = Math.round(v) + '%'; };
  bar('hull', m.hull); bar('res', m.resource);
  $('score').textContent = m.score;
  for (const t of m.targets) { const el = $('obj' + t.id); if (el) { el.className = t.done ? 'done' : ''; el.firstChild.textContent = t.done ? '✔' : '◯'; } }
  $('faultChips').innerHTML = m.activeFaults().map((f) => `<span class="chip">${m.platform.faultLabel(f.type)} · ${f.left.toFixed(0)}s</span>`).join('') + (m.safeHold ? '<span class="chip" style="background:#4c1219;color:#ff9aa4">SAFE-HOLD</span>' : '') + (m.queue.length ? `<span class="chip" style="background:#0d3a52;color:#4cc9f0">${m.queue.length} cmd in flight</span>` : '');
  if (m.events.length !== lastEvents) {
    lastEvents = m.events.length;
    $('events').innerHTML = m.events.slice(-30).reverse().map((e) => `<div class="ev ${e.level}"><b>${e.t.toFixed(0).padStart(4)}s</b> ${e.text}</div>`).join('');
  }
  const ban = $('banner');
  if (m.status !== 'running') { ban.hidden = false; ban.className = m.status; ban.textContent = m.status === 'complete' ? `MISSION COMPLETE · SCORE ${m.score}` : `MISSION FAILED · ${m.failReason.toUpperCase()}`; } else ban.hidden = true;

  $('health').innerHTML = healthHtml(m); renderCoach();
  const dof = m.model.neesIdx.length;
  $('score-table').tBodies[0].innerHTML = [...m.fs.values()].map((f) => {
    const s = m.summary(f.id); if (!s) return '';
    const cls = s.anees > 2 * dof ? 'over' : s.anees < 0.4 * dof ? 'under' : 'ok';
    const c = S.ui.colors[f.id];
    return `<tr><td><span class="tag" style="background:${c}"></span>${f.Cls.label.replace(' Kalman', '').replace('Particle Filter', 'PF').replace('Dead Reckoning', 'DR')}</td><td>${fmtLen(s.last.pos)}</td><td>${fmtLen(3 * s.last.sig)}</td><td>${fmtLen(s.rmse)}</td><td class="${cls}">${Number.isFinite(s.anees) ? s.anees.toFixed(1) : '–'}</td><td>${s.rejected}</td><td>${s.usPerStep.toFixed(0)}</td></tr>`;
  }).join('');
}
const fmtLen = (v) => (!Number.isFinite(v) ? '–' : v >= 1000 ? (v / 1000).toFixed(2) + 'k' : v >= 100 ? v.toFixed(0) : v.toFixed(2));

function updateCharts() {
  const h = m.hist, tMax = m.t, tMin = Math.max(0, tMax - 150);
  const win = h.filter((e) => e.t >= tMin);
  lineChart($('chErr'), { title: `position error [${m.platform.unit}] · all filters`, tMin, tMax, series: [...m.fs.keys()].map((id) => ({ color: S.ui.colors[id], width: id === m.primary ? 2 : 1.2, pts: win.map((e) => [e.t, e.f[id]?.pos]) })) });
  lineChart($('chSig'), { title: `primary: error vs 3σ bound [${m.platform.unit}]`, tMin, tMax, series: [
    { color: '#ffb454', dash: [4, 3], pts: win.map((e) => [e.t, 3 * (e.f[m.primary]?.sig ?? NaN)]) },
    { color: S.ui.colors[m.primary], width: 1.8, pts: win.map((e) => [e.t, e.f[m.primary]?.pos]) },
  ] });
  const [lo, hi] = neesBounds(m.model.neesIdx.length);
  lineChart($('chNees'), { title: 'primary: NEES vs 95% χ² band (green)', tMin, tMax, yMax: hi * 2.5, bands: [{ y0: lo, y1: hi, color: '#4ade8022' }], series: [{ color: S.ui.colors[m.primary], pts: win.map((e) => [e.t, e.f[m.primary]?.nees]) }] });
}

// ------------------------------------------------------------------ main loop
let acc = 0, last = performance.now(), hudT = 0, chT = 0;
/** Show a visible, copyable message instead of failing silently. */
function showError(where, err) {
  console.error(`[${where}]`, err);
  let el = document.getElementById('err3d');
  if (!el) {
    el = document.createElement('div'); el.id = 'err3d';
    el.style.cssText = 'position:absolute;left:50%;top:60px;transform:translateX(-50%);z-index:20;max-width:80%;background:#4c1219;color:#ffd7db;border:1px solid #ff5d6c;border-radius:8px;padding:8px 12px;font:12px/1.4 ui-monospace,monospace;user-select:text';
    (document.getElementById('mapwrap') || document.body).appendChild(el);
  }
  el.textContent = `3-D view stopped, switched to the 2-D map. Error: ${err?.message || err}`;
}
function frame(now) {
  requestAnimationFrame(frame); // schedule first: one bad frame must never stop the loop
  try { frameBody(now); } catch (err) { showError('frame', err); }
}
function frameBody(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (!S.paused) {
    acc += dt * S.speed;
    let n = 0;
    while (acc >= m.dt && n < 800) { m.step(); acc -= m.dt; n++; }
    if (n === 800) acc = 0;
  }
  const v3 = views3d[m.platform.id], use3d = S.view === '3d' && !!v3;
  $('map').style.display = use3d ? 'none' : 'block'; $('ov3d').style.display = use3d ? 'block' : 'none';
  for (const [pid, cid] of Object.entries(VIEW3D_CANVAS)) $(cid).style.display = use3d && pid === m.platform.id ? 'block' : 'none';
  $('viewsw').style.display = v3 ? 'flex' : 'none';
  if (v3 && $('viewsw').firstElementChild.textContent !== VIEW3D_LABEL[m.platform.id]) $('viewsw').firstElementChild.textContent = VIEW3D_LABEL[m.platform.id];
  if (use3d) {
    try { v3.render(m, S.ui); }
    catch (err) { showError('3-D view', err); delete views3d[m.platform.id]; S.view = '2d'; [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b.dataset.v === '2d')); setTimeout(() => view.resize(), 0); }
  } else view.draw(m, S.ui);
  if (now - hudT > 150) { hudT = now; updateHud(); }
  if (now - chT > 300) { chT = now; updateCharts(); }

}

// ------------------------------------------------------------------ input
function wire() {
  $('platform').onchange = (e) => {
    for (const id of S.custom) { unregisterFilter(id); S.enabled.delete(id); } S.custom = [];
    S.platform = e.target.value; fillScenarios(); S.scenario = $('scenario').value; newMission();
    setTimeout(() => { view.resize(); view.setWorld(m.world); }, 0);
  };
  $('scenario').onchange = (e) => { S.scenario = e.target.value; newMission(); };
  $('reset').onclick = newMission;
  $('play').onclick = () => { S.paused = !S.paused; $('play').textContent = S.paused ? '▶ Play' : '⏸ Pause'; };
  $('speed').onclick = (e) => { const v = e.target.dataset.v; if (!v) return; S.speed = +v; [...$('speed').children].forEach((b) => b.classList.toggle('on', b === e.target)); };
  $('learn').onclick = () => { $('modal').hidden = false; };
  $('modalClose').onclick = () => { $('modal').hidden = true; };
  $('modal').onclick = (e) => { if (e.target.id === 'modal') $('modal').hidden = true; };
  $('lessons').innerHTML = LESSONS;
  $('btnPlan').onclick = () => m.issue({ type: 'plan' });
  $('coach').onclick = (e) => {
    const a = e.target.dataset.act; if (!a) return;
    if (a === 'start') startMission(); else if (a === 'reset') newMission(); else if (a === 'stop') m.issue({ type: 'stop' });
    else if (a === 'learn') $('modal').hidden = false; else if (a === 'ekf') { m.setPrimary('ekf'); S.primary = 'ekf'; buildFilters(); }
  };
  $('btnUndo').onclick = () => m.issue({ type: 'undo' });
  $('btnStop').onclick = () => m.issue({ type: 'stop' });
  $('stopTop').onclick = () => m.issue({ type: 'stop' });
  $('tourBtn').onclick = tour;
  $('faults').onclick = (e) => { const f = e.target.dataset.f; if (f) m.injectFault(f); };
  $('btnCsv').onclick = exportCsv;
  const opt = (id, fn) => { $(id).onchange = (e) => fn(e.target.checked); };
  opt('optSafing', (v) => { S.safing = m.safing = v; });
  opt('optTruth', (v) => { S.ui.showTruth = v; });
  opt('optCov', (v) => { S.ui.showCov = v; });
  opt('optParticles', (v) => { S.ui.showParticles = v; });
  opt('optFollow', (v) => { view.follow = v; });
  $('delay').oninput = (e) => { S.delay = m.uplinkDelay = +e.target.value; $('delayVal').textContent = S.delay + ' s'; };

  $('filters').onclick = (e) => {
    const en = e.target.dataset.en, pri = e.target.dataset.pri;
    if (en) {
      if (e.target.checked) { S.enabled.add(en); m.addFilter(en); }
      else if (m.fs.size > 1) { S.enabled.delete(en); m.removeFilter(en); S.primary = m.primary; }
      else e.target.checked = true;
      buildFilters();
    } else if (pri) { if (!m.fs.has(pri)) { S.enabled.add(pri); m.addFilter(pri); } m.setPrimary(pri); S.primary = pri; buildFilters(); }
  };
  $('tuning').oninput = (e) => {
    const k = e.target.dataset.tune; if (!k) return;
    m.fs.get(m.primary).filter.setParam(k, +e.target.value); $('tv-' + k).textContent = (+e.target.value).toFixed(1);
  };

  // map interaction
  const cv = $('map'); let drag = null;
  cv.onmousedown = (e) => { drag = { x: e.offsetX, y: e.offsetY, moved: false }; };
  window.onmouseup = (e) => {
    if (drag && !drag.moved && e.target === cv) clickMap(e.offsetX, e.offsetY);
    drag = null;
  };
  cv.onmousemove = (e) => {
    if (!drag) return;
    const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 5) return;
    drag.moved = true; view.panBy(dx, dy); drag.x = e.offsetX; drag.y = e.offsetY;
    if (view.follow) { view.follow = false; $('optFollow').checked = false; }
  };
  cv.onwheel = (e) => { e.preventDefault(); view.zoom = Math.min(30, Math.max(0.4, view.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); };
  cv.oncontextmenu = (e) => { e.preventDefault(); m.issue({ type: 'undo' }); };

  // keyboard
  const typing = (e) => ['TEXTAREA', 'INPUT', 'SELECT'].includes(e.target.tagName) && e.target.type !== 'checkbox' && e.target.type !== 'range';
  const KEYMAP = { ArrowUp: 'up', w: 'up', ArrowDown: 'down', s: 'down', ArrowLeft: 'left', a: 'left', ArrowRight: 'right', d: 'right' };
  window.onkeydown = (e) => {
    if (typing(e)) return;
    const k = KEYMAP[e.key]; 
    if (k) { e.preventDefault(); if (!S.keys[k]) { S.keys[k] = true; drive(); } return; }
    if (e.key === ' ') { e.preventDefault(); $('play').click(); }
    else if (e.key === 'r') newMission();
    else if (e.key === 'p') $('btnPlan').click();
    else if (e.key === 'Escape') $('btnStop').click();
    else if (e.key === 'Backspace') { e.preventDefault(); $('btnUndo').click(); }
    else if (e.key === 't') { $('optTruth').click(); }
    else if (e.key === 'f') { $('optFollow').click(); }
  };
  window.onkeyup = (e) => { const k = KEYMAP[e.key]; if (k && S.keys[k]) { S.keys[k] = false; drive(); } };
  window.onresize = () => view.resize();
  $('viewsw').onclick = (e) => { const v = e.target.dataset.v; if (!v) return; S.view = v; [...$('viewsw').children].forEach((b) => b.classList.toggle('on', b === e.target)); if (v === '2d') setTimeout(() => view.resize(), 0); };
  // 3-D: click the ground (without dragging) to send a waypoint; near a flag = target that sample
  let down3d = null;
  for (const cid of Object.values(VIEW3D_CANVAS)) $(cid).addEventListener('pointerdown', (e) => { down3d = { x: e.offsetX, y: e.offsetY }; });
  for (const cid of Object.values(VIEW3D_CANVAS)) $(cid).addEventListener('pointerup', (e) => {
    const v3 = views3d[m.platform.id];
    if (!down3d || !v3 || Math.hypot(e.offsetX - down3d.x, e.offsetY - down3d.y) > 5) { down3d = null; return; }
    down3d = null;
    const p = v3.pick(e.offsetX, e.offsetY); if (!p) return;
    const near = m.targets.find((t) => !t.done && Math.hypot(t.x - p.x, t.y - p.y) < 3);
    m.issue(near ? { type: 'goto', x: near.x, y: near.y, kind: 'objective', targetId: near.id } : { type: 'goto', x: p.x, y: p.y, kind: 'goto' });
  });
  if (window.ResizeObserver) new ResizeObserver(() => view.resize()).observe($('mapwrap')); // coach text, lab drawer, window: keep the fit exact

  // lab
  $('labToggle').onclick = () => { $('lab').classList.toggle('collapsed'); setTimeout(() => view.resize(), 0); };
  let saved = null; try { saved = localStorage.getItem('labCode'); } catch {}
  $('labCode').value = saved || TEMPLATE;
  $('labReset').onclick = () => { $('labCode').value = TEMPLATE; $('labMsg').textContent = 'Template restored.'; };
  $('labCode').onkeydown = (e) => { if (e.key === 'Tab') { e.preventDefault(); const t = e.target, s = t.selectionStart; t.setRangeText('  ', s, t.selectionEnd, 'end'); } };
  $('labRun').onclick = () => {
    const code = $('labCode').value;
    try { localStorage.setItem('labCode', code); } catch {}
    try {
      const Cls = compileFilter(code, m.model, m.sensors);
      S.custom.push(Cls.id); S.enabled.add(Cls.id); colorsNow(); m.addFilter(Cls.id);
      $('labMsg').textContent = `✔ "${Cls.label}" compiled and running.\nIt starts from the primary's current belief; compare its RMSE and ANEES in the scoreboard.`;
      buildFilters();
    } catch (err) { $('labMsg').textContent = '✖ ' + err.message; }
  };
}

const drive = () => m.issue({ type: 'drive', cmd: m.platform.manualCmd(S.keys) });

function clickMap(px, py) {
  const [x, y] = view.s2w(px, py, view.center(m));
  const near = m.targets.find((t) => !t.done && Math.hypot(...view.w2s(t.x, t.y, view.center(m)).map((v, i) => v - [px, py][i])) < 16);
  m.issue(near ? { type: 'goto', x: near.x, y: near.y, kind: 'objective', targetId: near.id } : { type: 'goto', x, y, kind: 'goto' });
}

function exportCsv() {
  const ids = [...m.fs.keys()], rows = [['t', 'truth_x', 'truth_y', ...ids.flatMap((i) => [`${i}_x`, `${i}_y`, `${i}_err`, `${i}_3sigma`, `${i}_nees`])]];
  for (const e of m.hist) rows.push([e.t.toFixed(2), e.truth.x, e.truth.y, ...ids.flatMap((i) => { const f = e.f[i]; return f ? [f.x, f.y, f.pos, 3 * f.sig, f.nees] : ['', '', '', '', '']; })]);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.map((r) => r.join(',')).join('\n')], { type: 'text/csv' }));
  a.download = `mission_${m.platform.id}_${m.world.id}_seed${S.seed}.csv`; a.click();
}

// ------------------------------------------------------------------ boot
view = new MapView($('map'));
buildSelectors(); wire();
for (const [id, file, fn] of [['rover', './mars3d.js', 'createMars3D'], ['spacecraft', './space3d.js', 'createSpace3D']]) {
  import(file)
    .then((mod) => mod[fn]({ canvas: $(VIEW3D_CANVAS[id]), overlay: $('ov3d') }))
    .then((v) => { views3d[id] = v; })
    .catch((err) => showError(`3-D ${id} load`, err));
}
requestAnimationFrame(() => {
  view.resize(); newMission(); requestAnimationFrame(frame);
  const demo = new URLSearchParams(location.search).has('demo');
  let seen = false; try { seen = localStorage.getItem('briefSeen') === '1'; } catch {}
  $('briefBody').innerHTML = BRIEF;
  const close = (go) => { $('brief').hidden = true; try { localStorage.setItem('briefSeen', '1'); } catch {} if (go) startMission(); };
  $('briefGo').onclick = () => close(true); $('briefSkip').onclick = () => close(false); $('briefTour').onclick = () => { close(false); tour(); };
  if (demo) startMission(); else if (!seen) $('brief').hidden = false;
  $('learn').ondblclick = () => { $('brief').hidden = false; };
});
window.__view = () => view;
window.__mission = () => m; // handy in devtools: __mission().fs.get('ekf').filter.P
