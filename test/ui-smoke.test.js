// Drives the real UI modules against a stub DOM/canvas to catch runtime errors in render, HUD, charts, Lab.
// (Not a visual test: it proves the code paths execute without throwing on every platform.)
import test from 'node:test';
import assert from 'node:assert/strict';

const sink = () => new Proxy(function () {}, {
  get: (t, k) => (k === Symbol.toPrimitive ? () => 0 : k in t ? t[k] : (t[k] = typeof k === 'string' && /^(width|height|offsetX|offsetY)$/.test(k) ? 800 : sink())),
  set: (t, k, v) => { t[k] = v; return true; },
  apply: () => sink(),
});
const els = new Map();
const el = (id) => {
  if (!els.has(id)) {
    const e = { id, style: {}, dataset: {}, classList: { toggle() {}, add() {}, remove() {} }, tBodies: [{}], children: [], firstChild: {},
      getBoundingClientRect: () => ({ width: 800, height: 600 }), getContext: () => sink(), appendChild() {}, click() { this.onclick?.({ target: this }); },
      value: '', textContent: '', innerHTML: '', checked: true, hidden: false };
    els.set(id, e);
  }
  return els.get(id);
};
let rafQueue = [];
globalThis.window = globalThis;
globalThis.document = { getElementById: el, createElement: () => el('tmp'), addEventListener() {} };
globalThis.devicePixelRatio = 1;
globalThis.location = { search: '' };
globalThis.requestAnimationFrame = (f) => rafQueue.push(f);
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.URL = Object.assign(URL, { createObjectURL: () => 'blob:x' });

let clock = performance.now() + 1000; // monotonic fake clock shared across tick() calls
const tick = (n, dtMs = 100) => { for (let i = 0; i < n; i++) { clock += dtMs; const q = rafQueue; rafQueue = []; q.forEach((f) => f(clock)); } };

test('UI boots, runs and survives every platform, scenario, filter toggle and the Filter Lab', async () => {
  await import('../src/ui/app.js');
  tick(5);
  const m0 = () => window.__mission();
  assert.ok(m0().t > 0, 'mission is advancing');

  for (const plat of ['rover', 'jet', 'spacecraft']) {
    el('platform').onchange({ target: { value: plat } });
    tick(3);
    el('btnPlan').onclick();
    for (const sc of Object.keys(m0().platform.scenarios)) {
      el('scenario').onchange({ target: { value: sc } });
      el('btnPlan').onclick();
      tick(30, 200); // ~6 s of wall * speed 2
      assert.equal(m0().platform.id, plat);
      for (const f of Object.keys(m0().platform.faults)) el('faults').onclick({ target: { dataset: { f } } });
      tick(10, 200);
    }
    // toggle filters, change primary, tune, click on map, keys
    el('filters').onclick({ target: { dataset: { en: 'pf' }, checked: false } });
    el('filters').onclick({ target: { dataset: { en: 'pf' }, checked: true } });
    el('filters').onclick({ target: { dataset: { pri: 'ukf' } } });
    el('tuning').oninput({ target: { dataset: { tune: 'qScale' }, value: '10' } });
    el('map').onmousedown({ offsetX: 300, offsetY: 300 });
    window.onmouseup({ target: el('map') });
    window.onkeydown({ key: 'w', target: { tagName: 'BODY' }, preventDefault() {} });
    window.onkeyup({ key: 'w' });
    tick(10);
    const mm = m0();
    assert.ok(mm.summary(mm.primary), `${plat}: no summary; t=${mm.t} status=${mm.status} n=${mm.fs.get(mm.primary).m.n} primary=${mm.primary} reason=${mm.failReason}`);
    assert.ok(Number.isFinite(mm.summary(mm.primary).rmse));
    el('btnCsv').onclick();
  }

  // Filter Lab: starter template compiles and races on the current platform
  el('labRun').onclick();
  assert.match(el('labMsg').textContent, /compiled and running/, el('labMsg').textContent);
  tick(20);
  const custom = [...m0().fs.keys()].find((k) => k.startsWith('custom'));
  assert.ok(custom && Number.isFinite(m0().summary(custom).rmse), 'custom filter produces estimates');

  // a broken filter reports an error instead of crashing the app
  el('labCode').value = 'return { name: "bad", init(){}, predict(){}, update(){}, getState(){ return [NaN]; }, getCov(){ return null; } };';
  el('labRun').onclick();
  assert.match(el('labMsg').textContent, /✖/);
  tick(5);
});
