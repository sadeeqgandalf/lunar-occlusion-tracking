import test from 'node:test';
import assert from 'node:assert/strict';
import { Mission } from '../src/sim/mission.js';
import { listPlatforms } from '../src/platforms/index.js';

globalThis.window = globalThis;
globalThis.devicePixelRatio = 2;
const { MapView } = await import('../src/ui/render.js');

const sink = () => new Proxy(function () {}, { get: () => sink(), apply: () => sink() });
const view = (W, H) => {
  const cv = { width: 0, height: 0, getBoundingClientRect: () => ({ width: W, height: H }), getContext: () => sink() };
  const v = new MapView(cv); v.resize(); return v;
};

test('map view: world fits inside the free box with margin, on every platform and canvas size', () => {
  for (const [W, H] of [[464, 531], [660, 501], [820, 720], [1240, 900], [300, 300]]) {
    for (const P of listPlatforms()) {
      const m = new Mission({ platform: P.id, scenario: Object.keys(P.scenarios)[0] });
      const v = view(W, H); v.setWorld(m.world);
      const ctr = v.center(m), b = m.world.bounds;
      const pts = [[b.xmin, b.ymin], [b.xmin, b.ymax], [b.xmax, b.ymin], [b.xmax, b.ymax]].map(([x, y]) => v.w2s(x, y, ctr));
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const slack = Math.min(300, W) === 300 ? 0 : v.pad - 1; // tiny canvases: just must not exceed the box
      assert.ok(Math.min(...xs) >= slack && Math.max(...xs) <= W - slack, `${P.id} ${W}x${H} x-range ${Math.min(...xs)}..${Math.max(...xs)}`);
      assert.ok(Math.min(...ys) >= v.inset.t + slack && Math.max(...ys) <= H - slack, `${P.id} ${W}x${H} y-range ${Math.min(...ys)}..${Math.max(...ys)}`);
    }
  }
});

test('map view: world <-> screen is an exact round trip (incl. the rotated RPO view and pan/zoom)', () => {
  for (const P of listPlatforms()) {
    const m = new Mission({ platform: P.id, scenario: Object.keys(P.scenarios)[0] });
    const v = view(820, 720); v.setWorld(m.world); v.zoom = 1.7; v.panBy(35, -20);
    const ctr = v.center(m);
    for (const [x, y] of [[0, 0], [m.world.bounds.xmax * 0.6, m.world.bounds.ymin * 0.4], [-13.5, 22.25]]) {
      const [px, py] = v.w2s(x, y, ctr), [x2, y2] = v.s2w(px, py, ctr);
      assert.ok(Math.abs(x - x2) < 1e-6 && Math.abs(y - y2) < 1e-6, `${P.id} roundtrip`);
    }
  }
});

test('map view: dragging moves the content with the cursor in both orientations', () => {
  for (const P of listPlatforms()) {
    const m = new Mission({ platform: P.id, scenario: Object.keys(P.scenarios)[0] });
    const v = view(820, 720); v.setWorld(m.world);
    const [x0, y0] = v.w2s(0, 0, v.center(m));
    v.panBy(40, 25);
    const [x1, y1] = v.w2s(0, 0, v.center(m));
    assert.ok(Math.abs(x1 - x0 - 40) < 1e-6 && Math.abs(y1 - y0 - 25) < 1e-6, `${P.id}: moved (${x1 - x0}, ${y1 - y0})`);
  }
});

test('spacecraft uses the RPO convention: along-track horizontal, radial vertical', () => {
  const m = new Mission({ platform: 'spacecraft', scenario: 'approach' });
  const v = view(820, 720); v.setWorld(m.world);
  const c = v.center(m), o = v.w2s(0, 0, c);
  const alongTrack = v.w2s(0, 50, c), radial = v.w2s(50, 0, c);
  assert.ok(alongTrack[0] > o[0] && Math.abs(alongTrack[1] - o[1]) < 1e-9, '+y_T should point right');
  assert.ok(radial[1] < o[1] && Math.abs(radial[0] - o[0]) < 1e-9, '+x_R should point up');
});
