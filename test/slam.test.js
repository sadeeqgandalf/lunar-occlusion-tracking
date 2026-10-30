import { test } from 'node:test';
import assert from 'node:assert/strict';
import { umeyamaAlign3D, evaluateTrajectoryMetrics, LunarSlamSession, EkfSlam, covEllipse } from '../src/slam/engine.js';

test('umeyamaAlign3D recovers exact 3D scale, yaw rotation, and translation on Sim(3)', () => {
  const dst = [
    [0, 0, 0],
    [10, 0, 0],
    [10, 0, -10],
    [0, 0, -10],
    [5, 0, -15],
  ];
  const scaleTrue = 2.0;
  const theta = 0.45;
  const c = Math.cos(theta), s = Math.sin(theta);
  const src = dst.map(([x, y, z]) => {
    const rx = (c * x + s * z) / scaleTrue + 4.0;
    const rz = (-s * x + c * z) / scaleTrue - 7.0;
    return [rx, y, rz];
  });

  const res = umeyamaAlign3D(src, dst, true);
  assert.ok(Math.abs(res.s - scaleTrue) < 1e-4, `Expected scale ~2.0, got ${res.s}`);
  for (let i = 0; i < dst.length; i++) {
    const err = Math.hypot(
      res.aligned[i][0] - dst[i][0],
      res.aligned[i][1] - dst[i][1],
      res.aligned[i][2] - dst[i][2]
    );
    assert.ok(err < 1e-4, `Point ${i} alignment error too large: ${err}`);
  }
});

test('evaluateTrajectoryMetrics computes ATE RMSE, rawPoseErr, and RPE drift %', () => {
  const truth = [];
  const est = [];
  for (let i = 0; i <= 20; i++) {
    truth.push({ pos: [i * 1.0, 0, 0] });
    est.push({ pos: [i * 1.0, 0, i * 0.05] });
  }
  const m = evaluateTrajectoryMetrics(truth, est, false, 10.0);
  assert.ok(m.ateRmse >= 0 && m.ateRmse < 1.0);
  assert.ok(m.rpePct > 0);
  assert.ok(m.rawPoseErr > 0);
  assert.equal(m.aligned.length, 21);
});

test('EkfSlam: a landmark seen from two places is located, and seeing it again shrinks the pose uncertainty', () => {
  const f = new EkfSlam(0, 0, 0, 0.01, 0.005);
  const truth = { x: 6, y: 3 };
  const meas = (f, px, py, pth) => [Math.hypot(truth.x - px, truth.y - py), Math.atan2(truth.y - py, truth.x - px) - pth];
  f.add(7, ...meas(f, 0, 0, 0), 0.05, 0.01);
  assert.equal(f.n, 5);
  // Drive 2 m with odometry (exact here), so only the uncertainty grows.
  for (let i = 0; i < 20; i++) f.predict(1, 0, 0.1, 0.05, 0.02);
  const before = f.poseCov();
  const u = f.update(7, ...meas(f, 2, 0, 0), 0.05, 0.01);
  const after = f.poseCov();
  assert.ok(Math.abs(u.innovRange) < 1e-9 && Math.abs(u.innovBearing) < 1e-9, 'exact data gives zero innovation');
  assert.ok(after.xx + after.yy < before.xx + before.yy, 'a check against the map shrinks the pose uncertainty');
  const j = f.index.get(7);
  assert.ok(Math.hypot(f.mu[j] - truth.x, f.mu[j + 1] - truth.y) < 1e-9);
  for (let a = 0; a < f.n; a++) for (let b = 0; b < f.n; b++) assert.ok(Math.abs(f.get(a, b) - f.get(b, a)) < 1e-12, 'P stays symmetric');
});

test('EkfSlam: a wrong pose is pulled back by a well-known landmark', () => {
  const f = new EkfSlam(0, 0, 0, 0.01, 0.005);
  f.add(1, 10, 0, 0.02, 0.005); // landmark at (10, 0), well known
  f.mu[0] -= 0.8; // the pose drifts 0.8 m without knowing it ...
  f.P[0] += 0.5; // ... and admits it is unsure
  f.update(1, 10, 0, 0.02, 0.005); // true range 10 m from the true pose (0, 0)
  assert.ok(Math.abs(f.mu[0]) < 0.3, `pose should move back toward 0, got ${f.mu[0]}`);
});

test('covEllipse gives axis lengths and angle of a 2x2 covariance', () => {
  const e = covEllipse(4, 0, 1, 2);
  assert.ok(Math.abs(e.a - 4) < 1e-12 && Math.abs(e.b - 2) < 1e-12 && Math.abs(e.angle) < 1e-12);
  const r = covEllipse(2.5, 1.5, 2.5, 1); // eigenvalues 4 and 1 along the diagonals
  assert.ok(Math.abs(r.a - 2) < 1e-12 && Math.abs(r.b - 1) < 1e-12 && Math.abs(r.angle - Math.PI / 4) < 1e-12);
});

test('SLAM lab: after a 150 m loop SLAM stays close and honest, wheels alone drift, and a loop closure happens', () => {
  const s = new LunarSlamSession({ scenario: 'loop_home', seed: 3 });
  const [slam, wheels] = s.estimators;
  for (let i = 0; i < 1100; i++) s.step();
  assert.ok(s.distanceDriven > 90, `drove ${s.distanceDriven}`);
  assert.ok(slam.metrics.rawPoseErr < 1.5, `SLAM error ${slam.metrics.rawPoseErr}`);
  assert.ok(wheels.metrics.rawPoseErr > 3 * slam.metrics.rawPoseErr, 'wheels alone end up much further off');
  assert.ok(slam.metrics.insideRate > 0.8, `truth inside the 95% ellipse ${slam.metrics.insideRate}`);
  assert.ok(slam.mapLandmarks.size > 15 && slam.metrics.mapRmse < 1.5, 'SLAM builds an accurate landmark map');
  assert.equal(wheels.mapLandmarks.size, 0);
  assert.ok(s.loopClosures.length >= 1, 'recognising old landmarks closes a loop');
  const lc = s.loopClosures[0];
  assert.ok(lc.sigmaAfter < 0.75 * lc.sigmaBefore);
  assert.ok(s.grid.coveragePct > 40, 'the occupancy map grows as the rover explores');
});

test('SLAM lab: with the camera off, SLAM is no better than the wheels', () => {
  const s = new LunarSlamSession({ scenario: 'loop_home', seed: 3, useCamera: false });
  for (let i = 0; i < 600; i++) s.step();
  const [slam, wheels] = s.estimators;
  assert.equal(slam.mapLandmarks.size, 0);
  assert.ok(Math.abs(slam.metrics.rawPoseErr - wheels.metrics.rawPoseErr) < 1e-9);
});

test('SLAM lab: the map is drawn from the believed pose, so a lost rover draws rocks in the wrong place', () => {
  const s = new LunarSlamSession({ scenario: 'slippery', seed: 2 });
  for (let i = 0; i < 1000; i++) s.step();
  const occupied = (g) => { let k = 0; for (const v of g.logOdds) if (v > 0.35) k++; return k; };
  const truthRockCells = (g) => {
    let hit = 0, all = 0;
    for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
      if (g.logOdds[r * g.cols + c] <= 0.35) continue;
      all++;
      const x = g.xMin + (c + 0.5) * g.cellSize, y = g.yMin + (r + 0.5) * g.cellSize;
      if (s.boulders.some((b) => Math.hypot(x - b.x, y - b.y) < b.r + 1.5)) hit++;
    }
    return hit / all;
  };
  const [slam, wheels] = s.estimators;
  assert.ok(occupied(slam.grid) > 10 && occupied(wheels.grid) > 10);
  assert.ok(truthRockCells(slam.grid) > truthRockCells(wheels.grid) + 0.15,
    `SLAM's rock cells should sit on real rocks far more often (${truthRockCells(slam.grid)} vs ${truthRockCells(wheels.grid)})`);
});

test('SLAM lab: the rover never drives into a rock (hazard avoidance), and a rock stops it if driven at one by hand', () => {
  const gap = (s) => Math.min(...s.boulders.map((b) => Math.hypot(s.rover.x - b.x, s.rover.y - b.y) - b.r - 0.8));
  for (const scenario of ['loop_home', 'explore']) {
    const s = new LunarSlamSession({ scenario, seed: 3 });
    let worst = Infinity;
    for (let i = 0; i < 1500; i++) { s.step(); worst = Math.min(worst, gap(s)); }
    assert.ok(worst > 0, `${scenario}: body overlapped a rock by ${-worst} m`);
    assert.equal(s.bumps, 0);
  }
  const s = new LunarSlamSession({ scenario: 'loop_home', seed: 3 });
  const rock = s.boulders.reduce((a, b) => (Math.hypot(b.x - s.rover.x, b.y - s.rover.y) < Math.hypot(a.x - s.rover.x, a.y - s.rover.y) ? b : a));
  s.manualControl = { active: true, v: 1, w: 0 };
  s.rover.th = Math.atan2(rock.y - s.rover.y, rock.x - s.rover.x);
  for (let i = 0; i < 200; i++) s.step();
  assert.ok(s.bumps > 0 && gap(s) > -1e-9, 'driven straight at a rock, it stops at the rock');
});
