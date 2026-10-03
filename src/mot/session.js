// One experiment run: a world, its detections, and several trackers scored on the SAME detections.
import { MotWorld } from './world.js';
import { Tracker } from './tracker.js';
import { MotMetrics } from './metrics.js';
import { ParticlePHD } from './phd.js';

export class MotSession {
  constructor({ scenario = 'boulders', seed = 7, overrides = {}, trackerParams = {}, phd = false, lander = false, lidar = false } = {}) {
    this.world = new MotWorld(scenario, seed, overrides, { lander, lidar });
    const W = this.world, pd = W.cfg.pd, clutter = W.cfg.clutter, sun = W.cfg.shadows ? W.sun : null;
    // Ablation: identical trackers except for how they treat occlusion; the 4th also recognises people by appearance.
    this.runs = [
      { occlusionAware: true, negInfo: true },
      { occlusionAware: true, negInfo: false },
      { occlusionAware: false },
      { occlusionAware: true, negInfo: true, reid: true },
    ].map((v) => ({ tracker: new Tracker(W.cam, W.boulders, { ...v, pd, clutter, sun, ...trackerParams }), metrics: new MotMetrics() }));
    this.last = { dets: [], gt: [] };
    this.phd = phd ? new ParticlePHD(this.world, { seed }) : null;   // "where could anyone be?" density (display)
  }
  step() {
    const W = this.world;
    W.step();
    const { dets, gt } = W.sense();
    this.last = { dets, gt };
    const frames = [{ cam: W.cam, dets }];
    if (W.lander) { const L = W.senseLander(); this.last.ldets = L.dets; this.last.lgt = L.gt; frames.push({ cam: W.lander, dets: L.dets }); }
    if (W.lidar) { const D = W.senseLidar(); this.last.ddets = D.dets; this.last.dgt = D.gt; frames.push({ cam: W.lidar, dets: D.dets }); }
    if (this.phd) this.phd.step(frames);
    for (const r of this.runs) {
      // trackers never see identities; only the re-ID tracker gets each detection's colour signature
      const strip = (ds) => ds.map((d) => (r.tracker.reid ? { range: d.range, bearing: d.bearing, feat: d.feat } : { range: d.range, bearing: d.bearing }));
      if (frames.length === 1) r.tracker.step(strip(dets));
      else r.tracker.stepFrames(frames.map((f) => ({ cam: f.cam, dets: strip(f.dets) })));
      r.metrics.update(W, r.tracker);
    }
  }
  run(seconds) { const n = Math.round(seconds / this.world.dt); for (let i = 0; i < n; i++) this.step(); return this; }
  summaries() { return this.runs.map((r) => ({ name: r.tracker.name, ...r.metrics.summary() })); }
}
