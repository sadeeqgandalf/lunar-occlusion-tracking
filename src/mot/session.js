// One experiment run: a world, its detections, and several trackers scored on the SAME detections.
import { MotWorld } from './world.js';
import { Tracker } from './tracker.js';
import { MotMetrics } from './metrics.js';
import { ParticlePHD } from './phd.js';

export class MotSession {
  constructor({ scenario = 'boulders', seed = 7, overrides = {}, trackerParams = {}, phd = false } = {}) {
    this.world = new MotWorld(scenario, seed, overrides);
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
    if (this.phd) this.phd.step(dets);
    for (const r of this.runs) {
      // trackers never see identities; only the re-ID tracker gets each detection's colour signature
      r.tracker.step(dets.map((d) => (r.tracker.reid ? { range: d.range, bearing: d.bearing, feat: d.feat } : { range: d.range, bearing: d.bearing })));
      r.metrics.update(W, r.tracker);
    }
  }
  run(seconds) { const n = Math.round(seconds / this.world.dt); for (let i = 0; i < n; i++) this.step(); return this; }
  summaries() { return this.runs.map((r) => ({ name: r.tracker.name, ...r.metrics.summary() })); }
}
