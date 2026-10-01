// Line-of-sight occlusion for a camera on a 2-D ground plane.
// A target is a disk of radius rt; each boulder is a disk. Seen from the camera, every disk covers an angular
// interval. The target's visible fraction = 1 - |union of (closer boulders' intervals) ∩ (target interval)| / |target interval|.
import { wrapAngle } from '../core/linalg.js';

/** Angular interval [lo, hi] (relative to `ref` bearing, so no ±π seam issues near the optical axis). */
function interval(cam, ref, x, y, r) {
  const dx = x - cam.x, dy = y - cam.y, d = Math.hypot(dx, dy);
  const c = wrapAngle(Math.atan2(dy, dx) - ref), half = d <= r ? Math.PI / 2 : Math.asin(r / d);
  return { lo: c - half, hi: c + half, d };
}

/**
 * @returns {{inFov:boolean, range:number, bearing:number, visFrac:number, visCenter:number}}
 *  bearing: world bearing to target centre; visCenter: world bearing of the centre of its *visible* part
 *  (partial occlusion shifts the detected centroid toward the visible side, as a real detector does).
 */
export function visibility(cam, boulders, x, y, rt = 0.4) {
  const dx = x - cam.x, dy = y - cam.y, range = Math.hypot(dx, dy), bearing = Math.atan2(dy, dx);
  const off = wrapAngle(bearing - cam.th);
  const inFov = range <= cam.range && Math.abs(off) <= cam.fov / 2 && range > 0.5;
  const t = interval(cam, bearing, x, y, rt);
  const cuts = [];
  for (const b of boulders) {
    const bi = interval(cam, bearing, b.x, b.y, b.r);
    if (bi.d - b.r >= t.d) continue;                 // boulder is not in front of the target
    const lo = Math.max(bi.lo, t.lo), hi = Math.min(bi.hi, t.hi);
    if (hi > lo) cuts.push([lo, hi]);
  }
  cuts.sort((a, b) => a[0] - b[0]);
  let covered = 0, curLo = -Infinity, curHi = -Infinity;
  const visible = [];
  let cursor = t.lo;
  for (const [lo, hi] of cuts) {
    if (lo > curHi) { if (curHi > curLo) covered += curHi - curLo; curLo = lo; curHi = hi; } else curHi = Math.max(curHi, hi);
  }
  if (curHi > curLo) covered += curHi - curLo;
  // visible sub-intervals -> centroid of what the camera actually sees
  const merged = [];
  for (const [lo, hi] of cuts) { const l = merged[merged.length - 1]; if (l && lo <= l[1]) l[1] = Math.max(l[1], hi); else merged.push([lo, hi]); }
  for (const [lo, hi] of merged) { if (lo > cursor) visible.push([cursor, lo]); cursor = Math.max(cursor, hi); }
  if (cursor < t.hi) visible.push([cursor, t.hi]);
  const width = t.hi - t.lo, visFrac = Math.max(0, 1 - covered / width);
  let num = 0, den = 0;
  for (const [lo, hi] of visible) { num += (lo + hi) / 2 * (hi - lo); den += hi - lo; }
  const visCenter = wrapAngle(bearing + (den > 0 ? num / den : 0));
  return { inFov, range, bearing, visFrac, visCenter };
}

/** Detection probability as a function of how much of the object is visible. */
export const detectionProb = (inFov, visFrac, pdMax) => (!inFov ? 0 : visFrac < 0.25 ? 0 : pdMax * Math.min(1, (visFrac - 0.25) / 0.45));
