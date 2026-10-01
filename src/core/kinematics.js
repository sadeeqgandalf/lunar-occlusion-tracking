// Exact constant-speed, constant-turn-rate arc (Probabilistic Robotics eq. 5.9) + analytic partials.
// Shared by every planar vehicle (ground rover, aircraft). Straight-line limit via midpoint expansion.
import { wrapAngle } from './linalg.js';

export const W_EPS = 1e-3;

export function arc(x, y, th, v, w, dt) {
  if (Math.abs(w) < W_EPS) {
    const m = th + (w * dt) / 2;
    return [x + v * dt * Math.cos(m), y + v * dt * Math.sin(m), wrapAngle(th + w * dt)];
  }
  return [
    x + (v / w) * (Math.sin(th + w * dt) - Math.sin(th)),
    y + (v / w) * (Math.cos(th) - Math.cos(th + w * dt)),
    wrapAngle(th + w * dt),
  ];
}

/** Partials of the arc's (x', y') w.r.t. heading, turn rate and speed. */
export function arcPartials(th, v, w, dt) {
  if (Math.abs(w) < W_EPS) {
    const m = th + (w * dt) / 2, s = Math.sin(m), c = Math.cos(m);
    return { dxdth: -v * dt * s, dydth: v * dt * c, dxdw: (-v * dt * dt * s) / 2, dydw: (v * dt * dt * c) / 2, dxdv: dt * c, dydv: dt * s };
  }
  const s0 = Math.sin(th), c0 = Math.cos(th), s1 = Math.sin(th + w * dt), c1 = Math.cos(th + w * dt), r = v / w;
  return {
    dxdth: r * (c1 - c0), dydth: r * (s1 - s0),
    dxdw: (-v / (w * w)) * (s1 - s0) + r * dt * c1,
    dydw: (-v / (w * w)) * (c0 - c1) + r * dt * s1,
    dxdv: (s1 - s0) / w, dydv: (c0 - c1) / w,
  };
}
