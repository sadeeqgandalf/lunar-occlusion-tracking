import { Mat, wrapAngle } from './linalg.js';

/**
 * Central-difference Jacobian of fn: R^n -> R^m at x. `wrapOut` lists output indices that are angles,
 * so differences across the +/-pi seam do not blow up. Lets a new vehicle skip hand-derived Jacobians.
 */
export function numJac(fn, x, wrapOut = [], eps = 1e-6) {
  const n = x.length, f0 = fn(x), m = f0.length, J = Mat.zeros(m, n);
  for (let j = 0; j < n; j++) {
    const h = eps * Math.max(1, Math.abs(x[j]));
    const xp = Array.from(x), xm = Array.from(x);
    xp[j] += h; xm[j] -= h;
    const fp = fn(xp), fm = fn(xm);
    for (let i = 0; i < m; i++) {
      let d = fp[i] - fm[i];
      if (wrapOut.includes(i)) d = wrapAngle(d);
      J.set(i, j, d / (2 * h));
    }
  }
  return J;
}
