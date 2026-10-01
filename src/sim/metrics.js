import { Mat } from '../core/linalg.js';
import { resid } from '../filters/base.js';

// 95% two-sided chi-square bounds for NEES by degrees of freedom.
const NEES_TABLE = { 1: [0.001, 5.024], 2: [0.051, 7.378], 3: [0.216, 9.348], 4: [0.484, 11.143], 5: [0.831, 12.833] };
export const neesBounds = (dof) => NEES_TABLE[dof] || NEES_TABLE[3];

/** Normalised estimation error squared over the model's `neesIdx` subset. NaN if covariance is singular. */
export function nees(model, truth, est, P) {
  const idx = model.neesIdx, k = idx.length;
  try {
    const e = resid(model.angleStates, truth, est);
    const ev = Mat.from(idx.map((i) => [e[i]]));
    const Ps = Mat.zeros(k, k);
    idx.forEach((a, i) => idx.forEach((b, j) => Ps.set(i, j, P.get(a, b))));
    return ev.T().mul(Ps.inv()).mul(ev).d[0];
  } catch { return NaN; }
}

/** 1-sigma radius of the largest position-uncertainty eigenvalue. */
export function sigmaMax(model, P) {
  const [i, j] = model.posIdx;
  const a = P.get(i, i), b = P.get(i, j), c = P.get(j, j);
  return Math.sqrt(Math.max((a + c) / 2 + Math.sqrt(((a - c) / 2) ** 2 + b * b), 0));
}
