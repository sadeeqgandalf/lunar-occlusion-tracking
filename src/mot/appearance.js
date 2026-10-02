// Appearance sensor: a colour signature for each detection, so a tracker can recognise a person it lost.
// Signature = 18-bin hue histogram of the person's saturated pixels (their suit stripes) + 4 lightly weighted brightness
// bins, unit length, exactly as computed from rendered pixels in the PyTorch3D lab (pytorch3d/lunar3d/detector.py).
// Here it is modelled rather than measured: a hue bump at the person's stripe colour plus sensor noise that grows when
// the person is partly hidden (fewer stripe pixels) or in shadow (washed-out colour). Calibrated to the PyTorch3D
// measurements: same person ~0.95-0.99 cosine similarity, different people ~0.1, person vs false alarm ~0.05.
// Uses its OWN random stream, so detections and every earlier result are unchanged.
import { STRIPES } from '../ui/idcolor.js';

export const HUE_BINS = 18;

function hueOf(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255, g = parseInt(hex.slice(3, 5), 16) / 255, b = parseInt(hex.slice(5, 7), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  if (d === 0) return 0;
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return (h / 6 + 1) % 1;
}
const STRIPE_HUE = STRIPES.map((s) => hueOf(s.hex));

function unit(f) { const n = Math.hypot(...f) || 1; return f.map((v) => v / n); }

/** Signature of person `id` seen with visible fraction `vis` and shadow fraction `shadow`. */
export function personSignature(id, vis, shadow, rng) {
  const h = STRIPE_HUE[(id - 1) % STRIPE_HUE.length] * HUE_BINS, noise = 0.025 + 0.08 * (1 - vis) + 0.08 * shadow;
  const f = [];
  for (let k = 0; k < HUE_BINS; k++) {
    let d = Math.abs(k + 0.5 - h); d = Math.min(d, HUE_BINS - d);                 // hue is circular
    f.push(Math.exp(-0.5 * (d / 0.55) ** 2) + Math.abs(noise * rng.randn()));
  }
  for (let k = 0; k < 4; k++) f.push(0.05 * (k === 3 ? 1 : 0.2) + 0.02 * Math.abs(rng.randn()));   // white suit, small weight
  return unit(f);
}

/** Signature of a false alarm: grey rock or regolith, almost no saturated colour. */
export function clutterSignature(rng) {
  const f = [];
  for (let k = 0; k < HUE_BINS; k++) f.push(0.02 * Math.abs(rng.randn()));
  for (let k = 0; k < 4; k++) f.push(0.35 * (k === 1 || k === 2 ? 1 : 0.2) + 0.05 * Math.abs(rng.randn()));   // grey: brightness only
  return unit(f);
}

export const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
