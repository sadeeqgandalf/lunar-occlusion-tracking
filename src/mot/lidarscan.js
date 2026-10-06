// Cast the lidar's real beams against the scene: ground (flat in the work area), half-buried ellipsoid rocks and
// people (upright cylinders, 1.8 m). Used to DRAW what the lidar sees (point cloud, range image, bird's-eye view);
// the tracker's detections come from the sensor model in world.js, which uses the same line-of-sight geometry.
//
// Each return also gets a signal strength, as in a real time-of-flight lidar:
//   signal = reflectivity x cos(incidence)^0.7 x (fullRange / r)^2 x per-beam gain
// Dark regolith seen at a grazing angle returns little light, so the ground fades out with distance; rocks facing
// the sensor and white suits return strongly. Weak returns drop out at random and every range carries a few cm of
// noise. The noise is a hash of (beam, column, sim frame): it changes once per scan and replays identically.
const rnd = (a, b, c) => { let h = (Math.imul(a + 1, 374761393) ^ Math.imul(b + 1, 668265263) ^ Math.imul(c + 1, 2147483647)) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
const vnoise = (x, y) => {                                   // smooth value noise in [0, 1): surface mottling
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = rnd(xi, yi, 7), b = rnd(xi + 1, yi, 7), c = rnd(xi, yi + 1, 7), d = rnd(xi + 1, yi + 1, 7);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
};
const RHO_GROUND = 0.11, RHO_ROCK = 0.2, RHO_SUIT = 0.8;     // 905 nm reflectivity: regolith is dark, suits are white
const MIN_SIGNAL = 0.03;                                     // detection threshold: a 10% target at normal incidence reaches max range

const cache = new WeakMap();                                 // one scan per world per sim frame (both views draw the same scan)
export function lidarScan(w) {
  const c = cache.get(w); if (c && c.frame === w.frame) return c.scan;
  const scan = castScan(w); cache.set(w, { frame: w.frame, scan }); return scan;
}
function castScan(w) {
  const L = w.lidar, ox = L.x, oy = L.y, oz = L.h, nAz = Math.round(L.fov / L.azStep) + 1, nEl = L.channels, fr = w.frame | 0;
  const ret = [], img = new Float32Array(nAz * nEl), who = new Int16Array(nAz * nEl), inten = new Float32Array(nAz * nEl);
  for (let e = 0; e < nEl; e++) {
    const el = L.elevMax - (e * (L.elevMax - L.elevMin)) / (nEl - 1), ce = Math.cos(el), dz = Math.sin(el);
    const gain = 0.94 + 0.12 * rnd(e, 3, 11);                 // each laser/detector pair is slightly different: faint row banding
    for (let a = 0; a < nAz; a++) {
      const az = L.th + L.fov / 2 - a * L.azStep, dx = ce * Math.cos(az), dy = ce * Math.sin(az);
      let best = dz < 0 ? -oz / dz : Infinity, hit = 0, cosI = -dz, rho = 0;
      for (let bi = 0; bi < w.boulders.length; bi++) {
        const b = w.boulders[bi], h = b.h ?? 1e6, px = (ox - b.x) / b.r, py = (oy - b.y) / b.r, pz = oz / h, qx = dx / b.r, qy = dy / b.r, qz = dz / h;
        const A = qx * qx + qy * qy + qz * qz, B = 2 * (px * qx + py * qy + pz * qz), C = px * px + py * py + pz * pz - 1, D = B * B - 4 * A * C;
        if (D > 0) {
          const t = (-B - Math.sqrt(D)) / (2 * A);
          if (t > 0 && t < best) {                             // surface normal of the ellipsoid at the hit point
            const nx = (ox + t * dx - b.x) / (b.r * b.r), ny = (oy + t * dy - b.y) / (b.r * b.r), nz = (oz + t * dz) / (h * h), nn = Math.hypot(nx, ny, nz) || 1;
            best = t; hit = -1; cosI = Math.abs(nx * dx + ny * dy + nz * dz) / nn; rho = RHO_ROCK * (0.75 + 0.5 * rnd(bi, 5, 13));
          }
        }
      }
      for (const p of w.targets) {
        const fx = ox - p.x, fy = oy - p.y, A = dx * dx + dy * dy, B = 2 * (fx * dx + fy * dy), C = fx * fx + fy * fy - 0.0784, D = B * B - 4 * A * C;
        if (D > 0) {
          const t = (-B - Math.sqrt(D)) / (2 * A), z = oz + t * dz;
          if (t > 0 && t < best && z > 0 && z < 1.8) {
            best = t; hit = p.id; cosI = Math.abs((fx + t * dx) * dx + (fy + t * dy) * dy) / 0.28;
            rho = z < 0.18 ? 0.3 : z > 1.5 && z < 1.72 ? 0.45 : RHO_SUIT;      // boots and visor are darker than the suit
          }
        }
      }
      if (!(best < L.range)) continue;
      const hx = ox + best * dx, hy = oy + best * dy;
      if (hit === 0) rho = RHO_GROUND * (0.7 + 0.6 * (0.6 * vnoise(hx * 0.9, hy * 0.9) + 0.4 * vnoise(hx * 4.1, hy * 4.1)));
      else if (hit < 0) rho *= 0.8 + 0.4 * vnoise(hx * 3.3 + best, hy * 3.3);
      const refl = rho * Math.pow(Math.max(cosI, 0.001), 0.7) * gain, signal = refl * (L.fullRange / best) ** 2;
      const k = e * nAz + a, n1 = rnd(a, e, fr), n2 = rnd(a, e, fr + 9973);
      if (signal < MIN_SIGNAL * (0.6 + 0.8 * n1)) continue;                    // too little light came back: no return
      const snr = Math.min(1, signal / (8 * MIN_SIGNAL)), r = best + (n2 - 0.5) * (0.03 + 0.09 * (1 - snr));   // 1.5-6 cm of range noise
      img[k] = r; who[k] = hit; inten[k] = Math.max(0, refl * (1 + (n1 - 0.5) * (0.08 + 0.5 * (1 - snr))));
      ret.push(ox + r * dx, oy + r * dy, Math.max(0, oz + r * dz), hit, a, e);   // x, y, z, what, column, row
    }
  }
  return { ret, img, who, inten, nAz, nEl };
}

/** A smooth "turbo-like" colour ramp for distances and heights. */
export const turbo = (t) => { t = Math.min(1, Math.max(0, t)); return [Math.max(0, Math.min(1, 1.6 * t - 0.2 + 0.5 * Math.sin(3.1 * t))), Math.max(0, Math.sin(Math.PI * t)), Math.max(0, Math.min(1, 1.2 - 1.6 * t))]; };
