// Cast the lidar's real beams against the scene: ground (flat in the work area), half-buried ellipsoid rocks and
// people (upright cylinders, 1.8 m). Used to DRAW what the lidar sees (point cloud, range image, bird's-eye view);
// the tracker's detections come from the sensor model in world.js, which uses the same line-of-sight geometry.
export function lidarScan(w) {
  const L = w.lidar, ox = L.x, oy = L.y, oz = L.h, nAz = Math.round(L.fov / L.azStep) + 1, nEl = L.channels;
  const ret = [], img = new Float32Array(nAz * nEl), who = new Int16Array(nAz * nEl);
  for (let e = 0; e < nEl; e++) {
    const el = L.elevMax - (e * (L.elevMax - L.elevMin)) / (nEl - 1), ce = Math.cos(el), dz = Math.sin(el);
    for (let a = 0; a < nAz; a++) {
      const az = L.th + L.fov / 2 - a * L.azStep, dx = ce * Math.cos(az), dy = ce * Math.sin(az);
      let best = dz < 0 ? -oz / dz : Infinity, hit = 0;
      for (const b of w.boulders) {
        const h = b.h ?? 1e6, px = (ox - b.x) / b.r, py = (oy - b.y) / b.r, pz = oz / h, qx = dx / b.r, qy = dy / b.r, qz = dz / h;
        const A = qx * qx + qy * qy + qz * qz, B = 2 * (px * qx + py * qy + pz * qz), C = px * px + py * py + pz * pz - 1, D = B * B - 4 * A * C;
        if (D > 0) { const t = (-B - Math.sqrt(D)) / (2 * A); if (t > 0 && t < best) { best = t; hit = -1; } }
      }
      for (const p of w.targets) {
        const fx = ox - p.x, fy = oy - p.y, A = dx * dx + dy * dy, B = 2 * (fx * dx + fy * dy), C = fx * fx + fy * fy - 0.0784, D = B * B - 4 * A * C;
        if (D > 0) { const t = (-B - Math.sqrt(D)) / (2 * A), z = oz + t * dz; if (t > 0 && t < best && z > 0 && z < 1.8) { best = t; hit = p.id; } }
      }
      if (!(best < L.range)) continue;
      img[e * nAz + a] = best; who[e * nAz + a] = hit;
      ret.push(ox + best * dx, oy + best * dy, Math.max(0, oz + best * dz), hit, a, e);   // x, y, z, what, column, row
    }
  }
  return { ret, img, who, nAz, nEl };
}

/** A smooth "turbo-like" colour ramp for distances and heights. */
export const turbo = (t) => { t = Math.min(1, Math.max(0, t)); return [Math.max(0, Math.min(1, 1.6 * t - 0.2 + 0.5 * Math.sin(3.1 * t))), Math.max(0, Math.sin(Math.PI * t)), Math.max(0, Math.min(1, 1.2 - 1.6 * t))]; };
