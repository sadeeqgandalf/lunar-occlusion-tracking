// 3-D line-of-sight occlusion for a mast camera on (locally) flat ground.
//
// Geometry
//   camera eye     (cam.x, cam.y, cam.h ?? CAM_H)
//   person         upright body: radius rt, height PERSON_H
//   boulder        half-buried ellipsoid: centre (b.x, b.y, 0), horizontal radius b.r, height b.h
//                  (b.h undefined => effectively infinite: a full-height occluder, i.e. the old planar model)
// Visibility = fraction of sight lines (4 heights x 4 lateral offsets across the body) that reach the person
// without entering any boulder. A low rock therefore hides legs but not heads, as a real camera sees it.
//
// Photometric occlusion (optional): under a low sun, a person standing in a boulder's cast shadow is much
// harder for an optical camera to detect. shadowFraction() casts rays from the body toward the sun.
import { wrapAngle } from '../core/linalg.js';

export const CAM_H = 2.2, PERSON_H = 1.8;
const RAY_H = [0.3, 0.8, 1.3, 1.7], RAY_L = [-0.28, -0.09, 0.09, 0.28];
const SHADOW_H = [0.4, 0.9, 1.4], SHADOW_LEN = 80;

/** Does the segment E->P pass through boulder b's ellipsoid? (strict interior, segment parameter in (0,1)) */
function segmentHits(ex, ey, ez, px, py, pz, b) {
  const h = b.h ?? 1e6;
  const ox = (ex - b.x) / b.r, oy = (ey - b.y) / b.r, oz = ez / h;
  const dx = (px - ex) / b.r, dy = (py - ey) / b.r, dz = (pz - ez) / h;
  const a = dx * dx + dy * dy + dz * dz, bq = 2 * (ox * dx + oy * dy + oz * dz), c = ox * ox + oy * oy + oz * oz - 1;
  const disc = bq * bq - 4 * a * c;
  if (disc <= 0) return false;
  const s = Math.sqrt(disc), t1 = (-bq - s) / (2 * a), t2 = (-bq + s) / (2 * a);
  return t1 < 1 && t2 > 0; // overlap of (t1,t2) with (0,1)
}

/**
 * @returns {{inFov, range, bearing, visFrac, visCenter}}
 *  visCenter: bearing of the centre of the VISIBLE part (partial occlusion shifts a detector's centroid).
 */
export function visibility(cam, boulders, x, y, rt = 0.4) {
  const dx = x - cam.x, dy = y - cam.y, range = Math.hypot(dx, dy), bearing = Math.atan2(dy, dx);
  const inFov = range <= cam.range && Math.abs(wrapAngle(bearing - cam.th)) <= cam.fov / 2 && range > 0.5;
  // only boulders in front of the person and overlapping their angular extent can block anything
  const halfT = Math.asin(Math.min(1, rt / Math.max(range, rt)));
  const near = boulders.filter((b) => {
    const bd = Math.hypot(b.x - cam.x, b.y - cam.y);
    if (bd - b.r >= range) return false;
    const halfB = bd <= b.r ? Math.PI : Math.asin(b.r / bd);
    return Math.abs(wrapAngle(Math.atan2(b.y - cam.y, b.x - cam.x) - bearing)) <= halfB + halfT;
  });
  if (!near.length) return { inFov, range, bearing, visFrac: 1, visCenter: bearing };
  const ez = cam.h ?? CAM_H, nx = -Math.sin(bearing), ny = Math.cos(bearing), sc = (rt / 0.4);
  let vis = 0, latSum = 0;
  for (const l0 of RAY_L) {
    const l = l0 * sc, px = x + l * nx, py = y + l * ny;
    for (const hz of RAY_H) {
      if (!near.some((b) => segmentHits(cam.x, cam.y, ez, px, py, hz, b))) { vis++; latSum += l; }
    }
  }
  const n = RAY_L.length * RAY_H.length, visFrac = vis / n;
  const visCenter = vis ? wrapAngle(bearing + Math.atan2(latSum / vis, range)) : bearing;
  return { inFov, range, bearing, visFrac, visCenter };
}

/** Fraction of the body that lies in a boulder's cast shadow, for a sun at azimuth `az`, elevation `el` (radians). */
export function shadowFraction(boulders, x, y, sun) {
  if (!sun) return 0;
  const cx = Math.cos(sun.el) * Math.cos(sun.az), cy = Math.cos(sun.el) * Math.sin(sun.az), cz = Math.sin(sun.el);
  const near = boulders.filter((b) => {
    // boulder must lie roughly toward the sun from the person, within the shadow length
    const bx = b.x - x, by = b.y - y, along = bx * Math.cos(sun.az) + by * Math.sin(sun.az);
    return along > -b.r && along < SHADOW_LEN && Math.abs(-bx * Math.sin(sun.az) + by * Math.cos(sun.az)) < b.r + 0.5;
  });
  if (!near.length) return 0;
  let dark = 0;
  for (const hz of SHADOW_H) {
    const ex = x + cx * SHADOW_LEN, ey = y + cy * SHADOW_LEN, ez = hz + cz * SHADOW_LEN;
    if (near.some((b) => segmentHits(x, y, hz, ex, ey, ez, b))) dark++;
  }
  return dark / SHADOW_H.length;
}

/**
 * Detection probability. Geometric: needs >= 25% of the body in view, full rate from 70%.
 * Photometric: a fully shadowed person is detected 80% less often by an optical camera.
 */
export function detectionProb(inFov, visFrac, pdMax, shadowFrac = 0) {
  if (!inFov || visFrac < 0.25) return 0;
  return pdMax * Math.min(1, (visFrac - 0.25) / 0.45) * (1 - 0.8 * shadowFrac);
}

/**
 * Detection probability for a given sensor.
 *  passive camera (default): needs light, so cast shadow cuts it (detectionProb above).
 *  active lidar (sensor.active): brings its own light, so no shadow penalty; but a far person returns fewer points,
 *  so it fades linearly from full at sensor.fullRange to zero at sensor.range. Both need line of sight past rocks.
 */
export function sensorPd(sensor, v, pdDefault, shadowFrac = 0, vel = null) {
  if (sensor && sensor.event) {
    // Event camera: reports brightness CHANGES, so it sees motion, not presence. A person standing still is in plain
    // view and produces no events. High dynamic range: shadow costs little (EVENT_SHADOW).
    if (!v.inFov || v.visFrac < 0.25) return 0;
    return (sensor.pd ?? pdDefault) * Math.min(1, (v.visFrac - 0.25) / 0.45) * (1 - EVENT_SHADOW * shadowFrac) * motionSignal(sensor, v, vel);
  }
  if (sensor && sensor.active) {
    if (!v.inFov || v.visFrac < 0.25) return 0;
    const rf = v.range <= sensor.fullRange ? 1 : Math.max(0, 1 - (v.range - sensor.fullRange) / (sensor.range - sensor.fullRange));
    return sensor.pd * Math.min(1, (v.visFrac - 0.25) / 0.45) * rf;
  }
  return detectionProb(v.inFov, v.visFrac, sensor?.pd ?? pdDefault, shadowFrac);
}

// Event-camera response to apparent motion, fitted to a real recording (EVOS dataset, Crain & Ulrich 2025, run
// CC-T-NOM, DVXplorer Micro; fit by fit_motion.py: correlation 0.97 with the measured event rate):
//   signal = 1 - exp(-(omega + EVENT_LOOM * loom) / EVENT_S0)
//   omega = speed across the view [rad/s], loom = |range rate| / range [1/s] (growing or shrinking in the image)
// ponytail: measured on a 0.3 m foil-wrapped spacecraft model at 2 m, applied to people at 10-45 m, and it ignores
// limb motion; re-fit on event recordings of walking people when such data is available.
export const EVENT_S0 = 0.0077, EVENT_LOOM = 0.10;
// Dark vs nominal light on the same EVOS manoeuvre: about 10% fewer events at 0.6 lux (CIRC-TR-DARK vs CIRC-TR-NOM).
export const EVENT_SHADOW = 0.1;

/** How strongly an event camera responds to a person at view v moving with velocity vel = [vx, vy] (0..1). */
export function motionSignal(sensor, v, vel) {
  if (!vel) return 1;
  const r = Math.max(v.range, 1), bx = Math.cos(v.bearing ?? v.visCenter), by = Math.sin(v.bearing ?? v.visCenter);
  const radial = vel[0] * bx + vel[1] * by, across = -vel[0] * by + vel[1] * bx;
  return 1 - Math.exp(-(Math.abs(across) / r + EVENT_LOOM * Math.abs(radial) / r) / EVENT_S0);
}

/** "Hidden" for scoring and display: in view of the camera's field, but effectively undetectable. */
export const isHidden = (v) => v.inFov && (v.pd ?? (v.visFrac >= 0.25 ? 1 : 0)) < 0.15;

/** Why a hidden person is hidden, for display: measured cast-shadow darkness, otherwise the rock in the way. */
export const hiddenCause = (v) => ((v.shadow ?? 0) > 0.3 ? 'shadow' : 'rock');
