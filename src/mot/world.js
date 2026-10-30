// Lunar-surface scene: a rover mast camera watches astronauts / rovers walking among boulders.
import { RNG } from '../core/rng.js';
import { clamp, wrapAngle } from '../core/linalg.js';
import { visibility, detectionProb, shadowFraction, sensorPd } from './occlusion.js';
import { personSignature, clutterSignature } from './appearance.js';

export const MOT_SCENARIOS = {
  // 'work' = the boulder field, but people stop for about 6 s at each worksite (as on a real EVA)
  boulders: { label: 'Artemis EVA · Boulder field', blurb: 'Five crew/rovers walk through a boulder field. They keep disappearing behind rocks.', targets: 5, boulders: 9, pd: 0.95, clutter: 0.8 },
  crossing: { label: 'Worksite · Crossing paths', blurb: 'Eight targets, paths cross often behind cover. Hardest for keeping identities.', targets: 8, boulders: 7, pd: 0.92, clutter: 1.2 },
  polar: { label: 'South pole · Long shadows', blurb: 'Same boulder field under the low polar sun: people also vanish into long cast shadows, not just behind rocks.', targets: 5, boulders: 9, pd: 0.95, clutter: 0.8, shadows: true },
  work: { label: 'Artemis EVA · Stop and work', blurb: 'The boulder field again, but people stop for a few seconds at each worksite. An ordinary camera still sees them; an event camera does not.', targets: 5, boulders: 9, pd: 0.95, clutter: 0.8, dwell: 6 },
  // 'ltv' = the boulder field with two Lunar Terrain Vehicles (Artemis crew rovers) driving among the walkers
  ltv: { label: 'Artemis EVA · Lunar Terrain Vehicle', blurb: 'Two crew vehicles drive between worksites at walking-to-jogging speed, turning around rocks, while four astronauts walk.', targets: 4, vehicles: 2, boulders: 9, pd: 0.95, clutter: 0.8 },
  // 'drive' = the camera rover itself drives between worksites (for SLAM: it must work out where it is from what it sees)
  drive: { label: 'Rover drives · SLAM', blurb: 'The camera rover drives around the boulder field while astronauts walk and two vehicles drive. Where is the rover? That is the SLAM question.', targets: 4, vehicles: 2, boulders: 9, pd: 0.95, clutter: 0.8, roverDrives: true },
  open: { label: 'Open plain · Sparse cover', blurb: 'A single boulder: fewer, shorter occlusions. Shows how results scale with how much cover there is.', targets: 5, boulders: 1, pd: 0.95, clutter: 0.8 },
};

export const CAMERA = {
  x: 0, y: -4, th: Math.PI / 2, fov: (100 * Math.PI) / 180, range: 48,
  sigB: 0.004,                       // bearing noise [rad] (~2.5 px)
  sigR0: 0.08, sigRk: 0.0012,        // stereo depth noise: sigR = sigR0 + sigRk * r^2  [m]
  imgW: 960, imgH: 300, targetH: 1.8, targetR: 0.4,
};
// 3-D scanning lidar on the rover mast: 64 laser channels swept over 140 deg. Active sensor: works in darkness.
// Centroid of a person's returns: ~5 cm in range, ~0.17 deg in bearing; fewer false alarms than the camera.
export const LIDAR = { x: 0, y: -4, h: 2.0, th: Math.PI / 2, fov: (140 * Math.PI) / 180, range: 45, fullRange: 32, active: true,
  pd: 0.97, clutter: 0.3, sigB: 0.003, sigR0: 0.05, sigRk: 0.00002, imgW: 960, imgH: 300, targetH: 1.8, targetR: 0.4,
  modelInflate: 2,   // tracker's noise model x2: covers the centroid shift of a partly hidden person (else precise returns get gated out)
  channels: 64, elevMin: (-25 * Math.PI) / 180, elevMax: (4 * Math.PI) / 180, azStep: (0.25 * Math.PI) / 180 };
// Event camera on the rover mast, in place of the ordinary camera: same view and (stereo pair) depth noise, but it
// detects motion, not presence (occlusion.js: sensorPd, motionSignal). No colour, so no appearance signature.
// ponytail: modelled as a stereo pair with the frame camera's depth noise; a single event camera gives bearing only.
export const EVENT = { event: true, sigB: 0.003 };
export const LANDER = { x: 24, y: 46, h: 6, fov: (90 * Math.PI) / 180, range: 60 };   // lander camera, 6 m up
export const LANDER_AIM = { x: 0, y: 18 };                                             // ... aimed at the worksite
// Lunar Terrain Vehicle (truth model). NASA's LTV requirement is ~15 km/h top speed; on a worksite it drives slower.
// Car-like: it cannot turn on the spot, so yaw rate <= v / minTurnR. rt: half-width seen by the camera (body ~2.5 m wide).
// It drives like an autonomous rover: plans a route around the rocks (grid A* + shortcutting), follows it with pure
// pursuit, slows for curves (lateral acceleration aLat) and before stopping, stops for anyone in its path, and parks
// for a few seconds at each worksite.
export const VEHICLE = { vMin: 2.0, vMax: 3.5, minTurnR: 4.0, rt: 1.2, margin: 0.8, aAcc: 0.6, aDec: 0.8, aLat: 1.0,
  park: [3, 8], yieldDist: 5, yieldCone: (35 * Math.PI) / 180 };
// The camera rover when it drives (scenario 'drive'): small and slow, like NASA's RASSOR. Same planner as the vehicles.
// ahead: next worksite within 70 deg of the heading (a full U-turn at a 4 m turning radius drives a circle, which is
// both unlike a real rover and the worst motion for a single camera: nearly pure rotation)
export const ROVER = { vc: 1.0, rt: 0.8, ahead: (70 * Math.PI) / 180 };
export const sigRange = (cam, r) => cam.sigR0 + cam.sigRk * r * r;

export class MotWorld {
  constructor(scenario = 'boulders', seed = 7, overrides = {}, { lander = false, lidar = false, eyes = 'camera' } = {}) {
    this.cfg = { ...MOT_SCENARIOS[scenario] || MOT_SCENARIOS.boulders, ...overrides };
    this.id = scenario; this.seed = seed;
    this.cam = eyes === 'event' ? { ...CAMERA, ...EVENT } : { ...CAMERA };
    this.cam0 = { ...this.cam };   // where the rover started: worksites stay in this field even when the rover drives
    this.bounds = { xmin: -38, xmax: 38, ymin: 0, ymax: 46 };
    this.rng = new RNG(seed * 9973 + 17);   // world + truth motion
    this.srng = new RNG(seed * 7717 + 5);   // sensor noise (separate stream: same truth regardless of detector settings)
    this.arng = new RNG(seed * 4111 + 3);   // appearance noise (own stream: adding signatures changes nothing else)
    // low polar sun, matching the 3-D render: azimuth/elevation in the simulator frame
    this.sun = { az: Math.atan2(-25, -60), el: Math.atan2(9, Math.hypot(60, 25)) };
    this.t = 0; this.dt = 0.1; this.frame = 0;
    // optional second camera on a lander at 6 m, looking back across the worksite (own random streams, so the
    // rover's detections, and every single-camera result, are unchanged)
    this.lander = lander ? { ...CAMERA, ...LANDER, th: Math.atan2(LANDER_AIM.y - LANDER.y, LANDER_AIM.x - LANDER.x) } : null;
    if (lander) { this.lrng = new RNG(seed * 6151 + 11); this.larng = new RNG(seed * 3313 + 29); }
    this.lidar = lidar ? { ...LIDAR } : null;                 // own random stream too
    if (lidar) this.drng = new RNG(seed * 2711 + 13);
    this._placeBoulders(); this._spawnTargets();
    // a driving rover is a vehicle the cameras ride on; it is not a target (nobody tracks it, SLAM estimates it)
    if (this.cfg.roverDrives) this.rover = { kind: 'vehicle', rt: ROVER.rt, x: this.cam.x, y: this.cam.y, h: this.cam.th, v: 0, vc: ROVER.vc, w: 0, pause: 0, trail: [] };
  }

  _placeBoulders() {
    const r = this.rng, cam = this.cam; this.boulders = [];
    for (let k = 0; this.boulders.length < this.cfg.boulders && k < 2000; k++) {
      const range = r.uniform(9, 36), off = r.uniform(-0.42, 0.42) * cam.fov, br = r.uniform(1.4, 3.2);
      // ~35% are low rocks a camera on a 2.2 m mast can see heads over; the rest are full occluders
      const h = r.next() < (this.cfg.lowFrac ?? 0.35) ? r.uniform(0.7, 1.4) : r.uniform(2.4, 3.6);
      const b = { x: cam.x + range * Math.cos(cam.th + off), y: cam.y + range * Math.sin(cam.th + off), r: br, h };
      if (this.boulders.some((o) => Math.hypot(o.x - b.x, o.y - b.y) < o.r + b.r + 3.5)) continue;
      this.boulders.push(b);
    }
  }

  _spawnTargets() {
    const r = this.rng; this.targets = [];
    for (let i = 0; i < this.cfg.targets; i++) {
      let p;
      for (let k = 0; k < 500; k++) {
        { const rr = r.uniform(12, this.cam.range - 8), oo = r.uniform(-0.38, 0.38) * this.cam.fov; p = { x: this.cam.x + rr * Math.cos(this.cam.th + oo), y: this.cam.y + rr * Math.sin(this.cam.th + oo) }; }
        if (!this.boulders.some((b) => Math.hypot(b.x - p.x, b.y - p.y) < b.r + 1.2)) break;
      }
      const h = r.uniform(-Math.PI, Math.PI), v = r.uniform(0.6, 1.3);
      this.targets.push({ id: i + 1, x: p.x, y: p.y, h, v, trail: [], vis: this.view(p.x, p.y) });
    }
    // vehicles after the people, so every walker-only scenario draws exactly the same random numbers as before
    for (let i = 0; i < (this.cfg.vehicles || 0); i++) {
      let p;
      for (let k = 0; k < 500; k++) {
        const rr = r.uniform(14, this.cam.range - 8), oo = r.uniform(-0.38, 0.38) * this.cam.fov;
        p = { x: this.cam.x + rr * Math.cos(this.cam.th + oo), y: this.cam.y + rr * Math.sin(this.cam.th + oo) };
        if (!this.boulders.some((b) => Math.hypot(b.x - p.x, b.y - p.y) < b.r + VEHICLE.rt + 1)) break;
      }
      const h = r.uniform(-Math.PI, Math.PI), vc = r.uniform(VEHICLE.vMin, VEHICLE.vMax);   // vc: cruise speed
      this.targets.push({ id: this.targets.length + 1, kind: 'vehicle', rt: VEHICLE.rt, x: p.x, y: p.y, h, v: 0, vc, w: 0, pause: 0, trail: [], vis: this.view(p.x, p.y, this.cam, null, VEHICLE.rt) });
    }
  }

  _goal(clear = 1.5, from = null, minDist = 0, ahead = null) {
    const r = this.rng, cam = this.cam0;
    for (let k = 0; k < 200; k++) {
      const rr = r.uniform(11, cam.range - 6), oo = r.uniform(-0.4, 0.4) * cam.fov;
      const g = { x: cam.x + rr * Math.cos(cam.th + oo), y: cam.y + rr * Math.sin(cam.th + oo) };
      if (from && Math.hypot(g.x - from.x, g.y - from.y) < minDist) continue;
      // ahead (radians): only worksites within this angle of the heading, as a rover's planner avoids U-turns
      // (relaxed after 150 tries, so a rover facing out of the field still finds somewhere to go)
      if (ahead && k < 150 && Math.abs(wrapAngle(Math.atan2(g.y - from.y, g.x - from.x) - from.h)) > ahead) continue;
      if (!this.boulders.some((b) => Math.hypot(b.x - g.x, b.y - g.y) < b.r + clear)) return g;
    }
    return { x: cam.x, y: cam.y + 25 };
  }

  /** Truth motion: goal-directed walking between worksites (EVA-like), small heading noise, steering around boulders. */
  step() {
    const dt = this.dt, r = this.rng;
    if (this.rover) {                     // move the rover first; every camera on it moves with it
      const R = this.rover; this._drive(R);
      for (const c of [this.cam, this.lidar]) if (c) { c.x = R.x; c.y = R.y; c.th = R.h; }
    }
    for (const t of this.targets) {
      if (t.kind === 'vehicle') { this._drive(t); continue; }
      if (!t.goal || Math.hypot(t.goal.x - t.x, t.goal.y - t.y) < 1.5) {
        t.goal = this._goal();
        // cfg.dwell > 0: stop and work at each site for about that many seconds (off by default: no extra random draws)
        if (this.cfg.dwell) t.pause = this.cfg.dwell * r.uniform(0.5, 1.5);
      }
      if (t.pause > 0) {                                   // standing still: in plain view, but not moving
        t.pause -= dt; t.vel = [0, 0]; t.vis = this.view(t.x, t.y, this.cam, t.vel);
        if (this.lander || this.lidar) this._fuse(t);
        continue;
      }
      let sx = Math.cos(Math.atan2(t.goal.y - t.y, t.goal.x - t.x)), sy = Math.sin(Math.atan2(t.goal.y - t.y, t.goal.x - t.x));
      for (const b of this.boulders) {                     // walk around rocks, not through them
        const dx = t.x - b.x, dy = t.y - b.y, d = Math.hypot(dx, dy) - b.r;
        if (d < 2.0) { const w = 1.6 * (2.0 - Math.max(d, 0.05)) / 2.0; sx += (dx / (d + b.r)) * w; sy += (dy / (d + b.r)) * w; }
      }
      t.h += clamp(wrapAngle(Math.atan2(sy, sx) - t.h), -1, 1) * 1.2 * dt + 0.08 * Math.sqrt(dt) * r.randn();
      t.x += t.v * Math.cos(t.h) * dt; t.y += t.v * Math.sin(t.h) * dt;
      for (const b of this.boulders) { // never inside a rock
        const dx = t.x - b.x, dy = t.y - b.y, d = Math.hypot(dx, dy);
        if (d < b.r + 0.4) { t.x = b.x + (dx / d) * (b.r + 0.4); t.y = b.y + (dy / d) * (b.r + 0.4); }
      }
      t.vel = [t.v * Math.cos(t.h), t.v * Math.sin(t.h)];
      t.vis = this.view(t.x, t.y, this.cam, t.vel);
      if (this.lander || this.lidar) this._fuse(t);
      if (this.frame % 3 === 0) { t.trail.push([t.x, t.y]); if (t.trail.length > 120) t.trail.shift(); }
    }
    this.t += dt; this.frame++;
  }

  /** Truth motion of a vehicle: plan a route, follow it (pure pursuit), manage speed, yield, park. */
  _drive(t) {
    const dt = this.dt, r = this.rng, V = VEHICLE;
    // Closed-loop navigation (slam/drive_closed_loop.mjs): the rover plans and steers from `nav.belief`, the pose its
    // SLAM (or its wheels) reports, while the truth below moves it. Rocks close by are still avoided from live
    // sensing (the arc check uses the true pose: that is what hazard cameras measure). Without nav: unchanged.
    const nav = t === this.rover ? this.roverNav : null, me = nav ? nav.belief : t;
    if (nav && nav.idx >= nav.mission.length) { t.v = 0; t.w = 0; t.vel = [0, 0]; return; }   // mission done: stop
    if (t.pause > 0) t.pause -= dt;                                   // parked at a worksite
    else {
      if (!t.route || this.t > t.legDeadline) {                    // new leg: a worksite at least 12 m away, then a route to it
        t.goal = nav ? nav.mission[nav.idx] : this._goal(V.rt + V.margin + 1.5, t, 12, t === this.rover ? ROVER.ahead : null);
        t.route = this._plan(me, t.goal); t.wp = 0;
        let len = Math.hypot(t.route[0].x - me.x, t.route[0].y - me.y);
        for (let i = 1; i < t.route.length; i++) len += Math.hypot(t.route[i].x - t.route[i - 1].x, t.route[i].y - t.route[i - 1].y);
        t.legDeadline = this.t + 30 + (3 * len) / t.vc;              // circling a spot it cannot reach: give up, pick another
      }
      // pure pursuit (Coulter 1992): steer onto the arc through a route point one look-ahead distance away
      const Ld = clamp(0.8 * t.v + 2, 2.5, 5);                     // short look-ahead: cuts corners less, so it stays clear of rocks
      while (t.wp < t.route.length - 1 && Math.hypot(t.route[t.wp].x - me.x, t.route[t.wp].y - me.y) < Ld) t.wp++;
      const p = t.route[t.wp], dp = Math.max(Math.hypot(p.x - me.x, p.y - me.y), 0.5), alpha = wrapAngle(Math.atan2(p.y - me.y, p.x - me.x) - me.h);
      // curvature [1/m]; target behind: full lock toward it (sin(alpha) ~ 0 there would mean "drive straight away")
      const kpp = Math.abs(alpha) > Math.PI / 2 ? Math.sign(alpha || 1) / V.minTurnR : clamp((2 * Math.sin(alpha)) / dp, -1 / V.minTurnR, 1 / V.minTurnR);
      // safety layer (dynamic-window style, Fox et al. 1997): of 13 steering arcs, take the one nearest the pursuit arc
      // that stays clear of rocks over the stopping distance; if none is clear, the one that gets furthest, slowly
      // other vehicles (and the rover) are obstacles too. Right of way: the rover, then the lower id.
      const others = [...this.targets.filter((o) => o.kind === 'vehicle' && o !== t), ...(this.rover && this.rover !== t ? [this.rover] : [])];
      const prio = (o) => (o === this.rover ? 0 : o.id);
      const horizon = 2.5 + (t.v * t.v) / (2 * V.aDec) + t.v, free = (kk) => this._arcFree(t, kk, horizon, others);
      let k = kpp, room = free(kpp);
      if (room < horizon) {
        let best = null;
        for (let i = 0; i <= 12; i++) {
          const kk = (i / 6 - 1) / V.minTurnR, f = free(kk);
          if (f >= horizon ? !best || best.f < horizon || Math.abs(kk - kpp) < Math.abs(best.k - kpp) : !best || (best.f < horizon && f > best.f)) best = { k: kk, f };
        }
        k = best.k; room = best.f;
      }
      room = free(k); const blocker = this._hit;                       // what limits the chosen arc, if anything
      let left = dp; for (let i = t.wp; i < t.route.length - 1; i++) left += Math.hypot(t.route[i + 1].x - t.route[i].x, t.route[i + 1].y - t.route[i].y);
      // speed: cruise, but slow for the curve and to stop at the worksite; crawl while turning around
      let vT = Math.min(t.vc, Math.sqrt(V.aLat / Math.max(Math.abs(k), 1e-3)), Math.sqrt(2 * V.aDec * Math.max(left - 0.5, 0)),
        Math.max(0.4, Math.sqrt(2 * V.aDec * Math.max(room - 1.5, 0))));      // and to stop short of a rock (creep, never stall)
      if (Math.abs(alpha) > Math.PI / 2) vT = Math.min(vT, 1.0);
      for (const o of this.targets) {                                 // yield: stop for an astronaut in front
        if (o === t || o.kind === 'vehicle') continue;
        const d = Math.hypot(o.x - t.x, o.y - t.y);
        if (d < V.yieldDist + V.rt && Math.abs(wrapAngle(Math.atan2(o.y - t.y, o.x - t.x) - t.h)) < V.yieldCone) vT = 0;
      }
      for (const o of others) {                                       // give way to a moving vehicle with right of way just ahead
        const d = Math.hypot(o.x - t.x, o.y - t.y);
        if (prio(o) < prio(t) && o.v > 0.2 && d < 8 + o.rt + t.rt && Math.abs(wrapAngle(Math.atan2(o.y - t.y, o.x - t.x) - t.h)) < Math.PI / 4) vT = 0;
      }
      // boxed in by a vehicle: wait if it has right of way, else edge round it on the freest arc (never both waiting)
      if (blocker && room < 1.0) vT = prio(blocker) < prio(t) ? 0 : Math.min(vT, 0.4);
      // too close to a vehicle with right of way: back off slowly in a straight line to make room (a car cannot
      // turn tighter than its radius, so the other one may need the space to get round)
      if (others.some((o) => prio(o) < prio(t) && Math.hypot(o.x - t.x, o.y - t.y) < o.rt + t.rt + 0.5)) { vT = -0.4; k = 0; }
      t.v += clamp(vT - t.v, -V.aDec * dt, V.aAcc * dt);
      t.w = t.v * k;                                                   // |w| <= v / minTurnR: a car cannot turn on the spot
      t.h += t.w * dt + 0.01 * Math.sqrt(dt) * r.randn();
      // regolith slip (closed loop only): the wheels turn for v, the rover moves less, and the heading creeps
      const slip = nav ? nav.slip : 0;
      if (nav) t.h += nav.yawBias * dt;
      t.x += t.v * (1 - slip) * Math.cos(t.h) * dt; t.y += t.v * (1 - slip) * Math.sin(t.h) * dt;
      if (left < 1.5 || Math.hypot(t.goal.x - me.x, t.goal.y - me.y) < 3) {   // arrived (a car parks within ~3 m; it cannot pivot onto a point): park and work
        if (nav) nav.arrivals.push({ idx: nav.idx, t: this.t, missM: Math.hypot(t.goal.x - t.x, t.goal.y - t.y) }), nav.idx++;
        t.v = 0; t.w = 0; t.route = null; t.pause = r.uniform(...V.park);
      }
    }
    if (t.pause > 0) { t.v = 0; t.w = 0; }
    for (const o of [...this.targets, this.rover]) {                 // safety net: two vehicles never overlap
      if (!o || o === t || o.kind !== 'vehicle') continue;
      const dx = t.x - o.x, dy = t.y - o.y, d = Math.hypot(dx, dy) || 1e-6, m = t.rt + o.rt;
      if (d < m) { t.x = o.x + (dx / d) * m; t.y = o.y + (dy / d) * m; }
    }
    for (const b of this.boulders) {                                 // safety net only; the planner keeps clear of rocks
      const dx = t.x - b.x, dy = t.y - b.y, d = Math.hypot(dx, dy);
      if (d < b.r + V.rt) { t.x = b.x + (dx / d) * (b.r + V.rt); t.y = b.y + (dy / d) * (b.r + V.rt); if (nav) nav.bumps++; }
    }
    t.vel = [t.v * Math.cos(t.h), t.v * Math.sin(t.h)];
    t.vis = this.view(t.x, t.y, this.cam, t.vel, t.rt);
    if (this.lander || this.lidar) this._fuse(t);
    if (this.frame % 3 === 0) { t.trail.push([t.x, t.y]); if (t.trail.length > 120) t.trail.shift(); }
  }

  /** How far (m, up to `len`) the vehicle can drive along an arc of curvature k before its body would touch a rock. */
  _arcFree(t, k, len, others = []) {
    const clear = VEHICLE.rt + 0.3;
    this._hit = null;                                                // the vehicle that blocks this arc, if one does
    for (let s = 0.5; s <= len; s += 0.5) {
      const h = t.h + k * s, x = Math.abs(k) < 1e-6 ? t.x + s * Math.cos(t.h) : t.x + (Math.sin(h) - Math.sin(t.h)) / k;
      const y = Math.abs(k) < 1e-6 ? t.y + s * Math.sin(t.h) : t.y - (Math.cos(h) - Math.cos(t.h)) / k;
      if (this.boulders.some((b) => Math.hypot(b.x - x, b.y - y) < b.r + clear)) return s - 0.5;
      // other vehicles: where they are now and where they will be in 1 s (constant velocity), with a 0.6 m gap
      const o = others.find((v) => { const g = v.rt + t.rt + 0.6, vx = v.vel ? v.vel[0] : 0, vy = v.vel ? v.vel[1] : 0;
        return Math.hypot(v.x - x, v.y - y) < g || Math.hypot(v.x + vx - x, v.y + vy - y) < g; });
      if (o) { this._hit = o; return s - 0.5; }
    }
    return len;
  }

  /**
   * Route from a to b around the boulders for a vehicle: 8-connected A* on a 1 m grid (rocks grown by the vehicle's
   * half-width + margin), then shortcut to the furthest point in clear line of sight. Returns [{x, y}, ..., b].
   * ponytail: linear-scan open list, fine for ~3,500 cells planned every ~15 s; use a binary heap for bigger maps.
   */
  _plan(a, b) {
    const V = VEHICLE, B = this.bounds, R = 1, x0 = B.xmin - 6, y0 = B.ymin - 8, nx = Math.ceil((B.xmax - x0 + 6) / R), ny = Math.ceil((B.ymax - y0 + 6) / R);
    const grow = (o) => o.r + V.rt + V.margin;
    const free = (x, y) => !this.boulders.some((o) => Math.hypot(o.x - x, o.y - y) < grow(o));
    const clearSeg = (p, q) => this.boulders.every((o) => {
      const vx = q.x - p.x, vy = q.y - p.y, L2 = vx * vx + vy * vy || 1e-9, s = clamp(((o.x - p.x) * vx + (o.y - p.y) * vy) / L2, 0, 1);
      return Math.hypot(p.x + s * vx - o.x, p.y + s * vy - o.y) >= grow(o) + 0.3;   // extra room: the follower cuts corners a little
    });
    const cell = (p) => [clamp(Math.round((p.x - x0) / R), 0, nx - 1), clamp(Math.round((p.y - y0) / R), 0, ny - 1)];
    const [si, sj] = cell(a), [gi, gj] = cell(b), id = (i, j) => j * nx + i, goal = id(gi, gj);
    const g = new Float64Array(nx * ny).fill(Infinity), from = new Int32Array(nx * ny).fill(-1), done = new Uint8Array(nx * ny);
    const hh = (i, j) => { const dx = Math.abs(i - gi), dy = Math.abs(j - gj); return Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy); };
    g[id(si, sj)] = 0; const open = [id(si, sj)];
    while (open.length) {
      let bi = 0; for (let k = 1; k < open.length; k++) if (g[open[k]] + hh(open[k] % nx, (open[k] / nx) | 0) < g[open[bi]] + hh(open[bi] % nx, (open[bi] / nx) | 0)) bi = k;
      const c = open[bi]; open.splice(bi, 1);
      if (done[c]) continue; done[c] = 1;
      if (c === goal) break;
      const ci = c % nx, cj = (c / nx) | 0;
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
        const i = ci + di, j = cj + dj; if ((!di && !dj) || i < 0 || j < 0 || i >= nx || j >= ny) continue;
        const n = id(i, j); if (done[n] || (n !== goal && !free(x0 + i * R, y0 + j * R))) continue;
        const ng = g[c] + (di && dj ? Math.SQRT2 : 1) * R;
        if (ng < g[n]) { g[n] = ng; from[n] = c; open.push(n); }
      }
    }
    if (from[goal] < 0) return [{ x: b.x, y: b.y }];                   // no route found: head straight for it
    const cells = []; for (let c = goal; c >= 0; c = from[c]) cells.push({ x: x0 + (c % nx) * R, y: y0 + ((c / nx) | 0) * R });
    cells.reverse(); cells[cells.length - 1] = { x: b.x, y: b.y };
    const route = []; let p = { x: a.x, y: a.y }, k = 0;
    while (k < cells.length - 1) {                                     // shortcut: jump to the furthest visible cell
      let far = k + 1; for (let m = cells.length - 1; m > k + 1; m--) if (clearSeg(p, cells[m])) { far = m; break; }
      route.push(cells[far]); p = cells[far]; k = far;
    }
    return route.length ? route : [{ x: b.x, y: b.y }];
  }

  /** With a lander: what it sees of person t, and the fused view used for scoring (hidden = hidden from BOTH cameras). */
  _fuse(t) {
    const views = [t.vis];
    if (this.lander) views.push(t.visL = this.view(t.x, t.y, this.lander, null, t.rt));
    if (this.lidar) views.push(t.visD = this.view(t.x, t.y, this.lidar, null, t.rt));
    t.visFused = { inFov: views.some((v) => v.inFov), pd: Math.max(...views.map((v) => (v.inFov ? v.pd : 0))),
      visFrac: Math.max(...views.map((v) => v.visFrac)), shadow: t.vis.shadow };
  }

  /** What a camera can see of a person at (x, y): geometry, optional cast-shadow darkness, and detection probability. */
  view(x, y, cam = this.cam, vel = null, rt = this.cam.targetR) {
    const v = visibility(cam, this.boulders, x, y, rt);
    v.shadow = this.cfg.shadows ? shadowFraction(this.boulders, x, y, this.sun) : 0;
    v.pd = sensorPd(cam, v, this.cfg.pd, cam.active ? 0 : v.shadow, vel);   // the lidar brings its own light; an event camera needs motion
    return v;
  }

  /**
   * One camera frame of detections. The tracker gets ONLY `dets` (no identities). `gt[i]` is the true target id
   * of dets[i] (0 = clutter), kept separate for scoring.
   */
  sense() { return this._sense(this.cam, 'vis', this.srng, this.cam.event ? null : this.arng); }
  /** The lander camera's frame (same detector model, its own viewpoint and random streams). */
  senseLander() { return this._sense(this.lander, 'visL', this.lrng, this.larng); }
  /** The lidar's frame: centroids of person-shaped point clusters. No colour, so no appearance signature. */
  senseLidar() { return this._sense(this.lidar, 'visD', this.drng, null); }

  _sense(cam, field, s, as) {
    const dets = [], gt = [];
    if (field !== 'vis') for (const t of this.targets) if (!t[field]) this._fuse(t);
    for (const t of this.targets) {
      const v = t[field] || this.view(t.x, t.y, cam, null, t.rt);
      if (s.next() >= v.pd) continue;
      const sr = sigRange(cam, v.range);
      const sb = cam.sigB * (v.visFrac < 0.9 ? 2 : 1);  // partially hidden: centroid jitters more
      dets.push(this._det(v.range + sr * s.randn(), v.visCenter + sb * s.randn(), cam));
      if (as) dets[dets.length - 1].feat = personSignature(t.id, v.visFrac, v.shadow ?? 0, as);   // colour signature (cameras only)
      gt.push(t.id);
    }
    const nFalse = poisson(cam.clutter ?? this.cfg.clutter, s);
    for (let k = 0; k < nFalse; k++) {
      dets.push(this._det(s.uniform(4, cam.range), cam.th + s.uniform(-cam.fov / 2, cam.fov / 2), cam));
      if (as) dets[dets.length - 1].feat = clutterSignature(as);
      gt.push(0);
    }
    return { dets, gt };
  }

  _det(range, bearing, cam = this.cam) {
    const f = cam.imgW / 2 / Math.tan(cam.fov / 2), phi = wrapAngle(bearing - cam.th);
    return { range, bearing, u: cam.imgW / 2 - f * Math.tan(phi), hPx: (f * cam.targetH) / Math.max(range, 1) };
  }
}

function poisson(lam, rng) { let k = 0, p = Math.exp(-lam), s = p; const u = rng.next(); while (u > s && k < 50) { k++; p *= lam / k; s += p; } return k; }
