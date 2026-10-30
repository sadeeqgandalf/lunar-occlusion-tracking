// The lab camera as a real camera: shared by record_drive.mjs (open loop) and drive_closed_loop.mjs (closed loop).
// Camera model (what a real camera adds to the rendered scene):
//  * auto-exposure: meters the ground (lower 60% of the image; the sky is black), aims its median at mid-grey, moves
//    at most ~40% per frame (settles in about a second) and is limited to 1/4x .. 16x of the base exposure;
//  * sensor noise: shot noise + read noise. Full well 10,000 e-, read noise 3 e- (small CMOS pixel); a darker scene
//    needs more gain, so it gets grainier, as on a real camera.
//    ponytail: noise added after the renderer's tone curve (approximate); a raw-pipeline sim would add it before.
// No motion blur or lens distortion: the rover drives at <= 1 m/s and the lab camera is an ideal pinhole.
export const AE = { base: 1.15, target: 110, min: 0.25, max: 16, step: 1.4 };
export const SENSOR = { fullWell: 10000, read: 3 };

// Runs inside the lab page (Playwright page.evaluate): advance the world `steps` x 0.1 s, render the rover camera,
// apply auto-exposure and sensor noise, return PNGs (base64) and the camera's true pose.
// Closed loop: if world.roverNav is set, its belief is dead-reckoned from the wheels after every step (the rover
// steers by it); render: false skips the camera entirely (wheels-only and truth drivers need no images).
export const shootInPage = ({ steps, W, H, AE, SENSOR, seed, i, settleOnly, baseline, render = true }) => {
    const L = window.__lab, s = L.sess;
    const nav = s.world.roverNav;
    if (!settleOnly) for (let k = 0; k < steps; k++) {
      s.step();
      if (nav) {   // wheel odometry: what the wheels say the rover did (slip is invisible to them)
        const R = s.world.rover, dt = s.world.dt, b = nav.belief;
        b.h += R.w * dt; b.x += R.v * Math.cos(b.h) * dt; b.y += R.v * Math.sin(b.h) * dt;
      }
    }
    const R = s.world.rover, rover = R && { x: R.x, y: R.y, h: R.h, v: R.v };
    if (!render) return { t: s.world.t, rover, nav: nav && { idx: nav.idx, arrivals: nav.arrivals, bumps: nav.bumps } };
    L.w3d.render(s, { heat: null, sel: 0, color: '#ffffff', showTruth: false, marks: [] });
    L.exp ??= AE.base;
    const c = L.w3d.capture(s, W, H, L.exp); if (!c) return null;
    const px = c.px, used = L.exp;
    // auto-exposure for the NEXT frame, from this frame's ground (GL rows are bottom-up: rows 0..0.6H are the bottom)
    const lum = []; for (let y = 0; y < Math.floor(0.6 * H); y += 2) for (let x = 0; x < W; x += 4) { const k = 4 * (y * W + x); lum.push(0.299 * px[k] + 0.587 * px[k + 1] + 0.114 * px[k + 2]); }
    lum.sort((a, b) => a - b); const med = Math.max(lum[lum.length >> 1], 1);
    L.exp = Math.min(AE.max * AE.base, Math.max(AE.min * AE.base, L.exp * Math.min(AE.step, Math.max(1 / AE.step, (AE.target / med) ** 0.6))));
    if (settleOnly) return { settle: true };
    // sensor noise, seeded per frame and eye (same recording every time; the two eyes get independent noise)
    const G = used / AE.base;
    const develop = (px, eye) => {
      let a = (seed * 1000003 + i * 7919 + eye * 104729) >>> 0; const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
      const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
      const g = cv.getContext('2d'), im = g.createImageData(W, H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const k = 4 * ((H - 1 - y) * W + x), o = 4 * (y * W + x);
        for (let ch = 0; ch < 3; ch++) {
          const e = Math.pow(px[k + ch] / 255, 2.2) * SENSOR.fullWell / G;                    // electrons collected
          const ne = e + Math.sqrt(e + SENSOR.read * SENSOR.read) * gauss();
          im.data[o + ch] = Math.round(255 * Math.pow(Math.min(1, Math.max(0, (ne * G) / SENSOR.fullWell)), 1 / 2.2));
        }
        im.data[o + 3] = 255;
      }
      g.putImageData(im, 0, 0);
      return cv.toDataURL('image/png').split(',')[1];
    };
    // right eye: same instant, same exposure (stereo pairs share exposure)
    const right = baseline ? develop(L.w3d.capture(s, W, H, used, baseline).px, 1) : null;
    return { png: develop(px, 0), right, t: s.world.t, pos: c.pos, quat: c.quat, exposure: used, rover, nav: nav && { idx: nav.idx, arrivals: nav.arrivals, bumps: nav.bumps } };
  };
