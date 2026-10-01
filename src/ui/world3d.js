// Real-time 3-D view of the Occlusion Lab (Three.js r169, vendored).
// The simulation stays the source of truth; this module only draws it:
//   sim ground (x, y) -> three (x, 0, -y), y-up. Heights in metres.
// Official NASA models (assets/nasa): astronaut, RASSOR (camera robot), Apollo Lunar Module.
import { THREE, toV, hash, fbm, makeRenderer, fitRenderer, loadModel, starfield, drawLabels } from './three-common.js';
import { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
import { isHidden, hiddenCause } from '../mot/occlusion.js';

const MAST_H = 2.2, PERSON_H = 1.8;

/** Ground height: gentle regolith undulation (kept <~0.3 m inside the work area so the planar sim stays valid) + scenery craters outside it. */
function makeHeight(craters) {
  return (x, y) => {
    let h = (fbm(x * 0.08, y * 0.08) - 0.5) * 0.5;
    for (const c of craters) {
      const d = Math.hypot(x - c.x, y - c.y) / c.r;
      if (d < 1.35) h += d < 1 ? -c.depth * (1 - d * d) : c.depth * 0.35 * Math.cos(((d - 1) / 0.35) * Math.PI / 2) ** 2; // bowl + raised rim
    }
    return h;
  };
}

export async function createWorld3D({ mainCanvas, mainOverlay, camCanvas, camOverlay }) {
  const R1 = makeRenderer(mainCanvas), R2 = makeRenderer(camCanvas);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);

  // lunar south-pole lighting: sun ~7 deg above the horizon -> long shadows; faint earthshine fill
  const sun = new THREE.DirectionalLight(0xfff4e0, 3.2);
  sun.position.set(-60, 9, 25); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -70, right: 70, top: 70, bottom: -70, near: 1, far: 220 });
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.04;
  scene.add(sun, sun.target); sun.target.position.set(0, 0, -20);
  scene.add(new THREE.HemisphereLight(0x8fa8ff, 0x202020, 0.12));

  // stars + Earth low on the horizon (as seen from the lunar south pole)
  scene.add(starfield());
  const earth = new THREE.Mesh(new THREE.SphereGeometry(18, 48, 32), new THREE.MeshStandardMaterial({ color: 0x3a6ea5, emissive: 0x0c2340, roughness: 0.8 }));
  earth.position.set(140, 30, -420); scene.add(earth);

  // models
  const [astro, rassor, lm] = await Promise.all([
    loadModel('assets/nasa/astronaut.glb', PERSON_H),
    loadModel('assets/nasa/rassor.glb', 1.7, 'xz'),
    loadModel('assets/nasa/apollo_lunar_module.glb', 7.0),
  ]);

  const dyn = new THREE.Group(); scene.add(dyn);
  let world = null, height = () => 0, people = new Map(), staticGroup = null;
  const pool = { ring: [], area: [], ping: [] };

  function setWorld(w) {
    world = w;
    if (staticGroup) scene.remove(staticGroup);
    staticGroup = new THREE.Group(); scene.add(staticGroup);
    for (const p of people.values()) dyn.remove(p); people.clear();
    for (const k of Object.keys(pool)) { for (const m of pool[k]) dyn.remove(m); pool[k] = []; }

    // scenery craters kept OUTSIDE the camera's field of view so the planar simulation stays exact
    const cam = w.cam, craters = [];
    for (let i = 0; craters.length < 14 && i < 400; i++) {
      const x = (hash(i, 7) - 0.5) * 150, y = (hash(i, 13) - 0.3) * 110, r = 3 + hash(i, 19) * 9;
      const off = Math.atan2(y - cam.y, x - cam.x) - cam.th, rng = Math.hypot(x - cam.x, y - cam.y);
      const inFov = Math.abs(Math.atan2(Math.sin(off), Math.cos(off))) < cam.fov / 2 + 0.25 && rng < cam.range + r + 4;
      if (!inFov && rng > r + 8) craters.push({ x, y, r, depth: r * 0.18 });
    }
    height = makeHeight(craters);
    // light from the simulator's sun direction, so rendered shadows fall exactly where the experiment computes them
    if (w.sun) { const K = 70; sun.position.set(K * Math.cos(w.sun.el) * Math.cos(w.sun.az), K * Math.sin(w.sun.el), -K * Math.cos(w.sun.el) * Math.sin(w.sun.az)); }

    // terrain
    const G = new THREE.PlaneGeometry(200, 160, 220, 176); G.rotateX(-Math.PI / 2);
    const pos = G.attributes.position, col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = -pos.getZ(i) + 20; // shift so the work area sits near the middle
      pos.setZ(i, -y); pos.setY(i, height(x, y));
      const g = 0.42 + (fbm(x * 0.6, y * 0.6) - 0.5) * 0.18; col.set([g, g, g * 0.97], i * 3);
    }
    G.setAttribute('color', new THREE.BufferAttribute(col, 3)); G.computeVertexNormals();
    const ground = new THREE.Mesh(G, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }));
    ground.receiveShadow = true; staticGroup.add(ground);

    // boulders: irregular rocks at the simulator's exact positions and heights (half-buried ellipsoids, as in the
    // visibility model), so what the 3-D camera shows is what the experiment measures
    w.boulders.forEach((b, i) => {
      const geo = new THREE.IcosahedronGeometry(1, 4), p = geo.attributes.position;
      for (let k = 0; k < p.count; k++) {
        const v = new THREE.Vector3().fromBufferAttribute(p, k), n = 0.78 + 0.45 * fbm(v.x * 1.7 + i * 3, v.y * 1.7 + v.z * 1.3);
        p.setXYZ(k, v.x * n, v.y * n, v.z * n);
      }
      geo.computeVertexNormals();
      const rock = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x7c7f86, roughness: 0.95, flatShading: true }));
      const hb = b.h ?? Math.max(2.6, b.r * 1.15);
      rock.scale.set(b.r, hb, b.r); rock.position.copy(toV(b.x, b.y, height(b.x, b.y)));
      rock.rotation.y = hash(i, 3) * Math.PI * 2; rock.castShadow = rock.receiveShadow = true; staticGroup.add(rock);
    });

    // camera robot (NASA RASSOR) with a mast, and the Apollo LM as a landmark outside the work area
    const base = toV(cam.x, cam.y, height(cam.x, cam.y));
    if (rassor) { const r = rassor.clone(); r.position.copy(base); r.rotation.y = 0; staticGroup.add(r); }
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, MAST_H, 12), new THREE.MeshStandardMaterial({ color: 0xd9d9d9, metalness: 0.6, roughness: 0.4 }));
    mast.position.copy(base).add(new THREE.Vector3(0, MAST_H / 2, 0)); mast.castShadow = true; staticGroup.add(mast);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.22, 0.28), new THREE.MeshStandardMaterial({ color: 0x222428, metalness: 0.5, roughness: 0.5 }));
    head.position.copy(base).add(new THREE.Vector3(0, MAST_H, 0)); staticGroup.add(head);
    if (lm) { const m = lm.clone(); m.position.copy(toV(-26, -9, height(-26, -9))); m.rotation.y = 0.6; staticGroup.add(m); }

    // people (NASA astronaut model), one per simulated target
    for (const t of w.targets) {
      const g = astro ? astro.clone() : new THREE.Mesh(new THREE.CapsuleGeometry(0.35, 1.1, 4, 12), new THREE.MeshStandardMaterial({ color: 0xffffff }));
      dyn.add(g); people.set(t.id, g);
    }
  }

  // ---------- cameras
  const free = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
  free.position.set(7, 11, 9);   // just behind and above the camera robot, looking into the boulder field
  const controls = new OrbitControls(free, mainCanvas);
  controls.target.set(0, 0.8, -19); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495; controls.minDistance = 4; controls.maxDistance = 160;
  const roverCam = new THREE.PerspectiveCamera(60, 1, 0.1, 2000);

  const getPool = (kind, make) => { const i = (pool[kind].used = (pool[kind].used || 0) + 1) - 1; if (!pool[kind][i]) { pool[kind][i] = make(); dyn.add(pool[kind][i]); } pool[kind][i].visible = true; return pool[kind][i]; };
  const resetPools = () => { for (const k of Object.keys(pool)) { pool[k].used = 0; for (const m of pool[k]) m.visible = false; } };

  function render(sess, { sel, color, showTruth, marks = [] }) {
    if (!world || sess.world !== world) setWorld(sess.world);
    const w = sess.world, col = new THREE.Color(color);
    resetPools();
    const labels1 = [], labels2 = [];

    // people follow the simulation (heading: sim angle -> three yaw; NASA model faces +z)
    for (const t of w.targets) {
      const g = people.get(t.id); if (!g) continue;
      g.position.copy(toV(t.x, t.y, height(t.x, t.y))); g.rotation.y = Math.atan2(Math.cos(t.h), -Math.sin(t.h));
      g.visible = true;
      const hid = !t.vis.inFov || isHidden(t.vis);
      // the 3-D model IS the truth; only hidden people get a label, at their feet, so it never collides with the tracker's tag
      if (showTruth && hid) labels1.push({ p: toV(t.x, t.y, height(t.x, t.y) - 0.2), text: `person ${t.id} · ${hiddenCause(t.vis) === 'shadow' ? 'in shadow' : 'behind rock'}`, color: '#ffffff', bg: '#e8ecf2cc', font: '11px system-ui', alpha: 0.9 });
    }
    // camera pings (orange discs on the ground)
    sess.last.dets.forEach((d) => {
      const x = w.cam.x + d.range * Math.cos(d.bearing), y = w.cam.y + d.range * Math.sin(d.bearing);
      const m = getPool('ping', () => new THREE.Mesh(new THREE.RingGeometry(0.25, 0.42, 24), new THREE.MeshBasicMaterial({ color: 0xffb454, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false })));
      m.rotation.x = -Math.PI / 2; m.position.copy(toV(x, y, height(x, y) + 0.05)); m.renderOrder = 5;
    });
    // tracker beliefs: solid ground ring + name tag when seen; translucent search area when hidden
    for (const t of sess.runs[sel].tracker.tracks) {
      if (!t.confirmed) continue;
      const lost = t.lastSeen > 3, x = t.x[0], y = t.x[1], gh = height(x, y);
      if (lost) {
        const a = t.P.get(0, 0), b = t.P.get(0, 1), d2 = t.P.get(1, 1), mm = (a + d2) / 2, q = Math.sqrt(((a - d2) / 2) ** 2 + b * b), k = Math.sqrt(5.991);
        const m = getPool('area', () => new THREE.Mesh(new THREE.CircleGeometry(1, 48), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide })));
        m.material.color.copy(col); m.rotation.set(-Math.PI / 2, 0, 0.5 * Math.atan2(2 * b, a - d2));
        m.scale.set(Math.max(0.6, k * Math.sqrt(mm + q)), Math.max(0.6, k * Math.sqrt(Math.max(mm - q, 0))), 1);
        m.position.copy(toV(x, y, gh + 0.06));
        labels1.push({ p: toV(x, y, gh + 1.0), text: `#${t.id} lost sight · searching`, color, dashed: true });
        labels2.push({ p: toV(x, y, gh + PERSON_H + 0.4), text: `#${t.id} behind rock?`, color, dashed: true });
      } else {
        const m = getPool('ring', () => new THREE.Mesh(new THREE.RingGeometry(0.55, 0.78, 40), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false })));
        m.material.color.copy(col); m.rotation.x = -Math.PI / 2; m.position.copy(toV(x, y, gh + 0.07)); m.renderOrder = 6;
        labels1.push({ p: toV(x, y, gh + PERSON_H + 1.05), text: `#${t.id}`, color });
        labels2.push({ p: toV(x, y, gh + PERSON_H + 0.4), text: `#${t.id}`, color });
      }
    }
    for (const mk of marks) labels1.push({ p: toV(mk.x, mk.y, height(mk.x, mk.y) + PERSON_H + 1.9), text: mk.kept ? '✔ same ID kept' : '✘ lost them: new ID', color: mk.kept ? '#4ade80' : '#ff8a8a', bg: mk.kept ? '#14532d' : '#4c1219', fg: mk.kept ? '#4ade80' : '#ff8a8a', font: 'bold 13px system-ui', alpha: mk.alpha });

    // rover camera: at the mast head, looking along the simulated optical axis, same horizontal field of view
    const cam = w.cam, eye = toV(cam.x, cam.y, height(cam.x, cam.y) + MAST_H);
    roverCam.position.copy(eye); roverCam.lookAt(eye.clone().add(new THREE.Vector3(Math.cos(cam.th), -0.05, -Math.sin(cam.th))));
    if (fitRenderer(R2, camCanvas, roverCam)) {
      roverCam.fov = (2 * Math.atan(Math.tan(cam.fov / 2) / roverCam.aspect) * 180) / Math.PI; roverCam.updateProjectionMatrix();
      for (const p of people.values()) p.visible = true;
      const rings = [...pool.ring, ...pool.area, ...pool.ping]; rings.forEach((m) => { m.userData.v = m.visible; m.visible = false; }); // camera image = the scene only
      R2.render(scene, roverCam); rings.forEach((m) => { m.visible = m.userData.v; });
      drawLabels(camOverlay, roverCam, labels2);
    }
    for (const p of people.values()) p.visible = showTruth;
    if (fitRenderer(R1, mainCanvas, free)) { controls.update(); R1.render(scene, free); drawLabels(mainOverlay, free, labels1); }
  }

  return { render, setWorld, ok: !!(astro && rassor) };
}
