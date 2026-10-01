// Mission Control 3-D view (spacecraft platform): rendezvous and docking with NASA's Gateway model.
// Hill frame -> scene: along-track y_T -> three x (horizontal), radial x_R -> three y (up), out-of-plane = z.
// Chaser: NASA ESAS crew module (Orion's design predecessor). Model sizes are approximate (see constants).
import { isLost, GFX, THREE, hash, fbm, makeRenderer, fitRenderer, loadModel, starfield, drawLabels, makeEnvironment } from './three-common.js';
import { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
import { covEllipse } from '../filters/base.js';

const GATEWAY_SPAN = 40;   // m, approximate overall span incl. arrays (exact figure varies by configuration)
const CAPSULE_SIZE = 5;    // m, Orion-class crew module diameter
const H = (xR, yT, z = 0) => new THREE.Vector3(yT, xR, z);

export async function createSpace3D({ canvas, overlay }) {
  const R = makeRenderer(canvas, 1.1);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
  scene.add(starfield(GFX.low ? 1500 : 4000, 900));
  scene.environment = makeEnvironment(R); scene.environmentIntensity = 0.9; // lets NASA's metallic materials read correctly
  const sun = new THREE.DirectionalLight(0xffffff, 3.0); sun.position.set(300, 120, 260); scene.add(sun);
  scene.add(new THREE.HemisphereLight(0x8899aa, 0x222222, 0.25));

  // the Moon far below (radial is "up", so the Moon sits below the orbit)
  const mg = new THREE.SphereGeometry(1, 128, 96), mp = mg.attributes.position, mc = new Float32Array(mp.count * 3);
  for (let i = 0; i < mp.count; i++) { const v = new THREE.Vector3().fromBufferAttribute(mp, i), g = 0.5 + (fbm(v.x * 6 + 3, v.y * 6 + v.z * 4) - 0.5) * 0.6; mc.set([g, g, g * 0.98], i * 3); }
  mg.setAttribute('color', new THREE.BufferAttribute(mc, 3));
  const moon = new THREE.Mesh(mg, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }));
  moon.scale.setScalar(1700); moon.position.set(-150, -1950, -400); scene.add(moon);

  const [gateway, capsule] = await Promise.all([
    loadModel('assets/nasa/gateway.glb', GATEWAY_SPAN, 'xz'),
    loadModel('assets/nasa/esas_crew_module.glb', CAPSULE_SIZE, 'xz'),
  ]);
  const dyn = new THREE.Group(); scene.add(dyn);
  let world = null, staticGroup = null, chaser = null, framed = false;
  const ghosts = new Map(), discs = [], trails = new Map();
  let truthTrail = null, gates = [];

  function setWorld(w) {
    world = w;
    if (staticGroup) scene.remove(staticGroup);
    staticGroup = new THREE.Group(); scene.add(staticGroup);
    // Gateway at the docking target (Hill-frame origin), centred on its own bounding box
    const gw = gateway ? gateway.clone() : new THREE.Mesh(new THREE.CylinderGeometry(2.5, 2.5, 18, 24), new THREE.MeshStandardMaterial({ color: 0xcccccc }));
    gw.position.set(0, gateway ? -GATEWAY_SPAN * 0.12 : 0, 0); staticGroup.add(gw);
    // debris fields (the simulator's hazards): tumbling metallic fragments filling each keep-out sphere
    const fragGeo = new THREE.IcosahedronGeometry(1, 0), fragMat = new THREE.MeshStandardMaterial({ color: 0x9a9fa8, metalness: 0.7, roughness: 0.35, flatShading: true });
    w.hazards.forEach((h, k) => {
      const shell = new THREE.Mesh(new THREE.SphereGeometry(h.r, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffb454, transparent: true, opacity: 0.06, depthWrite: false }));
      shell.position.copy(H(h.x, h.y)); staticGroup.add(shell);
      for (let i = 0; i < 26; i++) {
        const f = new THREE.Mesh(fragGeo, fragMat), s = 0.2 + hash(k * 31 + i, 1) * 1.1;
        const u = hash(k, i) * 2 - 1, t = hash(i, k) * 6.283, rr = h.r * Math.cbrt(hash(i, k + 7)), q = Math.sqrt(1 - u * u);
        f.position.copy(H(h.x + rr * u, h.y + rr * q * Math.cos(t), rr * q * Math.sin(t))); f.scale.set(s, s * 0.4, s * 0.8);
        f.userData.spin = new THREE.Vector3(hash(i, 2) - 0.5, hash(i, 3) - 0.5, hash(i, 4) - 0.5).multiplyScalar(0.8);
        staticGroup.add(f);
      }
    });
    // approach gates as hoops to fly through
    gates = w.targets.map((t) => {
      const hoop = new THREE.Mesh(new THREE.TorusGeometry(4, 0.18, 12, 64), new THREE.MeshStandardMaterial({ color: 0xffd166, emissive: 0x5a4400 }));
      hoop.position.copy(H(t.x, t.y)); hoop.rotation.y = Math.PI / 2; staticGroup.add(hoop); return { t, hoop };
    });
    if (chaser) dyn.remove(chaser);
    chaser = capsule ? capsule.clone() : new THREE.Mesh(new THREE.ConeGeometry(2.5, 3.5, 24), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    dyn.add(chaser);
    for (const g of ghosts.values()) dyn.remove(g); ghosts.clear();
    for (const t of trails.values()) dyn.remove(t); trails.clear();
    framed = false;
  }

  const cam = new THREE.PerspectiveCamera(50, 1, 0.5, 6000);
  const controls = new OrbitControls(cam, canvas); controls.enableDamping = true; controls.minDistance = 8; controls.maxDistance = 900;
  const lineOf = (pts, color, opacity = 1) => new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  const updLine = (l, pts) => { l.geometry.dispose(); l.geometry = new THREE.BufferGeometry().setFromPoints(pts); };

  function render(m, ui) {
    if (m.world !== world) setWorld(m.world);
    const labels = [], x = m.truth, tpos = H(x[0], x[1]);
    // chaser: nose toward the docking target, as a real approach would be flown
    chaser.position.copy(tpos); chaser.lookAt(H(0, 0)); chaser.visible = ui.showTruth;
    if (ui.showTruth) labels.push({ p: tpos.clone().add(new THREE.Vector3(0, -8.5, 0)), text: `crew module (true) · ${Math.hypot(x[0], x[1]).toFixed(0)} m to Gateway, ${Math.hypot(x[2], x[3]).toFixed(2)} m/s`, color: '#fff', bg: '#ffffffd8', font: '11px system-ui' });
    labels.push({ p: H(GATEWAY_SPAN * 0.35, 0), text: 'Gateway (docking target)', color: '#7cf', bg: '#0b2a3a', fg: '#bfe9ff' });
    // camera: start behind and above the chaser looking at Gateway; then follow the chaser
    const tgt = tpos.clone().lerp(H(0, 0), 0.35);
    if (!framed) { controls.target.copy(tgt); cam.position.copy(tpos).add(new THREE.Vector3(-35, 18, 45)); framed = true; }
    const d = tgt.clone().sub(controls.target); controls.target.copy(tgt); cam.position.add(d);
    // debris tumbles
    staticGroup.children.forEach((o) => { if (o.userData.spin) { o.rotation.x += o.userData.spin.x * 0.02; o.rotation.y += o.userData.spin.y * 0.02; } });
    // beliefs
    discs.forEach((q) => (q.visible = false)); let used = 0;
    const primary = m.fs.get(m.primary), pp = primary && primary.filter.getState(), live = new Set();
    for (const f of m.fs.values()) {
      live.add(f.id);
      const color = new THREE.Color(ui.colors[f.id] || '#fff'), s = f.filter.getState(), pos = H(s[0], s[1]);
      let g = ghosts.get(f.id);
      if (!g) { g = new THREE.Mesh(new THREE.OctahedronGeometry(0.9, 0), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5 })); dyn.add(g); ghosts.set(f.id, g); }
      g.position.copy(pos).add(new THREE.Vector3(0, 4.5, 0)); g.rotation.y += 0.03;
      if (ui.showCov) {
        const Pm = f.filter.getCov(), e = covEllipse({ get: (a, b) => Pm.get(a, b) }), k = Math.sqrt(5.991);
        if (!discs[used]) { discs[used] = new THREE.Mesh(new THREE.CircleGeometry(1, 48), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide })); dyn.add(discs[used]); }
        const q = discs[used++]; q.visible = true; q.material.color.copy(color);
        q.material.opacity = 0.2 * Math.min(1, 12 / Math.max(k * Math.sqrt(e.l1), 1)); // huge uncertainty fades instead of hiding the scene
        // ellipse is in the (x_R, y_T) orbital plane = scene (y, x) plane
        q.rotation.set(0, 0, Math.PI / 2 - e.angle); q.scale.set(Math.max(0.4, k * Math.sqrt(e.l1)), Math.max(0.4, k * Math.sqrt(e.l2)), 1); q.position.copy(pos);
      }
      const tp = f.trail.map(([a, b]) => H(a, b));
      let tl = trails.get(f.id);
      if (!tl && tp.length > 1) { tl = lineOf(tp, color, 0.85); dyn.add(tl); trails.set(f.id, tl); } else if (tl && tp.length > 1) updLine(tl, tp);
      const apart = pp ? Math.hypot(s[0] - pp[0], s[1] - pp[1]) : 0;
      if (f.id === m.primary || apart > 3) labels.push({ p: pos.clone().add(new THREE.Vector3(0, 6.5, 0)), text: f.id === m.primary ? `${f.Cls.label} · steering` : `${f.Cls.label} · ${apart.toFixed(1)} m off`, color: ui.colors[f.id], font: f.id === m.primary ? 'bold 12px system-ui' : '11px system-ui' });
    }
    for (const [id, g] of ghosts) if (!live.has(id)) { dyn.remove(g); ghosts.delete(id); const t = trails.get(id); if (t) { dyn.remove(t); trails.delete(id); } }
    const tt = m.truthTrail.map(([a, b]) => H(a, b));
    if (!truthTrail && tt.length > 1) { truthTrail = lineOf(tt, 0xffffff, 0.6); dyn.add(truthTrail); } else if (truthTrail && tt.length > 1) updLine(truthTrail, tt);
    if (truthTrail) truthTrail.visible = ui.showTruth;
    for (const { t, hoop } of gates) {
      const done = m.targets.find((q) => q.id === t.id)?.done;
      hoop.material.color.setHex(done ? 0x4ade80 : 0xffd166); hoop.material.emissive.setHex(done ? 0x0f4020 : 0x5a4400);
      labels.push({ p: hoop.position.clone().add(new THREE.Vector3(0, -5.5, 0)), text: `${t.id === m.targets.length ? 'Dock' : `Gate ${t.id}`}${done ? ' ✔' : ''}`, color: done ? '#4ade80' : '#ffd166' });
    }
    R.toneMappingExposure = 1.1; // the renderer is shared with the other 3-D view on this canvas
    if (fitRenderer(R, canvas, cam)) { controls.update(); R.render(scene, cam); drawLabels(overlay, cam, labels); }
  }

  /** Clicking in space has no ground to hit: return null (use the Map view to place waypoints). */
  const pick = () => null;
  return { render, pick, ok: !!(gateway && capsule), isLost: () => isLost(R) };
}
