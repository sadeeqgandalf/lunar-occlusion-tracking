// Mission Control 3-D view (rover platform): NASA's Mars 2020 Perseverance model driving Martian terrain,
// with every estimator's belief drawn as a coloured ghost marker + 95% uncertainty disc on the ground.
// The mission engine stays the source of truth; this module only draws it.
import { THREE, toV, hash, fbm, makeRenderer, fitRenderer, loadModel, drawLabels } from './three-common.js';
import { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
import { covEllipse } from '../filters/base.js';

const ROVER_LEN = 3.0; // Perseverance is ~3 m long

export async function createMars3D({ canvas, overlay }) {
  const R = makeRenderer(canvas, 1.05);
  const scene = new THREE.Scene();
  const sky = new THREE.Color(0xc79b72); scene.background = sky; scene.fog = new THREE.Fog(sky, 90, 260);

  const sun = new THREE.DirectionalLight(0xfff0dc, 2.6);
  sun.position.set(-50, 45, 30); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -110, right: 110, top: 110, bottom: -110, near: 1, far: 300 });
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0005; sun.shadow.normalBias = 0.05;
  scene.add(sun, sun.target);
  scene.add(new THREE.HemisphereLight(0xf2c9a0, 0x5a3220, 0.55));

  const percy = await loadModel('assets/nasa/perseverance.glb', ROVER_LEN, 'xz');
  const dyn = new THREE.Group(); scene.add(dyn);
  let world = null, height = () => 0, staticGroup = null;
  const ghosts = new Map(), trails = new Map(), pool = { disc: [] };
  let truthTrail = null, wpLine = null, particles = null, rover = null;

  function setWorld(w) {
    world = w;
    if (staticGroup) scene.remove(staticGroup);
    staticGroup = new THREE.Group(); scene.add(staticGroup);

    // terrain: rusty regolith, real crater bowls at the simulator's hazards, sand patches as paler smooth ground
    height = (x, y) => {
      let h = (fbm(x * 0.05, y * 0.05) - 0.5) * 0.9;
      for (const c of w.hazards) {
        const d = Math.hypot(x - c.x, y - c.y) / c.r;
        if (d < 1.4) h += d < 1 ? -0.45 * c.r * (1 - d * d) * 0.5 : 0.12 * c.r * Math.cos(((d - 1) / 0.4) * Math.PI / 2) ** 2;
      }
      return h;
    };
    const B = w.bounds, W = B.xmax - B.xmin + 80, H = B.ymax - B.ymin + 80, cx = (B.xmin + B.xmax) / 2, cy = (B.ymin + B.ymax) / 2;
    const G = new THREE.PlaneGeometry(W, H, 260, 180); G.rotateX(-Math.PI / 2);
    const pos = G.attributes.position, col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) + cx, y = -pos.getZ(i) + cy;
      pos.setX(i, x); pos.setZ(i, -y); pos.setY(i, height(x, y));
      let sand = 0; for (const p of w.patches) sand = Math.max(sand, Math.max(0, 1 - Math.hypot(x - p.x, y - p.y) / p.r));
      const n = (fbm(x * 0.4, y * 0.4) - 0.5) * 0.12;
      col.set([0.58 + n + sand * 0.18, 0.33 + n * 0.8 + sand * 0.17, 0.2 + n * 0.5 + sand * 0.1], i * 3);
    }
    G.setAttribute('color', new THREE.BufferAttribute(col, 3)); G.computeVertexNormals();
    const ground = new THREE.Mesh(G, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }));
    ground.receiveShadow = true; ground.name = 'ground'; staticGroup.add(ground);

    // scattered small rocks for scale (decor only, kept off the route and crater bowls)
    const rockGeo = new THREE.DodecahedronGeometry(1, 1), rockMat = new THREE.MeshStandardMaterial({ color: 0x6b4330, roughness: 1, flatShading: true });
    for (let i = 0; i < 160; i++) {
      const x = B.xmin + hash(i, 5) * (B.xmax - B.xmin), y = B.ymin + hash(i, 9) * (B.ymax - B.ymin), s = 0.15 + hash(i, 11) * 0.45;
      if (w.hazards.some((c) => Math.hypot(x - c.x, y - c.y) < c.r * 1.3)) continue;
      const r = new THREE.Mesh(rockGeo, rockMat); r.scale.set(s, s * 0.6, s * 1.2); r.position.copy(toV(x, y, height(x, y) + s * 0.2)); r.rotation.y = hash(i, 3) * 6; r.castShadow = true; staticGroup.add(r);
    }
    // navigation beacons (the landmarks the filters range to)
    const postMat = new THREE.MeshStandardMaterial({ color: 0xcfd3d8, metalness: 0.5, roughness: 0.4 }), capMat = new THREE.MeshStandardMaterial({ color: 0x4cc9f0, emissive: 0x2a8bb0 });
    for (const l of w.landmarks) {
      const g = height(l.x, l.y), post = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 1.4, 10), postMat);
      post.position.copy(toV(l.x, l.y, g + 0.7)); post.castShadow = true; staticGroup.add(post);
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.16, 14, 10), capMat); cap.position.copy(toV(l.x, l.y, g + 1.45)); staticGroup.add(cap);
    }
    for (const m of [...ghosts.values()]) dyn.remove(m); ghosts.clear();
    for (const t of [...trails.values()]) dyn.remove(t); trails.clear();
    if (rover) dyn.remove(rover);
    rover = percy ? percy.clone() : new THREE.Mesh(new THREE.BoxGeometry(2.7, 1.2, 3), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    dyn.add(rover);
    for (const k of Object.keys(pool)) { for (const m of pool[k]) dyn.remove(m); pool[k] = []; }
    flags = []; for (const t of w.targets) { const f = makeFlag(); f.position.copy(toV(t.x, t.y, height(t.x, t.y))); staticGroup.add(f); flags.push({ t, f }); }
    framed = false;
  }
  let flags = [], framed = false;
  function makeFlag() {
    const g = new THREE.Group(), pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6, 8), new THREE.MeshStandardMaterial({ color: 0xeeeeee }));
    pole.position.y = 0.8; g.add(pole);
    const cloth = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.38), new THREE.MeshStandardMaterial({ color: 0xffd166, side: THREE.DoubleSide, emissive: 0x403000 }));
    cloth.position.set(0.3, 1.4, 0); g.add(cloth); g.userData.cloth = cloth;
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.1, 40), new THREE.MeshBasicMaterial({ color: 0xffd166, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.05; g.add(ring); g.userData.ring = ring;
    return g;
  }

  const cam = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  cam.position.set(-12, 9, 12);
  const controls = new OrbitControls(cam, canvas);
  controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.48; controls.minDistance = 4; controls.maxDistance = 220;

  const lineOf = (pts, color, opacity = 1) => {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  };
  const updLine = (line, pts) => { line.geometry.dispose(); line.geometry = new THREE.BufferGeometry().setFromPoints(pts); };

  /** Raycast a canvas click onto the terrain; returns simulator {x, y} or null. */
  const ray = new THREE.Raycaster();
  function pick(px, py) {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((px / r.width) * 2 - 1, -(py / r.height) * 2 + 1), cam);
    const hit = ray.intersectObject(staticGroup.getObjectByName('ground'))[0];
    return hit ? { x: hit.point.x, y: -hit.point.z } : null;
  }

  function render(m, ui) {
    if (m.world !== world) setWorld(m.world);
    const P = m.platform, labels = [], primary = m.fs.get(m.primary);
    // truth rover (NASA Perseverance)
    const tp = P.pose(m.truth), tpos = toV(tp.x, tp.y, height(tp.x, tp.y));
    rover.position.copy(tpos); rover.rotation.y = Math.atan2(Math.cos(tp.th), -Math.sin(tp.th));
    rover.visible = ui.showTruth;
    // truth label at ground level so it never collides with the estimators' labels above the rover
    if (ui.showTruth) labels.push({ p: tpos.clone().add(new THREE.Vector3(0, -0.3, 0)), text: 'Perseverance (true position)', color: '#ffffff', bg: '#ffffffd8', font: '11px system-ui' });
    // follow the rover: keep the user's chosen viewing offset, move the target with the vehicle
    const followTarget = tpos.clone().add(new THREE.Vector3(0, 0.8, 0));
    if (!framed) { controls.target.copy(followTarget); cam.position.copy(followTarget).add(new THREE.Vector3(-7, 4.5, 8)); framed = true; } // start close behind the rover
    const delta = followTarget.clone().sub(controls.target); controls.target.copy(followTarget); cam.position.add(delta);

    // estimator beliefs: ghost marker + 95% disc, trail
    pool.disc.used = 0; pool.disc.forEach((d) => (d.visible = false));
    const live = new Set();
    for (const f of m.fs.values()) {
      live.add(f.id);
      const color = new THREE.Color(ui.colors[f.id] || '#ffffff'), x = f.filter.getState(), pose = P.pose(x), g = height(pose.x, pose.y);
      let gh = ghosts.get(f.id);
      if (!gh) {
        gh = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.3, 16), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, transparent: true, opacity: 0.85 }));
        gh.rotation.order = 'YXZ'; dyn.add(gh); ghosts.set(f.id, gh);
      }
      gh.position.copy(toV(pose.x, pose.y, g + 2.4)); gh.rotation.set(Math.PI, Math.atan2(Math.cos(pose.th), -Math.sin(pose.th)), 0);
      if (ui.showCov) {
        const Pm = f.filter.getCov(), [i, j] = m.model.posIdx, e = covEllipse({ get: (a, b) => Pm.get([i, j][a], [i, j][b]) }), k = Math.sqrt(5.991);
        const n = pool.disc.used++;
        if (!pool.disc[n]) { pool.disc[n] = new THREE.Mesh(new THREE.CircleGeometry(1, 48), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide })); dyn.add(pool.disc[n]); }
        const d = pool.disc[n]; d.visible = true; d.material.color.copy(color);
        d.rotation.set(-Math.PI / 2, 0, e.angle); d.scale.set(Math.max(0.3, k * Math.sqrt(e.l1)), Math.max(0.3, k * Math.sqrt(e.l2)), 1);
        d.position.copy(toV(pose.x, pose.y, g + 0.08 + n * 0.01));
      }
      const tpts = f.trail.map(([a, b]) => toV(a, b, height(a, b) + 0.15));
      let tl = trails.get(f.id);
      if (!tl) { tl = lineOf(tpts.length > 1 ? tpts : [tpos, tpos], color, 0.85); dyn.add(tl); trails.set(f.id, tl); } else if (tpts.length > 1) updLine(tl, tpts);
      // label the steering estimator always; others only when they DISAGREE with it by > 2 m (the moment worth seeing)
      const pp = primary && P.pose(primary.filter.getState()), apart = pp ? Math.hypot(pose.x - pp.x, pose.y - pp.y) : 0;
      if (f.id === m.primary || apart > 2) labels.push({ p: toV(pose.x, pose.y, g + 3.3), text: f.id === m.primary ? `${f.Cls.label} · steering` : `${f.Cls.label} · ${apart.toFixed(1)} m off`, color: ui.colors[f.id], font: f.id === m.primary ? 'bold 12px system-ui' : '11px system-ui' });
    }
    for (const [id, gh] of ghosts) if (!live.has(id)) { dyn.remove(gh); ghosts.delete(id); const t = trails.get(id); if (t) { dyn.remove(t); trails.delete(id); } }
    // truth trail
    const tt = m.truthTrail.map(([a, b]) => toV(a, b, height(a, b) + 0.12));
    if (!truthTrail && tt.length > 1) { truthTrail = lineOf(tt, 0xffffff, 0.7); dyn.add(truthTrail); } else if (truthTrail && tt.length > 1) updLine(truthTrail, tt);
    if (truthTrail) truthTrail.visible = ui.showTruth;
    // planned route from the steering estimate
    if (primary) {
      const pe = P.pose(primary.filter.getState()), pts = [toV(pe.x, pe.y, height(pe.x, pe.y) + 0.3), ...m.waypoints.map((w) => toV(w.x, w.y, height(w.x, w.y) + 0.3))];
      if (!wpLine) { wpLine = lineOf(pts.length > 1 ? pts : [pts[0], pts[0]], 0xffffff, 0.9); dyn.add(wpLine); } else updLine(wpLine, pts.length > 1 ? pts : [pts[0], pts[0]]);
    }
    // particles of any particle filter (subsampled)
    const pf = [...m.fs.values()].find((f) => f.filter.getParticles());
    if (pf && ui.showParticles) {
      const p = pf.filter.getParticles(), n = pf.filter.nx, [i, j] = m.model.posIdx, N = Math.floor(pf.filter.N / 3), arr = new Float32Array(N * 3);
      for (let k = 0; k < N; k++) { const a = p[k * 3 * n + i], b = p[k * 3 * n + j]; arr.set([a, height(a, b) + 0.2, -b], k * 3); }
      if (!particles) { particles = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, color: new THREE.Color(ui.colors[pf.id]) })); dyn.add(particles); }
      particles.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3)); particles.visible = true;
    } else if (particles) particles.visible = false;
    // science targets
    for (const { t, f } of flags) {
      const done = m.targets.find((x) => x.id === t.id)?.done, c = done ? 0x4ade80 : 0xffd166;
      f.userData.cloth.material.color.setHex(c); f.userData.ring.material.color.setHex(c);
      labels.push({ p: f.position.clone().add(new THREE.Vector3(0, 2.1, 0)), text: `${P.objective.noun} ${t.id}${done ? ' ✔' : ''}`, color: done ? '#4ade80' : '#ffd166' });
    }
    if (fitRenderer(R, canvas, cam)) { controls.update(); R.render(scene, cam); drawLabels(overlay, cam, labels); }
  }

  return { render, pick, ok: !!percy };
}
