// Shared Three.js plumbing for the 3-D views (Occlusion Lab, Mission Control).
import * as THREE from 'three';
import { GLTFLoader } from '../../vendor/three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from '../../vendor/three/examples/jsm/loaders/DRACOLoader.js';

export { THREE };

// deterministic value noise (terrain, rock shapes)
export const hash = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
const smooth = (t) => t * t * (3 - 2 * t);
export function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1), u = smooth(xf), v = smooth(yf);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
export const fbm = (x, y) => { let s = 0, a = 0.5, f = 1; for (let o = 0; o < 4; o++) { s += a * vnoise(x * f, y * f); a *= 0.5; f *= 2.1; } return s; };

/** Simulator ground (x, y) -> Three.js (x, h, -y), y-up, metres. */
export const toV = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

export function makeRenderer(canvas, exposure = 1.15) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true });
  r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
  r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = exposure; r.outputColorSpace = THREE.SRGBColorSpace;
  return r;
}

/** Resize a renderer to its canvas' CSS box and update the camera aspect. Returns false if not laid out yet. */
export function fitRenderer(r, canvas, cam) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return false;
  const pr = r.getPixelRatio();
  if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) r.setSize(w, h, false);
  cam.aspect = w / h; cam.updateProjectionMatrix(); return true;
}

let _loader = null;
function loader() {
  if (_loader) return _loader;
  const draco = new DRACOLoader(); draco.setDecoderPath('vendor/three/examples/jsm/libs/draco/gltf/');
  _loader = new GLTFLoader(); _loader.setDRACOLoader(draco);
  return _loader;
}

/**
 * Load an official NASA glTF and normalise it to real-world size, standing on y=0 and centred.
 * axis 'y' = targetSize is the height; 'xz' = targetSize is the longest horizontal extent.
 * Resolves to null if the file cannot be loaded (callers fall back to simple geometry).
 */
export function loadModel(url, targetSize, axis = 'y') {
  return new Promise((res) => loader().load(url, (g) => {
    const root = g.scene, box = new THREE.Box3().setFromObject(root), size = new THREE.Vector3(), ctr = new THREE.Vector3();
    box.getSize(size); box.getCenter(ctr);
    const k = targetSize / (axis === 'y' ? size.y : Math.max(size.x, size.z));
    root.scale.setScalar(k);
    root.position.set(-ctr.x * k, -box.min.y * k, -ctr.z * k);
    root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    const holder = new THREE.Group(); holder.add(root); res(holder);
  }, undefined, () => res(null)));
}

/**
 * A soft studio-like environment so PBR metals (NASA models use them) have something to reflect.
 * Without it, metallic surfaces render nearly black. Built procedurally: no extra asset files.
 */
export function makeEnvironment(renderer, top = 0xbfd6ff, bottom = 0x1a1a1a) {
  const env = new THREE.Scene(), g = new THREE.SphereGeometry(50, 32, 16), pos = g.attributes.position, col = new Float32Array(pos.count * 3);
  const ct = new THREE.Color(top), cb = new THREE.Color(bottom), c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) { const t = (pos.getY(i) / 50 + 1) / 2; c.copy(cb).lerp(ct, Math.pow(t, 1.5)); col.set([c.r, c.g, c.b], i * 3); }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  env.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(30, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  panel.position.set(20, 25, 20); panel.lookAt(0, 0, 0); env.add(panel); // a bright "sun" panel for highlights
  const pm = new THREE.PMREMGenerator(renderer), tex = pm.fromScene(env, 0.04).texture; pm.dispose();
  return tex;
}

export function starfield(n = 3000, r = 600) {
  const sp = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const u = hash(i, 1) * 2 - 1, t = hash(i, 2) * Math.PI * 2, s = Math.sqrt(1 - u * u); sp.set([r * s * Math.cos(t), Math.abs(r * u) * 0.9 + 5, r * s * Math.sin(t)], i * 3); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffffff, size: 1.2, sizeAttenuation: false }));
}

/** Draw pill-shaped labels for 3-D points on a 2-D overlay canvas. item: {p, text, color, bg?, fg?, dashed?, font?, alpha?} */
export function drawLabels(ov, cam, items) {
  const dpr = window.devicePixelRatio || 1, w = ov.clientWidth, h = ov.clientHeight;
  if (ov.width !== Math.round(w * dpr) || ov.height !== Math.round(h * dpr)) { ov.width = Math.round(w * dpr); ov.height = Math.round(h * dpr); }
  const c = ov.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  for (const it of items) {
    const v = it.p.clone().project(cam);
    if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) continue;
    const x = (v.x * 0.5 + 0.5) * w, y = (-v.y * 0.5 + 0.5) * h;
    c.font = it.font || 'bold 12px system-ui'; const tw = c.measureText(it.text).width + 12;
    c.globalAlpha = it.alpha ?? 1;
    c.beginPath(); c.roundRect(x - tw / 2, y - 20, tw, 19, 9.5);
    if (it.dashed) { c.fillStyle = '#0a0e16dd'; c.fill(); c.setLineDash([3, 3]); c.strokeStyle = it.color; c.lineWidth = 1.5; c.stroke(); c.setLineDash([]); c.fillStyle = it.color; }
    else { c.fillStyle = it.bg || it.color; c.fill(); c.fillStyle = it.fg || '#06101a'; }
    c.textAlign = 'center'; c.fillText(it.text, x, y - 6); c.textAlign = 'left'; c.globalAlpha = 1;
  }
}
