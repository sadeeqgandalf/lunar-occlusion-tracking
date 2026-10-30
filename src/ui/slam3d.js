// Real-Time 3-D WebGL / Three.js Renderer for the Incremental Lunar/Planetary SLAM Lab.
//
// Visualizes the core SLAM loop in real time:
//  1. Incremental Fog-of-War & 3-D Occupancy Grid Carpet:
//     - Unmapped terrain starts under a dark Fog-of-War veil (0% mapped).
//     - As the rover's 100° active camera frustum sweeps across the terrain, it carves out
//       FREE space (emerald/cyan grid) and OCCUPIED boulder footprints (amber/red).
//  2. Incremental 3-D Obstacle & Point-Cloud Reconstruction:
//     - Undiscovered boulders remain ghosted/hidden in the Fog-of-War until sensor rays strike them.
//     - Once discovered, boulders materialize with a glowing 3-D Reconstructed Map Wireframe Cage
//       and accumulate dense 3-D surface point clouds.
//  3. Live 3-D Landmark Triangulation & Shrinking Uncertainty Ellipsoids:
//     - Renders the selected estimator's private landmark estimates m_hat_j = (x, y, z)
//       together with 3-D Uncertainty Rings (radius = 2 * sqrt(sigma_XX)) that visibly shrink
//       as the rover observes landmarks from multiple angles.
//  4. Live Rover Pose Estimate & 2-Sigma Covariance Ellipse:
//     - Renders the estimator's live unaligned pose + 2-sigma covariance ellipse alongside
//       the physical rover, plus 3-D loop-closure arcs and keyframe camera frustums.

import { THREE, toV, hash, fbm, makeRenderer, loadModel, starfield, drawLabels, GFX } from './three-common.js';
import { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
import { covEllipse } from '../slam/engine.js';

const MAST_H = 2.1;

function makeTerrainHeight(craters) {
  return (x, y) => {
    let h = (fbm(x * 0.075, y * 0.075) - 0.5) * 0.42;
    for (const c of craters) {
      const d = Math.hypot(x - c.x, y - c.y) / c.r;
      if (d < 1.35) {
        h += d < 1
          ? -c.depth * (1 - d * d)
          : c.depth * 0.34 * Math.cos(((d - 1) / 0.35) * Math.PI * 0.5) ** 2;
      }
    }
    return h;
  };
}

function lunarLambertShader(sh) {
  sh.fragmentShader = sh.fragmentShader.replace(
    '#include <lights_physical_pars_fragment>',
    THREE.ShaderChunk.lights_physical_pars_fragment.replace(
      'vec3 irradiance = dotNL * directLight.color;',
      `
      float llMu = saturate( dot( geometryNormal, geometryViewDir ) );
      float llG = degrees( acos( clamp( dot( directLight.direction, geometryViewDir ), -1.0, 1.0 ) ) );
      float llL = clamp( 1.0 - 0.019 * llG + 0.000242 * llG * llG - 0.00000146 * llG * llG * llG, 0.0, 1.0 );
      vec3 irradiance = ( dotNL > 0.0 ? 2.0 * llL * dotNL / ( dotNL + llMu + 1e-4 ) + ( 1.0 - llL ) * dotNL : 0.0 ) * directLight.color;`
    )
  );
}

function buildRegolithDetailTexture(repeat) {
  const N = 256;
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const g = c.getContext('2d');
  const im = g.createImageData(N, N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = 0.8 + 0.18 * (hash(x, y) - 0.5) + 0.22 * (fbm(x * 0.09, y * 0.09) - 0.5);
      const k = 4 * (y * N + x);
      const byte = 255 * Math.min(1, Math.max(0, v));
      im.data[k] = im.data[k + 1] = im.data[k + 2] = byte;
      im.data[k + 3] = 255;
    }
  }
  g.putImageData(im, 0, 0);
  for (let i = 0; i < 280; i++) {
    const x = hash(i, 101) * N, y = hash(i, 202) * N, r = 0.8 + 2.5 * hash(i, 303) ** 2;
    g.fillStyle = 'rgba(35,35,40,0.58)';
    g.beginPath(); g.arc(x + r * 0.5, y, r, 0, Math.PI * 2); g.fill();
    g.fillStyle = 'rgba(245,245,240,0.82)';
    g.beginPath(); g.arc(x - r * 0.3, y, r * 0.75, 0, Math.PI * 2); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat * 0.8);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** Build a 3-D wireframe camera frustum pyramid to represent a SLAM keyframe pose in 3-D space. */
function makeFrustumGeometry(scale = 0.55) {
  const w = 0.65 * scale, h = 0.42 * scale, z = 0.85 * scale;
  const pts = [
    0, 0, 0, -w, -h, -z,
    0, 0, 0,  w, -h, -z,
    0, 0, 0,  w,  h, -z,
    0, 0, 0, -w,  h, -z,
    -w, -h, -z,  w, -h, -z,
     w, -h, -z,  w,  h, -z,
     w,  h, -z, -w,  h, -z,
    -w,  h, -z, -w, -h, -z,
  ];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

export async function createSlam3D({ canvas, mainEl, camEl, mainOverlay, camOverlay, onGroundClick }) {
  const R = makeRenderer(canvas, 1.18);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x03050a);

  const sun = new THREE.DirectionalLight(0xfff4e0, 3.3);
  sun.position.set(-65, 9, 25);
  sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -75, right: 75, top: 75, bottom: -75, near: 1, far: 240 });
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.04;
  scene.add(sun, sun.target);
  sun.target.position.set(0, 0, -18);

  const hemi = new THREE.HemisphereLight(0xc8c6c2, 0x3b3a37, 0.12);
  scene.add(hemi);

  const stars = starfield(GFX.low ? 1200 : 2800);
  scene.add(stars);

  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(16, 36, 24),
    new THREE.MeshStandardMaterial({ color: 0x3b82f6, emissive: 0x0c2448, roughness: 0.75 })
  );
  earth.position.set(130, 26, -390);
  scene.add(earth);

  const [rassor, lm] = await Promise.all([
    loadModel('assets/nasa/rassor.glb', 1.8, 'xz'),
    loadModel('assets/nasa/apollo_lunar_module.glb', 7.2),
  ]);

  const staticGroup = new THREE.Group();
  const dynGroup = new THREE.Group();
  scene.add(staticGroup, dynGroup);

  let groundMesh = null;
  let roverGroup = null;
  let boulderMeshes = [];
  let boulderWires = [];
  let heightFn = () => 0;
  let boundSession = null;

  // Dynamic 3-D Occupancy & Fog-of-War Carpet Texture
  const occCanvas = document.createElement('canvas');
  occCanvas.width = 192;
  occCanvas.height = 192;
  const occCtx = occCanvas.getContext('2d');
  const occTex = new THREE.CanvasTexture(occCanvas);
  occTex.minFilter = THREE.LinearFilter;
  occTex.magFilter = THREE.LinearFilter;
  let occCarpetMesh = null;
  let scanConeMesh = null;
  let frontierBeacon = null;
  let frontierNavLine = null;
  let poseEllipseLine = null;
  let landmarkCovLines = null;
  let mappedCloudPoints = null;

  // Dynamic 3-D SLAM visual layers
  const frustumGeo = makeFrustumGeometry(0.6);
  const frustumPools = new Map();
  const pathLines = new Map();
  let errorStruts = null;
  const loopArcsGroup = new THREE.Group();
  let rayLines = null;
  let landmarkPoints = null;
  let waypointBeacon = null;
  dynGroup.add(loopArcsGroup);

  // Cameras
  const freeCam = new THREE.PerspectiveCamera(46, 1, 0.1, 2000);
  freeCam.position.set(0, 25, 19);
  const controls = new OrbitControls(freeCam, mainEl);
  mainEl.addEventListener(
    'pointerdown',
    (e) => {
      if (e.target.closest('button, a, input, select, label')) e.stopPropagation();
    },
    true
  );
  controls.target.set(0, 0.8, -18);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.492;
  controls.minDistance = 4;
  controls.maxDistance = 170;

  const roverCam = new THREE.PerspectiveCamera(62, 1, 0.1, 1500);

  // Raycaster for click-to-drive waypoint on the 3-D lunar terrain
  const raycaster = new THREE.Raycaster();
  let downPos = null;
  mainEl.addEventListener('pointerdown', (e) => {
    if (e.button === 0 && !e.target.closest('button, a, input, select, label')) {
      downPos = [e.clientX, e.clientY];
    }
  });
  mainEl.addEventListener('pointerup', (e) => {
    if (!downPos || !groundMesh || !onGroundClick) return;
    const move = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]);
    downPos = null;
    if (move > 5) return;
    const rect = mainEl.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(ndc, freeCam);
    const hits = raycaster.intersectObject(groundMesh, false);
    if (hits.length > 0) {
      const p = hits[0].point;
      onGroundClick(p.x, -p.z);
    }
  });

  function updateSunDirection(sunElDeg, sunAzDeg = 22) {
    const el = (Math.max(0.8, sunElDeg) * Math.PI) / 180;
    const az = (sunAzDeg * Math.PI) / 180;
    const K = 80;
    sun.position.set(K * Math.cos(el) * Math.cos(az), K * Math.sin(el), -K * Math.cos(el) * Math.sin(az));
  }

  function buildSceneForSession(sess) {
    boundSession = sess;
    while (staticGroup.children.length) staticGroup.remove(staticGroup.children[0]);
    boulderMeshes = [];
    boulderWires = [];

    const craters = [
      { x: -34, y: 12, r: 7.5, depth: 1.3 },
      { x: 35, y: 16, r: 8.2, depth: 1.5 },
      { x: -28, y: 42, r: 6.8, depth: 1.2 },
      { x: 31, y: 40, r: 7.0, depth: 1.3 },
      { x: 0, y: -18, r: 6.0, depth: 1.0 },
      { x: 0, y: 58, r: 9.0, depth: 1.6 },
    ];
    heightFn = makeTerrainHeight(craters);
    updateSunDirection(sess.sunElDeg, sess.sunAzDeg);

    // Terrain mesh with McEwen Lunar-Lambert shader
    const G = GFX.low
      ? new THREE.PlaneGeometry(210, 170, 110, 88)
      : new THREE.PlaneGeometry(210, 170, 210, 170);
    G.rotateX(-Math.PI / 2);
    const pos = G.attributes.position;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = -pos.getZ(i) + 18;
      pos.setZ(i, -y);
      pos.setY(i, heightFn(x, y));
      const g = 0.43 + (fbm(x * 0.55, y * 0.55) - 0.5) * 0.18;
      col.set([g, g, g * 0.97], i * 3);
    }
    G.setAttribute('color', new THREE.BufferAttribute(col, 3));
    G.computeVertexNormals();

    const gmat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 1.0,
      metalness: 0.0,
      map: buildRegolithDetailTexture(210 / 4),
    });
    gmat.onBeforeCompile = (sh) => {
      lunarLambertShader(sh);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vWxz;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWxz = (modelMatrix * vec4(transformed, 1.0)).xz;');
      sh.fragmentShader = sh.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
varying vec2 vWxz;
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }`
        )
        .replace(
          '#include <map_fragment>',
          `#include <map_fragment>
diffuseColor.rgb *= 1.58 * texture2D(map, vMapUv * 0.618 + vec2(0.37, 0.11)).rgb
  * (0.78 + 0.22 * vnoise(vWxz * 0.9) + 0.12 * vnoise(vWxz * 3.7 + 17.0) - 0.06);`
        );
    };

    groundMesh = new THREE.Mesh(G, gmat);
    groundMesh.receiveShadow = true;
    staticGroup.add(groundMesh);

    // 3-D Live Occupancy Grid & Fog-of-War Carpet draped over the 60x60m mission sector
    const carpetGeo = new THREE.PlaneGeometry(60, 60, 48, 48);
    carpetGeo.rotateX(-Math.PI / 2);
    const cPos = carpetGeo.attributes.position;
    for (let i = 0; i < cPos.count; i++) {
      const x = cPos.getX(i);
      const y = -cPos.getZ(i) + 20; // Centered at y = 20 (covers y in [-10, 50])
      cPos.setZ(i, -y);
      cPos.setY(i, heightFn(x, y) + 0.06);
    }
    carpetGeo.computeVertexNormals();
    const carpetMat = new THREE.MeshBasicMaterial({
      map: occTex,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    });
    occCarpetMesh = new THREE.Mesh(carpetGeo, carpetMat);
    occCarpetMesh.renderOrder = 2;
    staticGroup.add(occCarpetMesh);

    // Boulders + 3-D Reconstructed Map Wireframe Cages
    sess.boulders.forEach((b, i) => {
      const geo = new THREE.IcosahedronGeometry(1, GFX.low ? 2 : 3);
      const p = geo.attributes.position;
      for (let k = 0; k < p.count; k++) {
        const v = new THREE.Vector3().fromBufferAttribute(p, k);
        const n = 0.78 + 0.42 * fbm(v.x * 1.7 + i * 3, v.y * 1.7 + v.z * 1.3);
        p.setXYZ(k, v.x * n, v.y * n, v.z * n);
      }
      geo.computeVertexNormals();
      const rmat = new THREE.MeshStandardMaterial({
        color: 0x80848c,
        roughness: 0.94,
        flatShading: true,
        transparent: true,
        opacity: 0.22, // Starts dimmed in Fog-of-War until discovered!
      });
      rmat.onBeforeCompile = lunarLambertShader;
      const rock = new THREE.Mesh(geo, rmat);
      rock.scale.set(b.r, b.h, b.r);
      rock.position.copy(toV(b.x, b.y, heightFn(b.x, b.y)));
      rock.rotation.y = hash(i, 5) * Math.PI * 2;
      rock.castShadow = rock.receiveShadow = true;
      staticGroup.add(rock);
      boulderMeshes.push(rock);

      // Reconstructed 3-D SLAM Obstacle Wireframe Shell (appears when sensor maps the boulder!)
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x34d399,
        wireframe: true,
        transparent: true,
        opacity: 0.55,
      });
      const wireMesh = new THREE.Mesh(geo, wireMat);
      wireMesh.scale.set(b.r * 1.03, b.h * 1.03, b.r * 1.03);
      wireMesh.position.copy(rock.position);
      wireMesh.rotation.y = rock.rotation.y;
      wireMesh.visible = false;
      staticGroup.add(wireMesh);
      boulderWires.push(wireMesh);
    });

    // Apollo Lunar Module base landmark
    if (lm) {
      const m = lm.clone();
      m.position.copy(toV(-26, -7, heightFn(-26, -7)));
      m.rotation.y = 0.55;
      staticGroup.add(m);
    }

    // RASSOR Rover + Stereo Mast Camera
    roverGroup = new THREE.Group();
    staticGroup.add(roverGroup);
    if (rassor) {
      const r = rassor.clone();
      roverGroup.add(r);
    } else {
      const ch = new THREE.Mesh(
        new THREE.BoxGeometry(1.3, 0.5, 1.6),
        new THREE.MeshStandardMaterial({ color: 0xd4d8de, metalness: 0.5, roughness: 0.4 })
      );
      ch.position.y = 0.5;
      roverGroup.add(ch);
    }
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.055, MAST_H, 12),
      new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.65, roughness: 0.35 })
    );
    mast.position.set(0, MAST_H * 0.5, 0);
    mast.castShadow = true;
    roverGroup.add(mast);

    const stereoBar = new THREE.Mesh(
      new THREE.BoxGeometry(0.46, 0.16, 0.22),
      new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.6, roughness: 0.4 })
    );
    stereoBar.position.set(0, MAST_H, 0);
    roverGroup.add(stereoBar);

    const puck = new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.11, 0.1, 16),
      new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.45 })
    );
    puck.position.set(0, MAST_H + 0.14, 0);
    roverGroup.add(puck);

    // Active 100° Sensor Scan Fan attached to the rover so the viewer sees the live mapping cone!
    const fanGeo = new THREE.BufferGeometry();
    const fanVerts = [0, 0.15, 0];
    const nFan = 24;
    const hfov = (100 * Math.PI) / 180;
    const range = 20.5;
    for (let i = 0; i <= nFan; i++) {
      const a = (i / nFan - 0.5) * hfov;
      fanVerts.push(Math.sin(a) * range, 0.15, -Math.cos(a) * range);
    }
    const fanIndices = [];
    for (let i = 1; i <= nFan; i++) {
      fanIndices.push(0, i, i + 1);
    }
    fanGeo.setAttribute('position', new THREE.Float32BufferAttribute(fanVerts, 3));
    fanGeo.setIndex(fanIndices);
    scanConeMesh = new THREE.Mesh(
      fanGeo,
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.11,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    scanConeMesh.renderOrder = 3;
    roverGroup.add(scanConeMesh);

    if (!waypointBeacon) {
      waypointBeacon = new THREE.Mesh(
        new THREE.CylinderGeometry(0.08, 0.35, 2.4, 16),
        new THREE.MeshBasicMaterial({ color: 0x4cc9f0, wireframe: true, transparent: true, opacity: 0.85 })
      );
      waypointBeacon.visible = false;
      dynGroup.add(waypointBeacon);
    }

    if (!frontierBeacon) {
      frontierBeacon = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.45, 0),
        new THREE.MeshBasicMaterial({ color: 0x34d399, wireframe: true, transparent: true, opacity: 0.85 })
      );
      dynGroup.add(frontierBeacon);
    }

    if (!frontierNavLine) {
      frontierNavLine = new THREE.Line(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.45 })
      );
      dynGroup.add(frontierNavLine);
    }
  }

  /** Paint the 48x48 Log-Odds Occupancy Grid + Fog-of-War onto the 3-D ground carpet texture. */
  function updateOccupancyCarpetTexture(sess, robotMapOnly) {
    const g = sess.grid;
    const W = occCanvas.width, H = occCanvas.height;
    const cw = W / g.cols, ch = H / g.rows;
    occCtx.clearRect(0, 0, W, H);

    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const lo = g.logOdds[r * g.cols + c];
        // Note: canvas y=0 is top (north = yMax = +50), y=H is bottom (south = yMin = -10)
        const py = (g.rows - 1 - r) * ch;
        const px = c * cw;

        if (Math.abs(lo) < 0.15) {
          // UNKNOWN (Unmapped Fog-of-War)
          occCtx.fillStyle = robotMapOnly ? 'rgba(2, 4, 9, 0.96)' : 'rgba(3, 6, 14, 0.72)';
          occCtx.fillRect(px, py, cw + 0.5, ch + 0.5);
        } else if (lo < -0.15) {
          // FREE TRAVERSABLE SPACE discovered by sensor rays
          occCtx.fillStyle = 'rgba(16, 185, 129, 0.12)';
          occCtx.fillRect(px, py, cw, ch);
          occCtx.strokeStyle = 'rgba(56, 189, 248, 0.18)';
          occCtx.lineWidth = 0.5;
          occCtx.strokeRect(px, py, cw, ch);
        } else {
          // OCCUPIED BOULDER FOOTPRINT mapped by sensor rays
          occCtx.fillStyle = 'rgba(245, 158, 11, 0.55)';
          occCtx.fillRect(px, py, cw, ch);
        }
      }
    }
    occTex.needsUpdate = true;
  }

  function getFrustum(poolKey, idx, colorHex) {
    if (!frustumPools.has(poolKey)) frustumPools.set(poolKey, []);
    const arr = frustumPools.get(poolKey);
    if (!arr[idx]) {
      const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.78 });
      const seg = new THREE.LineSegments(frustumGeo, mat);
      dynGroup.add(seg);
      arr[idx] = seg;
    }
    arr[idx].visible = true;
    return arr[idx];
  }

  function hideAllFrustums() {
    for (const arr of frustumPools.values()) {
      for (const f of arr) f.visible = false;
    }
  }

  function updateTrajectoryLine(key, points3D, colorHex, opacity = 0.95) {
    let line = pathLines.get(key);
    if (!line) {
      const geo = new THREE.BufferGeometry();
      const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity });
      line = new THREE.Line(geo, mat);
      line.renderOrder = 4;
      dynGroup.add(line);
      pathLines.set(key, line);
    }
    if (points3D.length < 2) {
      line.visible = false;
      return;
    }
    const flat = [];
    for (const p of points3D) {
      const gh = heightFn(p[0], -p[2]);
      flat.push(p[0], gh + 0.18, p[2]);
    }
    line.geometry.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    line.geometry.computeBoundingSphere();
    line.visible = true;
  }

  function render(sess, opts = {}) {
    const {
      selectedIdx = 0,
      showAllPaths = true,
      showFrustums = true,
      showRays = true,
      showLandmarks = true,
      showErrorStruts = true,
      robotMapOnly = false,
      followRover = false,
    } = opts;

    if (!boundSession || boundSession !== sess) {
      buildSceneForSession(sess);
    }
    updateSunDirection(sess.sunElDeg, sess.sunAzDeg);

    // Update 3-D Occupancy & Fog-of-War Carpet every 3 steps
    if (sess.stepCount % 3 === 0 || sess.stepCount <= 2) {
      updateOccupancyCarpetTexture(sess, robotMapOnly);
    }

    // Update boulder visibility & 3-D reconstructed wireframe shells based on live discovery!
    sess.boulders.forEach((b, i) => {
      const rock = boulderMeshes[i];
      const wire = boulderWires[i];
      if (!rock || !wire) return;
      if (b.discovered) {
        rock.visible = !robotMapOnly;
        rock.material.opacity = 1.0;
        rock.castShadow = true;
        wire.visible = true;
        wire.material.opacity = robotMapOnly ? 0.88 : 0.45;
      } else {
        // Undiscovered boulder in Fog-of-War!
        rock.visible = !robotMapOnly;
        rock.material.opacity = 0.15;
        rock.castShadow = !robotMapOnly;
        wire.visible = false;
      }
    });

    // Place physical rover
    const rx = sess.rover.x, ry = sess.rover.y, rgh = heightFn(rx, ry);
    roverGroup.position.copy(toV(rx, ry, rgh));
    roverGroup.rotation.y = sess.rover.th - Math.PI / 2;

    if (followRover) {
      const targetV = toV(rx, ry, rgh + 1.2);
      controls.target.lerp(targetV, 0.12);
    }
    controls.update();

    // Waypoint or Active Frontier Beacon
    if (sess.waypoint) {
      const wx = sess.waypoint.x, wy = sess.waypoint.y;
      waypointBeacon.position.copy(toV(wx, wy, heightFn(wx, wy) + 1.2));
      waypointBeacon.rotation.y += 0.06;
      waypointBeacon.visible = true;
    } else if (waypointBeacon) {
      waypointBeacon.visible = false;
    }

    if (sess.frontierTarget) {
      const fx = sess.frontierTarget.x, fy = sess.frontierTarget.y;
      const fgh = heightFn(fx, fy);
      frontierBeacon.position.copy(toV(fx, fy, fgh + 0.9));
      frontierBeacon.rotation.y += 0.05;
      frontierBeacon.material.color.setHex(
        sess.frontierTarget.mode === 'homing' ? 0xf59e0b : 0xa78bfa
      );
      frontierBeacon.visible = true;

      frontierNavLine.geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute([rx, rgh + 0.35, -ry, fx, fgh + 0.65, -fy], 3)
      );
      frontierNavLine.visible = true;
    } else if (frontierBeacon) {
      frontierBeacon.visible = false;
      frontierNavLine.visible = false;
    }

    // 1. Ground Truth & Estimator 3D Trajectories (Live Unaligned SLAM Pose Path!)
    const truthPts = sess.truthKeyframes.map((k) => k.pos);
    updateTrajectoryLine('truth', robotMapOnly ? [] : truthPts, 0xe2e8f0, 0.95);

    sess.estimators.forEach((est, idx) => {
      const show = showAllPaths || idx === selectedIdx;
      const pts = show ? est.keyframes.map((k) => k.pos) : [];
      updateTrajectoryLine(est.id, pts, est.color, idx === selectedIdx ? 1.0 : 0.65);
    });

    // 2. Selected Estimator's Live 2-Sigma Pose Covariance Ellipse on the 3-D Ground
    const selEst = sess.estimators[selectedIdx];
    if (!poseEllipseLine) {
      poseEllipseLine = new THREE.LineLoop(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.95 })
      );
      poseEllipseLine.renderOrder = 6;
      dynGroup.add(poseEllipseLine);
    }
    {
      const ex = selEst.pose.x;
      const ey = selEst.pose.y;
      const egh = heightFn(ex, ey);
      const e = covEllipse(selEst.poseCov.xx, selEst.poseCov.xy, selEst.poseCov.yy, 2);
      const A = Math.min(8, Math.max(0.25, e.a)), B = Math.min(8, Math.max(0.25, e.b));
      const ca = Math.cos(e.angle), sa = Math.sin(e.angle);
      const ellPts = [];
      const nSeg = 40;
      for (let i = 0; i < nSeg; i++) {
        const a = (i / nSeg) * Math.PI * 2;
        const u = Math.cos(a) * A, v = Math.sin(a) * B;
        ellPts.push(ex + ca * u - sa * v, egh + 0.26, -(ey + sa * u + ca * v));
      }
      poseEllipseLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(ellPts, 3));
      poseEllipseLine.material.color.set(selEst.color);
      poseEllipseLine.visible = true;
    }

    // 3. Keyframe 3-D Camera Frustums
    hideAllFrustums();
    if (showFrustums) {
      const stride = Math.max(1, Math.floor(sess.truthKeyframes.length / 24));
      let fCount = 0;
      for (let i = 0; i < sess.truthKeyframes.length; i += stride) {
        const tk = sess.truthKeyframes[i];
        if (!robotMapOnly) {
          const gh = heightFn(tk.pos[0], -tk.pos[2]);
          const tf = getFrustum('truth', fCount, 0x94a3b8);
          tf.position.set(tk.pos[0], gh + 0.65, tk.pos[2]);
          tf.rotation.set(0, tk.th - Math.PI / 2, 0);
        }

        const ek = selEst.keyframes[i];
        if (ek) {
          const ep = ek.pos;
          const egh = heightFn(ep[0], -ep[2]);
          const ef = getFrustum(selEst.id, fCount, selEst.color);
          ef.position.set(ep[0], egh + 0.65, ep[2]);
          ef.rotation.set(0, ek.th - Math.PI / 2, 0);
        }
        fCount++;
      }
    }

    // 4. Red 3-D Localization Error Struts between Ground Truth and Selected Estimator
    if (!errorStruts) {
      errorStruts = new THREE.LineSegments(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0xff5d6c, transparent: true, opacity: 0.75 })
      );
      errorStruts.renderOrder = 5;
      dynGroup.add(errorStruts);
    }
    if (showErrorStruts && !robotMapOnly && selEst.keyframes.length >= 2) {
      const strutPts = [];
      const n = Math.min(sess.truthKeyframes.length, selEst.keyframes.length);
      for (let i = 0; i < n; i += 2) {
        const tp = sess.truthKeyframes[i].pos;
        const ep = selEst.keyframes[i].pos;
        const tgh = heightFn(tp[0], -tp[2]);
        const egh = heightFn(ep[0], -ep[2]);
        strutPts.push(tp[0], tgh + 0.25, tp[2], ep[0], egh + 0.25, ep[2]);
      }
      errorStruts.geometry.setAttribute('position', new THREE.Float32BufferAttribute(strutPts, 3));
      errorStruts.visible = true;
    } else {
      errorStruts.visible = false;
    }

    // 5. Loop-Closure 3-D Glowing Constraint Arcs
    while (loopArcsGroup.children.length) {
      const c = loopArcsGroup.children[0];
      c.geometry.dispose();
      loopArcsGroup.remove(c);
    }
    for (const lc of sess.loopClosures) {
      const p0 = new THREE.Vector3(lc.fromPos[0], heightFn(lc.fromPos[0], -lc.fromPos[2]) + 0.3, lc.fromPos[2]);
      const p2 = new THREE.Vector3(lc.toPos[0], heightFn(lc.toPos[0], -lc.toPos[2]) + 0.3, lc.toPos[2]);
      const mid = p0.clone().add(p2).multiplyScalar(0.5);
      mid.y += 3.2;
      const curve = new THREE.QuadraticBezierCurve3(p0, mid, p2);
      const pts = curve.getPoints(20);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.95 });
      loopArcsGroup.add(new THREE.Line(geo, mat));
    }

    // 6. Incremental 3-D Reconstructed Surface Point Cloud + Estimated Landmarks + Shrinking 3-D Uncertainty Rings!
    if (!mappedCloudPoints) {
      mappedCloudPoints = new THREE.Points(
        new THREE.BufferGeometry(),
        new THREE.PointsMaterial({ size: 0.19, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.82 })
      );
      mappedCloudPoints.renderOrder = 5;
      dynGroup.add(mappedCloudPoints);
    }
    if (!landmarkPoints) {
      landmarkPoints = new THREE.Points(
        new THREE.BufferGeometry(),
        new THREE.PointsMaterial({ size: 0.34, vertexColors: true, sizeAttenuation: true })
      );
      landmarkPoints.renderOrder = 6;
      dynGroup.add(landmarkPoints);
    }
    if (!landmarkCovLines) {
      landmarkCovLines = new THREE.LineSegments(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.72 })
      );
      landmarkCovLines.renderOrder = 6;
      dynGroup.add(landmarkCovLines);
    }

    if (showLandmarks) {
      // A) Dense 3-D surface point cloud discovered so far
      const pcPos = [], pcCol = [];
      for (const pt of sess.mappedPointCloud) {
        const gh = heightFn(pt.x, pt.y);
        pcPos.push(pt.x, gh + pt.z, -pt.y);
        if (pt.kind === 'rock') {
          pcCol.push(0.96, 0.68, 0.22); // Warm amber reconstructed boulder point
        } else {
          pcCol.push(0.22, 0.78, 0.92); // Cyan reconstructed ground point
        }
      }
      mappedCloudPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pcPos, 3));
      mappedCloudPoints.geometry.setAttribute('color', new THREE.Float32BufferAttribute(pcCol, 3));
      mappedCloudPoints.visible = pcPos.length > 0;

      // B) Selected Estimator's Triangulated Landmarks (m_hat_j) + 3-D Uncertainty Rings (2*sqrt(sigma_XX))
      const lPos = [], lCol = [];
      const covPos = [], covCol = [];
      const status = new Map(sess.activeObservations.map((o) => [o.lm.id, o.status]));
      // Green = checked this frame, amber = just added, blue = on the map.
      const COL = { known: [0.2, 0.83, 0.6], new: [0.96, 0.62, 0.04], map: [0.22, 0.74, 0.97] };

      for (const [lmId, estLm] of selEst.mapLandmarks.entries()) {
        if (estLm.x === undefined) continue;
        const gh = heightFn(estLm.x, estLm.y);
        const py = gh + estLm.z;
        lPos.push(estLm.x, py, -estLm.y);
        const [rC, gC, bC] = COL[status.get(lmId)] || COL.map;
        lCol.push(rC, gC, bC);

        // 2-sigma ellipse of this landmark's position, flat at its height
        const e = covEllipse(estLm.sigmaXX, estLm.sigmaXY, estLm.sigmaYY, 2);
        const A = Math.min(4, Math.max(0.08, e.a)), B = Math.min(4, Math.max(0.08, e.b));
        const ca = Math.cos(e.angle), sa = Math.sin(e.angle);
        const segs = 16;
        const pt = (k) => {
          const a = (k / segs) * Math.PI * 2, u = Math.cos(a) * A, v = Math.sin(a) * B;
          return [estLm.x + ca * u - sa * v, py, -(estLm.y + sa * u + ca * v)];
        };
        for (let k = 0; k < segs; k++) {
          covPos.push(...pt(k), ...pt(k + 1));
          covCol.push(rC, gC, bC, rC, gC, bC);
        }
      }
      landmarkPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(lPos, 3));
      landmarkPoints.geometry.setAttribute('color', new THREE.Float32BufferAttribute(lCol, 3));
      landmarkPoints.visible = lPos.length > 0;

      landmarkCovLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(covPos, 3));
      landmarkCovLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(covCol, 3));
      landmarkCovLines.visible = covPos.length > 0;
    } else {
      mappedCloudPoints.visible = false;
      landmarkPoints.visible = false;
      landmarkCovLines.visible = false;
    }

    // 7. Active Sensor Measurement Rays from Rover Mast Camera to Observed Features
    if (!rayLines) {
      rayLines = new THREE.LineSegments(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.48 })
      );
      rayLines.renderOrder = 5;
      dynGroup.add(rayLines);
    }
    if (showRays && sess.activeObservations.length > 0) {
      const rPos = [], rCol = [];
      const eye = toV(rx, ry, rgh + MAST_H);
      const RAY = { known: [0.2, 0.92, 0.6], new: [0.96, 0.65, 0.18], missed: [0.95, 0.3, 0.35] };
      for (const ob of sess.activeObservations) {
        const c = RAY[ob.status];
        if (!c) continue; // skipped (over the budget) or ignored (camera off)
        const lgh = heightFn(ob.lm.x, ob.lm.y);
        rPos.push(eye.x, eye.y, eye.z, ob.lm.x, lgh + ob.lm.z, -ob.lm.y);
        rCol.push(...c, ...c);
      }
      rayLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(rPos, 3));
      rayLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(rCol, 3));
      rayLines.visible = true;
    } else {
      rayLines.visible = false;
    }

    // --- Render Both Viewports on Single WebGL Canvas ---
    const parentRect = canvas.parentElement.getBoundingClientRect();
    const pr = R.getPixelRatio();
    const fullW = Math.floor(parentRect.width);
    const fullH = Math.floor(parentRect.height);
    if (!fullW || !fullH) return;
    if (canvas.width !== Math.floor(fullW * pr) || canvas.height !== Math.floor(fullH * pr)) {
      R.setSize(fullW, fullH, false);
    }
    R.setScissorTest(true);

    // Viewport 1: Main 3-D Orbit View (Top)
    const mRect = mainEl.getBoundingClientRect();
    const mLeft = mRect.left - parentRect.left;
    const mBottom = parentRect.bottom - mRect.bottom;
    freeCam.aspect = Math.max(1, mRect.width) / Math.max(1, mRect.height);
    freeCam.updateProjectionMatrix();
    R.setViewport(mLeft, mBottom, mRect.width, mRect.height);
    R.setScissor(mLeft, mBottom, mRect.width, mRect.height);
    R.render(scene, freeCam);

    // 3-D floating pill labels on main view
    const labels3D = [];
    if (!robotMapOnly) {
      labels3D.push({
        p: toV(rx, ry, rgh + MAST_H + 0.9),
        text: `Real rover (${rx.toFixed(1)}, ${ry.toFixed(1)}) m`,
        color: '#e2e8f0',
        bg: '#0f172acc',
        fg: '#f8fafc',
      });
    }
    {
      const ex = selEst.pose.x, ey = selEst.pose.y;
      const egh = heightFn(ex, ey);
      labels3D.push({
        p: toV(ex, ey, egh + 1.55),
        text: `SLAM thinks it is here · off by ${selEst.metrics.rawPoseErr.toFixed(2)} m · ±${(selEst.metrics.poseSigmaM * 2).toFixed(2)} m`,
        color: selEst.color,
      });
    }
    if (sess.frontierTarget) {
      const fx = sess.frontierTarget.x, fy = sess.frontierTarget.y;
      labels3D.push({
        p: toV(fx, fy, heightFn(fx, fy) + 1.7),
        text: { route: 'Next waypoint', homing: 'Back to mapped ground', explore: 'Edge of the map (frontier)' }[sess.frontierTarget.mode],
        color: sess.frontierTarget.mode === 'homing' ? '#f59e0b' : '#a78bfa',
      });
    }
    for (const lc of sess.loopClosures.slice(-2)) {
      labels3D.push({
        p: new THREE.Vector3(
          (lc.fromPos[0] + lc.toPos[0]) * 0.5,
          heightFn(lc.toPos[0], -lc.toPos[2]) + 3.6,
          (lc.fromPos[2] + lc.toPos[2]) * 0.5
        ),
        text: `Loop closure at ${lc.t.toFixed(0)} s`,
        color: '#34d399',
        bg: '#064e3b',
        fg: '#6ee7b7',
      });
    }
    drawLabels(mainOverlay, freeCam, labels3D);

    // Viewport 2: Rover Stereo Mast Camera Feed (Bottom-Left)
    const cRect = camEl.getBoundingClientRect();
    if (cRect.width > 20 && cRect.height > 20) {
      const cLeft = cRect.left - parentRect.left;
      const cBottom = parentRect.bottom - cRect.bottom;
      const eye = toV(rx, ry, rgh + MAST_H);
      roverCam.position.copy(eye);
      roverCam.lookAt(
        eye.clone().add(new THREE.Vector3(Math.cos(sess.rover.th), Math.tan(-0.08), -Math.sin(sess.rover.th)))
      );
      roverCam.aspect = cRect.width / cRect.height;
      roverCam.updateProjectionMatrix();

      const prevRayVis = rayLines.visible;
      const prevLmVis = landmarkPoints.visible;
      const prevCovVis = landmarkCovLines.visible;
      const prevCloudVis = mappedCloudPoints.visible;
      const prevCarpetVis = occCarpetMesh.visible;
      const prevConeVis = scanConeMesh.visible;

      rayLines.visible = false;
      landmarkPoints.visible = false;
      landmarkCovLines.visible = false;
      mappedCloudPoints.visible = false;
      occCarpetMesh.visible = false;
      scanConeMesh.visible = false;
      stars.visible = false;

      R.setViewport(cLeft, cBottom, cRect.width, cRect.height);
      R.setScissor(cLeft, cBottom, cRect.width, cRect.height);
      R.render(scene, roverCam);

      rayLines.visible = prevRayVis;
      landmarkPoints.visible = prevLmVis;
      landmarkCovLines.visible = prevCovVis;
      mappedCloudPoints.visible = prevCloudVis;
      occCarpetMesh.visible = prevCarpetVis;
      scanConeMesh.visible = prevConeVis;
      stars.visible = true;

      drawMastCameraHud(camOverlay, roverCam, sess, heightFn);
    }
  }

  return {
    render,
    resetCamera: () => {
      freeCam.position.set(0, 25, 19);
      controls.target.set(0, 0.8, -18);
      controls.update();
    },
  };
}

/** Mark each feature in the camera image by what SLAM did with it this frame. */
function drawMastCameraHud(ov, roverCam, sess, heightFn) {
  const dpr = window.devicePixelRatio || 1;
  const w = ov.clientWidth, h = ov.clientHeight;
  if (!w || !h) return;
  if (ov.width !== Math.round(w * dpr) || ov.height !== Math.round(h * dpr)) {
    ov.width = Math.round(w * dpr);
    ov.height = Math.round(h * dpr);
  }
  const g = ov.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  for (const ob of sess.activeObservations) {
    const gh = heightFn(ob.lm.x, ob.lm.y);
    const p3 = toV(ob.lm.x, ob.lm.y, gh + ob.lm.z).project(roverCam);
    if (p3.z < -1 || p3.z > 1 || Math.abs(p3.x) > 0.98 || Math.abs(p3.y) > 0.96) continue;
    const x = (p3.x * 0.5 + 0.5) * w;
    const y = (-p3.y * 0.5 + 0.5) * h;
    g.lineWidth = 1.5;
    if (ob.status === 'known') {
      g.strokeStyle = '#34d399'; // on the map already: used as a check
      g.strokeRect(x - 5, y - 5, 10, 10);
      g.fillStyle = '#34d399';
      g.font = '9.5px ui-monospace, Menlo, monospace';
      g.fillText(`#${ob.lm.id}`, x + 7, y - 4);
    } else if (ob.status === 'new') {
      g.strokeStyle = '#f59e0b'; // first sighting: goes onto the map
      g.beginPath();
      g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath();
      g.stroke();
    } else if (ob.status === 'missed') {
      g.strokeStyle = '#ef4444'; // in view but not found (shadow or noise)
      g.beginPath();
      g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4);
      g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4);
      g.stroke();
    } else {
      g.strokeStyle = 'rgba(148,163,184,0.6)'; // found but not used this frame
      g.strokeRect(x - 3, y - 3, 6, 6);
    }
  }

  const tel = sess.cameraTelemetry || {};
  g.fillStyle = 'rgba(10, 14, 22, 0.86)';
  g.fillRect(8, h - 26, Math.min(w - 16, 470), 20);
  g.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  g.fillStyle = !sess.useCamera ? '#ef4444' : tel.corrected ? '#34d399' : '#fbbf24';
  g.fillText(
    sess.useCamera
      ? `IN VIEW ${tel.inView}  |  CHECKS ${tel.corrected}  |  NEW ${tel.added}  |  MISSED ${tel.missed}  |  IN SHADOW ${Math.round((tel.shadowFraction || 0) * 100)}%`
      : 'CAMERA CORRECTIONS OFF: the filter ignores what it sees',
    12,
    h - 12
  );
}
