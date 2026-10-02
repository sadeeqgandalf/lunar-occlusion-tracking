"""PyTorch3D rendering of the lunar scene (pattern from 3D Deep Learning with Python, ch. 2: Meshes + TexturesVertex +
MeshRasterizer + Phong shading), adapted to a z-up world by giving the cameras up=(0, 0, 1).

Speed: PyTorch3D's CPU rasteriser tests every pixel against every triangle. The cameras and scenery never move, so
the scenery is rendered ONCE per camera (RGB + depth). Each frame only small windows around each astronaut are
rendered, with the SAME pinhole (principal point shifted, so window pixels line up exactly with the full image), and
composited by depth: a person pixel wins only where it is nearer than whatever is already there.

Each frame returns the camera image, its depth map (what a stereo camera measures) and, per person, the visible
pixels and the full silhouette (3 faces per pixel, so a person behind a rock still shows in a deeper layer):
visible / silhouette = visible fraction, measured from pixels."""
import math
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras, look_at_view_transform, RasterizationSettings, MeshRasterizer,
                                TexturesVertex, DirectionalLights, HardPhongShader, BlendParams)
from . import shapes

DEV = torch.device('cpu')


def to_mesh(v, f, c):
    return Meshes(verts=[torch.tensor(v, dtype=torch.float32)], faces=[torch.tensor(f, dtype=torch.int64)],
                  textures=TexturesVertex(verts_features=[torch.tensor(c, dtype=torch.float32)]))


class Scene:
    """Static geometry built once; people re-posed every frame."""

    def __init__(self, world, lander=None, identical_suits=False):
        self.world = world
        static = [shapes.terrain()] + [shapes.rock(b, i) for i, b in enumerate(world.boulders)] + [shapes.rover(world.cam)]
        if lander is not None:
            static.append(shapes.lander(*lander))
        self.static = to_mesh(*shapes.merge(static))
        stripe = (lambda t: shapes.UNIFORM_STRIPE) if identical_suits else (lambda t: shapes.STRIPES[(t['id'] - 1) % len(shapes.STRIPES)])
        self.person_mesh = [shapes.astronaut(stripe(t)) for t in world.targets]
        sun = world.sun if world.cfg['shadows'] else dict(az=world.sun['az'], el=math.radians(32))
        self.sun_dir = np.array([math.cos(sun['el']) * math.cos(sun['az']), math.cos(sun['el']) * math.sin(sun['az']), math.sin(sun['el'])])
        self.lights = DirectionalLights(direction=[tuple(-self.sun_dir)], ambient_color=((0.50, 0.52, 0.58),),
                                        diffuse_color=((0.78, 0.75, 0.68),), specular_color=((0.05, 0.05, 0.05),), device=DEV)

    def people(self):
        """All astronauts this frame as one mesh, plus face -> person id."""
        parts, obj = [], []
        for t, m in zip(self.world.targets, self.person_mesh):
            parts.append(shapes.place(m, t['x'], t['y'], t['h'])); obj.append(np.full(len(m[1]), t['id']))
        return to_mesh(*shapes.merge(parts)), np.concatenate(obj)


class Camera:
    """Pinhole camera at `eye` looking at `at`, horizontal field of view `hfov` (rad), image W x H (screen-space intrinsics)."""

    def __init__(self, eye, at, hfov, W, H):
        self.eye, self.W, self.H, self.hfov = np.array(eye, float), W, H, hfov
        self.R, self.T = look_at_view_transform(eye=(tuple(eye),), at=(tuple(at),), up=((0.0, 0.0, 1.0),))
        self.f = (W / 2) / math.tan(hfov / 2)                 # focal length in pixels (square pixels)
        self.full = self._cams(0, 0, W, H)
        self.bg_rgb = self.bg_z = None

    def _cams(self, u0, v0, w, h):
        """Same pinhole as the full image, cropped to the window [u0, u0+w) x [v0, v0+h)."""
        return PerspectiveCameras(focal_length=((self.f, self.f),), principal_point=((self.W / 2 - u0, self.H / 2 - v0),),
                                  image_size=((h, w),), in_ndc=False, R=self.R, T=self.T, device=DEV)

    def _raster(self, cams, mesh, h, w, fpp):
        r = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(
            image_size=(h, w), blur_radius=0.0, faces_per_pixel=fpp, cull_backfaces=True, bin_size=0))
        return r(mesh)

    def _shade(self, cams, frags, mesh, lights):
        sh = HardPhongShader(device=DEV, cameras=cams, lights=lights, blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
        return sh(frags, mesh)[0, ..., :3].clamp(0, 1).numpy()

    def project(self, P):
        """World points [N,3] -> pixel u (right), v (down), depth."""
        X = torch.tensor(np.atleast_2d(P), dtype=torch.float32)
        p = self.full.transform_points_screen(X).numpy().reshape(-1, 3)          # third value is 1/z here, so
        z = self.full.get_world_to_view_transform().transform_points(X).numpy().reshape(-1, 3)[:, 2]   # take depth directly
        return p[:, 0], p[:, 1], z

    def render_background(self, scene):
        frags = self._raster(self.full, scene.static, self.H, self.W, 1)
        self.bg_rgb = self._shade(self.full, frags, scene.static, scene.lights)
        z = frags.zbuf[0, ..., 0].numpy().copy(); z[z < 0] = np.inf
        self.bg_z = z

    def render(self, scene):
        """Composite this frame. Returns rgb [H,W,3], depth [H,W], owner [H,W] (person id or 0), and per-person
        dict id -> (visible pixel count, silhouette pixel count, (u0, v0, u1, v1) box of visible pixels or None)."""
        if self.bg_rgb is None:
            self.render_background(scene)
        rgb, z, owner = self.bg_rgb.copy(), self.bg_z.copy(), np.zeros((self.H, self.W), int)
        pm, obj = scene.people()
        sil = {}
        for t in scene.world.targets:
            box = np.array([[t['x'] + dx, t['y'] + dy, hz] for dx in (-0.6, 0.6) for dy in (-0.6, 0.6) for hz in (0.0, 2.0)])
            u, v, d = self.project(box)
            if (d <= 0.3).any():
                continue
            u0, v0 = max(0, int(math.floor(u.min())) - 2), max(0, int(math.floor(v.min())) - 2)
            u1, v1 = min(self.W, int(math.ceil(u.max())) + 2), min(self.H, int(math.ceil(v.max())) + 2)
            if u1 - u0 < 2 or v1 - v0 < 2:
                continue
            cams = self._cams(u0, v0, u1 - u0, v1 - v0)
            frags = self._raster(cams, pm, v1 - v0, u1 - u0, 3)
            p2f = frags.pix_to_face[0].numpy(); zb = frags.zbuf[0].numpy()
            ids = np.where(p2f >= 0, obj[np.maximum(p2f, 0)], 0)
            sil[t['id']] = sil.get(t['id'], 0) + int((ids == t['id']).any(-1).sum())   # full outline (all layers)
            near = ids[..., 0]; nz = np.where(p2f[..., 0] >= 0, zb[..., 0], np.inf)
            win = (near > 0) & (nz < z[v0:v1, u0:u1])                                    # nearer than scenery / earlier people
            if win.any():
                col = self._shade(cams, frags, pm, scene.lights)
                rgb[v0:v1, u0:u1][win] = col[win]; z[v0:v1, u0:u1][win] = nz[win]; owner[v0:v1, u0:u1][win] = near[win]
        stats = {}
        for t in scene.world.targets:
            m = owner == t['id']; n = int(m.sum())
            box = None
            if n:
                vv, uu = np.nonzero(m); box = (uu.min(), vv.min(), uu.max() + 1, vv.max() + 1)
            stats[t['id']] = (n, max(sil.get(t['id'], 0), n), box)
        return rgb, z, owner, stats
