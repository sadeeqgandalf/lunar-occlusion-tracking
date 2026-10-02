"""One PyTorch3D experiment: a world, rendered camera frames, image-based detections, and several trackers scored on
the SAME detections. Cameras: the rover mast camera, optionally a second camera on a lander."""
import math
import numpy as np
from .world import LunarWorld
from .render import Scene, Camera
from .detector import Detector
from .tracker import Tracker
from .metrics import MotMetrics

LANDER = (24.0, 46.0, 6.0)          # lander position and camera height: looks back across the worksite


class Rig:
    """A simulator camera (for the tracker: position, field of view, noise) + its PyTorch3D renderer + detector."""

    def __init__(self, world, cam, look_at, W, H):
        self.cam = cam
        self.render_cam = Camera((cam['x'], cam['y'], cam['h']), look_at, cam['fov'], W, H)
        self.detector = Detector(world, cam, self.render_cam)
        self.last = None

    def frame(self, scene):
        rgb, depth, owner, stats = self.render_cam.render(scene)
        dets, feats, gt, boxes = self.detector.detect(rgb, depth, owner, stats)
        self.last = dict(rgb=rgb, depth=depth, owner=owner, stats=stats, dets=dets, feats=feats, gt=gt, boxes=boxes)
        return dets, feats


def lander_camera(base):
    x, y, h = LANDER
    th = math.atan2(18 - y, 0 - x)          # aim at the middle of the worksite
    return {**base, 'x': x, 'y': y, 'h': h, 'th': th, 'fov': math.radians(90), 'range': 60.0}


class Session:
    def __init__(self, scenario='boulders', seed=8, variants=('naive', 'negInfo', 'negInfo+reid'), two_cameras=False,
                 identical_suits=False, W=960, H=300):
        self.world = Wd = LunarWorld(scenario, seed)
        self.scene = Scene(Wd, lander=LANDER if two_cameras else None, identical_suits=identical_suits)
        c = Wd.cam
        rover_cam = {**c, 'y': c['y'] + 0.25}
        self.rigs = [Rig(Wd, rover_cam, (c['x'], c['y'] + 40, c['h'] - 2.2), W, H)]
        if two_cameras:
            lc = lander_camera(c)
            self.rigs.append(Rig(Wd, lc, (lc['x'] + 30 * math.cos(lc['th']), lc['y'] + 30 * math.sin(lc['th']), 0.0), W, H))
        sun = Wd.sun if Wd.cfg['shadows'] else None
        cams = [r.cam for r in self.rigs]
        self.runs = []
        for v in variants:
            base, _, extra = v.partition('+')
            T = Tracker(cams, Wd.boulders, base, sun=sun, pd=Wd.cfg['pd'], clutter=Wd.cfg['clutter'], reid=extra == 'reid')
            self.runs.append(dict(tracker=T, metrics=MotMetrics()))

    def step(self):
        self.world.step()
        frames = [(r.cam, *r.frame(self.scene)) for r in self.rigs]
        for run in self.runs:
            fr = frames if run['tracker'].p['reid'] else [(c, d, None) for c, d, _ in frames]
            run['tracker'].step(fr)
            run['metrics'].update(self.world, run['tracker'])

    def run(self, seconds, progress=False):
        n = round(seconds / self.world.dt)
        for i in range(n):
            self.step()
            if progress and i % 100 == 0:
                print(f'  t = {self.world.t:5.1f} s', flush=True)
        return self

    def table(self):
        rows = []
        for r in self.runs:
            s = r['metrics'].summary(); b = s['occByDuration']
            short = (sum(x['kept'] for x in b[:2]), sum(x['n'] for x in b[:2]))
            long_ = (sum(x['kept'] for x in b[2:]), sum(x['n'] for x in b[2:]))
            rows.append((r['tracker'].name, short, long_, s['idsw'], s['idf1'], s['mota'], len(r['tracker'].reidEvents)))
        return rows
