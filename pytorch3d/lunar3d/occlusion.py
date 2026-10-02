"""3-D line of sight, cast shadows and detection probability (port of src/mot/occlusion.js, vectorised with numpy)."""
import math
import numpy as np

RAY_H = np.array([0.3, 0.8, 1.3, 1.7])
RAY_L = np.array([-0.28, -0.09, 0.09, 0.28])


def wrap(a):
    """Same arithmetic as wrapAngle in the JS: fmod toward zero, then shift."""
    a = np.fmod(np.asarray(a, dtype=float) + math.pi, 2 * math.pi)
    a = np.where(a < 0, a + 2 * math.pi, a)
    return a - math.pi


def visibility(cam, boulders, X, Y, rt=0.4):
    """Fraction of 16 sight lines (4 heights x 4 lateral offsets) from the mast camera that reach each person."""
    X = np.atleast_1d(np.asarray(X, float)); Y = np.atleast_1d(np.asarray(Y, float))
    dx, dy = X - cam['x'], Y - cam['y']
    rng_, brg = np.hypot(dx, dy), np.arctan2(dy, dx)
    in_fov = (rng_ <= cam['range']) & (np.abs(wrap(brg - cam['th'])) <= cam['fov'] / 2) & (rng_ > 0.5)
    L = np.repeat(RAY_L, len(RAY_H)) * (rt / 0.4)            # lateral outer, height inner (JS loop order)
    H = np.tile(RAY_H, len(RAY_L))
    nx, ny = -np.sin(brg)[:, None], np.cos(brg)[:, None]
    PX, PY = X[:, None] + L * nx, Y[:, None] + L * ny
    hit = np.zeros(PX.shape, bool)
    ez = cam.get('h', 2.2)
    for b in boulders:
        h = b['h'] if np.isfinite(b['h']) else 1e6
        ox, oy, oz = (cam['x'] - b['x']) / b['r'], (cam['y'] - b['y']) / b['r'], ez / h
        ddx, ddy, ddz = (PX - cam['x']) / b['r'], (PY - cam['y']) / b['r'], (H - ez) / h
        a = ddx ** 2 + ddy ** 2 + ddz ** 2
        bq = 2 * (ox * ddx + oy * ddy + oz * ddz)
        c = ox ** 2 + oy ** 2 + oz ** 2 - 1
        disc = bq ** 2 - 4 * a * c
        s = np.sqrt(np.maximum(disc, 0))
        t1, t2 = (-bq - s) / (2 * a), (-bq + s) / (2 * a)
        hit |= (disc > 0) & (t1 < 1) & (t2 > 0)
    vis = (~hit).sum(1)
    vis_frac = vis / L.size
    lat = ((~hit) * L).sum(1)
    center = brg.copy()
    part = (vis > 0) & (vis < L.size)
    center[part] = wrap(brg[part] + np.arctan2(lat[part] / vis[part], rng_[part]))
    return dict(inFov=in_fov, range=rng_, bearing=brg, visFrac=vis_frac, visCenter=center)


def shadow_fraction(boulders, X, Y, sun):
    """Fraction of the body (3 heights) inside a boulder's cast shadow under a low sun."""
    X = np.atleast_1d(np.asarray(X, float)); Y = np.atleast_1d(np.asarray(Y, float))
    if sun is None:
        return np.zeros_like(X)
    SH, LEN = np.array([0.4, 0.9, 1.4]), 80.0
    cx, cy, cz = math.cos(sun['el']) * math.cos(sun['az']), math.cos(sun['el']) * math.sin(sun['az']), math.sin(sun['el'])
    dark = np.zeros((X.size, SH.size), bool)
    for b in boulders:
        h = b['h'] if np.isfinite(b['h']) else 1e6
        bx, by = b['x'] - X, b['y'] - Y
        along = bx * math.cos(sun['az']) + by * math.sin(sun['az'])
        near = (along > -b['r']) & (along < LEN) & (np.abs(-bx * math.sin(sun['az']) + by * math.cos(sun['az'])) < b['r'] + 0.5)
        if not near.any():
            continue
        for j, hz in enumerate(SH):
            ox, oy, oz = (X - b['x']) / b['r'], (Y - b['y']) / b['r'], hz / h
            ddx, ddy, ddz = cx * LEN / b['r'], cy * LEN / b['r'], cz * LEN / h
            a = ddx ** 2 + ddy ** 2 + ddz ** 2
            bq = 2 * (ox * ddx + oy * ddy + oz * ddz)
            c = ox ** 2 + oy ** 2 + oz ** 2 - 1
            disc = bq ** 2 - 4 * a * c
            s = np.sqrt(np.maximum(disc, 0))
            t1, t2 = (-bq - s) / (2 * a), (-bq + s) / (2 * a)
            dark[:, j] |= near & (disc > 0) & (t1 < 1) & (t2 > 0)
    return dark.sum(1) / SH.size


def detection_prob(in_fov, vis_frac, pd_max, shadow=0.0):
    """Needs >= 25% of the body in view, full rate from 70%; full shadow cuts detection by 80%."""
    pd = pd_max * np.minimum(1, (np.asarray(vis_frac) - 0.25) / 0.45) * (1 - 0.8 * np.asarray(shadow))
    return np.where(np.asarray(in_fov) & (np.asarray(vis_frac) >= 0.25), pd, 0.0)
