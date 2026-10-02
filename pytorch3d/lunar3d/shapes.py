"""Small mesh builders (numpy): cylinders, spheres, boxes, terrain, rocks, astronauts, rover, lander.
Every builder returns (verts[V,3], faces[F,3], colors[V,3]) in the simulator's world frame (x, y on the ground, z up)."""
import math
import numpy as np

WHITE, BOOT, VISOR, PACK = (0.93, 0.93, 0.94), (0.33, 0.33, 0.35), (0.85, 0.63, 0.12), (0.80, 0.81, 0.83)
# identity stripes: hues spread around the colour wheel and kept away from the gold visor's hue (~0.11), so a colour
# signature can tell people apart (the web lab's gold and orange stripes would be confused with the visor)
STRIPES = [(0.90, 0.16, 0.16), (0.16, 0.40, 0.92), (0.16, 0.72, 0.28), (0.88, 0.20, 0.72), (0.12, 0.74, 0.80),
           (0.52, 0.26, 0.88), (0.60, 0.86, 0.14), (0.95, 0.45, 0.62), (0.30, 0.30, 0.65), (0.70, 0.12, 0.30)]


def merge(parts):
    V, Fc, C, off = [], [], [], 0
    for v, f, c in parts:
        V.append(v); Fc.append(f + off); C.append(c); off += len(v)
    return np.concatenate(V), np.concatenate(Fc), np.concatenate(C)


def cylinder(r, z0, z1, color, n=14, cx=0.0, cy=0.0, caps=True):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    v = np.concatenate([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)], [[cx, cy, z0], [cx, cy, z1]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
        if caps:
            f += [[2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return v, np.array(f), np.tile(color, (len(v), 1))


def sphere(r, center, color, n=10, keep=None, scale=(1, 1, 1)):
    th, ph = np.linspace(0, math.pi, n + 1), np.linspace(0, 2 * math.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij')
    u = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    v = u * r * np.array(scale) + np.array(center)
    f, m = [], 2 * n
    for i in range(n):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            if keep is None or keep(u[a]) or keep(u[d]):
                f += [[a, c, d], [a, d, b]]
    return v, np.array(f), np.tile(color, (len(v), 1))


def box(center, size, color):
    s = np.array(size) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + np.array(center)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f, np.tile(color, (8, 1))


def astronaut(stripe):
    """1.8 m EVA suit, facing +x in its own frame. The stripe colour (waist, arms, legs, helmet) is the true identity,
    as on Apollo/Artemis suits."""
    parts = [cylinder(0.09, 0.10, 0.85, WHITE, cy=0.11), cylinder(0.09, 0.10, 0.85, WHITE, cy=-0.11),
             cylinder(0.10, 0.0, 0.12, BOOT, cy=0.11), cylinder(0.10, 0.0, 0.12, BOOT, cy=-0.11),
             cylinder(0.24, 0.80, 1.48, WHITE),
             cylinder(0.075, 0.95, 1.45, WHITE, cy=0.31), cylinder(0.075, 0.95, 1.45, WHITE, cy=-0.31),
             cylinder(0.249, 1.00, 1.12, stripe, caps=False),
             cylinder(0.08, 1.26, 1.36, stripe, cy=0.31, caps=False), cylinder(0.08, 1.26, 1.36, stripe, cy=-0.31, caps=False),
             cylinder(0.095, 0.48, 0.58, stripe, cy=0.11, caps=False), cylinder(0.095, 0.48, 0.58, stripe, cy=-0.11, caps=False),
             sphere(1.0, (0, 0, 1.47), WHITE, n=8, scale=(0.27, 0.33, 0.11)),
             box((-0.33, 0, 1.18), (0.2, 0.46, 0.6), PACK),
             sphere(0.19, (0, 0, 1.70), WHITE, n=10),
             sphere(0.197, (0, 0, 1.70), VISOR, n=10, keep=lambda u: u[0] > 0.3),
             sphere(0.2, (0, 0, 1.70), stripe, n=10, keep=lambda u: u[2] > 0.82)]
    return merge(parts)


def place(mesh, x, y, heading):
    v, f, c = mesh
    ch, sh = math.cos(heading), math.sin(heading)
    R = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    return v @ R.T + np.array([x, y, 0.0]), f, c


def terrain(xlim=(-60, 60), ylim=(-20, 70), step=1.5):
    xs, ys = np.arange(xlim[0], xlim[1] + step, step), np.arange(ylim[0], ylim[1] + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.035 * (np.sin(0.31 * X + 1.3) + np.sin(0.27 * Y + 0.4) + 0.6 * np.sin(0.71 * X + 0.53 * Y))
    alb = 0.50 + 0.03 * np.sin(0.23 * X + 0.3) * np.sin(0.19 * Y)
    for cx, cy, R, dep in [(-38, 58, 8, 0.9), (22, 63, 10, 1.1), (47, 22, 6, 0.6), (-50, 12, 6, 0.6), (30, -14, 7, 0.7), (52, 52, 9, 1)]:
        d = np.hypot(X - cx, Y - cy) / R
        Z += -dep * np.maximum(0, 1 - d ** 2) + 0.35 * dep * np.exp(-((d - 1) / 0.25) ** 2)
        alb -= 0.06 * np.maximum(0, 1 - d ** 2)
    ny, nx = X.shape
    v = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])
    col = np.c_[alb.ravel(), alb.ravel() * 0.975, alb.ravel() * 0.94]
    return v, f, col


def rock(b, seed):
    """Half-buried boulder: deformed sphere with a faceted look (radius b['r'], height b['h'])."""
    v, f, _ = sphere(1.0, (0, 0, 0), (0, 0, 0), n=9)
    k = 7.3 * b['x'] + 3.1 * b['y'] + seed
    X, Y, Z = v[:, 0], v[:, 1], v[:, 2]
    rn = 1 + 0.12 * np.sin(2.1 * X + 1.3 * Y + 0.9 * Z + k) + 0.07 * np.sin(3.7 * Y - 2.9 * Z + 1.7 * k) + 0.05 * np.sin(5.3 * X + 4.1 * Z - k)
    z = np.maximum(Z * rn * b['h'], -0.3)
    v = np.c_[b['x'] + X * rn * b['r'], b['y'] + Y * rn * b['r'], z]
    sh = 0.30 + 0.03 * math.sin(k) + 0.04 * np.sin(4 * X + 3 * Y + k)
    return v, f, np.c_[sh * 1.05, sh, sh * 0.94]


def rover(cam):
    x0, y0 = cam['x'], cam['y'] - 0.6
    parts = [box((x0, y0, 0.75), (1.5, 2.0, 0.45), (0.82, 0.83, 0.85)), box((x0, y0, 1.02), (1.7, 2.1, 0.06), (0.12, 0.18, 0.36)),
             cylinder(0.06, 1.0, cam['h'] - 0.1, (0.7, 0.7, 0.72), cx=x0, cy=cam['y'] - 0.15),
             box((x0, cam['y'] - 0.2, cam['h']), (0.55, 0.28, 0.24), (0.22, 0.23, 0.26))]
    for sx in (-0.9, 0.9):
        for sy in (-0.75, 0, 0.75):
            parts.append(box((x0 + sx, y0 + sy, 0.32), (0.24, 0.6, 0.6), (0.25, 0.25, 0.27)))
    return merge(parts)


def lander(x, y, h=6.0):
    """A lander with a camera on top, giving a second, higher view of the worksite."""
    gold = (0.80, 0.62, 0.20)
    parts = [box((x, y, 2.4), (3.2, 3.2, 2.4), gold), box((x, y, 4.2), (2.2, 2.2, 1.2), (0.85, 0.85, 0.87)),
             cylinder(0.08, 4.8, h, (0.7, 0.7, 0.72), cx=x, cy=y), box((x, y, h), (0.5, 0.5, 0.3), (0.22, 0.23, 0.26))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(cylinder(0.09, 0.0, 1.6, (0.75, 0.75, 0.77), cx=x + 2.0 * sx, cy=y + 2.0 * sy))
    return merge(parts)


UNIFORM_STRIPE = (0.90, 0.16, 0.16)   # 'identical suits' experiment: everyone wears the same red stripe
