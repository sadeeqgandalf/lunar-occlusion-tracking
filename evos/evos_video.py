"""Watch an EVOS recording: what the event camera sees (left) next to the motion-capture view of the test bed (right).

Usage: uv run --with aedat --with numpy --with pandas --with opencv-python-headless --with "imageio[ffmpeg]" python evos_video.py CIRC-TR-NOM [CC-T-NOM ...]
       ... python evos_video.py --lightings CIRC-TR      # the same manoeuvre in nominal, dark and sun-glare light, side by side
Each video frame covers 0.1 s of recording and plays at 30 frames a second, so the video runs at 3x real speed.
"""
import glob, sys
import aedat, cv2, imageio, numpy as np, pandas as pd

K = np.loadtxt("calib-intrinsics.csv", delimiter=",", encoding="utf-8-sig"); FX, CX = K[0, 0], K[0, 2]
STEP = 100_000                    # microseconds of recording per video frame
F = cv2.FONT_HERSHEY_SIMPLEX
LIGHT = {"NOM": "nominal light", "DARK": "low light, 0.6 lux", "SG": "sun glare, 452 lux"}


def load(exp):
    ev = np.concatenate([p["events"] for p in aedat.Decoder(glob.glob(f"{exp}/*.aedat4")[0]) if "events" in p])
    h = np.zeros((480, 640), int); np.add.at(h, (ev["y"], ev["x"]), 1)
    hot = h > np.percentile(h[h > 0], 99.5)                   # faulty pixels that fire constantly
    ev = ev[~hot[ev["y"], ev["x"]]]
    gt = pd.read_csv(glob.glob(f"{exp}/*.csv")[0]).drop_duplicates("system_time_us").reset_index(drop=True)
    return ev, gt


class EventView:
    """Events as a fading picture: brighter-than-before pixels warm, darker-than-before pixels cool. Old events fade
    over about a third of a second, so a target that stops moving visibly fades away."""
    def __init__(self):
        self.on, self.off = np.zeros((480, 640), np.float32), np.zeros((480, 640), np.float32)

    def step(self, e):
        self.on *= 0.72; self.off *= 0.72
        np.add.at(self.on, (e["y"][e["p"]], e["x"][e["p"]]), 1.0); np.add.at(self.off, (e["y"][~e["p"]], e["x"][~e["p"]]), 1.0)
        a, b = np.clip(self.on / 2.5, 0, 1), np.clip(self.off / 2.5, 0, 1)
        return (np.stack([a * 255 + b * 60, a * 150 + b * 150, a * 60 + b * 255], -1)).clip(0, 255).astype(np.uint8)


def at(gt, t):
    """Chaser and target (x, y in mm, heading in rad) at time t, interpolated from the motion capture."""
    g = lambda c: np.interp(t, gt.system_time_us, gt[c])
    return (g("red_x"), g("red_y"), g("red_angle")), (g("black_x"), g("black_y"), np.interp(t, gt.system_time_us, np.unwrap(gt.black_angle)))


def map_view(gt, t, trail):
    """The 3.5 m x 2.4 m table from above."""
    img = np.full((480, 640, 3), (18, 20, 26), np.uint8); s = 165.0; ox, oy = 30, 440
    P = lambda x, y: (int(ox + x / 1000 * s), int(oy - y / 1000 * s))
    cv2.rectangle(img, P(0, 0), P(3500, 2400), (60, 66, 78), 1)
    (rx, ry, ra), (bx, by, ba) = at(gt, t)
    half = np.arctan(320 / FX)                                 # half of the camera's horizontal field of view
    wedge = np.array([P(rx, ry)] + [P(rx + 3300 * np.cos(ra + a), ry + 3300 * np.sin(ra + a)) for a in np.linspace(-half, half, 12)], np.int32)
    ov = img.copy(); cv2.fillPoly(ov, [wedge], (70, 90, 120)); img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    trail.append(P(bx, by))
    for k in range(max(1, len(trail) - 250), len(trail)):
        cv2.line(img, trail[k - 1], trail[k], (150, 120, 60), 1, cv2.LINE_AA)
    box = lambda x, y, a, w, col, th: cv2.polylines(img, [np.array([P(x + w * (cx_ * np.cos(a) - sy_ * np.sin(a)), y + w * (cx_ * np.sin(a) + sy_ * np.cos(a))) for cx_, sy_ in ((1, 1), (1, -1), (-1, -1), (-1, 1))], np.int32)], True, col, th, cv2.LINE_AA)
    box(rx, ry, ra, 150, (90, 170, 255), 2); box(bx, by, ba, 150, (255, 200, 80), 2)
    cv2.line(img, P(bx, by), P(bx + 260 * np.cos(ba), by + 260 * np.sin(ba)), (255, 200, 80), 2, cv2.LINE_AA)   # which way the target faces
    cv2.putText(img, "chaser (camera)", (P(rx, ry)[0] - 150, P(rx, ry)[1] - 34), F, 0.5, (90, 170, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "target", (P(bx, by)[0] - 26, P(bx, by)[1] - 36), F, 0.5, (255, 200, 80), 1, cv2.LINE_AA)
    rng = np.hypot(bx - rx, by - ry) / 1000
    cv2.putText(img, f"motion capture: target is {rng:.2f} m away", (16, 26), F, 0.55, (220, 225, 235), 1, cv2.LINE_AA)
    cv2.putText(img, "shaded wedge = what the camera can see", (16, 48), F, 0.45, (150, 160, 175), 1, cv2.LINE_AA)
    return img, (rx, ry, ra), (bx, by, ba)


def one(exp):
    ev, gt = load(exp); view = EventView(); trail = []
    t0, t1 = int(gt.system_time_us.iloc[0]), int(gt.system_time_us.iloc[-1])
    out = f"{exp}.mp4"; w = imageio.get_writer(out, fps=30, quality=7, macro_block_size=2)
    for t in range(t0, t1, STEP):
        a, b = np.searchsorted(ev["t"], [t, t + STEP]); cam = view.step(ev[a:b])
        top, (rx, ry, ra), (bx, by, ba) = map_view(gt, t, trail)
        bearing = np.arctan2(by - ry, bx - rx) - ra; bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        u = int(CX - FX * np.tan(bearing))                    # where the motion capture says the target is (camera assumed to look straight ahead)
        if 0 <= u < 640:
            cv2.line(cam, (u, 0), (u, 14), (255, 200, 80), 2); cv2.line(cam, (u, 465), (u, 479), (255, 200, 80), 2)
        cv2.rectangle(cam, (0, 0), (640, 0), (0, 0, 0), -1)
        cv2.putText(cam, f"event camera  |  {exp}  |  t = {(t - t0) / 1e6:5.1f} s  |  {(b - a) * 10 / 1000:5.0f}k events/s", (12, 40), F, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(cam, "yellow ticks: where the motion capture says the target is", (12, 458), F, 0.45, (255, 200, 80), 1, cv2.LINE_AA)
        w.append_data(np.hstack([cam, top]))
    w.close(); print("wrote", out)


def lightings(man):
    runs = [(k, *load(f"{man}-{k}"), EventView()) for k in LIGHT]
    n = min(int((g.system_time_us.iloc[-1] - g.system_time_us.iloc[0]) // STEP) for _, _, g, _ in runs)
    out = f"{man}-lightings.mp4"; w = imageio.get_writer(out, fps=30, quality=7, macro_block_size=2)
    for i in range(n):
        tiles = []
        for k, ev, gt, view in runs:
            t = int(gt.system_time_us.iloc[0]) + i * STEP; a, b = np.searchsorted(ev["t"], [t, t + STEP]); cam = view.step(ev[a:b])
            cv2.putText(cam, f"{LIGHT[k]}  |  {(b - a) * 10 / 1000:4.0f}k events/s", (12, 30), F, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(cam)
        frame = np.hstack(tiles); cv2.putText(frame, f"{man}: same manoeuvre, three recordings (not the same instant)   t = {i * STEP / 1e6:5.1f} s", (12, 468), F, 0.55, (200, 205, 215), 1, cv2.LINE_AA)
        w.append_data(cv2.resize(frame, (1440, 360), interpolation=cv2.INTER_AREA))
    w.close(); print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "--lightings":
        lightings(sys.argv[2])
    else:
        for exp in sys.argv[1:]:
            one(exp)
