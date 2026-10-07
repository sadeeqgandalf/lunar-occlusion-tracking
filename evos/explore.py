"""First look at EVOS: what the event camera sees in three lightings, and whether it agrees with the motion capture.
Usage: uv run --with aedat --with numpy --with pandas --with matplotlib python explore.py   (writes evos_first_look.png)
"""
import glob
import aedat, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

K = np.loadtxt("calib-intrinsics.csv", delimiter=",", encoding="utf-8-sig"); fx, cx = K[0, 0], K[0, 2]
LIGHT = {"NOM": "nominal lab light", "DARK": "low light (0.6 lux)", "SG": "sun glare (452 lux, one hard source)"}


def load(exp):
    ev = np.concatenate([p["events"] for p in aedat.Decoder(glob.glob(f"{exp}/*.aedat4")[0]) if "events" in p])
    gt = pd.read_csv(glob.glob(f"{exp}/*.csv")[0]).drop_duplicates("system_time_us")
    return ev, gt


def frame(ev, t, dt=30_000):
    """Events in [t, t+dt) as a picture: ON red, OFF blue."""
    a, b = np.searchsorted(ev["t"], [t, t + dt]); e = ev[a:b]
    img = np.zeros((480, 640, 3), np.uint8)
    img[e["y"][e["p"]], e["x"][e["p"]]] = (255, 80, 60)
    img[e["y"][~e["p"]], e["x"][~e["p"]]] = (70, 150, 255)
    return img, b - a


fig, ax = plt.subplots(2, 3, figsize=(17, 9.6))
rates = {}
for j, (key, name) in enumerate(LIGHT.items()):
    ev, gt = load(f"CIRC-TR-{key}")
    t_mid = int(gt.system_time_us.iloc[len(gt) // 3])
    img, n = frame(ev, t_mid)
    ax[0, j].imshow(img); ax[0, j].set_title(f"{name}\n{n:,} events in 30 ms", fontsize=11); ax[0, j].axis("off")
    tb = np.arange(ev["t"][0], ev["t"][-1], 1_000_000)
    rates[name] = (np.diff(np.searchsorted(ev["t"], tb)) / 1e3, len(ev), (ev["t"][-1] - ev["t"][0]) / 1e6)
    if key == "NOM":
        nom = (ev, gt)

ev, gt = nom
a = ax[1, 0]                                              # the test bed from above, from the motion capture
a.plot(gt.black_x / 1000, gt.black_y / 1000, color="#b4691a", lw=1.5, label="target path")
a.plot(gt.red_x.iloc[0] / 1000, gt.red_y.iloc[0] / 1000, "s", color="#1f5f8b", ms=10, label="chaser (camera), not moving")
h = gt.red_angle.iloc[0]; a.arrow(gt.red_x.iloc[0] / 1000, gt.red_y.iloc[0] / 1000, 0.35 * np.cos(h), 0.35 * np.sin(h), color="#1f5f8b", head_width=0.05)
a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.set_title("Motion capture: the test bed from above"); a.legend(frameon=False, fontsize=9)

a = ax[1, 1]
for (name, (r, n, dur)), c in zip(rates.items(), ("#1f5f8b", "#47515e", "#b4691a")):
    a.plot(np.arange(len(r)), r, color=c, lw=1, label=f"{name}: {n / 1e6:.1f} M events in {dur:.0f} s")
a.set_xlabel("time in recording (s)"); a.set_ylabel("thousand events per second"); a.set_title("How busy the sensor is"); a.legend(frameon=False, fontsize=9)

# does the camera agree with the motion capture? where the target should be in the image (from mocap bearing) vs where the events are
dx, dy = gt.black_x - gt.red_x, gt.black_y - gt.red_y
bearing = np.arctan2(dy, dx) - gt.red_angle
bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
u_pred = cx - fx * np.tan(bearing)                        # camera looks along the chaser's heading; image x grows to the right
tt = gt.system_time_us.values
u_meas = np.array([np.median(ev["x"][slice(*np.searchsorted(ev["t"], [t - 50_000, t + 50_000]))]) if np.diff(np.searchsorted(ev["t"], [t - 50_000, t + 50_000]))[0] > 200 else np.nan for t in tt])
ok = ~np.isnan(u_meas)
a = ax[1, 2]; s = (tt - tt[0]) / 1e6
a.plot(s, u_pred, color="#b4691a", lw=1.6, label="predicted from motion capture")
a.plot(s[ok], u_meas[ok], ".", color="#1f5f8b", ms=3, label="median column of the events")
r = np.corrcoef(u_pred[ok], u_meas[ok])[0, 1]; off = np.median(u_meas[ok] - u_pred[ok]); res = np.median(np.abs(u_meas[ok] - u_pred[ok] - off))
a.set_xlabel("time (s)"); a.set_ylabel("image column (pixels)"); a.set_title(f"Camera vs motion capture (nominal light)\ncorrelation {r:.3f}, offset {off:.0f} px, median error after offset {res:.0f} px", fontsize=11); a.legend(frameon=False, fontsize=9)
for a in ax[1]:
    a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.2)
fig.suptitle("EVOS, experiment CIRC-TR (target circles and spins): event camera on a still chaser", fontsize=13, y=0.995)
fig.tight_layout(); fig.savefig("evos_first_look.png", dpi=95)
print("corr", round(r, 3), "offset px", round(off, 1), "median abs err px", round(res, 1), "| range m", round(np.hypot(dx, dy).min() / 1000, 2), "-", round(np.hypot(dx, dy).max() / 1000, 2))
for name, (r_, n, dur) in rates.items(): print(name, n, round(dur), "s", round(n / dur), "ev/s")
