"""Fit how an event camera's signal depends on the target's apparent motion, from an EVOS recording.

signal(t) = (events/s - background) / (95th percentile - background), clipped to 0..1
model     = 1 - exp(-(omega + k * loom) / s0)
  omega = how fast the target moves across the view [rad/s]; loom = |range rate| / range [1/s] (growing or shrinking)
Usage: uv run --with aedat --with numpy --with pandas python fit_motion.py CC-T-NOM
"""
import glob, sys
import aedat, numpy as np, pandas as pd

exp = sys.argv[1] if len(sys.argv) > 1 else "CC-T-NOM"
ev = np.concatenate([p["events"] for p in aedat.Decoder(glob.glob(f"{exp}/*.aedat4")[0]) if "events" in p])
h = np.zeros((480, 640), int); np.add.at(h, (ev["y"], ev["x"]), 1)
ev = ev[~(h > np.percentile(h[h > 0], 99.5))[ev["y"], ev["x"]]]                       # drop hot pixels
gt = pd.read_csv(glob.glob(f"{exp}/*.csv")[0]).drop_duplicates("system_time_us").reset_index(drop=True)
t = gt.system_time_us.values.astype(np.int64); s = (t - t[0]) / 1e6
sm = lambda x: pd.Series(x).rolling(5, center=True, min_periods=1).mean().values
n = sm(np.array([np.diff(np.searchsorted(ev["t"], [a - 50_000, a + 50_000]))[0] for a in t]) / 0.1)
dx, dy = gt.black_x - gt.red_x, gt.black_y - gt.red_y; rng = np.hypot(dx, dy).values
omega = sm(np.abs(np.gradient(np.unwrap(np.arctan2(dy, dx)), s)))
loom = sm(np.abs(np.gradient(rng, s)) / rng)
bg, top = np.percentile(n, 2), np.percentile(n, 95)
y = np.clip((n - bg) / (top - bg), 0, 1)
best = min(((np.mean((1 - np.exp(-(omega + k * loom) / s0) - y) ** 2), k, s0) for k in np.linspace(0, 1.5, 31) for s0 in np.geomspace(5e-4, 5e-2, 60)))
mse, k, s0 = best
pred = 1 - np.exp(-(omega + k * loom) / s0)
print(f"{exp}: k = {k:.2f}, s0 = {s0:.4f} rad/s ({np.degrees(s0):.2f} deg/s), rms error {np.sqrt(mse):.3f}, corr {np.corrcoef(pred, y)[0, 1]:.3f}, n = {len(y)}")
print(f"  background {bg:.0f} ev/s, 95th pct {top:.0f} ev/s | omega range {omega.min():.4f}-{omega.max():.4f} rad/s | loom range {loom.min():.4f}-{loom.max():.4f} /s")
for a in (0.0, 0.5, 1, 2, 5, 10): print(f"  apparent motion {a * s0:.4f} rad/s ({np.degrees(a * s0):.2f} deg/s) -> signal {1 - np.exp(-a):.2f}")
