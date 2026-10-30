"""Closed-loop missions: where each brain really drove vs where it thought it was, per sun angle.
    python slam/plot_closed_loop.py [seed]      -> slam/runs/closed/closed_loop_seed<seed>.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt

root = Path(__file__).parent / "runs" / "closed"
seed = sys.argv[1] if len(sys.argv) > 1 else "1"
brains = [("truth", "steers by the truth"), ("wheels", "wheels only"), ("orb", "ORB-SLAM3 stereo, live")]
suns = [30, 7, 2]


def rows(p):
    return [l.split() for l in p.read_text().splitlines() if not l.startswith("#")]


fig, axes = plt.subplots(len(suns), len(brains), figsize=(15, 4.6 * len(suns)), squeeze=False)
for i, sun in enumerate(suns):
    for j, (brain, label) in enumerate(brains):
        ax, run = axes[i][j], root / f"{brain}_seed{seed}_sun{sun}"
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        if not (run / "result.json").exists():
            ax.set_title(f"{label}, sun {sun}°: not run")
            continue
        r = json.loads((run / "result.json").read_text())
        t, b = rows(run / "poses_truth_rover.txt"), rows(run / "poses_belief_rover.txt")
        ax.plot([float(x[1]) for x in t], [float(x[2]) for x in t], color="#111", lw=2, label="where it really drove")
        if brain != "truth":
            ax.plot([float(x[1]) for x in b], [float(x[2]) for x in b], color="#d62728" if brain == "wheels" else "#1f77b4",
                    lw=1.4, ls="--", label="where it thought it was")
        rocks = next(root.glob(f"*_seed{seed}_sun*/boulders.json"), None)
        rocks = json.loads(rocks.read_text()) if rocks else []
        for o in rocks:
            ax.add_patch(plt.Circle((o["x"], o["y"]), o["r"], color="#8a8a8a", alpha=0.6))
        # closest the rover's body (half-width 0.8 m) came to a rock's edge
        gap = min((((float(x[1]) - o["x"]) ** 2 + (float(x[2]) - o["y"]) ** 2) ** 0.5 - o["r"] - 0.8 for x in t for o in rocks), default=float("nan"))
        m = json.loads((run / "mission.json").read_text())
        for k, p in enumerate(m):
            ax.add_patch(plt.Circle((p["x"], p["y"]), 3, fill=False, ls=":", color="green"))
            ax.text(p["x"] + 1.2, p["y"] + 1.2, "home" if k == len(m) - 1 else str(k + 1), color="green", fontsize=10)
        done = "mission done" if r["missionDone"] else "DID NOT FINISH"
        extra = f", tracked {r['trackedPct']}%" if "trackedPct" in r else ""
        ax.set_title(f"{label}, sun {sun}°\n{done}; ends {r['homeMissM']} m from home; {r['seconds']:.0f} s\npose error max {r['poseErrMaxM']} m; closest to a rock {gap:.2f} m{extra}", fontsize=10)
        if i == 0 and j > 0:
            ax.legend(fontsize=8, loc="lower left")
fig.suptitle(f"Closed loop: the rover steers by its own pose estimate (seed {seed}, 5% wheel slip; green rings = stops, 3 m arrival radius)", fontsize=13)
plt.tight_layout()
out = root / f"closed_loop_seed{seed}.png"
plt.savefig(out, dpi=110)
print(out)
