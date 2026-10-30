"""Top-down plot of the true camera path and the aligned SLAM path (ATE alignment), one panel per run, labelled
for non-specialists. Marks two things that explain most failures: tight full circles (mostly rotation: one camera
gets no depth) and long stretches with no keyframes.
    python slam/plot_paths.py OUT.png RUN_DIR [RUN_DIR ...]"""
import json
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'slam')
from score_ate import umeyama

GAP_M = 8.0          # keyframes further apart than this (true distance) are marked
LOOP_DEG = 330.0     # turning at least this much ...
LOOP_M = 30.0        # ... within this much driving = a tight full circle (radius under ~5 m)

out, runs = sys.argv[1], sys.argv[2:]
fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.8), squeeze=False)
for ax, run in zip(axes[0], runs):
    T = np.loadtxt(f'{run}/poses_truth.txt')[:, 1:4]
    E = np.loadtxt(f'{run}/poses_vggt.txt'); ids = E[:, 0].astype(int); P = E[:, 1:4]
    s, R, t = umeyama(P, T[ids]); A = (s * (R @ P.T)).T + t
    X, Y, AX, AY = T[:, 0], -T[:, 2], A[:, 0], -A[:, 2]  # three.js ground plane: x right, -z forward
    seg = np.linalg.norm(np.diff(T, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])

    # tight full circles: heading of travel turns >= LOOP_DEG within LOOP_M of driving
    mv = seg > 0.02
    hd = np.unwrap(np.arctan2(np.diff(Y)[mv], np.diff(X)[mv])); hc = cum[1:][mv]
    loops, j = [], 0
    for i in range(len(hd)):
        while hc[i] - hc[j] > LOOP_M:
            j += 1
        if abs(hd[i] - hd[j]) >= np.radians(LOOP_DEG) and not (loops and hc[j] <= loops[-1][1]):
            loops.append((hc[j], hc[i]))
    for a, b in loops:
        k = (cum >= a) & (cum <= b)
        ax.plot(X[k], Y[k], color='#f5a524', lw=7, alpha=0.35, solid_capstyle='round', zorder=1)
    ax.plot(X, Y, color='#8a9099', lw=1.8, zorder=2)
    ax.plot(AX, AY, '-o', color='#0b84f3', ms=3.5, lw=1.3, zorder=4)
    for k, i in enumerate(ids):
        ax.plot([X[i], AX[k]], [Y[i], AY[k]], color='#e5484d', lw=0.8, zorder=3)
    ax.plot(X[0], Y[0], 's', color='k', ms=7, zorder=5)
    ax.annotate('start', (X[0], Y[0]), xytext=(6, -12), textcoords='offset points', fontsize=8)

    meta = json.load(open(f'{run}/meta.json'))
    err = np.linalg.norm(A - T[ids], axis=1); rmse = np.sqrt((err ** 2).mean())
    name = re.sub(r'.*/', '', run.rstrip('/'))
    ax.set_title(f"Drive {meta['seed']} · sun {meta['sunElevationDeg']}° · error {rmse:.1f} m", fontsize=10)
    ax.set_xlabel('metres', fontsize=8); ax.set_ylabel('metres', fontsize=8)
    ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)

handles = [plt.Line2D([], [], color='#8a9099', lw=1.8, label='true path'),
           plt.Line2D([], [], color='#0b84f3', marker='o', ms=4, lw=1.3, label='SLAM estimate'),
           plt.Line2D([], [], color='#e5484d', lw=1.2, label='error'),
           plt.Line2D([], [], color='#f5a524', lw=7, alpha=0.35, label='rover drove in circles')]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, frameon=False)
fig.suptitle('Rover path seen from above: truth vs VGGT-SLAM', fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(out, dpi=120)
print('wrote', out)
