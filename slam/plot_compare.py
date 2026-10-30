"""One drive seen from above: true path, VGGT-SLAM (one camera; aligned with scale, as one camera cannot know size)
and stereo ORB-SLAM3 (two cameras; aligned without scale, so a wrong size counts as error).
    python slam/plot_compare.py OUT.png MONO_RUN_DIR STEREO_RUN_DIR [MONO_RUN_DIR STEREO_RUN_DIR ...]"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'slam')
from score_ate import umeyama

out, pairs = sys.argv[1], list(zip(sys.argv[2::2], sys.argv[3::2]))
fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5.4), squeeze=False)
for ax, (mono, stereo) in zip(axes[0], pairs):
    T = np.loadtxt(f'{mono}/poses_truth.txt')[:, 1:4]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(T, axis=0), axis=1))])
    xy = lambda Q: (Q[:, 0], -Q[:, 2])                    # three.js ground plane: x right, -z forward
    ax.plot(*xy(T), color='#8a9099', lw=2.2, label='true path')
    for run, name, scale, col in ((mono, 'VGGT-SLAM, 1 camera', True, '#0b84f3'), (stereo, 'ORB-SLAM3, 2 cameras', False, '#e8590c')):
        E = np.loadtxt(f'{run}/{"poses_vggt.txt" if scale else "poses_orb.txt"}'); ids = E[:, 0].astype(int)
        s, R, t = umeyama(E[:, 1:4], T[ids], scale); A = (s * (R @ E[:, 1:4].T)).T + t
        e = np.sqrt((np.linalg.norm(A - T[ids], axis=1) ** 2).mean())
        ax.plot(*xy(A), color=col, lw=1.4, label=f'{name}: off by {100 * e / cum[-1]:.1f}%')
    ax.plot(*xy(T[:1]), 's', color='k', ms=7)
    ax.annotate('start', (T[0, 0], -T[0, 2]), xytext=(6, -12), textcoords='offset points', fontsize=8)
    ax.set_title(f"Drive {json.load(open(f'{mono}/meta.json'))['seed']} · sun {json.load(open(f'{mono}/meta.json'))['sunElevationDeg']}°", fontsize=11)
    ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.tick_params(labelsize=8); ax.set_xlabel('metres', fontsize=8)
    ax.legend(fontsize=8, loc='best')
fig.suptitle('Rover path seen from above: one camera vs two cameras', fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=120)
print('wrote', out)
