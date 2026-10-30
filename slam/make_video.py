"""Video of VGGT-SLAM on one recorded drive: left, the rover camera (what SLAM is given); right, from above, the map
SLAM built, the true path and SLAM's path, drawn as the drive goes on.
    python slam/make_video.py RUN_DIR OUT.mp4
RUN_DIR: frames/, poses_truth.txt, poses_vggt.txt, meta.json, map_small.npy (SLAM point cloud, voxel-downsampled,
columns x y z [r g b], in SLAM's own frame). The SLAM path is its final (optimised) estimate, revealed keyframe by
keyframe; map points appear with the keyframe nearest to them.
"""
import json
import sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'slam')
from score_ate import umeyama

run, out = sys.argv[1], sys.argv[2]
meta = json.load(open(f'{run}/meta.json'))
T = np.loadtxt(f'{run}/poses_truth.txt')[:, 1:4]
E = np.loadtxt(f'{run}/poses_vggt.txt'); ids = E[:, 0].astype(int)
s, R, t = umeyama(E[:, 1:4], T[ids])
A = (s * (R @ E[:, 1:4].T)).T + t                       # SLAM keyframe positions, in the truth frame
M = np.load(f'{run}/map_small.npy')
P = (s * (R @ M[:, :3].T)).T + t                         # map points, same alignment
col = np.clip(M[:, 3:6], 0, 1) if M.shape[1] >= 6 else np.full((len(M), 3), 0.5)

# top-down: three.js ground plane, x right and -z forward
gx, gy = lambda Q: Q[:, 0], lambda Q: -Q[:, 2]
path_xy = np.c_[gx(T), gy(T)]
lo, hi = path_xy.min(0) - 12, path_xy.max(0) + 12
keep = (gx(P) > lo[0]) & (gx(P) < hi[0]) & (gy(P) > lo[1]) & (gy(P) < hi[1]) & (P[:, 1] < 6)   # drop far/sky outliers
P, col = P[keep], col[keep]
near = np.argmin(((P[:, None, [0, 2]] - A[None, :, [0, 2]]) ** 2).sum(-1), axis=1)   # nearest keyframe per point
order = np.argsort(near, kind='stable'); P, col, near = P[order], col[order], near[order]
cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(T, axis=0), axis=1))])

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='#0d1117')
fig.suptitle(f"VGGT-SLAM 2.0 on a simulated lunar rover drive · sun {meta['sunElevationDeg']}° above the horizon",
             color='white', fontsize=14, y=0.97)
axc = fig.add_axes([0.02, 0.22, 0.52, 0.62]); axc.axis('off')
axm = fig.add_axes([0.57, 0.12, 0.41, 0.76], facecolor='#161b22')
axc.set_title('rover camera: the only input SLAM gets', color='#c9d1d9', fontsize=11)
axm.set_title('seen from above', color='#c9d1d9', fontsize=11)
for sp in axm.spines.values():
    sp.set_color('#30363d')
axm.tick_params(colors='#8b949e', labelsize=8); axm.set_xlim(lo[0], hi[0]); axm.set_ylim(lo[1], hi[1]); axm.set_aspect('equal')
axm.set_xlabel('metres', color='#8b949e', fontsize=8)

img = axc.imshow(imageio.imread(f'{run}/frames/000000.png'))
badge = axc.text(0.02, 0.95, '', transform=axc.transAxes, color='#0d1117', fontsize=11, weight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#58a6ff', ec='none'))
mapsc = axm.scatter(np.empty(0), np.empty(0), s=1.2, color='#9aa0a6', alpha=0.55, linewidths=0)
truth, = axm.plot([], [], color='#8b949e', lw=2, label='true path')
slam, = axm.plot([], [], '-o', color='#58a6ff', ms=3, lw=1.4, label='SLAM estimate')
errl, = axm.plot([], [], color='#ff7b72', lw=1.5, label='error now')
rov, = axm.plot([], [], marker=(3, 0, 0), color='white', ms=11, lw=0)
axm.legend(loc='upper right', fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
txt = fig.text(0.02, 0.12, '', color='#c9d1d9', fontsize=13, family='monospace')
fig.text(0.02, 0.04, 'Grey dots: the 3-D map SLAM builds from the images. Simulation with exact ground truth; '
         'SLAM path aligned to the truth (one camera cannot know absolute size).', color='#8b949e', fontsize=9)

writer = imageio.get_writer(out, fps=15, codec='libx264', quality=7, macro_block_size=8)
for i in range(len(T)):
    k = np.searchsorted(ids, i, side='right')            # keyframes so far
    img.set_data(imageio.imread(f'{run}/frames/{i:06d}.png'))
    badge.set_text('KEYFRAME → VGGT' if i in set(ids) else ''); badge.set_visible(i in set(ids))
    n = np.searchsorted(near, k - 1, side='right') if k else 0
    mapsc.set_offsets(np.c_[gx(P[:n]), gy(P[:n])]); mapsc.set_facecolors(np.c_[col[:n] * 0.6 + 0.35, np.full(n, 1.0)] if n else np.empty((0, 4)))
    truth.set_data(gx(T[:i + 1]), gy(T[:i + 1]))
    slam.set_data(gx(A[:k]), gy(A[:k]))
    hdg = np.degrees(np.arctan2(gy(T[i:i + 3])[-1] - gy(T[i:i + 1])[0], gx(T[i:i + 3])[-1] - gx(T[i:i + 1])[0] + 1e-9))
    rov.set_data([gx(T[i:i + 1])[0]], [gy(T[i:i + 1])[0]]); rov.set_marker((3, 0, hdg - 90))
    if k:
        j = ids[k - 1]; errl.set_data([gx(A[k - 1:k])[0], gx(T[j:j + 1])[0]], [gy(A[k - 1:k])[0], gy(T[j:j + 1])[0]])
        e = np.linalg.norm(A[k - 1] - T[j])
    else:
        e = 0.0
    txt.set_text(f'time {i / meta["fps"]:5.1f} s   driven {cum[i]:5.1f} m   keyframes {k:3d}   error now {e:4.1f} m')
    fig.canvas.draw()
    writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
writer.close()
print('wrote', out)
