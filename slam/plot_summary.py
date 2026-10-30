"""Error vs sun height, one dot per drive: VGGT-SLAM (1 camera) vs ORB-SLAM3 (2 cameras). Log scale: the two differ ~50x.
    python slam/plot_summary.py OUT.png MONO_ROOT STEREO_ROOT"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'slam')
from score_ate import score

out, mono, stereo = sys.argv[1:4]
suns, seeds = (30, 7, 2), range(1, 6)
fig, ax = plt.subplots(figsize=(7, 4.8))
for k, (root, est, scale, col, name) in enumerate(((mono, 'poses_vggt.txt', True, '#0b84f3', 'VGGT-SLAM, 1 camera'),
                                                    (stereo, 'poses_orb.txt', False, '#e8590c', 'ORB-SLAM3, 2 cameras'))):
    for j, sun in enumerate(suns):
        pct = [100 * o['ate_rmse_m'] / o['driven_m'] for o in (score(f'{root}/seed{s}_sun{sun}', est, scale) for s in seeds)]
        x = j + (k - 0.5) * 0.28
        ax.scatter(np.full(5, x) + np.linspace(-0.05, 0.05, 5), pct, s=40, color=col, edgecolor='white', lw=1, zorder=3,
                   label=name if j == 0 else None)
        m = np.median(pct); ax.plot([x - 0.11, x + 0.11], [m, m], color=col, lw=2.5, zorder=2)
        ax.annotate(f'{m:.1f}%', (x + 0.12, m), va='center', fontsize=9, color='#333')
ax.set_yscale('log'); ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(['0.1%', '1%', '10%'])
ax.set_xticks(range(3)); ax.set_xticklabels([f'sun {s}° above horizon' for s in suns]); ax.set_xlim(-0.6, 2.7)
ax.set_ylabel('how far off the path is\n(% of distance driven)')
ax.grid(axis='y', alpha=0.3); ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_title('Lunar rover SLAM in simulation: one camera vs two\n(dots = 5 drives, bar = middle drive; lower is better)', fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=130); print('wrote', out)
