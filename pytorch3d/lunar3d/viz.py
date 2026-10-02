"""Video of a PyTorch3D session: rendered overview + rover camera (+ lander camera) with tracker overlays, a scoreboard,
the People table and a feed. Every image of the moon is rendered by PyTorch3D; matplotlib only lays out and annotates.

    python -m lunar3d.viz --seconds 90 --out lunar_reid.mp4
"""
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import imageio.v2 as imageio
from .session import Session
from .render import Camera
from .phd import ParticlePHD
from matplotlib.patches import Circle
from . import shapes

BG, FG, DIM = '#0a0d14', '#dce3ef', '#8e9bb2'
TRK = ['#ff6b6b', '#4cc9f0', '#4ade80', '#bd8cff']
PAL = ['#F3C300', '#A1CAF1', '#F38400', '#E68FAC', '#00C27C', '#C2B280', '#FF5C6C', '#8DB600', '#4FA3E0', '#F99379',
       '#D97BA4', '#DCD300', '#B98AD4', '#E25822', '#2EC4B6', '#B794F6']
PLAIN = {'Naive': 'forgets anyone it cannot see',
         'Aware + neg. info': 'waits, and searches the blind spot (motion only)',
         'Aware + neg. info + re-ID': '+ recognises people by their suit stripes'}


def hexrgb(c):
    return tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))


class Labels:
    """Display IDs #11, #12 ... per tracker, in order of confirmation (same rule as the web and MATLAB labs)."""

    def __init__(self):
        self.m = {}

    def __call__(self, tid):
        if tid not in self.m:
            self.m[tid] = 11 + len(self.m)
        return self.m[tid]


def stripe_hex(pid, identical):
    c = shapes.UNIFORM_STRIPE if identical else shapes.STRIPES[(pid - 1) % len(shapes.STRIPES)]
    return '#%02x%02x%02x' % tuple(int(255 * v) for v in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='boulders'); ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--seconds', type=float, default=90); ap.add_argument('--out', default='lunar_reid.mp4')
    ap.add_argument('--two-cameras', action='store_true'); ap.add_argument('--identical-suits', action='store_true')
    ap.add_argument('--every', type=int, default=2, help='draw every Nth simulation frame (0.1 s each)')
    ap.add_argument('--phd', action='store_true', help='add the "where could anyone be?" PHD density map')
    a = ap.parse_args()

    S = Session(a.scenario, a.seed, two_cameras=a.two_cameras, identical_suits=a.identical_suits)
    W = S.world; c = W.cam
    over = Camera((-9.0, -30.0, 22.0), (0.0, 19.0, 0.0), math.radians(62), 1280, 600)     # orbit-style overview
    sel = len(S.runs) - 1                                   # draw the most capable tracker
    labels = [Labels() for _ in S.runs]
    hist = [{} for _ in S.runs]; prev = [{} for _ in S.runs]; feed = []; seen_re = 0; arcs = []
    prev_hidden = {}

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    ax_o = fig.add_axes([0.005, 0.40, 0.665, 0.555]); ax_c = fig.add_axes([0.005, 0.01, 0.665, 0.37])
    if a.phd:
        ax_h = fig.add_axes([0.685, 0.505, 0.305, 0.45], facecolor='#05070b')
        ax_p = fig.add_axes([0.68, 0.01, 0.315, 0.48]); ax_p.axis('off'); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1)
        phd = ParticlePHD([r.cam for r in S.rigs], W.boulders, sun=W.sun if W.cfg['shadows'] else None, pd=W.cfg['pd'], clutter=W.cfg['clutter'])
    else:
        ax_p = fig.add_axes([0.68, 0.01, 0.315, 0.945]); ax_p.axis('off'); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); phd = None
    writer = imageio.get_writer(a.out, fps=10 // max(1, a.every // 2) if a.every <= 2 else 5, codec='libx264', quality=8)

    def T(): return W.t

    def push(txt, col):
        feed.insert(0, (f'{T():5.0f}s', txt, col)); del feed[11:]

    nframes = round(a.seconds / W.dt)
    for i in range(nframes):
        S.step()
        if phd is not None:
            phd.step([(rg.cam, rg.last['dets'], None) for rg in S.rigs])
        # ------------------------------------------------ feed bookkeeping (every simulation frame)
        for k, r in enumerate(S.runs):
            M = r['metrics']
            for gid, kid in M.lastMatch.items():
                n = labels[k](kid); h = hist[k].setdefault(gid, [])
                if not h or h[-1] != n:
                    h.append(n)
                b = prev[k].get(gid)
                if k == sel and b is not None and b != kid:
                    push(f'P{gid}  #{labels[k](b)} → #{n}  ID switch', '#ff9aa4')
                prev[k][gid] = kid
        for t in W.targets:
            hid = W.hidden(t['vis'])
            if hid and not prev_hidden.get(t['id'], False):
                push(f"P{t['id']}  {'in shadow' if t['vis']['shadow'] > 0.3 else 'behind rock'}", '#ffd166')
            prev_hidden[t['id']] = hid
        Tsel = S.runs[sel]['tracker']
        for (tt, tid, sim, gap) in Tsel.reidEvents[seen_re:]:
            push(f'#{labels[sel](tid)}  recognised again after {gap:.0f} s (match {sim:.2f})' if gap >= 1 else
                 f'#{labels[sel](tid)}  recognised again after <1 s (match {sim:.2f})', '#7ef0b0')
            arcs.append(dict(t=tt, tid=tid))
        seen_re = len(Tsel.reidEvents)
        if i % a.every:
            continue

        # ------------------------------------------------ overview (rendered) + overlays
        rgb_o, _, _, _ = over.render(S.scene)
        ax_o.clear(); ax_o.imshow(rgb_o, interpolation='bilinear', aspect='auto'); ax_o.axis('off')
        th = np.linspace(0, 2 * np.pi, 40)
        for b in W.boulders:                               # camera blind zones behind tall rocks (on the ground)
            if b['h'] < c['h']:
                continue
            d = math.hypot(b['x'] - c['x'], b['y'] - c['y']); half = math.asin(min(1, b['r'] / d)); cb = math.atan2(b['y'] - c['y'], b['x'] - c['x'])
            ang = np.linspace(cb - half, cb + half, 12); tl = d * math.cos(half)
            P = np.r_[[[c['x'] + tl * math.cos(cb - half), c['y'] + tl * math.sin(cb - half), 0.05]],
                      np.c_[c['x'] + c['range'] * np.cos(ang), c['y'] + c['range'] * np.sin(ang), np.full(12, 0.05)],
                      [[c['x'] + tl * math.cos(cb + half), c['y'] + tl * math.sin(cb + half), 0.05]]]
            u, v, _ = over.project(P)
            ax_o.add_patch(Polygon(np.c_[u, v], closed=True, fc=(0.55, 0.35, 0.95, 0.16), ec=(0.7, 0.5, 1, 0.4), ls='--', lw=1))
        for t in W.targets:                                 # truth labels at the feet
            u, v, _ = over.project(np.array([[t['x'], t['y'], 0.0]]))
            txt = f"P{t['id']}" + (' hidden' if W.hidden(t['vis']) else '')
            ax_o.text(u[0] + 8, v[0] + 4, txt, fontsize=8, color='#0a0d14', weight='bold',
                      bbox=dict(boxstyle='round,pad=0.15', fc=stripe_hex(t['id'], a.identical_suits) if not W.hidden(t['vis']) else '#dfe3ea', ec='none'))
        for tr in Tsel.confirmed():
            n = labels[sel](tr['id']); col = PAL[(n - 11) % len(PAL)]; x, y = tr['x'][0], tr['x'][1]
            if tr['lastSeen'] > 3:                          # hidden: 95% search area on the ground
                Pm = tr['P'][:2, :2]; ev, evec = np.linalg.eigh((Pm + Pm.T) / 2)
                e = evec @ np.diag(np.sqrt(np.maximum(ev, 0) * 5.991)) @ np.vstack([np.cos(th), np.sin(th)])
                e *= np.maximum(1, 0.8 / np.maximum(np.linalg.norm(e, axis=0), 1e-9))
                u, v, _ = over.project(np.c_[x + e[0], y + e[1], np.full(40, 0.05)])
                ax_o.add_patch(Polygon(np.c_[u, v], closed=True, fc=hexrgb(col) + (0.22,), ec=col, ls='--', lw=1.6))
                u, v, _ = over.project(np.array([[x, y, 1.2]]))
                ax_o.text(u[0], v[0], f'#{n}?', color=col, fontsize=10, weight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=col, ls='--'))
            else:
                u, v, _ = over.project(np.c_[x + 0.75 * np.cos(th), y + 0.75 * np.sin(th), np.full(40, 0.08)])
                ax_o.plot(u, v, color=col, lw=2.6)
                u, v, _ = over.project(np.array([[x, y, 2.6]]))
                ax_o.text(u[0], v[0], f'#{n}', color='#0a0d14', fontsize=10, weight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', fc=col, ec='none'))
        arcs[:] = [r for r in arcs if T() - r['t'] < 3]
        for r in arcs:                                      # a re-identification: flash the recovered ID
            tr = next((q for q in Tsel.tracks if q['id'] == r['tid']), None)
            if tr is None:
                continue
            u, v, _ = over.project(np.array([[tr['x'][0], tr['x'][1], 3.4]]))
            ax_o.text(u[0], v[0], f"re-ID  #{labels[sel](r['tid'])}  ✔", color='#7ef0b0', fontsize=11, weight='bold', ha='center',
                      bbox=dict(boxstyle='round,pad=0.25', fc='#0d3320', ec='#7ef0b0'))
        ax_o.text(10, 22, f"{W.cfg['label']}  ·  seed {a.seed}  ·  T+{T():5.1f} s     rendered with PyTorch3D", color=FG, fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='none', alpha=0.8))
        ax_o.text(10, 52, 'P1–P5 real astronaut (stripe)   #11 tracker ID   #11? hidden: best guess   purple = camera blind zone',
                  color=FG, fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='none', alpha=0.8))

        # ------------------------------------------------ rover camera (+ lander camera) with boxes
        ax_c.clear(); ax_c.axis('off')
        imgs = [rg.last['rgb'] for rg in S.rigs]
        canvas = np.concatenate(imgs, 1) if len(imgs) > 1 else imgs[0]
        ax_c.imshow(canvas, interpolation='bilinear', aspect='auto')
        lm = S.runs[sel]['metrics'].lastMatch; live = {q['id'] for q in Tsel.reported()}
        for ri, rg in enumerate(S.rigs):
            off = ri * rg.render_cam.W
            for pid, (n, _sil, box) in rg.last['stats'].items():
                if box is None or n < 6:
                    continue
                kid = lm.get(pid); has = kid is not None and kid in live and not W.hidden(next(t for t in W.targets if t['id'] == pid)['vis'])
                col = PAL[(labels[sel](kid) - 11) % len(PAL)] if has else '#9aa3b2'
                u0, v0, u1, v1 = box
                ax_c.add_patch(Rectangle((off + u0 - 2, v0 - 2), u1 - u0 + 4, v1 - v0 + 4, fill=False, ec=col, lw=1.8, ls='-' if has else '--'))
                ax_c.text(off + (u0 + u1) / 2, v0 - 5, f'#{labels[sel](kid)}' if has else 'no ID', color='#0a0d14' if has else col,
                          fontsize=8, weight='bold', ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.12', fc=col if has else BG, ec='none' if has else col))
            name = 'ROVER CAMERA · 2.2 m mast' if ri == 0 else 'LANDER CAMERA · 6 m'
            ax_c.text(off + 8, 18, name, color='#ffd166', fontsize=9, weight='bold', bbox=dict(boxstyle='round,pad=0.25', fc=BG, ec='none', alpha=0.8))

        # ------------------------------------------------ PHD density map (top-down)
        if phd is not None:
            ax_h.clear(); Dn, ext = phd.density()
            ax_h.imshow(Dn, extent=ext, origin='lower', cmap='inferno', vmin=0, vmax=0.9, interpolation='bilinear', aspect='equal')
            for rg in S.rigs:
                cm = rg.cam; ang = cm['th'] + np.linspace(-cm['fov'] / 2, cm['fov'] / 2, 30)
                ax_h.plot(np.r_[cm['x'], cm['x'] + cm['range'] * np.cos(ang), cm['x']], np.r_[cm['y'], cm['y'] + cm['range'] * np.sin(ang), cm['y']],
                          color='#4cc9f0', lw=0.8, alpha=0.6)
                ax_h.plot(cm['x'], cm['y'], 's', color='#4cc9f0', ms=6)
            for b in W.boulders:
                ax_h.add_patch(Circle((b['x'], b['y']), b['r'], fc='#6b6f78' if b['h'] >= c['h'] else '#3c4048', ec='#9aa3b2', lw=0.6))
            for t in W.targets:
                ax_h.plot(t['x'], t['y'], 'o', ms=7, mfc=stripe_hex(t['id'], a.identical_suits), mec='white', mew=1.2)
                ax_h.text(t['x'] + 1, t['y'] + 1, f"P{t['id']}", color='white', fontsize=8, weight='bold')
            ax_h.set_xlim(ext[0], ext[1]); ax_h.set_ylim(ext[2], ext[3]); ax_h.set_xticks([]); ax_h.set_yticks([])
            for sp in ax_h.spines.values():
                sp.set_color('#2a3142')
            ax_h.set_title(f'WHERE COULD ANYONE BE?  PHD density · expects {phd.expected_count:.1f} people (truth {len(W.targets)})',
                           color=FG, fontsize=10, loc='left')
            ax_h.text(ext[0] + 1, ext[2] + 1, 'bright = someone is probably here · glow behind rocks = hidden but not forgotten',
                      color=DIM, fontsize=8)

        # ------------------------------------------------ side panel
        ax_p.clear(); ax_p.axis('off'); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1)
        y = 0.985
        sc = 0.55 if phd is not None else 1.0         # compress the text panel when the map is shown
        ax_p.text(0, y, 'TRACKERS · same rendered detections', color=DIM, fontsize=10, weight='bold', va='top')
        for k, r in enumerate(S.runs):
            s = r['metrics'].summary(); b = s['occByDuration']
            sk, sn = sum(x['kept'] for x in b[:2]), sum(x['n'] for x in b[:2]); lk, ln = sum(x['kept'] for x in b[2:]), sum(x['n'] for x in b[2:])
            y -= 0.045 / sc ** 0.3; ax_p.text(0, y, ('◉ ' if k == sel else '') + r['tracker'].name, color=TRK[k % len(TRK)], fontsize=12, weight='bold', va='top')
            y -= 0.03; ax_p.text(0.02, y, PLAIN.get(r['tracker'].name, ''), color=DIM, fontsize=9, va='top')
            y -= 0.03; ax_p.text(0.02, y, f'same ID after a short hide {sk}/{sn}   after a long hide (>5 s) {lk}/{ln}   ID switches {s["idsw"]}',
                                 color=FG, fontsize=9.5, va='top')
        y -= 0.06; ax_p.text(0, y, f'PEOPLE · IDs given by {Tsel.name}', color=DIM, fontsize=10, weight='bold', va='top')
        for t in W.targets:
            y -= 0.034
            ax_p.add_patch(Rectangle((0, y - 0.018), 0.018, 0.016, color=stripe_hex(t['id'], a.identical_suits)))
            h = hist[sel].get(t['id'], [])
            now = 'hidden' if W.hidden(t['vis']) else (f"#{labels[sel](lm[t['id']])}" if t['id'] in lm and lm[t['id']] in live else '–')
            ids = ' '.join(f'#{n}' for n in h[-5:]) or '–'
            sw = f'   {len(h) - 1} ✘' if len(h) > 1 else ''
            ax_p.text(0.03, y, f"P{t['id']}   now {now:7s}   IDs: {ids}{sw}", color=FG, fontsize=10, va='top', family='monospace')
        y -= 0.06; ax_p.text(0, y, 'WHAT HAPPENED', color=DIM, fontsize=10, weight='bold', va='top')
        for tt, txt, col in (feed[:6] if phd is not None else feed):
            y -= 0.03 / sc ** 0.6; ax_p.text(0, y, f'{tt}  {txt}', color=col, fontsize=9.5, va='top', family='monospace')

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        if i % 100 == 0:
            print(f'  T+{T():5.1f} s', flush=True)
    writer.close()
    for row in S.table():
        print(row)
    print('video:', a.out)


if __name__ == '__main__':
    main()
