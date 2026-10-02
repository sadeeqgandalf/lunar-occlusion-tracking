"""Pool experiments/results.jsonl into a Markdown table with 95% Wilson intervals (writes experiments/results.md)."""
import json, math, os, sys
from collections import defaultdict

path = os.path.join(os.path.dirname(__file__), 'results.jsonl')
rows = [json.loads(l) for l in open(path)]


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


groups = defaultdict(list)
for r in rows:
    groups[(r['scenario'], r['identical'], r['cameras'], r['tracker'])].append(r)
out = ['# PyTorch3D results: re-identification on rendered detections', '',
       'Every detection comes from a PyTorch3D render (visible pixels, depth). All trackers in a row group see the same detections.',
       '"Same ID after a hide": of people who were tracked, hid for >= 1 s and came back, the share that came back with the same ID.', '']
order = ['Naive', 'Aware + neg. info', 'Aware + neg. info + re-ID']
for key in sorted({k[:3] for k in groups}):
    scen, ident, cams = key
    seeds = sorted({r['seed'] for k, v in groups.items() if k[:3] == key for r in v})
    secs = rows[0]['seconds']
    out += [f"## {scen} · {'identical suits' if ident else 'distinct suit stripes'} · {cams} camera{'s' if cams > 1 else ''} · seeds {seeds} × {secs:.0f} s", '',
            '| Tracker | Same ID, hide 1–5 s | Same ID, hide > 5 s | ID switches per run | IDF1 |', '|---|---|---|---|---|']
    for name in order:
        g = groups.get((scen, ident, cams, name))
        if not g:
            continue
        sk, sn = sum(r['short_kept'] for r in g), sum(r['short_n'] for r in g)
        lk, ln = sum(r['long_kept'] for r in g), sum(r['long_n'] for r in g)
        a, b = wilson(sk, sn); c, d = wilson(lk, ln)
        out.append(f"| {name} | {100*sk/max(sn,1):.0f}% ({100*a:.0f}–{100*b:.0f}), {sk}/{sn} | {100*lk/max(ln,1):.0f}% ({100*c:.0f}–{100*d:.0f}), {lk}/{ln} | "
                   f"{sum(r['idsw'] for r in g)/len(g):.1f} | {sum(r['idf1'] for r in g)/len(g):.3f} |")
    out.append('')
open(os.path.join(os.path.dirname(__file__), 'results.md'), 'w').write('\n'.join(out))
print('\n'.join(out))
