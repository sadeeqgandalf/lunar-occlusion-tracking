"""One seed of the re-ID experiment on PyTorch3D-rendered detections; writes a JSON line of results.
    python experiments/run_reid.py --scenario boulders --seed 3 --seconds 300 [--identical-suits] [--two-cameras]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lunar3d.session import Session

ap = argparse.ArgumentParser()
ap.add_argument('--scenario', default='boulders'); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--seconds', type=float, default=300); ap.add_argument('--identical-suits', action='store_true')
ap.add_argument('--two-cameras', action='store_true'); ap.add_argument('--out', default='experiments/results.jsonl')
a = ap.parse_args()
t0 = time.time()
S = Session(a.scenario, a.seed, two_cameras=a.two_cameras, identical_suits=a.identical_suits).run(a.seconds)
for r in S.runs:
    s = r['metrics'].summary(); b = s['occByDuration']
    row = dict(scenario=a.scenario, seed=a.seed, seconds=a.seconds, identical=a.identical_suits, cameras=2 if a.two_cameras else 1,
               tracker=r['tracker'].name, idsw=s['idsw'], idf1=s['idf1'], mota=s['mota'],
               short_kept=sum(x['kept'] for x in b[:2]), short_n=sum(x['n'] for x in b[:2]),
               long_kept=sum(x['kept'] for x in b[2:]), long_n=sum(x['n'] for x in b[2:]), reids=len(r['tracker'].reidEvents))
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')
print(f'seed {a.seed} identical={a.identical_suits} cams={2 if a.two_cameras else 1}: {time.time() - t0:.0f} s')
