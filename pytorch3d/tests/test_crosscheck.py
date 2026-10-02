"""The Python port must reproduce the web lab exactly on the web lab's own sensor model."""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lunar3d.world import LunarWorld
from lunar3d.tracker import Tracker
from lunar3d.metrics import MotMetrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def run(scenario='boulders', seed=7, seconds=60):
    js = json.loads(subprocess.check_output(['node', 'js_reference.mjs', scenario, str(seed), str(seconds)],
                                            cwd=os.path.join(ROOT, 'matlab', 'lunar')))
    W = LunarWorld(scenario, seed)
    sun = W.sun if W.cfg['shadows'] else None
    runs = [(Tracker(W.cam, W.boulders, v, sun=sun, pd=W.cfg['pd'], clutter=W.cfg['clutter']), MotMetrics())
            for v in ('negInfo', 'aware', 'naive')]
    nd = 0
    for _ in range(round(seconds / W.dt)):
        W.step(); d, _ = W.sense(); nd += len(d)
        for T, M in runs:
            T.step(d); M.update(W, T)
    truth = np.array([[t['x'], t['y']] for t in W.targets])
    assert nd == js['nDets'], (nd, js['nDets'])
    assert np.abs(truth - np.array(js['truth'])).max() < 1e-6
    for (T, M), j in zip(runs, js['runs']):
        s = M.summary()
        assert (s['idsw'], s['occKept'], s['occEvents']) == (j['idsw'], j['occKept'], j['occEvents']), (T.name, s['idsw'], j['idsw'])
        assert abs(s['idf1'] - j['idf1']) < 1e-9 and abs(s['mota'] - j['mota']) < 1e-9
    return [(T.name, M.summary()['idsw']) for T, M in runs]


def test_boulders():
    run('boulders', 7, 60)


def test_polar():
    run('polar', 3, 60)


if __name__ == '__main__':
    print(run(*(sys.argv[1:2] or ['boulders']), seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7,
              seconds=float(sys.argv[3]) if len(sys.argv) > 3 else 60))
    print('Python port reproduces the web lab exactly')
