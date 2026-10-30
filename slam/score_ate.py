"""Score a SLAM trajectory against the lab's true camera path.

ATE (absolute trajectory error, Sturm et al. 2012): align the estimate to the truth with the best similarity
transform (rotation, translation, scale; Umeyama 1991, since one camera cannot know absolute scale), then take the
RMS of the remaining position errors. Also reported as % of the distance driven, which compares runs of any length.

    python slam/score_ate.py [--est poses_orb.txt] [--se3] RUN_DIR [RUN_DIR ...]
--se3: align with rotation + translation only (scale fixed at 1), for stereo, which measures true size; a wrong
size then counts as error instead of being aligned away. Default --est is poses_vggt.txt.
RUN_DIR holds poses_truth.txt (t x y z qx qy qz qw, one line per frame i) and poses_vggt.txt (frame_id x y z ...,
frame_id = index of the image, 000123.jpg -> 123). Only positions are scored.

RPE (relative error, drift per distance; after Geiger et al. 2012 / KITTI): for every pair of SLAM keyframes about
SEG metres of true driving apart, compare the estimated displacement (scaled and rotated by the ATE alignment) with
the true one; report the mean error as % of SEG. ATE says where it ended up, RPE how fast it drifts.
ponytail: compares displacements in the aligned world frame, not in each keyframe's own frame (full RPE also scores
rotation); add that if heading drift needs its own number.
"""
import sys
import numpy as np


def umeyama(src, dst, scale=True):
    """s, R, t minimising |dst - (s R src + t)|^2 over all points (rows); scale=False keeps s = 1."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


SEG = 10.0


def rpe(P, T, s, R, along):
    """Mean |aligned estimated displacement - true displacement| / true distance driven, for keyframe pairs ~SEG m apart."""
    errs = []
    for a in range(len(T)):
        b = np.searchsorted(along, along[a] + SEG)
        if b >= len(T):
            break
        d = along[b] - along[a]
        errs.append(np.linalg.norm(s * R @ (P[b] - P[a]) - (T[b] - T[a])) / d)
    return float(np.mean(errs)) if errs else float('nan')


def score(run, est_name='poses_vggt.txt', scale=True):
    truth = np.loadtxt(f'{run}/poses_truth.txt')[:, 1:4]
    est = np.loadtxt(f'{run}/{est_name}')
    ids = est[:, 0].astype(int)
    P, T = est[:, 1:4], truth[ids]
    s, R, t = umeyama(P, T, scale)
    err = np.linalg.norm((s * (R @ P.T)).T + t - T, axis=1)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))])
    driven = cum[-1]
    return dict(rpe_pct=100 * rpe(P, T, s, R, cum[ids]), run=run.rstrip('/').split('/')[-1], frames=len(truth), posed=len(ids), driven_m=driven,
                ate_rmse_m=float(np.sqrt((err ** 2).mean())), ate_max_m=float(err.max()))


if __name__ == '__main__':
    # self-check: a scaled, rotated, shifted, slightly noisy copy of a path must score ~ its noise
    rng = np.random.default_rng(0)
    A = np.cumsum(rng.normal(size=(200, 3)), 0)
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    B = (0.3 * (Rz @ A.T)).T + [5, -2, 1] + rng.normal(scale=0.01, size=A.shape)
    s, R, t = umeyama(B, A)
    assert np.sqrt((np.linalg.norm((s * (R @ B.T)).T + t - A, axis=1) ** 2).mean()) < 0.06
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
    assert rpe(B, A, s, R, cum) < 0.01                                  # a near-perfect copy barely drifts
    assert abs(umeyama(B, A, scale=False)[0] - 1) < 1e-12 and np.allclose(umeyama(A + 3, A, scale=False)[2], -3)   # SE(3): no scale
    assert rpe((Rz @ A.T).T * 0.3 + 0.002 * np.arange(200)[:, None] ** 2, A, *umeyama((Rz @ A.T).T * 0.3, A)[:2], cum) > 0.01   # growing drift shows up
    print(f'| run | frames | posed by SLAM | driven (m) | ATE RMSE (m) | ATE as % of distance | worst error (m) | drift per {SEG:.0f} m (RPE) |\n|---|---|---|---|---|---|---|---|')
    args = sys.argv[1:]
    est_name = args.pop(args.index('--est') + 1) if '--est' in args else 'poses_vggt.txt'
    scale = '--se3' not in args
    for r in [a for a in args if not a.startswith('--')]:
        try:
            o = score(r, est_name, scale)
            print(f"| {o['run']} | {o['frames']} | {o['posed']} | {o['driven_m']:.1f} | {o['ate_rmse_m']:.2f} | {100 * o['ate_rmse_m'] / o['driven_m']:.1f}% | {o['ate_max_m']:.2f} | {o['rpe_pct']:.1f}% |")
        except Exception as e:  # a failed run is a result too: say so instead of hiding it
            print(f"| {r.rstrip('/').split('/')[-1]} | failed: {e} |")
