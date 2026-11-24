"""The eval metrics (eval/common/metrics.py) on made-up arrays. Run:  python tests/test_eval_metrics.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "common"))
import metrics as M  # noqa: E402

rng = np.random.default_rng(0)


def scene_depth(T=3, H=40, W=60):
    d = 2 + 8 * rng.random((T, H, W))
    d[:, :5] = np.inf                                       # sky
    return d


def test_depth_is_one_for_the_reference_at_any_scale():
    ref = scene_depth()
    est = np.where(np.isfinite(ref), ref, 50.0)
    assert np.allclose(M.depth_accuracy(est, ref), 1.0)
    assert np.allclose(M.depth_accuracy(est * 7.3, ref), 1.0)             # one scale per frame is removed


def test_depth_counts_the_wrong_pixels():
    ref = scene_depth()
    est = np.where(np.isfinite(ref), ref, 50.0)
    est[:, 5:, :15] *= 2.0                                                # a quarter of the columns are twice too far
    assert np.allclose(M.depth_accuracy(est, ref), 0.75, atol=0.02)
    region = np.zeros(ref.shape, bool)
    region[:, 5:, :15] = True
    assert np.allclose(M.depth_accuracy(est, ref, region=region), 0.0)    # counted only where it is wrong


def test_depth_inverse_relative_fits_scale_and_shift():
    ref = scene_depth()
    est = np.where(np.isfinite(ref), 3.0 / ref + 0.4, 0.0)                # inverse depth, scaled and shifted
    assert np.allclose(M.depth_accuracy(est, ref, kind="inverse_relative"), 1.0)
    assert M.depth_accuracy(est, ref, kind="depth").max() < 0.9           # read as depth, the same array is wrong


def test_mask_iou():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b[4:8, 2:6] = True
    assert M.mask_iou(a, a) == 1.0 and abs(M.mask_iou(a, b) - 8 / 24) < 1e-9 and np.isnan(M.mask_iou(~a & a, ~a & a))


def test_tracks():
    N, T, size = 5, 20, (832, 480)
    ref = np.cumsum(rng.normal(0, 3, (N, T, 2)), 1) + 200
    vis = np.ones((N, T), bool)
    q = np.zeros(N, int)
    d, per, occ = M.track_accuracy(ref.copy(), vis, ref, vis, q, 0, T - 1, size)
    assert d == 1.0 and occ == 1.0
    off = ref + np.array([10 * 832 / 256, 0])                             # 10 px off at 256 px scale: only the 16 px bin passes
    d, per, _ = M.track_accuracy(off, vis, ref, vis, q, 0, T - 1, size)
    assert per == {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0, 16: 1.0} and abs(d - 0.2) < 1e-9
    d, _, _ = M.track_accuracy(off, vis, ref, vis, np.full(N, T - 1), 0, T - 1, size)
    assert np.isnan(d)                                                    # nothing after the query frame


def test_harmonic_mean_is_pulled_to_the_weakest():
    assert abs(M.harmonic_mean([1, 1, 1]) - 1) < 1e-9 and M.harmonic_mean([0.9, 0.9, 0.1]) < 0.25 and M.harmonic_mean([0.5, 0.0]) == 0.0


def look_at(pos, target):
    """Blender camera (looks down -Z, +Y up) at pos, aimed at target."""
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = np.stack([r, np.cross(r, f), -f], 1), pos
    return m


def render_plane(K, c2w, H, W):
    """Depth (along the view axis) of the ground plane z = 0 for one camera."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u)], -1) @ c2w[:3, :3].T
    s = -c2w[2, 3] / ray[..., 2]
    return np.where((ray[..., 2] < 0), s, np.inf)


def test_correspondences_land_on_the_same_ground_point():
    H, W = 60, 80
    K = np.array([[60.0, 0, W / 2 - 0.5], [0, 60.0, H / 2 - 0.5], [0, 0, 1]])
    ca, cb = look_at(np.array([0.0, -6, 4]), [0, 0, 0]), look_at(np.array([5.0, 2, 3]), [0, 0, 0])
    da, db = render_plane(K, ca, H, W), render_plane(K, cb, H, W)
    inst = np.full((H, W), 100, np.uint8)
    ua, va, cov = M.correspondences(db, K, cb, inst, da, K, ca, inst)
    assert cov.mean() > 0.3

    def ground(u, v, c2w, d):                                             # the world point a pixel sees
        p = np.array([(u - K[0, 2]) / K[0, 0] * d, -(v - K[1, 2]) / K[1, 1] * d, -d, 1.0])
        return (c2w @ p)[:3]
    for y, x in zip(*np.nonzero(cov)[0:2]):
        if (y * W + x) % 97:
            continue
        pb = ground(x, y, cb, db[y, x])
        xa, ya = int(round(ua[y, x])), int(round(va[y, x]))
        pa = ground(ua[y, x], va[y, x], ca, da[ya, xa])
        assert np.linalg.norm(pa - pb) < 0.25 and abs(pb[2]) < 1e-6
    hidden = da.copy()
    hidden[:] = 0.5                                                       # something close in front of camera A everywhere
    assert not M.correspondences(db, K, cb, inst, hidden, K, ca, inst)[2].any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
