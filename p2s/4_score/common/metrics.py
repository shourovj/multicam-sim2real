"""The metrics, as plain functions of arrays (numpy only): no model, no scene knowledge, no file access.

Per camera (Eval2), each in [0, 1] with 1 best:
  depth_accuracy     share of pixels whose estimated depth is within a factor of the rendered depth
  mask_iou           overlap of each object's estimated mask with its rendered mask
  track_accuracy     share of simulator surface points the tracker finds near where the simulator puts them (TAP-Vid)
Across cameras (Eval3):
  correspondences    which pixel of camera A shows the same surface point as each pixel of camera B, from the
                     rendered depth and the cameras; the two cross-camera scores compare features at these pairs
"""

import numpy as np

TAPVID_THRESHOLDS = (1, 2, 4, 8, 16)       # pixels, with the frame rescaled to 256 x 256 (TAP-Vid's convention)


def harmonic_mean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return float("nan")
    return 0.0 if min(xs) <= 0 else len(xs) / sum(1.0 / x for x in xs)


def align_depth(est, ref, valid, kind):
    """Bring one frame's estimate onto the reference's scale. kind "depth": the estimate is depth up to a scale, so
    multiply by median(ref) / median(est). kind "inverse_relative": the estimate is inverse depth up to a scale and a
    shift (Depth Anything V2, MiDaS), so fit both to 1 / ref by least squares."""
    if kind == "depth":
        return est * (np.median(ref[valid]) / max(np.median(est[valid]), 1e-9))
    A = np.stack([est[valid], np.ones(int(valid.sum()))], 1)
    a, b = np.linalg.lstsq(A, 1.0 / ref[valid], rcond=None)[0]
    return 1.0 / np.maximum(a * est + b, 1e-6)


def depth_ok(est, ref, kind="depth", ratio=1.25):
    """One frame: (valid, ok) boolean maps. valid = the reference has a surface there; ok = valid and the aligned
    estimate is within `ratio` of it. (None, None) if the frame has too few valid pixels to align."""
    valid = np.isfinite(ref) & (ref > 0) & np.isfinite(est)
    if valid.sum() < 16:
        return None, None
    d = np.maximum(align_depth(est.astype(np.float64), ref.astype(np.float64), valid, kind), 1e-6)
    r = np.where(valid, ref, 1.0)
    return valid, (np.maximum(d / r, r / d) < ratio) & valid


def depth_accuracy(est, ref, kind="depth", ratio=1.25, region=None):
    """est, ref [T, H, W]; ref is inf where there is no surface. region [T, H, W] bool restricts the count (the
    alignment always uses every valid pixel). Returns the per-frame share of pixels with max(est / ref, ref / est) < ratio;
    nan for a frame with nothing to count."""
    out = np.full(len(ref), np.nan)
    for t in range(len(ref)):
        valid, ok = depth_ok(est[t], ref[t], kind, ratio)
        if valid is None:
            continue
        count = valid if region is None else valid & region[t]
        if count.sum() >= 16:
            out[t] = (ok & count).sum() / count.sum()
    return out


def mask_iou(est, ref):
    """IoU of two boolean masks; nan if both are empty."""
    union = (est | ref).sum()
    return float((est & ref).sum() / union) if union else float("nan")


def track_accuracy(est, est_visible, ref, ref_visible, query_t, first, last, size):
    """TAP-Vid's position accuracy and occlusion accuracy for the points of one shot.
    est, ref [N, T, 2] pixels; *_visible [N, T]; query_t [N] the frame each point was handed to the tracker;
    first, last: the shot's frame range (one pair, or one per point); size (W, H). Frames up to and including the query frame are not counted.
    Returns (delta_avg, {threshold: share}, occlusion accuracy), nan if there is nothing to count."""
    N, T = ref.shape[:2]
    t = np.arange(T)[None]
    first, last = np.asarray(first).reshape(-1, 1), np.asarray(last).reshape(-1, 1)      # one range, or one per point
    counted = (t > query_t[:, None]) & (t >= first) & (t <= last)
    sel = counted & ref_visible
    if not sel.any():
        return float("nan"), {}, float("nan")
    scale = np.array([256.0 / size[0], 256.0 / size[1]])
    err = np.linalg.norm((est - ref) * scale, axis=-1)[sel]
    per = {p: float((err < p).mean()) for p in TAPVID_THRESHOLDS}
    occ = float((est_visible[counted] == ref_visible[counted]).mean())
    return float(np.mean(list(per.values()))), per, occ


def correspondences(depth_b, K_b, c2w_b, inst_b, depth_a, K_a, c2w_a, inst_a, rel_tol=0.04, abs_tol=0.08):
    """For every pixel of camera B: the pixel of camera A that shows the same surface point, at the same instant.
    depth_* [H, W] (inf = nothing), K_* [3, 3], c2w_* [4, 4] (Blender camera), inst_* [H, W].
    Returns ua, va [H, W] float (A's pixel coordinates) and covisible [H, W] bool: B's pixel has a surface, it projects
    inside A, A's rendered surface there is the same one (depth agrees, same instance id)."""
    H, W = depth_b.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    valid = np.isfinite(depth_b) & (depth_b > 0)
    z = np.where(valid, depth_b, 1.0)
    pc = np.stack([(u - K_b[0, 2]) / K_b[0, 0] * z, -(v - K_b[1, 2]) / K_b[1, 1] * z, -z, np.ones_like(z)], -1)
    pa = pc @ c2w_b.T @ np.linalg.inv(c2w_a).T
    za = -pa[..., 2]
    zz = np.maximum(za, 1e-6)
    ua = K_a[0, 0] * pa[..., 0] / zz + K_a[0, 2]
    va = K_a[1, 1] * -pa[..., 1] / zz + K_a[1, 2]
    Ha, Wa = depth_a.shape
    inb = valid & (za > 0.05) & (ua >= 0) & (ua <= Wa - 1) & (va >= 0) & (va <= Ha - 1)
    ui, vi = np.clip(np.round(ua).astype(int), 0, Wa - 1), np.clip(np.round(va).astype(int), 0, Ha - 1)
    da = depth_a[vi, ui]
    cov = inb & np.isfinite(da) & (np.abs(da - za) < rel_tol * za + abs_tol) & (inst_a[vi, ui] == inst_b)
    return ua, va, cov


def cosine(a, b):
    """Row-wise cosine similarity of two [N, D] arrays."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return (a * b).sum(-1)
