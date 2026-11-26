"""A camera's scene bundle from its raw render: the same for every scene and both routes (numpy + OpenCV).

  export(bundle_dir, n_frames, world_checks)     called by p2s/2_render/export_x.py and legacy/sim/world/export.py

Reads rgb.npy, depth.npy, instance.npy, timeline.json (p2s/2_render/render_x.py or legacy/sim/render_world.py). Writes:
  rgb.mp4, frame0.png
  boxes.json        per object (instance id), per frame [x0, y0, x1, y1] from instance.npy, or null
  visibility.json   per object: visible pixels per frame, frames with >= MIN_PIXELS, first / last such frame
  tracks.npz        points [N,T,2] (pixels), visible [N,T], object_id [N], part [N] (index into parts), depth [N,T]
  overlay.mp4       RGB with boxes and track points (filled = visible, ring = occluded)
  controls/         depth, seg, edge, normals, rgb control videos + controls_sheet.jpg (p2s/2_render/control_videos.py)
  checks.json       the world's checks plus any `visible` checks for this camera, now evaluated
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import control_videos as ctl  # noqa: E402

write_mp4 = ctl.write_mp4

MIN_PIXELS = 30            # an object counts as visible in a frame with at least this many pixels
VIS_DEPTH_TOL_M = 0.08


def project(K, c2w, pts):
    """pts [N, T, 3] world, c2w [T, 4, 4] (Blender camera) -> uv [N, T, 2], z [N, T]."""
    w2c = np.linalg.inv(c2w)
    pc = np.einsum("tij,ntj->nti", w2c[:, :3, :3], pts) + w2c[None, :, :3, 3]
    x, y, z = pc[..., 0], -pc[..., 1], -pc[..., 2]
    zz = np.maximum(z, 1e-6)
    return np.stack([K[0, 0] * x / zz + K[0, 2], K[1, 1] * y / zz + K[1, 2]], -1), z


def palette(i):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def export(out, n_frames, world_checks):
    """The bundle from one camera's raw render in `out`. Scene-independent: p2s/2_render/export_x.py calls it too."""
    tl = json.load(open(os.path.join(out, "timeline.json")))
    rgb, depth, inst = (np.load(os.path.join(out, f)) for f in ("rgb.npy", "depth.npy", "instance.npy"))
    T, H, W, _ = rgb.shape
    assert T == n_frames, "preview renders are not exported"
    K, c2w = np.asarray(tl["camera"]["K"]), np.asarray(tl["camera"]["cam2world_blender"])
    write_mp4(os.path.join(out, "rgb.mp4"), rgb, tl["fps"])
    cv2.imwrite(os.path.join(out, "frame0.png"), cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR))
    normals = np.load(os.path.join(out, "normals.npy")) if os.path.exists(os.path.join(out, "normals.npy")) else None
    cs = ctl.build_all(rgb, depth, inst, normals)
    os.makedirs(os.path.join(out, "controls"), exist_ok=True)
    for name, v in cs.items():
        ctl.write_mp4(os.path.join(out, "controls", f"{name}.mp4"), v, tl["fps"])
    cv2.imwrite(os.path.join(out, "controls", "controls_sheet.jpg"), cv2.cvtColor(ctl.sheet(cs), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    boxes, vis = {}, {}
    for o in tl["objects"]:
        i = o["instance_id"]
        per, px = [], []
        for t in range(T):
            ys, xs = np.nonzero(inst[t] == i)
            px.append(int(len(xs)))
            per.append(None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        boxes[str(i)] = per
        good = [t for t in range(T) if px[t] >= MIN_PIXELS]
        vis[o["id"]] = {"instance_id": i, "pixels": px, "frames_visible": len(good),
                        "first": good[0] if good else None, "last": good[-1] if good else None, "hidden_from_camera": o["hidden"]}
    json.dump(boxes, open(os.path.join(out, "boxes.json"), "w"))
    json.dump(vis, open(os.path.join(out, "visibility.json"), "w"), indent=1)

    pts_all, vis_all, ids, part_idx, z_all, part_names = [], [], [], [], [], []
    for o in tl["objects"]:
        if o["hidden"]:
            continue
        for pname, p in o["parts"].items():
            local = np.asarray(p["track_points_local"])
            if len(local) == 0:
                continue
            M = np.asarray(p["local2world"])                                     # [T, 4, 4]
            if tl.get("track_points_frame") != "part":      # a render from before 2026-11-23: points relative to the last pose
                local = local @ M[-1, :3, :3].T + M[-1, :3, 3]
            world = np.einsum("tij,nj->nti", M[:, :3, :3], local) + M[None, :, :3, 3]
            uv, z = project(K, c2w, world)
            ui = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
            inb = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1) & (z > 0)
            tt = np.broadcast_to(np.arange(T), ui.shape)
            v = inb & (inst[tt, vi, ui] == o["instance_id"]) & (np.abs(z - depth[tt, vi, ui]) < VIS_DEPTH_TOL_M)
            pts_all.append(uv); vis_all.append(v); z_all.append(z)
            ids += [o["instance_id"]] * len(local)
            part_idx += [len(part_names)] * len(local)
            part_names.append(f"{o['id']}/{pname}")
    if not pts_all:                                         # a scene whose objects have no rigid parts (only fluid, cloth)
        pts_all, vis_all, z_all = [np.zeros((0, T, 2))], [np.zeros((0, T), bool)], [np.zeros((0, T))]
    np.savez(os.path.join(out, "tracks.npz"), points=np.concatenate(pts_all).astype(np.float32), visible=np.concatenate(vis_all),
             object_id=np.asarray(ids, np.int32), part=np.asarray(part_idx, np.int32), depth=np.concatenate(z_all).astype(np.float32),
             part_names=np.asarray(part_names))

    # checks: the world's, plus `visible` checks for this camera
    checks = []
    for c in world_checks:
        if c["type"] == "visible":
            if c["camera"] != tl["camera"]["id"]:
                continue
            got = {oid: vis[oid]["frames_visible"] for oid in c["objects"]}
            c = {**c, "ok": all(v >= c["min_frames"] for v in got.values()), "detail": {"frames_visible": got}}
        checks.append(c)
    json.dump(checks, open(os.path.join(out, "checks.json"), "w"), indent=1)
    for c in checks:
        if c["type"] == "visible":
            print(f"[export] {tl['camera']['id']}: visible check {'ok' if c['ok'] else 'FAILED'} {c['detail']}")

    points = np.concatenate(pts_all)
    visible = np.concatenate(vis_all)
    frames = []
    for t in range(T):
        f = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        for i, b in boxes.items():
            if b[t] is not None:
                cv2.rectangle(f, tuple(b[t][:2]), tuple(b[t][2:]), palette(i), 1)
        for n in range(len(points)):
            c = (int(round(points[n, t, 0])), int(round(points[n, t, 1])))
            if 0 <= c[0] < W and 0 <= c[1] < H:
                cv2.circle(f, c, 2, palette(ids[n]), -1 if visible[n, t] else 1, cv2.LINE_AA)
        cv2.putText(f, f"{tl['camera']['id']} t={t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(f)
    write_mp4(os.path.join(out, "overlay.mp4"), np.stack(frames)[..., ::-1], tl["fps"], crf=18)
    print(f"[export] {out}: {T} frames, {len(points)} track points")

