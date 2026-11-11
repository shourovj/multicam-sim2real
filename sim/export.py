"""Second half of the scene bundle (plan §4, sim/export.py): from the raw render to the files every
condition and the eval read.

  envs/eval/.venv/bin/python sim/export.py <bundle_dir>

Reads rgb.npy, depth.npy, instance.npy, timeline.json (sim/render_scene.py). Writes:
  rgb.mp4, rgb/%04d.png, frame0.png    condition R and the image input for B, C, E
  tracks.npz      points [N,T,2] (pixels, x then y), visible [N,T] bool, object_id [N],
                  depth [N,T] (camera z of the point, metres)
  boxes.json      per object, per frame [x0, y0, x1, y1] from instance.npy, or null
  overlay.mp4     RGB with boxes, track points (filled = visible, ring = occluded) and short trails
  checks.json     in-frame / size checks, the observed crossing time and the frames where each object is
                  fully hidden; the crossing and the hidden ranges are also added to timeline.json as events
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

VIS_DEPTH_TOL_M = 0.05
PALETTE = {1: (0, 140, 255), 2: (255, 200, 0), 3: (80, 220, 80)}   # BGR per instance id, used in every overlay


def project(K, cam2world, pts_world):
    """Blender camera (looks down -Z, +Y up) -> OpenCV pixels. Returns uv [..., 2], z [...] (planar)."""
    w2c = np.linalg.inv(cam2world)
    pc = pts_world @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = pc[..., 0], -pc[..., 1], -pc[..., 2]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.stack([u, v], -1), z


def write_mp4(path, frames_rgb, fps, crf=12):
    T, H, W, _ = frames_rgb.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames_rgb).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {path}"


def compute_tracks(tl, depth, inst):
    K = np.asarray(tl["camera"]["K"])
    c2w = np.asarray(tl["camera"]["cam2world_blender"])
    T, H, W = depth.shape
    pts_all, vis_all, ids, z_all = [], [], [], []
    for o in tl["objects"]:
        local = np.asarray(o["track_points_local"])                       # [n, 3]
        mats = np.asarray(o["local2world"])                               # [T, 4, 4]
        world = np.einsum("tij,nj->nti", mats[:, :3, :3], local) + mats[None, :, :3, 3]  # [n, T, 3]
        uv, z = project(K, c2w, world)
        ui = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1)
        vi = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
        inb = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1) & (z > 0)
        tt = np.broadcast_to(np.arange(T), ui.shape)
        d_here = depth[tt, vi, ui]
        same = inst[tt, vi, ui] == o["instance_id"]
        vis = inb & same & (np.abs(z - d_here) < VIS_DEPTH_TOL_M)
        pts_all.append(uv); vis_all.append(vis); z_all.append(z)
        ids += [o["instance_id"]] * len(local)
    return (np.concatenate(pts_all).astype(np.float32), np.concatenate(vis_all),
            np.asarray(ids, np.int32), np.concatenate(z_all).astype(np.float32))


def boxes_from_instance(inst, ids):
    out = {}
    for i in ids:
        per = []
        for m in inst:
            ys, xs = np.nonzero(m == i)
            per.append(None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        out[str(i)] = per
    return out


def draw_overlay(rgb, points, visible, object_id, boxes, trail=12):
    """BGR frames with boxes, points (filled = visible, ring = occluded) and trails."""
    frames = []
    for t in range(len(rgb)):
        f = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        for i, b in boxes.items():
            if b[t] is not None:
                cv2.rectangle(f, tuple(b[t][:2]), tuple(b[t][2:]), PALETTE.get(int(i), (255, 255, 255)), 1)
        for n in range(len(points)):
            col = PALETTE.get(int(object_id[n]), (255, 255, 255))
            for s in range(max(0, t - trail), t):
                p0, p1 = points[n, s], points[n, s + 1]
                cv2.line(f, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 1, cv2.LINE_AA)
            c = (int(round(points[n, t, 0])), int(round(points[n, t, 1])))
            cv2.circle(f, c, 2, col, -1 if visible[n, t] else 1, cv2.LINE_AA)
        cv2.putText(f, f"t={t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(f)
    return np.stack(frames)


def main(out):
    tl = json.load(open(os.path.join(out, "timeline.json")))
    rgb = np.load(os.path.join(out, "rgb.npy"))
    depth = np.load(os.path.join(out, "depth.npy"))
    inst = np.load(os.path.join(out, "instance.npy"))
    T, H, W, _ = rgb.shape
    fps = tl["fps"]
    assert T == tl["frames"], "preview bundles can't be exported"

    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    for t in range(T):
        cv2.imwrite(os.path.join(out, "rgb", f"{t:04d}.png"), cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "frame0.png"), cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR))
    write_mp4(os.path.join(out, "rgb.mp4"), rgb, fps)

    points, visible, object_id, zc = compute_tracks(tl, depth, inst)
    np.savez(os.path.join(out, "tracks.npz"), points=points, visible=visible, object_id=object_id, depth=zc)
    ids = [o["instance_id"] for o in tl["objects"]]
    boxes = boxes_from_instance(inst, ids)
    json.dump(boxes, open(os.path.join(out, "boxes.json"), "w"))

    # Checks: every drone fully in frame (box off the border) and >= 40 px wide on every frame;
    # tracked points land on their own mask when visible; crossing = image-x order of box centres swaps.
    checks = {}
    for i in ids:
        b = np.array([bb if bb is not None else [-1] * 4 for bb in boxes[str(i)]])
        b = b[b[:, 0] >= 0]            # frames where the object is fully hidden by an occluder have no box
        widths = b[:, 2] - b[:, 0] + 1
        checks[f"id{i}"] = {
            "frames_present": len(b),
            "fully_in_frame": bool(((b[:, 0] > 0) & (b[:, 1] > 0) & (b[:, 2] < W - 1) & (b[:, 3] < H - 1)).all()),
            "min_width_px": int(widths.min()), "max_width_px": int(widths.max()),
            "visible_point_frac": float(visible[object_id == i].mean()),
            "frame0_visible_points": int(visible[object_id == i, 0].sum()),
        }
    # Occlusion ground truth: the frame ranges where an object is fully hidden (no pixel of it is visible).
    names = {o["instance_id"]: o["id"] for o in tl["objects"]}
    hidden_events = []
    for i in ids:
        gone = [bb is None for bb in boxes[str(i)]]
        runs = [[t, t] for t in range(T) if gone[t] and (t == 0 or not gone[t - 1])]
        for r in runs:
            while r[1] + 1 < T and gone[r[1] + 1]:
                r[1] += 1
        checks[f"id{i}"]["hidden_frames"] = runs
        hidden_events += [{"name": f"{names[i]}_hidden", "t": float(tl["times_s"][a]), "until": float(tl["times_s"][b]),
                           "frames": [a, b]} for a, b in runs]
    if hidden_events:
        tl["events"] = [e for e in tl["events"] if not e["name"].endswith("_hidden")] + hidden_events
        json.dump(tl, open(os.path.join(out, "timeline.json"), "w"))
    if len(ids) == 2 and all(bb is not None for i in ids for bb in boxes[str(i)]):
        cx = {i: np.array([(bb[0] + bb[2]) / 2 for bb in boxes[str(i)]]) for i in ids}
        s = np.sign(cx[ids[0]] - cx[ids[1]])
        swap = np.nonzero(s[1:] != s[:-1])[0]
        if len(swap):
            t_cross = float(tl["times_s"][int(swap[0]) + 1])
            checks["crossing_observed"] = {"frame": int(swap[0]) + 1, "t": t_cross}
            tl["events"] = [e for e in tl["events"] if e["name"] != "crossing_observed"] + \
                           [{"name": "crossing_observed", "t": t_cross, "frame": int(swap[0]) + 1,
                             "note": "image-x order of the two box centres swaps"}]
            json.dump(tl, open(os.path.join(out, "timeline.json"), "w"))
    json.dump(checks, open(os.path.join(out, "checks.json"), "w"), indent=2)
    print(json.dumps(checks, indent=2))

    ov = draw_overlay(rgb, points, visible, object_id, boxes)
    write_mp4(os.path.join(out, "overlay.mp4"), ov[..., ::-1], fps, crf=16)
    for t in (0, T // 2, T - 1):
        cv2.imwrite(os.path.join(out, f"overlay_{t:04d}.png"), ov[t])


if __name__ == "__main__":
    main(sys.argv[1])
