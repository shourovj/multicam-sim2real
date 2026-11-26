"""Per-model control inputs from a scene bundle (plan §5, legacy/sim/controls.py).

Runs in VACE's venv (numpy, OpenCV, imageio and VACE's annotator utils), from any cwd:

  $S2R_ROOT/VACE/.venv/bin/python legacy/sim/controls.py <bundle_dir> <poc_dir> [--only X ...]

Writes:
  <bundle>/controls/vace_depth.mp4    81 frames @ 16 fps: 1/z, sky masked to 0, min-max per frame,
                                      near = bright, 3-channel (Depth Anything V2 style, as VACE trained)
  <bundle>/controls/cosmos_depth.mp4  153 frames @ 30 fps: 1/z, sky 0, min-max over the whole clip,
                                      near = bright, 3-channel (matches Sim-Transfer's example2 depth)
  <poc>/B/inputs/src_video.mp4, src_mask.mp4   frame 0 + 80 gray frames (VACE's frameref CLI)
  <poc>/C/inputs/src_video.mp4, src_mask.mp4   frame 0, then vace_depth frames 1..80
                                               (VACE CompositionAnnotator, frameref + depth)
  <poc>/C0/inputs/src_video.mp4, src_mask.mp4  vace_depth for all 81 frames, mask all white
                                               (VACE's plain depth task, no first frame)
  <poc>/Bbox/inputs/src_video.mp4, src_mask.mp4  frame 0, then the simulator boxes drawn in VACE's
                                               layout_track format (3 px label colour on white)
  <poc>/P1/inputs/src_video.mp4, src_mask.mp4  as Bbox for the first second (frames 1..16), then gray:
                                               the model gets the start of the motion and continues it
  <poc>/E/inputs/tracks.npy [81, N, 2] float32, visibility.npy [81, N] bool, points.json
       Wan-Move tracks: frame-0 mask pixels back-projected through the depth, carried rigidly with
       their drone, at most one point per 8x8 latent cell per drone, all visible at frame 0.

--frame0 IMAGE uses that image (e.g. a hand-edited, realistic frame 0) in place of the bundle's
frame0.png for B, Bbox and P1, and --suffix names those folders (B_edit, ...), so both versions can
sit side by side. C always uses the simulator's frame 0, because its depth has to match the frame.

Every 16 fps stream picks bundle frames by timestamp: output frame i <- bundle frame round(i * 30/16),
so frame 80 is bundle frame 150 and motion keeps its real speed.
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.environ.get("S2R_ROOT", os.path.expanduser("~/code/sim2real-videogen"))
VACE_DIR = os.path.join(ROOT, "VACE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export import project  # noqa: E402

WAN_N, WAN_FPS = 81, 16
WANMOVE_HW = (464, 832)     # Wan-Move's working size for an 832x480 image (gotchas §3.4)
LATENT_STRIDE = 8


def wan_indices(fps_src, n_src, n=WAN_N, fps=WAN_FPS):
    idx = np.round(np.arange(n) * fps_src / fps).astype(int)
    assert idx[-1] < n_src
    return idx


def inverse_depth_u8(depth, per_frame):
    """depth [T,H,W] planar metres (inf = sky) -> uint8 [T,H,W,3], near bright, sky 0."""
    sky = ~np.isfinite(depth)
    inv = np.where(sky, 0.0, 1.0 / np.where(sky, 1.0, depth))
    out = np.zeros(depth.shape, np.float32)
    if per_frame:
        for t in range(len(depth)):
            v = inv[t][~sky[t]]
            lo, hi = v.min(), v.max()
            out[t] = (inv[t] - lo) / (hi - lo)
    else:
        v = inv[~sky]
        out = (inv - v.min()) / (v.max() - v.min())
    out[sky] = 0.0
    u8 = np.clip(np.round(out * 255), 0, 255).astype(np.uint8)
    return np.repeat(u8[..., None], 3, -1)


def save_mp4(path, frames_rgb, fps):
    from annotators.utils import save_one_video  # VACE's writer (imageio, quality as VACE uses)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_one_video(path, list(frames_rgb), fps=fps)


def load_frame0(path, H, W):
    """Frame 0 as RGB at the clip size. An edited frame may come back from the editor at another size."""
    im = cv2.imread(path)
    assert im is not None, f"can't read {path}"
    if im.shape[:2] != (H, W):
        assert abs(im.shape[1] / im.shape[0] - W / H) < 0.02, f"{path} has a different aspect ratio: {im.shape[1]}x{im.shape[0]}"
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def build_vace_b(frame0, out, H, W):
    """Text + first frame (VACE's frameref CLI): `frame0` then 80 gray frames, written to `out`."""
    os.makedirs(out, exist_ok=True)
    image = os.path.join(out, "frame0.png")
    cv2.imwrite(image, cv2.cvtColor(load_frame0(frame0, H, W), cv2.COLOR_RGB2BGR))
    subprocess.run([os.path.join(VACE_DIR, ".venv/bin/python"), "vace/vace_preproccess.py",
                    "--task", "frameref", "--mode", "firstframe", "--expand_num", str(WAN_N - 1),
                    "--image", image, "--save_fps", str(WAN_FPS),
                    "--pre_save_dir", out], check=True, cwd=VACE_DIR)
    os.replace(os.path.join(out, "src_video-frameref.mp4"), os.path.join(out, "src_video.mp4"))
    os.replace(os.path.join(out, "src_mask-frameref.mp4"), os.path.join(out, "src_mask.mp4"))


def build_vace_c(bundle, poc, vace_depth, H, W):
    from annotators.composition import CompositionAnnotator
    from annotators.utils import read_video_frames
    out = os.path.join(poc, "C", "inputs")
    os.makedirs(out, exist_ok=True)
    build_vace_b(os.path.join(bundle, "frame0.png"), os.path.join(out, "frameref"), H, W)   # simulator frame 0, always
    ff_video = os.path.join(out, "frameref", "src_video.mp4")
    ff_mask = os.path.join(out, "frameref", "src_mask.mp4")
    v1 = read_video_frames(ff_video, use_type="cv2")
    v2 = read_video_frames(vace_depth, use_type="cv2")
    binar = lambda ms: [np.where(m > 127, 1, 0).astype(np.uint8) for m in ms]  # as VACE's gradio tab
    m1 = binar(read_video_frames(ff_mask, use_type="cv2"))
    m2 = [np.ones((H, W, 3), np.uint8)] * WAN_N                   # control task: mask all white
    vid, mask = CompositionAnnotator({}).forward("frameref", "depth", v1, v2, m1, m2)
    save_mp4(os.path.join(out, "src_video.mp4"), [np.asarray(f, np.uint8) for f in vid], WAN_FPS)
    save_mp4(os.path.join(out, "src_mask.mp4"), [np.asarray(m, np.uint8) for m in mask], WAN_FPS)


def build_vace_c0(poc, vace_depth, H, W):
    """VACE plain depth control, no first frame: src_video = the depth video, src_mask all white."""
    import shutil
    out = os.path.join(poc, "C0", "inputs")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(vace_depth, os.path.join(out, "src_video.mp4"))
    save_mp4(os.path.join(out, "src_mask.mp4"), [np.full((H, W, 3), 255, np.uint8)] * WAN_N, WAN_FPS)


# Box colour per object type: 'drone', 'army tank' and 'truck' in VACE-Annotators layout/ram_tag_color_list.txt.
TAG_RGB = {"quadcopter": (201, 103, 188), "tank": (67, 213, 201), "truck": (67, 42, 201)}


def build_vace_bbox(bundle, out, idx, H, W, frame0, types, until=None, pad=0):
    """First frame + box layout, VACE layout_track (bboxtrack) format composed like move_anything.

    layout_track draws the external contour of each tracked mask, 3 px, in the label's RAM-tag
    colour on white. Our masks are the simulator boxes (bboxtrack + a 'bbox' mask_cfg gives the
    same). Each drone is drawn on its own and the canvases are merged, so the two boxes stay
    separate where they overlap. Frame 0 is the real frame; mask black on frame 0, white after.
    A fully hidden object has no box on that frame. `types` maps instance id -> asset type (box colour).
    `until`: boxes are drawn up to this output frame (inclusive) and the later frames are gray, VACE's
    "no control" value. Use a multiple of 4, so the cut falls between two of the model's latent frames.
    `pad`: grow every box by this many pixels on each side. The simulator's box is tight around the rendered
    object; a little room keeps the model from clipping parts (rotor tips) that stick out in a realistic frame 0.
    """
    from annotators.layout import LayoutMaskAnnotator
    import cv2 as _cv2
    anno = LayoutMaskAnnotator({"USE_AUG": False})
    boxes = json.load(open(os.path.join(bundle, "boxes.json")))
    frames = []
    for n, i in enumerate(idx[1:], start=1):
        if until is not None and n > until:
            frames.append(np.full((H, W, 3), 127, np.uint8))
            continue
        canv = np.full((H, W, 3), 255, np.uint8)
        for oid, per in boxes.items():
            b = per[i]
            if b is None:
                continue
            m = np.zeros((H, W), np.uint8)
            m[max(b[1] - pad, 0):b[3] + 1 + pad, max(b[0] - pad, 0):b[2] + 1 + pad] = 1
            canv = np.minimum(canv, np.asarray(anno.forward(mask=m, color=TAG_RGB[types.get(oid, "quadcopter")]), np.uint8))
        frames.append(canv)
    f0 = load_frame0(frame0, H, W)
    os.makedirs(out, exist_ok=True)
    save_mp4(os.path.join(out, "src_video.mp4"), [f0] + frames, WAN_FPS)
    save_mp4(os.path.join(out, "src_mask.mp4"),
             [np.zeros((H, W, 3), np.uint8)] + [np.full((H, W, 3), 255, np.uint8)] * len(frames), WAN_FPS)


def build_wanmove(bundle, poc, tl, depth, inst, idx, min_cover=0.35):
    """Back-project frame-0 mask pixels per drone, one per latent cell, carry them rigidly."""
    K = np.asarray(tl["camera"]["K"])
    c2w = np.asarray(tl["camera"]["cam2world_blender"])
    T, H, W = depth.shape
    sy = WANMOVE_HW[0] / H                                    # cells are 8x8 in Wan-Move's resized frame
    tracks, vis, owner = [], [], []
    for o in tl["objects"]:
        oid = o["instance_id"]
        mask = inst[0] == oid
        ys, xs = np.nonzero(mask)
        cells = {}
        for y, x in zip(ys, xs):
            cells.setdefault((int(x // LATENT_STRIDE), int(y * sy // LATENT_STRIDE)), []).append((y, x))
        chosen = []
        for thr in (min_cover, 0.25, 0.0):                    # relax coverage if the drone is tiny
            chosen = []
            for (cx, cy), pix in cells.items():
                area = LATENT_STRIDE * LATENT_STRIDE / sy
                if len(pix) / area < thr:
                    continue
                cc = np.array([(cx + 0.5) * LATENT_STRIDE, (cy + 0.5) * LATENT_STRIDE / sy])
                p = np.array(pix)
                y, x = p[np.argmin(((p[:, 1] - cc[0]) ** 2 + (p[:, 0] - cc[1]) ** 2))]
                chosen.append((y, x))
            if len(chosen) >= 6:
                break
        # back-project pixel centres with the rendered planar depth (OpenCV camera -> Blender -> world)
        yx = np.array(chosen, float)
        z = depth[0, yx[:, 0].astype(int), yx[:, 1].astype(int)]
        xc = (yx[:, 1] - K[0, 2]) / K[0, 0] * z
        yc = (yx[:, 0] - K[1, 2]) / K[1, 1] * z
        p_blcam = np.stack([xc, -yc, -z], -1)
        p_world = p_blcam @ c2w[:3, :3].T + c2w[:3, 3]
        mats = np.asarray(o["local2world"])
        local = (p_world - mats[0, :3, 3]) @ mats[0, :3, :3]   # inverse of a rigid transform
        world_t = np.einsum("tij,nj->tni", mats[idx, :3, :3], local) + mats[idx, None, :3, 3]
        uv, zt = project(K, c2w, world_t)                       # [81, n, 2], [81, n]
        ui = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1)
        vi = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
        fr = np.broadcast_to(idx[:, None], ui.shape)
        inb = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
        v = inb & (inst[fr, vi, ui] == oid) & (np.abs(zt - depth[fr, vi, ui]) < 0.08)
        v[0] = True                                             # by construction; Wan-Move requires it
        tracks.append(uv); vis.append(v); owner += [oid] * len(chosen)
    tracks = np.concatenate(tracks, 1).astype(np.float32)
    vis = np.concatenate(vis, 1)
    out = os.path.join(poc, "E", "inputs")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "tracks.npy"), tracks)
    np.save(os.path.join(out, "visibility.npy"), vis)
    info = {"shape": list(tracks.shape), "points_per_object": {str(i): owner.count(i) for i in set(owner)},
            "object_id": owner, "visible_frac": float(vis.mean()), "bundle_frames": idx.tolist()}
    json.dump(info, open(os.path.join(out, "points.json"), "w"), indent=1)
    print("[wanmove]", {k: info[k] for k in ("shape", "points_per_object", "visible_frac")})
    return tracks, vis, owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("poc")
    ap.add_argument("--only", nargs="*", default=["vace", "cosmos", "B", "C", "E"])
    ap.add_argument("--frame0", help="image used as frame 0 for B, Bbox and P1 (default: the bundle's frame0.png)")
    ap.add_argument("--suffix", default="", help="added to the B, Bbox and P1 folder names, e.g. _edit")
    ap.add_argument("--box_pad", type=int, default=0, help="pixels added on each side of every box in Bbox and P1")
    a = ap.parse_args()
    frame0 = a.frame0 or os.path.join(a.bundle, "frame0.png")
    cond_dir = lambda c: os.path.join(a.poc, c + a.suffix, "inputs")
    sys.path.insert(0, os.path.join(VACE_DIR, "vace"))
    tl = json.load(open(os.path.join(a.bundle, "timeline.json")))
    depth = np.load(os.path.join(a.bundle, "depth.npy"))
    inst = np.load(os.path.join(a.bundle, "instance.npy"))
    T, H, W = depth.shape
    idx = wan_indices(tl["fps"], T)
    cdir = os.path.join(a.bundle, "controls")
    vace_depth = os.path.join(cdir, "vace_depth.mp4")
    if "vace" in a.only:
        save_mp4(vace_depth, inverse_depth_u8(depth[idx], per_frame=True), WAN_FPS)
    if "cosmos" in a.only:
        save_mp4(os.path.join(cdir, "cosmos_depth.mp4"), inverse_depth_u8(depth, per_frame=False), tl["fps"])
    types = {str(o["instance_id"]): o.get("type", "quadcopter") for o in tl["objects"]}
    if "B" in a.only:
        build_vace_b(frame0, cond_dir("B"), H, W)
    if "C" in a.only:
        build_vace_c(a.bundle, a.poc, vace_depth, H, W)
    if "C0" in a.only:
        build_vace_c0(a.poc, vace_depth, H, W)
    if "Bbox" in a.only:
        build_vace_bbox(a.bundle, cond_dir("Bbox"), idx, H, W, frame0, types, pad=a.box_pad)
    if "P1" in a.only:
        build_vace_bbox(a.bundle, cond_dir("P1"), idx, H, W, frame0, types, until=16, pad=a.box_pad)
    if "E" in a.only:
        build_wanmove(a.bundle, a.poc, tl, depth, inst, idx)


if __name__ == "__main__":
    main()
