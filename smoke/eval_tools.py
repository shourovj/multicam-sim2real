"""Smoke test for the eval tools: SAM 2.1 (video masks) and CoTracker3 offline (point tracks).

Run with the shared eval env:

    $S2R_ROOT/envs/eval/.venv/bin/python smoke/eval_tools.py \
        --video $S2R_ROOT/co-tracker/assets/apple.mp4 \
        --out $S2R_DATA/smoke/eval

Outputs in --out:
    sam2_masks.npz       masks [T, O, H, W] bool, obj_ids [O]
    sam2_overlay.mp4
    cotracker_tracks.npz tracks [N, T, 2] float32 (x, y pixels), visibility [N, T] bool, queries [N, 3] (t, x, y)
    cotracker_overlay.mp4
    timing.json          wall time and peak CUDA memory per tool

--video may be an .mp4 or a directory of JPEG frames named <int>.jpg (e.g. sam2/notebooks/videos/bedroom).
The default prompt (box + one positive click) fits co-tracker/assets/apple.mp4 (1296x720, 50 frames).
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(os.environ.get("S2R_ROOT", Path.home() / "code" / "sim2real-videogen"))
SAM2_CKPT = ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
COTRACKER_CKPT = ROOT / "checkpoints" / "cotracker" / "scaled_offline.pth"

# Default prompt for co-tracker/assets/apple.mp4 (from assets/apple_mask.png): bbox and its centroid.
DEFAULT_BOX = [318, 129, 642, 453]
DEFAULT_POINTS = [[480, 291]]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help=".mp4 file or directory of JPEG frames")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda:0", help="torch device (default cuda:0)")
    p.add_argument("--tools", default="sam2,cotracker", help="comma list of tools to run: sam2,cotracker")
    p.add_argument("--max-frames", type=int, default=100, help="use at most this many frames from the start")
    p.add_argument("--fps", type=float, default=None, help="overlay fps (default: source fps or 10)")
    # SAM 2 prompt on frame 0. Pass --box none and/or --points none to drop the default for that prompt type.
    p.add_argument("--points", default=None,
                   help="positive clicks on frame 0 as 'x,y;x,y' (pixels). Default: apple centroid. 'none' to disable")
    p.add_argument("--neg-points", default=None, help="negative clicks on frame 0 as 'x,y;x,y'")
    p.add_argument("--box", default=None, help="box on frame 0 as 'x0,y0,x1,y1'. Default: apple bbox. 'none' to disable")
    # CoTracker queries.
    p.add_argument("--track-mode", choices=["grid", "mask"], default="grid",
                   help="grid: grid_size x grid_size grid on frame 0; mask: --num-points random points inside the "
                        "frame-0 SAM 2 mask (needs sam2 in --tools)")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--num-points", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_xy_list(s):
    return [[float(v) for v in pt.split(",")] for pt in s.split(";") if pt.strip()]


def load_frames(path, max_frames):
    """Returns uint8 frames [T, H, W, 3] (RGB), fps, and a JPEG dir usable by SAM 2 (or None)."""
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted((f for f in path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")), key=lambda f: int(f.stem))
        files = files[:max_frames]
        frames = np.stack([imageio.imread(f)[..., :3] for f in files])
        # SAM 2 reads every JPEG in the dir, so reuse it only if we keep all of them.
        n_total = sum(1 for f in path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg"))
        return frames, None, (path if len(files) == n_total else None)
    reader = imageio.get_reader(str(path))
    fps = reader.get_meta_data().get("fps")
    frames = []
    for i, fr in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(fr[..., :3])
    reader.close()
    return np.stack(frames), fps, None


def write_jpeg_dir(frames, d):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.jpg"):
        f.unlink()
    for i, fr in enumerate(frames):
        cv2.imwrite(str(d / f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return d


def write_mp4(path, frames, fps):
    # macro_block_size=1 avoids resizing odd-sized frames.
    imageio.mimwrite(str(path), list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_gib(device):
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def fmt_mem(m):
    return f"{m:.2f} GiB" if m is not None else "n/a (cpu)"


PALETTE = np.array([[255, 64, 64], [64, 160, 255], [64, 255, 128], [255, 200, 0], [200, 64, 255]], dtype=np.float32)


def run_sam2(args, frames, jpeg_dir, device, out):
    from sam2.build_sam import build_sam2_video_predictor

    points = DEFAULT_POINTS if args.points is None else ([] if args.points == "none" else parse_xy_list(args.points))
    neg = parse_xy_list(args.neg_points) if args.neg_points else []
    box = DEFAULT_BOX if args.box is None else (None if args.box == "none" else [float(v) for v in args.box.split(",")])
    if not points and not neg and box is None:
        raise SystemExit("SAM 2 needs at least one click or a box")

    reset_peak(device)
    t0 = time.perf_counter()
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)
    cuda_sync(device)
    t_load = time.perf_counter() - t0

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    T, H, W = frames.shape[:3]
    masks = np.zeros((T, 1, H, W), dtype=bool)
    t0 = time.perf_counter()
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=str(jpeg_dir))
        pts = points + neg
        kw = {}
        if pts:
            kw["points"] = np.array(pts, dtype=np.float32)
            kw["labels"] = np.array([1] * len(points) + [0] * len(neg), dtype=np.int32)
        if box is not None:
            kw["box"] = np.array(box, dtype=np.float32)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, **kw)
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            masks[fidx] = (logits > 0.0).cpu().numpy()[:, 0]
    cuda_sync(device)
    t_run = time.perf_counter() - t0
    mem = peak_gib(device)

    np.savez_compressed(out / "sam2_masks.npz", masks=masks, obj_ids=np.array([1]),
                        prompt_points=np.array(pts, dtype=np.float32).reshape(-1, 2),
                        prompt_labels=np.array([1] * len(points) + [0] * len(neg)),
                        prompt_box=np.array(box if box is not None else [], dtype=np.float32))
    overlay = frames.astype(np.float32).copy()
    for o in range(masks.shape[1]):
        m = masks[:, o]
        overlay[m] = 0.5 * overlay[m] + 0.5 * PALETTE[o % len(PALETTE)]
    overlay = overlay.astype(np.uint8)
    for t in range(T):
        for o in range(masks.shape[1]):
            cnts, _ = cv2.findContours(masks[t, o].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay[t], cnts, -1, (255, 255, 255), 2)
    for x, y in points:
        cv2.drawMarker(overlay[0], (int(x), int(y)), (0, 255, 0), cv2.MARKER_STAR, 20, 2)
    for x, y in neg:
        cv2.drawMarker(overlay[0], (int(x), int(y)), (255, 0, 0), cv2.MARKER_STAR, 20, 2)
    if box is not None:
        cv2.rectangle(overlay[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    write_mp4(out / "sam2_overlay.mp4", overlay, args.fps)

    area = masks[:, 0].reshape(T, -1).mean(1)
    print(f"[sam2] frames={T} mask area frac: frame0={area[0]:.4f} min={area.min():.4f} max={area.max():.4f} "
          f"empty_frames={(area == 0).sum()}")
    del predictor, state
    return masks, {"load_s": t_load, "run_s": t_run, "peak_mem_gib": mem}


def run_cotracker(args, frames, device, out, mask0=None):
    from cotracker.predictor import CoTrackerPredictor

    reset_peak(device)
    t0 = time.perf_counter()
    model = CoTrackerPredictor(checkpoint=str(COTRACKER_CKPT), offline=True, window_len=60).to(device)
    cuda_sync(device)
    t_load = time.perf_counter() - t0

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T 3 H W, 0..255
    T = frames.shape[0]
    t0 = time.perf_counter()
    if args.track_mode == "mask":
        if mask0 is None or not mask0.any():
            raise SystemExit("--track-mode mask needs a non-empty frame-0 SAM 2 mask (include sam2 in --tools)")
        ys, xs = np.nonzero(mask0)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(xs), size=min(args.num_points, len(xs)), replace=False)
        q = np.stack([np.zeros(len(idx)), xs[idx], ys[idx]], 1).astype(np.float32)
        queries = torch.from_numpy(q)[None].to(device)
        tracks, vis = model(video, queries=queries)
    else:
        tracks, vis = model(video, grid_size=args.grid_size)
        queries = None
    cuda_sync(device)
    t_run = time.perf_counter() - t0
    mem = peak_gib(device)

    tracks = tracks[0].permute(1, 0, 2).float().cpu().numpy()  # N T 2
    vis = vis[0].permute(1, 0).cpu().numpy().astype(bool)  # N T
    q = queries[0].cpu().numpy() if queries is not None else np.concatenate(
        [np.zeros((tracks.shape[0], 1), np.float32), tracks[:, 0]], 1)
    np.savez_compressed(out / "cotracker_tracks.npz", tracks=tracks, visibility=vis, queries=q)

    overlay = frames.copy()
    N = tracks.shape[0]
    colors = (cv2.applyColorMap(np.linspace(0, 255, max(N, 1)).astype(np.uint8)[:, None], cv2.COLORMAP_HSV)[:, 0, ::-1]).tolist()
    for t in range(T):
        for n in range(N):
            x, y = tracks[n, t]
            c = tuple(int(v) for v in colors[n])
            if t > 0:  # short trail
                for s in range(max(1, t - 8), t + 1):
                    p0, p1 = tracks[n, s - 1], tracks[n, s]
                    cv2.line(overlay[t], (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), c, 1, cv2.LINE_AA)
            cv2.circle(overlay[t], (int(x), int(y)), 4, c, -1 if vis[n, t] else 1, cv2.LINE_AA)
    write_mp4(out / "cotracker_overlay.mp4", overlay, args.fps)
    print(f"[cotracker] tracks {tracks.shape} visibility {vis.shape} mean visible={vis.mean():.3f}")
    del model
    return {"load_s": t_load, "run_s": t_run, "peak_mem_gib": mem, "num_points": int(N)}


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()  # reset_peak_memory_stats raises "Invalid device argument" before CUDA init
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    frames, fps, jpeg_dir = load_frames(args.video, args.max_frames)
    args.fps = args.fps or fps or 10.0
    print(f"video {args.video}: {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]} fps={args.fps} "
          f"device={device} torch={torch.__version__}")

    timing = {}
    mask0 = None
    if "sam2" in tools:
        if jpeg_dir is None:
            jpeg_dir = write_jpeg_dir(frames, out / "frames_jpg")
        masks, timing["sam2"] = run_sam2(args, frames, jpeg_dir, device, out)
        mask0 = masks[0, 0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if "cotracker" in tools:
        timing["cotracker"] = run_cotracker(args, frames, device, out, mask0)

    for k, v in timing.items():
        print(f"[{k}] load {v['load_s']:.2f}s  run {v['run_s']:.2f}s  "
              f"torch.cuda.max_memory_allocated {fmt_mem(v['peak_mem_gib'])}")
    (out / "timing.json").write_text(json.dumps(
        {"video": str(args.video), "frames": int(frames.shape[0]), "hw": list(frames.shape[1:3]),
         "device": str(device), "torch": torch.__version__, "tools": timing}, indent=2))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
