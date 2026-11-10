"""Labelled side-by-side grid video of every condition (plan §9 video grid, PoC version).

  envs/eval/.venv/bin/python eval/grid.py <poc_scene_dir> [--scale 0.75] [--cols 4]

Every clip is resampled to 81 frames at 16 fps over the same time window by timestamp (frame i <-
source frame round(i * fps_src / 16)), resized to the bundle's 832x480 (Wan-Move's 832x464 output is
padded, not stretched) and tiled. Clips that don't exist yet are shown as a "still running" tile.
Writes <poc_scene_dir>/grid.mp4 and grid_frames.png (frames 0, 40, 80 of every tile).
"""

import argparse
import os
import subprocess

import cv2
import numpy as np

N, FPS, W, H = 81, 16, 832, 480


def read(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return np.stack(frames), fps


def fit(f):
    h, w = f.shape[:2]
    if (w, h) == (W, H):
        return f
    if w == W and h < H:                      # pad (Wan-Move 832x464)
        top = (H - h) // 2
        return cv2.copyMakeBorder(f, top, H - h - top, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)


def resample(frames, fps):
    idx = np.round(np.arange(N) * fps / FPS).astype(int)
    if idx[-1] >= len(frames):
        raise ValueError(f"{len(frames)} frames at {fps} fps don't cover {N / FPS:.2f} s")
    return np.stack([fit(frames[i]) for i in idx])


def label(tile, text, sub=""):
    bar = np.full((34, tile.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (tile.shape[1] - 8 - 7 * len(sub), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return np.concatenate([bar, tile], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poc")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    P = lambda *p: os.path.join(a.poc, *p)
    tiles = [
        ("R  raw render", "simulator", P("bundle", "rgb.mp4")),
        ("depth control (VACE fmt)", "input to C, C0", P("bundle", "controls", "vace_depth.mp4")),
        ("A  VACE-14B text", "", P("A", "14B-bf16", "out_video.mp4")),
        ("A+  VACE-14B detailed text", "path in words", P("Aplus", "14B-bf16", "out_video.mp4")),
        ("B  VACE-14B text+frame0", "", P("B", "14B-bf16", "out_video.mp4")),
        ("B-box  VACE-14B frame0+boxes", "layout_track", P("Bbox", "14B-bf16", "out_video.mp4")),
        ("C  VACE-14B frame0+depth", "", P("C", "14B-bf16", "out_video.mp4")),
        ("C0  VACE-14B depth+text", "no frame0", P("C0", "14B-bf16", "out_video.mp4")),
        ("D  Cosmos Sim-Transfer depth+text", "", P("D", "D_seed0", "vision.mp4")),
        ("E  Wan-Move frame0+tracks", "832x464 padded", P("E", "seed0.mp4")),
        ("R + simulator tracks", "", P("bundle", "overlay.mp4")),
    ]
    tw, th = int(W * a.scale) // 2 * 2, int(H * a.scale) // 2 * 2
    vids, missing = [], []
    for name, sub, path in tiles:
        if os.path.exists(path):
            fr, fps = read(path)
            v = resample(fr, fps)
            print(f"{name}: {path} ({len(fr)} frames @ {fps:.2f} fps, {fr.shape[2]}x{fr.shape[1]})")
        else:
            v = np.zeros((N, H, W, 3), np.uint8)
            cv2.putText(v[0], "still running", (W // 2 - 150, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (200, 200, 200), 2, cv2.LINE_AA)
            v[:] = v[0]
            missing.append(name)
        vids.append(np.stack([label(cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA), name, sub) for f in v]))
    while len(vids) % a.cols:
        vids.append(np.zeros_like(vids[0]))
    rows = [np.concatenate(vids[r:r + a.cols], 2) for r in range(0, len(vids), a.cols)]
    grid = np.concatenate(rows, 1)                             # [N, Hg, Wg, 3] BGR
    for t in range(N):
        cv2.putText(grid[t], f"{t / FPS:.2f}s", (grid.shape[2] - 80, grid.shape[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = P("grid.mp4")
    Hg, Wg = grid.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{Wg}x{Hg}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(grid).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    cv2.imwrite(P("grid_frames.png"), np.concatenate([grid[0], grid[40], grid[80]], 0))
    print("wrote", out, f"{Wg}x{Hg}", "missing:", missing or "none")


if __name__ == "__main__":
    main()
