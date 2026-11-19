"""Control videos from one camera's render (numpy + OpenCV): what a video model can be conditioned on, all exact,
because the simulator knows the geometry. Written by sim/world/export.py into <bundle>/controls/, at the bundle's fps.

  depth.mp4     inverse depth, near bright, sky black, one min-max scale for the whole clip (Cosmos / Round 2 convention)
  seg.mp4       one flat colour per instance id: moving objects (1-99) and static parts (100+: road, sidewalk, each
                building, each signal, props), sky black. Colours are fixed per id, so an object has the same colour in
                every camera; Cosmos trained on arbitrary mask colours, so the colours are identities, not appearance.
  edge.mp4      geometry edges only, white on black: object boundaries, depth steps and creases (normal changes).
                No texture edges, unlike Canny on a textured render.
  normals.mp4   surface normals as colour ((n + 1) / 2), for models or checks that use them
  rgb.mp4       the render itself (textured or clay, per the spec's render.style)
"""

import subprocess

import cv2
import numpy as np

CONTROLS = ("depth", "seg", "edge", "normals", "rgb")


def palette(n=256, seed=7):
    """Fixed id -> colour table: well-separated hues for objects, softer colours for static parts, black for sky (0)."""
    rng = np.random.default_rng(seed)
    pal = np.zeros((n, 3), np.uint8)
    for i in range(1, n):
        h = (i * 0.61803398875) % 1.0
        s, v = (0.85, 0.95) if i < 100 else (rng.uniform(0.35, 0.6), rng.uniform(0.55, 0.85))
        hsv = np.uint8([[[int(h * 179), int(s * 255), int(v * 255)]]])
        pal[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return pal


PALETTE = palette()


def depth_control(depth):
    sky = ~np.isfinite(depth)
    inv = np.where(sky, 0.0, 1.0 / np.where(sky, 1.0, np.maximum(depth, 1e-3)))
    v = inv[~sky]
    out = (inv - v.min()) / max(v.max() - v.min(), 1e-9)
    out[sky] = 0.0
    u8 = np.clip(np.round(out * 255), 0, 255).astype(np.uint8)
    return np.repeat(u8[..., None], 3, -1)


def seg_control(inst):
    return PALETTE[inst]


def edge_control(depth, inst, normals=None, depth_rel=0.04, crease_deg=35.0, width=1):
    """Edges where the instance id changes, depth jumps by > depth_rel (relative), or the normal turns by > crease_deg."""
    T = len(depth)
    d = np.where(np.isfinite(depth), depth, 1e4).astype(np.float32)
    out = np.zeros(depth.shape, np.uint8)
    cos_lim = np.cos(np.deg2rad(crease_deg))
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8) if width > 0 else None
    for t in range(T):
        e = np.zeros(d[t].shape, bool)
        for ax in (0, 1):
            e |= np.diff(inst[t], axis=ax, prepend=inst[t].take([0], axis=ax)) != 0
            dd = np.abs(np.diff(np.log(d[t]), axis=ax, prepend=np.log(d[t]).take([0], axis=ax)))
            e |= dd > np.log1p(depth_rel)
            if normals is not None:
                n = normals[t].astype(np.float32)
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
                nb = np.roll(n, 1, axis=ax)
                e |= (n * nb).sum(-1) < cos_lim
        e[0, :] = e[:, 0] = False
        m = e.astype(np.uint8) * 255
        out[t] = cv2.dilate(m, k) if k is not None and width > 1 else m
    return np.repeat(out[..., None], 3, -1)


def normals_control(normals):
    n = normals.astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return np.clip(np.round((n + 1) / 2 * 255), 0, 255).astype(np.uint8)


def write_mp4(path, frames, fps, crf=12):
    T, H, W, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {path}"


def read_mp4(path):
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                          capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, H = int(info[0]), int(info[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def build_all(rgb, depth, inst, normals):
    out = {"depth": depth_control(depth), "seg": seg_control(inst), "edge": edge_control(depth, inst, normals), "rgb": rgb}
    if normals is not None:
        out["normals"] = normals_control(normals)
    return out


def sheet(controls, frames=(0, 56, 112), w=277, h=160):
    """One contact sheet: a row per control, a column per frame."""
    rows = []
    for name in CONTROLS:
        if name not in controls:
            continue
        v = controls[name]
        tiles = []
        for f in frames:
            im = cv2.resize(np.ascontiguousarray(v[min(f, len(v) - 1)]), (w, h), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{name} f{f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(im)
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate(rows, 0)
