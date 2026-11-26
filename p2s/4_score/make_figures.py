"""Figures of the per-camera scores (Eval2), drawn in the house style of the paper figure guide (paperplot).
Runs nothing on the GPU: score the runs with p2s/4_score/run_eval.py first.

  $S2R_ROOT/envs/eval/.venv/bin/python p2s/4_score/make_figures.py --exp legacy/configs/experiments/multicam_v1_2.yaml \
      --scene intersection_4way --cond D --seed 0 --out docs/figures/<name>/intersection [--frame 38]
  $S2R_ROOT/envs/eval/.venv/bin/python p2s/4_score/make_figures.py --table docs/figures/<name>/scores_by_control.csv --out docs/figures/<name>

With --exp, one row per camera view (PDF for the paper + JPEG preview each):
  depth      generated frame | rendered depth | depth estimated from the video | where the two agree
  masks      generated frame | rendered object masks | the masks found in the video over the rendered outlines
  tracks     generated frame with each simulator surface point (ring) and where the tracker finds it (dot)
With --table (a CSV from p2s/4_score/summarize.py --out, kept next to the figure so every number can be redrawn):
  scores_by_control   one panel per score; rows are the control given to the model, grouped by model

The guide is the repo shourovj/paper-figure-guide, expected at docs/docs/paper-figure-guide (not tracked here).
Names shown in a figure come from the tables below, so no raw identifier reaches a label.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GUIDE = os.path.join(REPO, "docs", "docs", "paper-figure-guide")
if not os.path.exists(os.path.join(GUIDE, "paperplot.py")):
    sys.exit(f"missing {GUIDE}/paperplot.py: clone github.com/shourovj/paper-figure-guide into docs/docs/")
sys.path.insert(0, GUIDE)
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, os.path.join(REPO, "p2s", "3_generate", "common"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paperplot as pp  # noqa: E402
import multicam_common as mc  # noqa: E402
import metrics as M  # noqa: E402
from reference import load_view, scored  # noqa: E402
from video_io import read_video  # noqa: E402

# --- names and colour roles: one per model, the same in every figure (guide section 4)
MODELS = {   # model.name in an experiment file -> (name shown, paperplot role)
    "simulator render": ("Simulator render (reference)", "base"),
    "cosmos3-nano": ("Cosmos3-Nano", "a"),
    "cosmos3-nano-sim-transfer": ("Cosmos3-Nano-Sim-Transfer", "b"),
    "wan22-vace-fun-a14b": ("Wan2.2-VACE-Fun", "c"),
    "ltx25-distilled": ("LTX-2.5", "d"),
}
CONTROLS = {  # control label from p2s/4_score/summarize.py -> name shown, in order from the densest control to none
    "(reference: one world, by construction)": "the render, scored as it is",
    "the render itself": "the render itself",
    "depth + seg": "depth + segmentation",
    "depth": "depth",
    "seg": "segmentation",
    "edge": "edges",
    "text only": "text only",
}
SCORES = {"depth": "depth accuracy", "mask_iou": "mask IoU", "track": "track accuracy", "eval2": "Eval2 score"}
CAMERAS = {"cctv": "CCTV camera", "dashcam": "dashcam", "phone": "phone camera", "wrist": "wrist camera", "side": "side camera"}
OBJECT_COLOURS = ["#C9302C", "#2E7BD6", "#BF8700", "#A23BA6", "#159E86", "#8C8C8C", "#E07B39", "#5B8C2A", "#3D5A9A"]


def rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def frame_grid(rows, col_titles, row_names, notes, stem, col_h=1.02):
    """A grid of frames: column titles on the top row, the camera's name beside each row, an optional note under a tile."""
    n, m = len(rows), len(rows[0])
    fig, axes = plt.subplots(n, m, figsize=pp.size("double", h=col_h * n + 0.25), squeeze=False)
    for r in range(n):
        for c in range(m):
            ax = axes[r][c]
            ax.imshow(rows[r][c], interpolation="lanczos")
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(row_names[r])
            if notes and notes[r][c]:
                ax.set_xlabel(notes[r][c], fontsize=7)
    save_frames(fig, stem)


def save_frames(fig, stem):
    """A figure of video frames: the PDF for the paper (with the guide's checks) and a JPEG preview for the docs,
    which is a fraction of a PNG's size for photographs (guide section 11)."""
    pp.save(fig, stem, png=False)
    fig.savefig(stem + ".jpg", dpi=300, pil_kwargs={"quality": 88})
    plt.close(fig)
    print("wrote", stem + ".jpg")


def clip_figures(exp, scene, cond, seed, out, frame):
    jobs = [j for j in mc.jobs(exp) if (j["scene"], j["cond"], j["seed"], j["kind"]) == (scene, cond, seed, "cam")]
    assert jobs, f"no camera views for {scene} {cond} seed {seed} in this experiment"
    depth_rows, depth_notes, mask_rows, mask_notes, track_tiles, names = [], [], [], [], [], []
    for j in jobs:
        v = load_view(exp, j)
        d = mc.run_dir(exp, j)
        est = os.path.join(d, "est")
        assert os.path.exists(os.path.join(est, "depth.npy")), f"{est} has no reader outputs: run p2s/4_score/run_eval.py first"
        gen = read_video(os.path.join(d, "out_video.mp4"), (v.W, v.H))
        t = frame
        names.append(CAMERAS.get(j["view"], j["view"]))
        ids = [o["instance_id"] for o in v.objects if scored(v, o, t)]
        colour = {i: rgb(OBJECT_COLOURS[k % len(OBJECT_COLOURS)]) for k, i in enumerate(sorted(o["instance_id"] for o in v.objects))}
        # depth
        e = np.asarray(np.load(os.path.join(est, "depth.npy"), mmap_mode="r")[t], np.float32)
        kind = json.load(open(os.path.join(est, "depth.json")))["kind"]
        ref = v.depth[t]
        valid, ok = M.depth_ok(e, ref, kind)
        aligned = M.align_depth(e.astype(np.float64), ref.astype(np.float64), valid, kind)
        top = np.percentile(1 / ref[valid], 99)
        grey = lambda x: np.repeat((255 * np.clip(x / top, 0, 1)).astype(np.uint8)[..., None], 3, -1)
        verdict = np.full((v.H, v.W, 3), 235, np.uint8)
        verdict[ok], verdict[valid & ~ok] = rgb("#7FB7A6"), rgb("#C9302C")
        depth_rows.append([gen[t], grey(np.where(valid, 1 / ref, 0)), grey(np.where(valid, 1 / np.maximum(aligned, 1e-6), 0)), verdict])
        depth_notes.append(["", "", "", f"{100 * ok.sum() / valid.sum():.0f}% of pixels agree"])
        # masks
        z = np.load(os.path.join(est, "masks.npz"))
        ref_im = np.full((v.H, v.W, 3), 245, np.uint8)
        over, ious = gen[t].astype(np.float32), []
        for i in ids:
            ref_im[v.inst[t] == i] = colour[i]
            if str(i) not in z.files:
                continue
            m = np.unpackbits(z[str(i)])[: v.T * v.H * v.W].reshape(v.T, v.H, v.W)[t].astype(bool)
            over[m] = 0.45 * over[m] + 0.55 * colour[i]
            if (v.inst[t] == i).sum() >= 100:
                ious.append(M.mask_iou(m, v.inst[t] == i))
        over = over.astype(np.uint8)
        for i in ids:
            c, _ = cv2.findContours((v.inst[t] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, c, -1, (255, 255, 255), 3)
        mask_rows.append([gen[t], ref_im, over])
        mask_notes.append(["", "", f"mask IoU {np.mean(ious):.2f} in this frame" if ious else ""])
        # tracks
        tz = np.load(os.path.join(est, "tracks.npz"))
        tr = [x for x in v.tracks if len(x["points"])]
        im = gen[t].copy()
        if tr:
            ref_p, vis = np.concatenate([x["points"] for x in tr]), np.concatenate([x["visible"] for x in tr])
            q, obj = np.concatenate([x["query_t"] for x in tr]), np.concatenate([x["object_id"] for x in tr])
            for k in range(len(ref_p)):
                if not (vis[k, t] and t > q[k]):
                    continue
                col = tuple(int(c) for c in colour[int(obj[k])])
                for s in range(max(int(q[k]), t - 14), t):
                    cv2.line(im, tuple(tz["points"][k, s].astype(int)), tuple(tz["points"][k, s + 1].astype(int)), col, 3, cv2.LINE_AA)
                cv2.circle(im, tuple(ref_p[k, t].astype(int)), 9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(im, tuple(tz["points"][k, t].astype(int)), 5, col, -1, cv2.LINE_AA)
        track_tiles.append(im)
    pp.use("cvpr")
    frame_grid(depth_rows, ["generated video", "rendered depth", "depth estimated from the video", "where the two agree"],
               names, depth_notes, os.path.join(out, "depth"))
    frame_grid(mask_rows, ["generated video", "rendered object masks", "found (fill) and rendered (outline)"],
               names, mask_notes, os.path.join(out, "masks"), col_h=1.36)
    fig, axes = plt.subplots(1, len(track_tiles), figsize=pp.size("double", h=1.72), squeeze=False)
    for ax, im, name in zip(axes[0], track_tiles, names):
        ax.imshow(im, interpolation="lanczos")
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
        ax.set_title(name)
    from matplotlib.lines import Line2D
    grey = pp.ROLES["base"].color
    pp.legend_below(fig, [Line2D([], [], ls="none", marker="o", markerfacecolor="white", markeredgecolor=pp.INK, markeredgewidth=0.9, markersize=5.5,
                                 label="simulator point (ring)"),
                          Line2D([], [], ls="-", lw=1.4, marker="o", markersize=3.5, color=grey,
                                 label="the same point found in the video, with its last 14 frames (one colour per object)")])
    save_frames(fig, os.path.join(out, "tracks"))


def chart(table, out):
    """The claim: with one structure control a clip follows the simulation as closely as the simulator's own render does,
    and with text alone it does not. Horizontal bars from zero, one panel per score with shared limits, rows grouped
    by model, one colour and hatch per model."""
    pp.use("cvpr")
    rows = [r for r in csv.DictReader(open(table)) if r["control"] in CONTROLS and r["model"] in MODELS]
    order = list(CONTROLS)
    groups = []                                             # models in the order of MODELS, controls densest first
    for model in MODELS:
        g = sorted((r for r in rows if r["model"] == model), key=lambda r: order.index(r["control"]))
        if g:
            groups.append((model, g))
    y, ticks, pos = 0.0, [], []
    for model, g in groups:
        for r in g:
            pos.append((y, r, MODELS[model][1]))
            ticks.append((y, CONTROLS[r["control"]]))
            y += 1.0
        y += 0.55                                           # a gap between two models' rows
    fig, axes = plt.subplots(1, len(SCORES), figsize=pp.size("double", h=0.21 * len(pos) + 0.95), sharey=True)
    for ax, (key, name) in zip(axes, SCORES.items()):
        for yy, r, role in pos:
            val = float(r[key])
            pp.bar(ax, yy, val, 0.74, role, horizontal=True)
            ax.text(val + 0.025, yy, f"{val:.2f}", va="center", ha="left", fontsize=7)
        ax.set(xlim=(0, 1.22), xticks=[0, 0.5, 1.0], xlabel=name)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    axes[0].set_ylim(max(p[0] for p in pos) + 0.62, -0.62)
    axes[0].set_ylabel("control given to the model")
    pp.legend_below(fig, [pp.key(MODELS[m][1], MODELS[m][0], "patch") for m, _ in groups])
    stem = os.path.join(out, "scores_by_control")
    pp.save(fig, stem)
    plt.close(fig)
    print("wrote", stem + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--scene")
    ap.add_argument("--cond")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame", type=int, default=38)
    ap.add_argument("--table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.exp:
        exp = mc.load_exp(a.exp if os.path.isabs(a.exp) else os.path.join(mc.REPO, a.exp))
        clip_figures(exp, a.scene, a.cond, a.seed, a.out, a.frame)
    if a.table:
        chart(a.table, a.out)


if __name__ == "__main__":
    main()
