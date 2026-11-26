"""Score an experiment's finished runs against the simulator: three scores per camera (Eval2) and two across
cameras (Eval3). Nothing here is specific to a scene: every reference comes from the camera bundles.

  source ~/.sim2real.env
  $S2R_ROOT/envs/eval/.venv/bin/python p2s/4_score/run_eval.py --exp legacy/configs/experiments/cvp_v1/cosmos3_sim_transfer.yaml \
      [--only scene=..,view=..,cond=..,seed=..] [--force] [--on-render]

Per camera, each in [0, 1] with 1 best (p2s/4_score/common/metrics.py):
  depth        share of pixels whose depth, estimated from the video, is within 1.25x of the rendered depth
  mask_iou     overlap of each object's mask in the video with its rendered mask
  track        share of simulator surface points the tracker finds where the simulator puts them (TAP-Vid delta_avg)
  eval2        the harmonic mean of the three
Across the separate camera views of one condition and seed, each in [0, 1] with 1 best:
  object_colour    share of (object, camera pair, instant) where the object's colour inside its rendered mask is the
                   same in both cameras (CIELAB distance under 20)
  surface_colour   share of pixel pairs that show the same surface point (the rendered depth and the cameras say which)
                   and have the same colour in both videos
Both count only what is in place: an object whose mask in the video overlaps its rendered mask (IoU >= 0.5) in both
cameras, a pixel pair whose estimated depth passes in both cameras. Elsewhere the simulator's masks and pixel pairs
do not point at the same thing in the video, and comparing colours there says nothing (a text-only clip would score
as consistent because the mask lands on road in both views). object_in_place and surface_in_place give the share
that was counted; a score from a small share describes little of the scene.
Next to each, as diagnostics: the mean colour distance, and a DINOv2 feature similarity with its references. On the
textured render, whose cameras truly show one world, colour separates "same" from "different" far better than the
features do (they follow the viewing angle more than the object), which is why colour is the score.

It runs in two steps. Readers turn each video into arrays, cached in <run folder>/est/ (p2s/4_score/readers/): depth
(Depth Anything 3, its own env), masks (SAM 2.1) and point tracks (CoTracker3). Then the metrics compare those with
the reference. A reader runs again only if the video or the reader changed, or with --force.
--on-render scores the simulator's own render instead of the generated videos: what the readers give on a clip that
follows the simulator exactly, which is the ceiling to read the per-camera scores against.

--on-render scores the cross-camera pair too: on an experiment whose scenes are textured renders (multicam_v1.yaml)
that is what cameras that truly show one world score, the reference for the two cross-camera numbers.

Writes <scene>/r<rev>/scores/<experiment>/: eval2.csv (one row per run), eval2_objects.csv, eval3.csv (one row per
scene, condition and seed), eval3_objects.csv, eval_summary.json. With --on-render the files end in _render.
A GPU with a few GB free is needed; set CUDA_VISIBLE_DEVICES, or the one with the most free memory is used.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "3_generate", "common"))
sys.path.insert(0, HERE)
import multicam_common as mc  # noqa: E402
import metrics as M  # noqa: E402
from gpus import free_gib  # noqa: E402
from reference import load_view, scored  # noqa: E402
from video_io import cache_mark, cache_ok, read_video  # noqa: E402

MIN_MASK_PX = 100        # an (object, frame) pair counts for mask_iou from this many rendered pixels
MIN_CROP_PX = 400        # an object is compared across cameras when it is at least this large in both
CROSS_STEP = 4           # cross-camera scores use every 4th frame
SURFACE_PIXELS = 20000   # shared pixels sampled per camera pair and frame
SAME_COLOUR = 20.0       # CIELAB distance below which two colours count as the same (about 2 is just visible)
IN_PLACE_IOU = 0.5       # an object is compared across cameras only where its mask in the video overlaps the rendered one this much
DEPTH_READER = ("depth_anything3", 1)


def video_of(exp, j, on_render):
    """The clip to score and the folder for its reader outputs."""
    if not on_render:
        d = mc.run_dir(exp, j)
        return os.path.join(d, "out_video.mp4"), os.path.join(d, "est")
    root = mc.scene_root(exp, j["scene"])
    base = os.path.join(root, "cams", j["view"], "bundle") if j["kind"] == "cam" else os.path.join(root, "cut", j["view"])
    return os.path.join(base, "rgb.mp4"), os.path.join(root, "est_render", j["view"])


def run_depth_reader(todo):
    """todo: [(video, est_dir, stamp, (W, H))]. One subprocess in the Depth Anything 3 env for all of them."""
    if not todo:
        return
    py = os.path.join(os.environ["S2R_ROOT"], "envs", "da3", ".venv", "bin", "python")
    assert os.path.exists(py), f"missing {py} (setup/README.md: eval readers)"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"video": v, "out": d, "stamp": s, "size": list(size)} for v, d, s, size in todo], f)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_LIBRARY_PATH")}
    subprocess.run([py, os.path.join(HERE, "readers", "depth_anything3", "run.py"), f.name], check=True, env=env)
    os.remove(f.name)


def crop(frame, mask):
    """The object's box (padded 10 %) with everything outside its mask greyed, as a 224 x 224 image."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int(0.1 * (x1 - x0)) + 2, int(0.1 * (y1 - y0)) + 2
    x0, x1, y0, y1 = max(x0 - px, 0), min(x1 + px, frame.shape[1]), max(y0 - py, 0), min(y1 + py, frame.shape[0])
    near = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    im = np.where(near[..., None], frame[y0:y1, x0:x1], 127).astype(np.uint8)
    return cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)


def lab(rgb):
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def score_run(run, views):
    """The three per-camera scores of one run, from its cached reader outputs."""
    j, video, est = run["job"], run["video"], run["est"]
    v = views[(j["scene"], j["view"])]
    row = {"run_id": run["run_id"], "scene": j["scene"], "view": j["view"], "kind": j["kind"], "cond": j["cond"], "seed": j["seed"]}
    objects = []
    # depth
    d = np.load(os.path.join(est, "depth.npy")).astype(np.float32)
    kind = json.load(open(os.path.join(est, "depth.json")))["kind"]
    ids = {o["instance_id"]: o for o in v.objects}
    region = np.zeros(v.inst.shape, bool)
    for t in range(v.T):
        on = [i for i, o in ids.items() if scored(v, o, t)]
        region[t] = np.isin(v.inst[t], on)
    row["depth"] = float(np.nanmean(M.depth_accuracy(d, v.depth, kind)))
    do = M.depth_accuracy(d, v.depth, kind, region=region)
    row["depth_objects"] = float(np.nanmean(do)) if np.isfinite(do).any() else float("nan")
    # masks
    z = np.load(os.path.join(est, "masks.npz"))
    all_iou = []
    for o in v.objects:
        key = str(o["instance_id"])
        if key not in z.files:
            continue
        m = np.unpackbits(z[key], axis=None)[: v.T * v.H * v.W].reshape(v.T, v.H, v.W).astype(bool)
        ious = []
        for t in range(v.T):
            ref = v.inst[t] == o["instance_id"]
            if scored(v, o, t) and ref.sum() >= MIN_MASK_PX:
                ious.append(M.mask_iou(m[t], ref))
        if ious:
            objects.append({"run_id": run["run_id"], "object": o["id"], "type": o["type"], "frames": len(ious), "mask_iou": float(np.mean(ious))})
            all_iou += ious
    row["mask_iou"] = float(np.mean(all_iou)) if all_iou else float("nan")
    # tracks
    tz = np.load(os.path.join(est, "tracks.npz"))
    tr = [t for t in v.tracks if len(t["points"])]
    if tr:
        ref = np.concatenate([t["points"] for t in tr])
        rvis = np.concatenate([t["visible"] for t in tr])
        q = np.concatenate([t["query_t"] for t in tr])
        first = np.concatenate([np.full(len(t["points"]), t["first"]) for t in tr])
        last = np.concatenate([np.full(len(t["points"]), t["last"]) for t in tr])
        row["track"], per, row["track_occlusion"] = M.track_accuracy(tz["points"], tz["visible"], ref, rvis, q, first, last, (v.W, v.H))
        row["track_points"] = len(ref)
    else:
        row["track"], row["track_occlusion"], row["track_points"] = float("nan"), float("nan"), 0
    row["eval2"] = M.harmonic_mean([row["depth"], row["mask_iou"], row["track"]])
    return row, objects


def cross_view(group, views, dino):
    """The two cross-camera scores for the separate camera views of one scene, condition and seed."""
    cams = [r["job"]["view"] for r in group]
    scene = group[0]["job"]["scene"]
    V = {c: views[(scene, c)] for c in cams}
    T, H, W = V[cams[0]].inst.shape
    gen = {r["job"]["view"]: read_video(r["video"], (W, H)) for r in group}
    ts = list(range(0, T, CROSS_STEP))
    rng = np.random.default_rng(0)
    # --- the same object in every camera
    est = {r["job"]["view"]: r["est"] for r in group}
    keys, ims, colour, in_place = [], [], {}, {}
    for c in cams:
        z = np.load(os.path.join(est[c], "masks.npz"))
        for o in V[c].objects:
            if c not in o["cameras"]:
                continue
            found = np.unpackbits(z[str(o["instance_id"])])[: T * H * W].reshape(T, H, W).astype(bool) if str(o["instance_id"]) in z.files else None
            for t in ts:
                m = V[c].inst[t] == o["instance_id"]
                if m.sum() >= MIN_CROP_PX:
                    keys.append((o["id"], c, t))
                    ims.append(crop(gen[c][t], m))
                    colour[(o["id"], c, t)] = np.median(gen[c][t][m], 0)
                    in_place[(o["id"], c, t)] = found is not None and M.mask_iou(found[t], m) >= IN_PLACE_IOU
    feat = dict(zip(keys, dino.crops(ims)))
    types = {o["id"]: o["type"] for c in cams for o in V[c].objects}
    same, later, other, gaps, per_obj, pairs_all = [], [], [], [], {}, 0
    for (o, c, t), f in feat.items():
        if (o, c, t + 16) in feat:                                   # the same object, same camera, one second on
            later.append(float(f @ feat[(o, c, t + 16)]))
        for c2 in cams:
            if c2 <= c:
                continue
            if (o, c2, t) in feat:                                   # the same object, another camera, same instant
                pairs_all += 1
                if not (in_place[(o, c, t)] and in_place[(o, c2, t)]):
                    continue                                         # not where the simulator puts it in one of the two
                s = float(f @ feat[(o, c2, t)])
                same.append(s)
                g = float(np.linalg.norm(lab(colour[(o, c, t)]) - lab(colour[(o, c2, t)])))
                gaps.append(g)
                per_obj.setdefault(o, []).append((s, g, c, c2))
            for o2 in types:                                         # a different object of the same kind
                if o2 != o and types[o2] == types[o] and (o2, c2, t) in feat:
                    other.append(float(f @ feat[(o2, c2, t)]))
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    row = {"scene": scene, "cond": group[0]["job"]["cond"], "seed": group[0]["job"]["seed"], "cameras": " ".join(cams),
           "object_colour": mean([g < SAME_COLOUR for g in gaps]), "object_in_place": len(same) / pairs_all if pairs_all else float("nan"),
           "object_pairs": len(same), "object_colour_gap": mean(gaps),
           "object_feature": mean(same), "object_feature_same_camera_1s_later": mean(later), "object_feature_different_object": mean(other)}
    objects = [{"scene": scene, "cond": row["cond"], "seed": row["seed"], "object": o, "type": types[o], "pairs": len(v),
                "cameras": " ".join(sorted({c for x in v for c in x[2:]})), "colour": mean([x[1] < SAME_COLOUR for x in v]),
                "colour_gap": mean([x[1] for x in v]), "feature": mean([x[0] for x in v])} for o, v in per_obj.items()]
    # --- the same surface point in every camera
    sim, sim_obj, sim_bg, floor, dE, share, agree, agree_obj, agree_bg, placed = [], [], [], [], [], [], [], [], [], []
    dest = {c: (np.load(os.path.join(est[c], "depth.npy"), mmap_mode="r"), json.load(open(os.path.join(est[c], "depth.json")))["kind"]) for c in cams}
    for t in ts[::2]:
        fmap = {c: dino.dense(gen[c][t]) for c in cams}
        ok = {c: M.depth_ok(np.asarray(dest[c][0][t], np.float32), V[c].depth[t], dest[c][1])[1] for c in cams}
        for i, a in enumerate(cams):
            for b in cams[i + 1:]:
                ua, va, cov = M.correspondences(V[b].depth[t], V[b].K[t], V[b].c2w[t], V[b].inst[t], V[a].depth[t], V[a].K[t], V[a].c2w[t], V[a].inst[t])
                share.append(float(cov.mean()))
                if ok[a] is None or ok[b] is None or not cov.any():
                    continue
                ui, vi = np.clip(np.round(ua).astype(int), 0, W - 1), np.clip(np.round(va).astype(int), 0, H - 1)
                good = cov & ok[b] & ok[a][vi, ui]                   # the surface is in place in both videos
                placed.append(float(good.sum() / cov.sum()))
                ys, xs = np.nonzero(good)
                if len(xs) < 500:
                    continue
                pick = rng.choice(len(xs), size=min(SURFACE_PIXELS, len(xs)), replace=False)
                ys, xs = ys[pick], xs[pick]
                fb = dino.sample(fmap[b], xs.astype(np.float64), ys.astype(np.float64), (W, H))
                fa = dino.sample(fmap[a], ua[ys, xs], va[ys, xs], (W, H))
                s = M.cosine(fa, fb)
                is_obj = V[b].inst[t][ys, xs] < 100
                sim.append(s.mean())
                floor.append(M.cosine(fa[rng.permutation(len(fa))], fb).mean())
                if is_obj.sum() > 50:
                    sim_obj.append(s[is_obj].mean())
                if (~is_obj).sum() > 50:
                    sim_bg.append(s[~is_obj].mean())
                ca = cv2.remap(gen[a][t], ua.astype(np.float32), va.astype(np.float32), cv2.INTER_LINEAR)[ys, xs]
                gap = np.linalg.norm(lab(ca) - lab(gen[b][t][ys, xs]), axis=-1)
                dE.append(gap.mean())
                agree.append((gap < SAME_COLOUR).mean())
                if is_obj.sum() > 50:
                    agree_obj.append((gap[is_obj] < SAME_COLOUR).mean())
                if (~is_obj).sum() > 50:
                    agree_bg.append((gap[~is_obj] < SAME_COLOUR).mean())
    row.update(surface_colour=mean(agree), surface_in_place=mean(placed), surface_colour_objects=mean(agree_obj), surface_colour_background=mean(agree_bg),
               surface_colour_gap=mean(dE), surface_shared_pixels=mean(share), surface_feature=mean(sim),
               surface_feature_objects=mean(sim_obj), surface_feature_background=mean(sim_bg), surface_feature_random_pairs=mean(floor))
    return row, objects


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--only", default="", help="filters, e.g. scene=intersection_4way,cond=D,seed=0")
    ap.add_argument("--force", action="store_true", help="run every reader again")
    ap.add_argument("--on-render", action="store_true", help="score the simulator's own render (the readers' ceiling)")
    a = ap.parse_args()
    exp = mc.load_exp(a.exp if os.path.isabs(a.exp) else os.path.join(mc.REPO, a.exp))
    flt = dict(kv.split("=") for kv in a.only.split(",") if kv)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free = free_gib()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(max(free, key=free.get))
    os.environ.setdefault("TORCH_HOME", os.path.join(os.environ["S2R_CACHE"], "torch"))
    print(f"[eval] GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    runs, seen = [], set()
    for j in mc.jobs(exp):
        if not all(str(j[k]) == v for k, v in flt.items()):
            continue
        video, est = video_of(exp, j, a.on_render)
        if not os.path.exists(video) or (a.on_render and (j["scene"], j["view"]) in seen):
            continue
        seen.add((j["scene"], j["view"]))
        rid = mc.run_id(exp, j) if not a.on_render else f"render/{j['scene']}/{j['view']}"
        runs.append({"job": j, "video": video, "est": est, "run_id": rid, "sha": mc.sha256_file(video)})
    if not runs:
        sys.exit("[eval] no finished runs match")
    views = {}
    for r in runs:
        key = (r["job"]["scene"], r["job"]["view"])
        if key not in views:
            views[key] = load_view(exp, r["job"])
        os.makedirs(r["est"], exist_ok=True)
    print(f"[eval] {len(runs)} clips, {len(views)} views", flush=True)

    # --- readers
    stamp = lambda r, name, ver: f"{name}/{ver}/{r['sha']}"
    need = lambda r, f, name, ver: a.force or not cache_ok(r["est"], f, stamp(r, name, ver))
    run_depth_reader([(r["video"], r["est"], stamp(r, *DEPTH_READER), (views[(r["job"]["scene"], r["job"]["view"])].W, views[(r["job"]["scene"], r["job"]["view"])].H))
                      for r in runs if need(r, "depth", *DEPTH_READER)])
    from readers.cotracker3 import reader as ct  # noqa: E402
    from readers.dinov2 import reader as dn  # noqa: E402
    from readers.sam2 import reader as sm  # noqa: E402
    todo_m = [r for r in runs if need(r, "masks", sm.NAME, sm.VERSION)]
    todo_t = [r for r in runs if need(r, "tracks", ct.NAME, ct.VERSION)]
    if todo_m:
        sam = sm.Reader()
        for n, r in enumerate(todo_m):
            v = views[(r["job"]["scene"], r["job"]["view"])]
            masks = sam.run(read_video(r["video"], (v.W, v.H)), v)
            np.savez_compressed(os.path.join(r["est"], "masks.npz"), **{str(i): np.packbits(m) for i, m in masks.items()})
            cache_mark(r["est"], "masks", stamp(r, sm.NAME, sm.VERSION), reader=sm.NAME)
            print(f"[eval] masks {n + 1}/{len(todo_m)} {r['run_id']}", flush=True)
        del sam
    if todo_t:
        tracker = ct.Reader()
        for n, r in enumerate(todo_t):
            v = views[(r["job"]["scene"], r["job"]["view"])]
            pts, vis = tracker.run(read_video(r["video"], (v.W, v.H)), v)
            np.savez_compressed(os.path.join(r["est"], "tracks.npz"), points=pts, visible=vis)
            cache_mark(r["est"], "tracks", stamp(r, ct.NAME, ct.VERSION), reader=ct.NAME)
            print(f"[eval] tracks {n + 1}/{len(todo_t)} {r['run_id']}", flush=True)
        del tracker

    # --- per camera
    by_scene = {}
    for r in runs:
        row, objs = score_run(r, views)
        s = by_scene.setdefault(r["job"]["scene"], {"eval2": [], "objects": [], "eval3": [], "eval3_objects": [], "runs": []})
        s["eval2"].append(row)
        s["objects"] += objs
        s["runs"].append(r)
        print(f"[eval] {r['run_id']}: depth {row['depth']:.3f}  mask_iou {row['mask_iou']:.3f}  track {row['track']:.3f}  eval2 {row['eval2']:.3f}", flush=True)

    # --- across cameras: the separate camera views of each scene, condition and seed
    # (with --on-render: the render's own cameras against each other. On a textured render that is what views that
    #  truly show one world score under these viewpoint changes; on a clay render every surface is the same grey.)
    if True:
        dino = dn.Reader()
        for scene, s in by_scene.items():
            groups = {}
            for r in s["runs"]:
                if r["job"]["kind"] == "cam":
                    groups.setdefault((r["job"]["cond"], r["job"]["seed"]), []).append(r)
            for (cond, seed), g in sorted(groups.items()):
                if len(g) < 2:
                    continue
                row, objs = cross_view(sorted(g, key=lambda r: r["job"]["view"]), views, dino)
                s["eval3"].append(row)
                s["eval3_objects"] += objs
                print(f"[eval] {scene} {cond} seed {seed}: object_colour {row['object_colour']:.3f} (in place {row['object_in_place']:.0%})  "
                      f"surface_colour {row['surface_colour']:.3f} (in place {row['surface_in_place']:.0%})", flush=True)

    # --- write
    tag = "_render" if a.on_render else ""
    for scene, s in by_scene.items():
        out = os.path.join(mc.scene_root(exp, scene), "scores", exp["experiment"])
        os.makedirs(out, exist_ok=True)
        write_csv(os.path.join(out, f"eval2{tag}.csv"), s["eval2"])
        write_csv(os.path.join(out, f"eval2_objects{tag}.csv"), s["objects"])
        write_csv(os.path.join(out, f"eval3{tag}.csv"), s["eval3"])
        write_csv(os.path.join(out, f"eval3_objects{tag}.csv"), s["eval3_objects"])
        summary = {"experiment": exp["experiment"], "scene": scene, "on_render": a.on_render, "per_condition_and_seed": {}}
        for cond, seed in sorted({(r["cond"], r["seed"]) for r in s["eval2"]}):
            cam_rows = [r for r in s["eval2"] if (r["cond"], r["seed"], r["kind"]) == (cond, seed, "cam")]
            e = {k: float(np.nanmean([r[k] for r in cam_rows])) for k in ("depth", "mask_iou", "track")} if cam_rows else {}
            if e:                                      # averaged over cameras first, then the harmonic mean
                e["eval2"] = M.harmonic_mean([e["depth"], e["mask_iou"], e["track"]])
                e["cameras"] = [r["view"] for r in cam_rows]
            e["cuts"] = {r["view"]: {k: r[k] for k in ("depth", "mask_iou", "track", "eval2")} for r in s["eval2"] if (r["cond"], r["seed"], r["kind"]) == (cond, seed, "cut")}
            x = next((r for r in s["eval3"] if (r["cond"], r["seed"]) == (cond, seed)), None)
            if x:
                e["eval3"] = {k: x[k] for k in x if k not in ("scene", "cond", "seed")}
            summary["per_condition_and_seed"][f"{cond}/seed{seed}"] = e
        json.dump(summary, open(os.path.join(out, f"eval_summary{tag}.json"), "w"), indent=1)
        print(f"[eval] {scene}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
