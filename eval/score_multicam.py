"""Score multicam runs against the simulator (eval env: numpy + OpenCV + yaml).

  $S2R_ROOT/envs/eval/.venv/bin/python eval/score_multicam.py --exp configs/experiments/multicam_v1.yaml [--only scene=...]

For every finished run it reads the generated video and the simulator's instance masks for that view (a camera's
bundle, or the cut's frames joined by the same schedule) and writes to <scene>/r<rev>/scores/:
  colour.csv   per run x object (x shot for cuts): the colour name of the generated pixels inside the object's eroded
               simulator mask, next to the colour name of the simulator render's own pixels there
  lights.csv   per run x signal head (intersection): the lit lamp read from the generated pixels vs the simulator's state,
               on frames where the head is at least MIN_HEAD_PX tall
  summary.json per run: colour matches, cross-view agreement (separate views of one seed), across-cut agreement
These are coarse checks; the findings are read by eye first.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gen"))
import multicam_common as mc  # noqa: E402

MIN_PX = 150           # an object counts on a frame with at least this many (eroded) mask pixels
MIN_HEAD_PX = 16       # a signal head must be this tall (px) to read its lamps
HUES = [(15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (195, "cyan"), (255, "blue"), (290, "purple"), (335, "pink"), (360, "red")]


def colour_name(px):
    """Name the colour of a set of RGB pixels [N, 3]: by hue if enough of them are saturated, else by brightness.
    Shadows, windows and tyres pull a median toward grey, so only the brighter half of the pixels is used."""
    if len(px) == 0:
        return None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    h, sat, val = hsv[:, 0] * 2, hsv[:, 1] / 255, hsv[:, 2] / 255
    bright = val >= np.median(val)
    chroma = bright & (sat > 0.35) & (val > 0.15)
    if chroma.mean() > 0.25 / 2:                          # a quarter of the bright half is clearly coloured
        r = np.deg2rad(h[chroma])                         # circular mean, so reds near 0 and 360 average correctly
        hh = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360
        name = next(n for lim, n in HUES if hh % 360 <= lim)
        return ("dark " if np.median(val[chroma]) < 0.3 and name not in ("red",) else "") + name
    v = np.percentile(val, 75)
    return "white" if v > 0.62 else "light grey" if v > 0.42 else "dark grey" if v > 0.16 else "black"


def read_video(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, H = int(out[0]), int(out[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def colour_class(name):
    """Coarse class used for matching: hues, plus light (white, light grey) and dark (dark grey, black)."""
    if name is None:
        return None
    if name in ("white", "light grey"):
        return "light"
    if name in ("dark grey", "black"):
        return "dark"
    return name.replace("dark ", "")


def expected_name(o):
    """The colour the spec asks for: a car's paint, a person's top, a pitcher's glaze (Blender linear RGB -> sRGB)."""
    a = o["asset"]
    c = a.get("color") or a.get("colors", {}).get("top")
    if a["type"] == "robot":
        return "white"
    if c is None:
        return None
    lin = np.clip(np.asarray(c, float), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return colour_name(np.uint8(np.round(srgb * 255))[None])


def pixels(frames, masks, idx):
    px = [frames[t][masks[t]] for t in idx]
    return np.concatenate(px) if px else np.zeros((0, 3), np.uint8)


def view_masks(exp, j):
    root = mc.scene_root(exp, j["scene"])
    base = os.path.join(root, "cams", j["view"], "bundle") if j["kind"] == "cam" else os.path.join(root, "cut", j["view"])
    return np.load(os.path.join(base, "instance.npy")), np.load(os.path.join(base, "rgb.npy"), mmap_mode="r"), base


def lamp_state(frame, box):
    """Which third of a signal head's box (top red, middle amber, bottom green) is brightest. A lit lamp renders bright
    but pale, unlit ones dark but saturated, so brightness (not saturation) picks the lit one."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1 + 1, x0:x1 + 1]), cv2.COLOR_RGB2HSV).astype(float)
    score = hsv[..., 2]
    thirds = np.array_split(score, 3, axis=0)
    vals = [float(np.percentile(t, 95)) if t.size else 0 for t in thirds]
    return ["red", "yellow", "green"][int(np.argmax(vals))], vals


def score_run(exp, j, spec):
    d = mc.run_dir(exp, j)
    gen = read_video(os.path.join(d, "out_video.mp4"))
    inst, sim, base = view_masks(exp, j)
    T = min(len(gen), len(inst))
    if gen.shape[1:3] != inst.shape[1:3]:
        gen = np.stack([cv2.resize(f, (inst.shape[2], inst.shape[1])) for f in gen])
    k = np.ones((5, 5), np.uint8)
    shots = [(0, T - 1, j["view"])]
    if j["kind"] == "cut":
        shots = [(s["frames"][0], s["frames"][1], s["cam"]) for s in exp["cuts"][j["scene"]][j["view"]]]
    rows = []
    parent_of = {c["id"]: c.get("parent") for c in spec["cameras"]}     # a camera's own car / holder / robot is not scored in it
    for o in spec["objects"]:
        if o["asset"]["type"] in ("water_stream", "water_level", "glass"):
            continue
        masks = [cv2.erode((inst[t] == o["instance_id"]).astype(np.uint8), k).astype(bool) for t in range(T)]
        if o["asset"]["type"] == "pedestrian":                  # the clothing top: rows 18-50 % down the person's box
            for t in range(T):
                ys = np.nonzero(masks[t].any(1))[0]
                if len(ys):
                    y0, y1 = ys.min(), ys.max()
                    keep = np.zeros_like(masks[t])
                    keep[int(y0 + 0.18 * (y1 - y0)):int(y0 + 0.5 * (y1 - y0)) + 1] = True
                    masks[t] &= keep
        want = expected_name(o)
        for a, b, cam in shots:
            if parent_of.get(cam) == o["id"]:
                continue
            idx = [t for t in range(a, b + 1) if masks[t].sum() >= MIN_PX // (3 if o["asset"]["type"] == "pedestrian" else 1)]
            if not idx:
                continue
            gp, sp = pixels(gen, masks, idx), pixels(sim, masks, idx)
            gn, sn = colour_name(gp), colour_name(sp)
            ref = want or sn                                        # the spec's intended colour; else the render's own
            rows.append({"run_id": mc.run_id(exp, j), "view": j["view"], "shot_camera": cam, "object": o["id"], "frames": len(idx),
                         "gen_median_rgb": [int(x) for x in np.median(gp, 0)], "gen_name": gn, "gen_class": colour_class(gn),
                         "sim_name": sn, "expected_name": want, "expected_class": colour_class(ref),
                         "match": colour_class(gn) == colour_class(ref)})
    lights = []
    wj = json.load(open(os.path.join(mc.scene_root(exp, j["scene"]), "world", "world.json")))
    for a, b, cam in shots:
        cb = os.path.join(mc.scene_root(exp, j["scene"]), "cams", cam, "bundle")
        tl = json.load(open(os.path.join(cb, "timeline.json")))
        static = json.load(open(os.path.join(cb, "static_ids.json")))
        K, c2w = np.asarray(tl["camera"]["K"]), np.asarray(tl["camera"]["cam2world_blender"])
        for sig in wj.get("signals", []):
            for hd in sig["heads"]:
                hid = next((v["id"] for v in static.values() if v["what"] == f"signal {sig['id']} head {hd['lane']}"), None)
                c, yaw = np.asarray(hd["pos"]), hd["yaw"]
                face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                corners = np.array([c + face * 0.17 + np.array([-np.sin(yaw), np.cos(yaw), 0]) * dy + np.array([0, 0, dz])
                                    for dy in (-0.16, 0.16) for dz in (-0.5, 0.5)])
                for t in range(a, b + 1, 2):
                    tc = t                                             # a cut keeps the world clock: frame t is that camera's frame t
                    w2c = np.linalg.inv(c2w[tc])
                    cam_pos = c2w[tc][:3, 3]
                    if face @ (cam_pos - c) <= 0.7 * np.linalg.norm(cam_pos - c):
                        continue                                       # more than ~45 deg off its face: the visors hide the lamps
                    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
                    z = -pc[:, 2]
                    if (z <= 0.1).any():
                        continue
                    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
                    v = K[1, 1] * -pc[:, 1] / z + K[1, 2]
                    x0, x1, y0, y1 = int(u.min()), int(np.ceil(u.max())), int(v.min()), int(np.ceil(v.max()))
                    H, W = inst.shape[1:3]
                    if y1 - y0 < MIN_HEAD_PX or x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
                        continue
                    if hid is not None and (inst[t, y0:y1 + 1, x0:x1 + 1] == hid).mean() < 0.4:
                        continue                                       # something is in front of the head
                    box = (x0, y0, x1, y1)
                    gs, _ = lamp_state(gen[t], box)
                    ss, _ = lamp_state(sim[t], box)
                    lights.append({"run_id": mc.run_id(exp, j), "shot_camera": cam, "head": f"{sig['id']} {hd['lane']}", "frame": t,
                                   "height_px": y1 - y0, "truth": tl["signals"][sig["id"]][t], "gen": gs, "sim_read": ss,
                                   "match": gs == tl["signals"][sig["id"]][t], "sim_read_ok": ss == tl["signals"][sig["id"]][t]})
    return rows, lights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    exp = mc.load_exp(os.path.join(mc.REPO, a.exp) if not os.path.isabs(a.exp) else a.exp)
    flt = dict(kv.split("=") for kv in a.only.split(",") if kv)
    by_scene = {}
    for j in mc.jobs(exp):
        if not all(str(j[k]) == v for k, v in flt.items()):
            continue
        if not os.path.exists(os.path.join(mc.run_dir(exp, j), "out_video.mp4")):
            continue
        spec = mc.scene_spec(exp, j["scene"])
        rows, lights = score_run(exp, j, spec)
        s = by_scene.setdefault(j["scene"], {"colour": [], "lights": [], "runs": {}})
        s["colour"] += rows
        s["lights"] += lights
        s["runs"][mc.run_id(exp, j)] = j
        print(f"[score] {mc.run_id(exp, j)}: colour {sum(r['match'] for r in rows)}/{len(rows)}"
              + (f", lights {sum(l['match'] for l in lights)}/{len(lights)}" if lights else ""))
    for scene, s in by_scene.items():
        out = os.path.join(mc.scene_root(exp, scene), "scores")
        os.makedirs(out, exist_ok=True)
        for name, rows in (("colour", s["colour"]), ("lights", s["lights"])):
            if rows:
                with open(os.path.join(out, f"{name}.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        summary = {}
        for rid, j in s["runs"].items():
            mine = [r for r in s["colour"] if r["run_id"] == rid]
            lt = [l for l in s["lights"] if l["run_id"] == rid]
            entry = {"colour_match": f"{sum(r['match'] for r in mine)}/{len(mine)}",
                     "lights_match": f"{sum(l['match'] for l in lt)}/{len(lt)}" if lt else None,
                     "lights_read_on_sim": f"{sum(l['sim_read_ok'] for l in lt)}/{len(lt)}" if lt else None}
            if j["kind"] == "cut":                      # does each object keep one colour name across the shots it is in?
                per = {}
                for r in mine:
                    per.setdefault(r["object"], set()).add(r["gen_class"])
                multi = {o: n for o, n in per.items() if sum(1 for r in mine if r["object"] == o) > 1}
                entry["across_cut_consistent"] = f"{sum(len(n) == 1 for n in multi.values())}/{len(multi)}"
            summary[rid] = entry
        # cross-view: for the separate camera views of one condition and seed, does each object seen in 2+ views
        # get one colour name everywhere?
        groups = {}
        for r in s["colour"]:
            j = s["runs"][r["run_id"]]
            if j["kind"] == "cam":
                groups.setdefault((j["cond"], j["seed"]), {}).setdefault(r["object"], {})[j["view"]] = r["gen_class"]
        summary["cross_view"] = {}
        for (c, sd), objs in groups.items():
            multi = {o: v for o, v in objs.items() if len(v) > 1}
            summary["cross_view"][f"{c}/seed{sd}"] = {
                "objects_seen_in_2plus_views": len(multi),
                "same_colour_in_every_view": sum(len(set(v.values())) == 1 for v in multi.values()),
                "names": objs}
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        print(f"[score] {scene}: wrote {out}")


if __name__ == "__main__":
    main()
