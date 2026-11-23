"""What each camera sees, from the simulator: the facts a text-only prompt has to carry because no control does.

  python gen/camera_facts.py --exp configs/experiments/multicam_v1_3.yaml

For each scene and camera, writes <prompts dir>/facts/<camera>.json and a readable <camera>.txt:
  camera   where it is (height, compass direction it looks, tilt), whether and how far it moves
  objects  per moving object: when it is on screen, where in the frame (thirds) at entry, middle and exit,
           its largest size (share of frame width), and which way it moves across the frame
  static   per building, prop and signal head: share of the frame and its horizontal extent, at the first and last frame
The camera blocks of the text-to-video prompts are written from these files, so a placement claim in a prompt can
be checked against them. Reads the rendered bundles: cams/<camera>/bundle/{boxes,visibility,static_ids}.json, instance.npy.
"""

import argparse
import json
import os

import numpy as np

import multicam_common as mc

MIN_PX = 150          # an object counts as on screen from this many pixels (about a 12 x 12 px patch)
MIN_SHARE = 0.004     # a static part is listed from 0.4 % of the frame


def thirds(x, y):
    h = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "centre")
    v = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "middle")
    return f"{v} {h}"


def compass(v):
    ang = (np.degrees(np.arctan2(v[0], v[1])) + 360) % 360        # 0 = north (+Y), 90 = east (+X)
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((ang + 22.5) // 45) % 8]


def camera_facts(c2w, fps):
    pos, fwd = c2w[:, :3, 3], -c2w[:, :3, 2]
    tilt = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    moving = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps > 0.2        # m/s
    return {"height_m": [round(float(pos[0, 2]), 2), round(float(pos[-1, 2]), 2)],
            "looks": [compass(fwd[0]), compass(fwd[-1])], "tilt_deg": [round(float(tilt[0]), 1), round(float(tilt[-1]), 1)],
            "path_m": round(path, 2),
            "moving_s": ([round(float(np.argmax(moving)) / fps, 2), round(float(len(moving) - np.argmax(moving[::-1])) / fps, 2)]
                         if moving.any() else None)}


def object_facts(boxes, vis, W, H, fps):
    out = {}
    for name, v in vis.items():
        px = np.asarray(v["pixels"])
        on = np.where(px >= MIN_PX)[0]
        if len(on) == 0:
            out[name] = {"on_screen": False, "hidden_from_camera": v.get("hidden_from_camera", False)}
            continue
        b = boxes[str(v["instance_id"])]
        pick = [on[0], on[len(on) // 2], on[-1]]
        where, cx = [], []
        for t in pick:
            x0, y0, x1, y1 = b[t]
            where.append({"t_s": round(t / fps, 2), "at": thirds((x0 + x1) / 2 / W, (y0 + y1) / 2 / H),
                          "x": round((x0 + x1) / 2 / W, 2), "y": round((y0 + y1) / 2 / H, 2), "width": round((x1 - x0) / W, 2)})
            cx.append((x0 + x1) / 2 / W)
        widths = [(bb[2] - bb[0]) / W for bb in (b[t] for t in on) if bb]
        dx = cx[-1] - cx[0]
        gaps = int((np.diff(on) > 1).sum())
        out[name] = {"on_screen": True, "from_s": round(on[0] / fps, 2), "to_s": round(on[-1] / fps, 2),
                     "frames": int(len(on)), "gaps": gaps, "where": where, "max_width": round(max(widths), 2),
                     "moves": "left to right" if dx > 0.15 else ("right to left" if dx < -0.15 else "stays about in place")}
    return out


def static_facts(inst, static_ids, W, H):
    names = {}
    for part, d in static_ids.items():
        names.setdefault(d["id"], d["what"])
    out = {}
    for label, t in (("first", 0), ("last", len(inst) - 1)):
        f = np.asarray(inst[t])
        ids, counts = np.unique(f, return_counts=True)
        rows = []
        for i, n in zip(ids, counts):
            if int(i) not in names or n / f.size < MIN_SHARE:
                continue
            xs = np.where((f == i).any(axis=0))[0]
            rows.append({"what": names[int(i)], "share": round(float(n / f.size), 3),
                         "x_from": round(float(xs[0] / W), 2), "x_to": round(float(xs[-1] / W), 2)})
        out[label] = sorted(rows, key=lambda r: -r["share"])
    return out


def text(cam, f):
    c = f["camera"]
    lines = [f"camera {cam}: {c['height_m'][0]} m up, looks {c['looks'][0]} (tilt {c['tilt_deg'][0]} deg)"
             + (f", moves {c['path_m']} m between {c['moving_s'][0]} s and {c['moving_s'][1]} s, ends looking {c['looks'][1]}" if c["moving_s"] else ", static")]
    for o, d in f["objects"].items():
        if not d["on_screen"]:
            lines.append(f"  {o}: not on screen" + (" (hidden: it carries the camera)" if d.get("hidden_from_camera") else ""))
            continue
        w = "; ".join(f"{p['t_s']} s {p['at']} (w {p['width']})" for p in d["where"])
        lines.append(f"  {o}: {d['from_s']}-{d['to_s']} s, {d['moves']}, up to {d['max_width']} of frame width"
                     + (f", out of view {d['gaps']}x in between" if d["gaps"] else "") + f" | {w}")
    for label in ("first", "last"):
        lines.append(f"  static at {label} frame: " + "; ".join(f"{r['what']} {r['share']:.1%} x {r['x_from']}-{r['x_to']}" for r in f["static"][label]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    exp = mc.load_exp(a.exp if os.path.isabs(a.exp) else os.path.join(mc.REPO, a.exp))
    for scene in exp["scenes"]:
        root = mc.scene_root(exp, scene)
        spec = mc.scene_spec(exp, scene)
        W, H = spec["resolution"]
        fps = spec["fps"]
        z = np.load(os.path.join(root, "world", "world.npz"))
        outdir = os.path.join(mc.REPO, exp["scenes"][scene]["prompts"], "facts")
        os.makedirs(outdir, exist_ok=True)
        for cam in [c["id"] for c in spec["cameras"]]:
            B = os.path.join(root, "cams", cam, "bundle")
            f = {"_about": f"Simulator facts for {scene} camera {cam}, from gen/camera_facts.py; world.npz sha256 "
                           f"{mc.sha256_file(os.path.join(root, 'world', 'world.npz'))[:12]}",
                 "camera": camera_facts(z[f"cams/{cam}/cam2world"], fps),
                 "objects": object_facts(json.load(open(os.path.join(B, "boxes.json"))), json.load(open(os.path.join(B, "visibility.json"))), W, H, fps),
                 "static": static_facts(np.load(os.path.join(B, "instance.npy"), mmap_mode="r"), json.load(open(os.path.join(B, "static_ids.json"))), W, H)}
            json.dump(f, open(os.path.join(outdir, f"{cam}.json"), "w"), indent=1)
            t = text(cam, f)
            open(os.path.join(outdir, f"{cam}.txt"), "w").write(t + "\n")
            print(f"== {scene}\n{t}\n")


if __name__ == "__main__":
    main()
