"""One table from the scores of several experiments: a row per model and condition, ordered from the densest
control to text only. Reads what eval/run_eval.py wrote; runs nothing.

  $S2R_ROOT/envs/eval/.venv/bin/python eval/summarize.py --exp configs/experiments/multicam_v1.yaml \
      configs/experiments/multicam_v1_1.yaml configs/experiments/multicam_v1_2.yaml configs/experiments/multicam_v1_3.yaml \
      [--reference configs/experiments/multicam_v1.yaml] [--by-scene] [--out table.csv]

Each score is averaged over the camera views of a scene and seed first, then over seeds and scenes; eval2 is the
harmonic mean taken per scene and seed (eval_summary.json) and averaged the same way. Cut videos are not included.
--reference adds the row of the simulator's own render, from that experiment's --on-render scores.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gen", "common"))
sys.path.insert(0, os.path.join(HERE, "common"))
import multicam_common as mc  # noqa: E402
import metrics as M  # noqa: E402

DENSITY = {"blur": 0, "depth": 2, "seg": 3, "edge": 4}        # lower = the control carries more; several controls rank 1, text 9
COLS = ("depth", "mask_iou", "track", "eval2", "object_colour", "object_in_place", "surface_colour", "surface_in_place")


def control_label(cond_spec):
    if not cond_spec:
        return "text only", 9
    keys = list(cond_spec)
    if keys == ["blur"]:
        preset = cond_spec["blur"].get("preset_blur_strength", "none")
        return ("the render itself" if preset == "none" else f"the render, blurred ({preset})"), 0
    if len(keys) > 1:
        return " + ".join(keys), 1
    return keys[0], DENSITY.get(keys[0], 5)


def read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def collect(exp, render=False):
    """{(scene, cond, seed): {column: value}} from one experiment's score files."""
    tag = "_render" if render else ""
    out = {}
    for scene in exp["scenes"]:
        d = os.path.join(mc.scene_root(exp, scene), "scores", exp["experiment"])
        cams = {}
        for r in read_csv(os.path.join(d, f"eval2{tag}.csv")):
            if r["kind"] == "cam":
                cams.setdefault((scene, r["cond"], int(r["seed"])), []).append(r)
        for key, rows in cams.items():
            e = {k: float(np.nanmean([float(r[k]) for r in rows])) for k in ("depth", "mask_iou", "track")}
            e["eval2"] = M.harmonic_mean([e["depth"], e["mask_iou"], e["track"]])
            e["views"] = len(rows)
            out[key] = e
        for r in read_csv(os.path.join(d, f"eval3{tag}.csv")):
            key = (scene, r["cond"], int(r["seed"]))
            if key in out:
                out[key].update({k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
                                 for k in ("object_colour", "object_in_place", "surface_colour", "surface_in_place")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--reference", help="experiment whose --on-render scores give the simulator-render row")
    ap.add_argument("--by-scene", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: mc.load_exp(p if os.path.isabs(p) else os.path.join(mc.REPO, p))
    rows = []
    if a.reference:
        ref = load(a.reference)
        for (scene, cond, seed), e in collect(ref, render=True).items():
            rows.append(dict(rank=-1, model="simulator render", control="(reference: one world, by construction)", cond="render", scene=scene, seed=seed, **e))
    for p in a.exp:
        exp = load(p)
        specs = {j["cond"]: j["cond_spec"] for j in mc.jobs(exp)}
        for (scene, cond, seed), e in collect(exp).items():
            label, rank = control_label(specs.get(cond, {}))
            rows.append(dict(rank=rank, model=exp["model"]["name"], control=label, cond=cond, scene=scene, seed=seed, **e))
    if not rows:
        sys.exit("no scores found: run eval/run_eval.py on these experiments first")
    groups = {}
    for r in rows:
        key = (r["rank"], r["control"], r["model"], r["cond"]) + ((r["scene"],) if a.by_scene else ())
        groups.setdefault(key, []).append(r)
    table = []
    for key, g in sorted(groups.items()):
        t = {"control": key[1], "model": key[2], "condition": key[3]}
        if a.by_scene:
            t["scene"] = key[4]
        t["scenes x seeds"] = len(g)
        for c in COLS:
            v = [r[c] for r in g if c in r and not np.isnan(r[c])]
            t[c] = round(float(np.mean(v)), 3) if v else ""
        table.append(t)
    fields = list(table[0])
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(" --- " for _ in fields) + "|")
    for t in table:
        print("| " + " | ".join(str(t[f]) for f in fields) + " |")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(table)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
