"""Depth reader: Depth Anything 3, one frame at a time (monocular). Runs in its own env ($S2R_ROOT/envs/da3).

  $S2R_ROOT/envs/da3/.venv/bin/python eval/readers/depth_anything3/run.py <jobs.json> [--model depth-anything/DA3METRIC-LARGE]

jobs.json: [{"video": path, "out": folder, "stamp": text, "size": [W, H]}]. For each it writes
  <out>/depth.npy    float16 [T, H, W], the model's depth at the simulator's frame size
  <out>/depth.json   the stamp, the model, and kind = "depth": depth up to one scale factor per frame, which is
                     what eval/common/metrics.depth_accuracy removes with a median ratio
DA3METRIC-LARGE and DA3MONO-LARGE are Apache-2.0; DA3-LARGE and DA3-GIANT are CC BY-NC 4.0 (their model cards).
The metric model's output is metres only after multiplying by focal / 300; the median ratio makes that factor moot.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

REVISIONS = {"depth-anything/DA3METRIC-LARGE": None}      # pin a revision here once the benchmark is frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    from depth_anything_3.api import DepthAnything3
    kw = {"revision": REVISIONS[a.model]} if REVISIONS.get(a.model) else {}
    model = DepthAnything3.from_pretrained(a.model, **kw).to("cuda").eval()
    for j in jobs:
        cap, frames = cv2.VideoCapture(j["video"]), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        W, H = j["size"]
        out = np.empty((len(frames), H, W), np.float16)
        with torch.inference_mode():
            for t, f in enumerate(frames):
                d = np.asarray(model.inference([f]).depth[0], np.float32)
                out[t] = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        os.makedirs(j["out"], exist_ok=True)
        np.save(os.path.join(j["out"], "depth.npy"), out)
        json.dump({"stamp": j["stamp"], "reader": "depth_anything3", "model": a.model, "kind": "depth"},
                  open(os.path.join(j["out"], "depth.json"), "w"), indent=1)
        print(f"[depth_anything3] {j['video']} -> {len(frames)} frames", flush=True)


if __name__ == "__main__":
    sys.exit(main())
