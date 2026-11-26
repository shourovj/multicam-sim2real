"""Export one camera's render of a prompt-to-sim world to a scene bundle (p2s/2_render/bundle.py does the work).

  $S2R_ROOT/envs/eval/.venv/bin/python p2s/2_render/export_x.py <world_dir> <bundle_dir>

Reads what p2s/2_render/render_x.py wrote in <bundle_dir> and world/spec.yaml + world.json; writes rgb.mp4, boxes.json,
visibility.json, tracks.npz, overlay.mp4, checks.json and controls/ (depth, seg, edge, normals, rgb videos).
"""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import export  # noqa: E402

if __name__ == "__main__":
    world, out = sys.argv[1:3]
    spec = yaml.safe_load(open(os.path.join(world, "spec.yaml")))
    export(out, spec["frames"], json.load(open(os.path.join(world, "world.json")))["checks"])
