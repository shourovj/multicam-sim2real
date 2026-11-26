"""Per-camera scene bundle from one render of a spec v2 world. The work is done by p2s/2_render/bundle.py,
which both routes share; this adds the spec v2 entry point.

  $S2R_ROOT/envs/eval/.venv/bin/python legacy/sim/export_world.py <spec.yaml> <world_dir> <bundle_dir>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "p2s", "2_render"))
sys.path.insert(0, os.path.dirname(HERE))
from bundle import export, palette, project  # noqa: E402,F401
from world.spec import load  # noqa: E402


def main(spec_path, world_dir, out):
    spec, _ = load(spec_path)
    wj = json.load(open(os.path.join(world_dir, "world.json")))
    export(out, spec["frames"], wj["checks"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
