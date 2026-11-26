"""Export one camera's render to a scene bundle. See p2s/2_render/bundle.py.

  $S2R_ROOT/envs/eval/.venv/bin/python legacy/sim/export_world.py <spec.yaml> <world_dir> <bundle_dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world.export import main  # noqa: E402

if __name__ == "__main__":
    main(*sys.argv[1:4])
