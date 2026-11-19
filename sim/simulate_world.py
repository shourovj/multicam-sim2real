"""Simulate a scene spec v2 once (all objects, all cameras). See sim/world/simulate.py.

  $S2R_ROOT/envs/eval/.venv/bin/python sim/simulate_world.py configs/scenes/<scene>.yaml <world_dir> [--allow-failed-checks]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world.simulate import simulate  # noqa: E402

if __name__ == "__main__":
    info, failed = simulate(sys.argv[1], sys.argv[2])
    if failed and "--allow-failed-checks" not in sys.argv:
        sys.exit(f"[simulate] {len(failed)} check(s) failed: {[c['type'] for c in failed]}")
