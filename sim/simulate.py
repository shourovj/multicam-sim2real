"""Run a scene's physics without rendering and print what happens: a quick check while writing a scene.

  $S2R_ROOT/envs/eval/.venv/bin/python sim/simulate.py configs/scenes/<scene>.yaml

Prints, for every object moved by sim/physics, its position and speed over the clip, its peak speed and tilt,
and the events the physics reports (lift-off, release, landing).
"""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion  # noqa: E402
import physics  # noqa: E402


def run(spec_path):
    spec = yaml.safe_load(open(spec_path))
    road = motion.scene_road(spec, os.environ.get("S2R_DATA", ""))
    driven = [o for o in motion.scene_objects(spec) if "motion" in o]
    return spec, physics.simulate(driven, spec["fps"], spec["frames"], road=road, seed=spec["seed"])


if __name__ == "__main__":
    spec, out = run(sys.argv[1])
    t = np.arange(spec["frames"]) / spec["fps"]
    rows = np.linspace(0, spec["frames"] - 1, 11).round().astype(int)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for i, o in out["objects"].items():
        print(f"\n{i}: peak speed {o['speed'].max():.2f} m/s" + (f", peak tilt {o['tilt_deg'].max():.1f} deg" if "tilt_deg" in o else ""))
        print("  t (s)   ", t[rows])
        if "road_m" in o:
            print("  road (m)", o["road_m"][rows])
        else:
            for k, ax in enumerate("xyz"):
                print(f"  {ax} (m)   ", o["pos"][rows, k])
        print("  speed   ", o["speed"][rows])
    print("\nevents:", ", ".join(f"{e['name']} at {e['t']} s" for e in out["events"]) or "none")
