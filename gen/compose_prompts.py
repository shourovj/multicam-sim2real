"""Write the exact caption Cosmos receives for every view of an experiment into the repo, so the prompt is a
tracked file, not only something composed at run time.

  $S2R_ROOT/envs/eval/.venv/bin/python gen/compose_prompts.py --exp configs/experiments/multicam_v1.yaml [--check]

For each scene and view: <prompts dir>/composed/<view>.json = multicam_common.compose_prompt (world block + that
camera's or cut's block + duration, fps, resolution). The negative prompt is configs/prompts/cosmos3_negative_prompt.json
(a frozen copy of Cosmos3's cookbook negative prompt). gen/run_multicam.py refuses to launch a job whose composed file
is missing or differs, so edit the blocks, re-run this, and commit both. --check only reports differences.
"""

import argparse
import json
import os
import sys

import multicam_common as mc


def composed_path(exp, j):
    return os.path.join(mc.REPO, exp["scenes"][j["scene"]]["prompts"], "composed", f"{j['view']}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    exp = mc.load_exp(os.path.join(mc.REPO, a.exp) if not os.path.isabs(a.exp) else a.exp)
    seen, bad = set(), []
    for j in mc.jobs(exp):
        key = (j["scene"], j["view"])
        if key in seen:
            continue
        seen.add(key)
        cap, p = mc.compose_prompt(exp, j), composed_path(exp, j)
        same = os.path.exists(p) and json.load(open(p)) == cap
        if a.check:
            print(f"{'ok     ' if same else 'DIFFERS'} {os.path.relpath(p, mc.REPO)}")
            bad += [] if same else [p]
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(cap, open(p, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {os.path.relpath(p, mc.REPO)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
