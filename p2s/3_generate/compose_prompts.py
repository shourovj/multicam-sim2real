"""Write the exact prompt the model receives for every view of an experiment into the repo, so the prompt is a
tracked file, not only something composed at run time.

  $S2R_ROOT/envs/eval/.venv/bin/python p2s/3_generate/compose_prompts.py --exp legacy/configs/experiments/multicam_v1.yaml [--check]

For each scene and view it writes <prompts dir>/composed/<view>.json or <view>.txt, in the format the experiment's
model takes (multicam_common.FAMILIES):
  .json  Cosmos: the structured caption = world block + that camera's or cut's block + duration, fps, resolution
  .txt   Wan, LTX: the same blocks as one paragraph (multicam_common.prose_prompt)
Negative prompts are frozen files in p2s/3_generate/negative_prompts/, named by the experiment file. p2s/3_generate/run_multicam.py refuses to
launch a job whose composed file is missing or differs, so edit the blocks, re-run this, and commit both.
--check only reports differences.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import multicam_common as mc  # noqa: E402


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
        cap, p = mc.model_prompt(exp, j), mc.composed_path(exp, j)
        same = mc.read_composed(p) == cap
        if a.check:
            print(f"{'ok     ' if same else 'DIFFERS'} {os.path.relpath(p, mc.REPO)}")
            bad += [] if same else [p]
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if isinstance(cap, str):
            open(p, "w").write(cap + "\n")
        else:
            json.dump(cap, open(p, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {os.path.relpath(p, mc.REPO)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
