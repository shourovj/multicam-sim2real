"""Starter caption blocks for the video models, from a prompt file alone (no LLM, nothing scene-specific).

  $S2R_ROOT/envs/eval/.venv/bin/python p2s/3_generate/write_captions.py p2s/prompts/<name>.yaml [--out p2s/captions/<name>] [--force]

Writes the blocks p2s/3_generate/common/multicam_common.py composes a view's prompt from:
  world.json        shared by every view: the optional `setting:` of the prompt file, and the realism sentence
  cams/<cam>.json   one per view: what the view is (the prompt file's words for that camera) followed by the prompt,
                    and whether the camera is fixed, rides on something, or is handheld (from world/spec.yaml)
Every view is therefore told the same events in the same words, and differs only in its camera sentence.
These are a floor, not the captions a careful run wants: they say nothing the prompt does not (no colours or timings
from the simulation). Edit them, or replace them with blocks written from the report's fact card (world/r.md), then
run p2s/3_generate/compose_prompts.py for each experiment that uses them.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import promptfile as c  # noqa: E402

MOTION = {"fixed": "Static", "attached": "Moves with the {parent} it is mounted on", "handheld": "Handheld, slight natural shake",
          "orbit": "Circles the scene", "follow": "Follows the subject"}


def sentence(text):
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = c.load_prompt(a.prompt)
    out = os.path.join(c.REPO, a.out or os.path.join("p2s", "captions", p["name"]))
    if os.path.exists(os.path.join(out, "world.json")) and not a.force:
        sys.exit(f"[captions] {out} already has blocks; --force overwrites them")
    spec_path = os.path.join(c.scene_root(p), "world", "spec.yaml")
    cams = {x["id"]: x for x in yaml.safe_load(open(spec_path))["cameras"]} if os.path.exists(spec_path) else {}
    os.makedirs(os.path.join(out, "cams"), exist_ok=True)
    about = f"written by p2s/3_generate/write_captions.py from {p['path']} (starter block: only what the prompt file says)"
    world = {"_about": f"World block for {p['name']}, shared by every view; {about}."}
    if p.get("setting"):
        world["background_setting"] = sentence(" ".join(p["setting"].split()))
    world["context"] = "Realistic real-world video, natural lighting and motion, not a rendering."
    json.dump(world, open(os.path.join(out, "world.json"), "w"), indent=1)
    for cid, what in p["cameras"].items():
        cam = cams.get(cid, {})
        motion = MOTION.get(cam.get("type", "fixed"), "Static").format(parent=cam.get("parent", "object"))
        block = {"_about": f"Camera block: {cid} ({what}); {about}.",
                 "temporal_caption": f"{sentence('Filmed from ' + what)} {sentence(p['prompt'])}",
                 "cinematography": {"camera_motion": motion},
                 "style_medium": "Live-action video, photoreal"}
        json.dump(block, open(os.path.join(out, "cams", f"{cid}.json"), "w"), indent=1)
    print(f"[captions] {len(p['cameras'])} camera blocks + world block -> {os.path.relpath(out, c.REPO)}")


if __name__ == "__main__":
    main()
