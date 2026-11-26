"""Shared by the prompt-to-sim pipeline scripts (p2s): the prompt file, the folder layout, and how a PhysX
output X is described to the rest of the repo (gen/ and eval/). Stdlib + yaml, so it runs in every env.

A prompt file (p2s/prompts/<name>.yaml) is the only input of the pipeline:
  name, rev          the scene's folder: $S2R_DATA/<out_root>/<name>/r<rev>/   (out_root defaults to cv-project/p2s)
  prompt             what happens, in plain words
  cameras            {id: what this view is}: the views every clip is rendered and generated from
  clip               {fps, frames, resolution}: fixed for a benchmark, so every scene gives clips of one format
  scene              optional: a scene file S already written for this prompt (sim/physx/scenes/*.json). Without it
                     the SceneAgent (an LLM) writes S from the prompt.
  score_static       optional: static things that eval should score as objects (a glass that is poured into)

Layout under the scene's folder:
  agent/task.md, agent/it<N>/     the SceneAgent's iterations: scene.json (S), x.json + streams/ (X), r.md (report)
  world/                          the accepted iteration, plus spec.yaml and world.json (what gen/ and eval/ read)
  cams/<cam>/bundle/              render_x.py + export_x.py: the same bundle as legacy/sim/render_world.py + export_world.py
"""

import hashlib
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(REPO, "sim", "physx", "build")
DOCS = os.path.join(REPO, "sim", "physx", "docs")
OUT_ROOT = "cv-project/p2s"
OBJECT_IDS = range(1, 100)       # instance ids of scored objects, as in legacy/sim/world/spec.py; 100+ = scenery
GROUND_ID, FIRST_STATIC_ID, LAST_ID = 100, 101, 255
NO_GEOMETRY = ("magnet", "explosion", "joint")


class PromptError(ValueError):
    pass


def load_prompt(path):
    """The prompt file as a dict, checked, with `path` (relative to the repo) and `sha256` added."""
    text = open(path, "rb").read()
    p = yaml.safe_load(text)

    def need(cond, msg):
        if not cond:
            raise PromptError(f"{path}: {msg}")

    for k in ("name", "rev", "prompt", "cameras", "clip"):
        need(k in p, f"missing key {k}")
    need(isinstance(p["cameras"], dict) and p["cameras"], "cameras must be {id: what the view is}")
    c = p["clip"]
    need(all(k in c for k in ("fps", "frames", "resolution")), "clip needs fps, frames and resolution")
    need(c["frames"] % 4 == 1, f"clip.frames must be 1 + 4n (video models), got {c['frames']}")
    p["prompt"] = " ".join(p["prompt"].split())
    p["path"] = os.path.relpath(os.path.abspath(path), REPO)
    # of the content, not the bytes: editing a comment does not make an accepted world out of date
    p["sha256"] = hashlib.sha256(json.dumps({k: p[k] for k in sorted(p) if k != "path"}, sort_keys=True).encode()).hexdigest()
    return p


def scene_root(p):
    return os.path.join(os.environ["S2R_DATA"], p.get("out_root", OUT_ROOT), p["name"], f"r{p['rev']}")


def full_prompt(p):
    """The prompt with its views, as one paragraph: what the SceneAgent is given and what S records as meta.prompt."""
    views = "; ".join(f"{cid}: {what}" for cid, what in p["cameras"].items())
    return f"{p['prompt']} Views: {views}."


def task_text(p, run_dir, gpu):
    """The SceneAgent's whole instruction for this prompt: where the manual is, the prompt, what is fixed, how to run."""
    c = p["clip"]
    duration = (c["frames"] - 1) / c["fps"]
    cams = "\n".join(f"- `{cid}`: {what}" for cid, what in p["cameras"].items())
    return f"""# SceneAgent task: {p['name']}

You are the SceneAgent. Read these two files first, completely; they are your manual:

- `{os.path.relpath(os.path.join(DOCS, '2026-11-24-1859-sceneagent-prompt.md'), REPO)}` (the loop, the scene format, expectations)
- `{os.path.relpath(os.path.join(DOCS, '2026-11-24-1859-sceneagent-iteration.md'), REPO)}` (what to do after a FAIL)

`sim/physx/scenes/` has scenes that pass; start from the closest one.

## The prompt

{full_prompt(p)}

## Fixed for this task (do not change)

- `"name": "{p['name']}"`, and `meta.prompt` is the prompt above, word for word.
- `clock`: `"fps": {c['fps']}, "duration_s": {duration:g}` ({c['frames']} frames). Choose `substeps` yourself.
- Cameras: exactly these ids, each with `"resolution": {json.dumps(c['resolution'])}`:
{cams}
- Write at least one `in_view` expectation per camera and one expectation per thing the prompt says happens.

## How to run (this replaces section 1 of the manual; do not source sim/physx/env.sh)

Write iteration N's scene to `{run_dir}/it<N>/scene.json`, then run from the repository root:

```bash
p2s/1_scene/iterate.sh {run_dir} <N>
```

It simulates on GPU {gpu}, checks, and prints the status; the report is `{run_dir}/it<N>/r.md`. Exit code 0 is PASS,
1 is FAIL (read r.md, fix, write `it<N+1>/scene.json` with `meta.iteration` N+1, run again), 3 is STOP.

## When you are done

Stop at PASS or STOP. End your last message with one line: `RESULT: PASS it<N>` or `RESULT: STOP it<N>: <why>`.
"""


def last_iteration(run_dir):
    """(N, status) of the highest it<N> folder that has a report, or (None, None)."""
    best = (None, None)
    for d in os.listdir(run_dir) if os.path.isdir(run_dir) else []:
        rj = os.path.join(run_dir, d, "r.json")
        if d.startswith("it") and d[2:].isdigit() and os.path.exists(rj):
            n = int(d[2:])
            if best[0] is None or n > best[0]:
                best = (n, json.load(open(rj)).get("status"))
    return best


def describe(p, x):
    """What gen/ and eval/ need to know about a PhysX output X: the clip format, the cameras, and an instance id for
    everything that is drawn. Scored objects (ids 1-99) are everything that can move; scenery (100+) is the ground,
    static bodies and traffic signals. A soft body mounted on a host is part of its host. Returns the spec dict."""
    objects, static, ids = [], [], {}
    score_static = set(p.get("score_static") or [])
    known = {e["id"] for e in x["entities"]}
    for s in score_static:
        if s not in known:
            raise PromptError(f"{p['path']}: score_static names {s}, which is not in the scene")
    for e in x["entities"]:
        kind = e["kind"]
        if kind in NO_GEOMETRY or (kind == "soft_body" and e.get("host")):
            continue
        scenery = kind == "traffic_signal" or (kind == "rigid_body" and e.get("type") == "static" and e["id"] not in score_static)
        if scenery:
            i = GROUND_ID if e["id"] == "ground" else min(FIRST_STATIC_ID + len([s for s in static if s["id"] != "ground"]), LAST_ID)
            static.append({"id": e["id"], "instance_id": i, "what": kind})
        else:
            i = len(objects) + 1
            if i not in OBJECT_IDS:
                raise PromptError(f"{p['path']}: more than {len(OBJECT_IDS)} moving things; eval scores at most that many")
            objects.append({"id": e["id"], "instance_id": i, "asset": {"type": kind}})
        ids[e["id"]] = i
    part_of = {e["id"]: e["host"] for e in x["entities"] if e["kind"] == "soft_body" and e.get("host")}
    cams = [{"id": c["id"], "type": c["type"], "fov_deg": c["fov_deg"], **({"parent": c["spec"]["parent"]} if c["spec"].get("parent") else {})}
            for c in x["cameras"]]
    clock = x["clock"]
    return {"name": p["name"], "rev": p["rev"], "source": "p2s", "prompt_file": p["path"], "prompt_sha256": p["sha256"],
            "fps": int(round(clock["fps"])), "frames": clock["frames"], "resolution": list(x["cameras"][0]["resolution"]),
            "tracks": {"points_per_object": 24}, "cameras": cams, "objects": objects, "static": static, "part_of": part_of}


def check_format(p, x):
    """The accepted X against what the prompt file fixes. Returns a list of problems (empty = fine)."""
    c, bad = p["clip"], []
    if int(round(x["clock"]["fps"])) != c["fps"] or x["clock"]["frames"] != c["frames"]:
        bad.append(f"clock is {x['clock']['frames']} frames at {x['clock']['fps']:g} fps, the prompt file fixes {c['frames']} at {c['fps']}")
    got = [cam["id"] for cam in x["cameras"]]
    if sorted(got) != sorted(p["cameras"]):
        bad.append(f"cameras are {got}, the prompt file fixes {list(p['cameras'])}")
    for cam in x["cameras"]:
        if list(cam["resolution"]) != list(c["resolution"]):
            bad.append(f"camera {cam['id']} is {cam['resolution']}, the prompt file fixes {c['resolution']}")
    return bad
