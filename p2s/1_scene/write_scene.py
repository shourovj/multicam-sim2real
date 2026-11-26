"""Step 1 of the prompt-to-sim pipeline: from a prompt file to an accepted simulated world.

  $S2R_ROOT/envs/eval/.venv/bin/python p2s/1_scene/write_scene.py p2s/prompts/<name>.yaml [--gpu N] [--force]
      [--task-only] [--agent-cmd "<command>"] [--print-root]

The scene file S comes from the prompt file's `scene:` if it has one (a scene already written for this prompt), else
from the SceneAgent: an LLM that is given agent/task.md (p2s/promptfile.py: task_text) and iterates with
p2s/1_scene/iterate.sh until the checker says PASS. The agent is a command that reads the task on stdin; the default is
Claude Code run headless, allowed to read, write files and run iterate.sh only. --task-only writes task.md and stops,
so the task can be given to any other LLM tool by hand; run this script again afterwards to accept its result.

Accepting copies the last iteration (it must be PASS and match the prompt file's clip format and camera ids) to
world/ and writes world/spec.yaml and world/world.json, which gen/ and eval/ read in place of a spec v2 file.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import promptfile as c  # noqa: E402

AGENT_CMD = ("claude -p --allowedTools Read Write Edit Glob Grep 'Bash(p2s/1_scene/iterate.sh:*)' --add-dir {run_dir}")


def iterate(run_dir, n, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.run([os.path.join(c.REPO, "p2s", "1_scene", "iterate.sh"), run_dir, str(n)], env=env, cwd=c.REPO).returncode


def free_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return min((int(line.split(",")[1]), int(line.split(",")[0])) for line in out.strip().splitlines())[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="write the scene again even if a world is already accepted")
    ap.add_argument("--task-only", action="store_true")
    ap.add_argument("--agent-cmd", default=os.environ.get("P2S_AGENT_CMD", AGENT_CMD))
    ap.add_argument("--print-root", action="store_true")
    a = ap.parse_args()
    p = c.load_prompt(a.prompt)
    root = c.scene_root(p)
    if a.print_root:
        print(root)
        return
    run_dir, world = os.path.join(root, "agent"), os.path.join(root, "world")
    spec_path = os.path.join(world, "spec.yaml")
    if os.path.exists(spec_path) and not a.force:
        if yaml.safe_load(open(spec_path)).get("prompt_sha256") == p["sha256"]:
            print(f"[p2s] {p['name']} r{p['rev']}: world already accepted -> {root}")
            return
        sys.exit(f"[p2s] {spec_path} was made from a different version of {p['path']}. Raise `rev` in the prompt file "
                 "(a new folder), or pass --force to replace this world; its renders and clips then need redoing.")
    os.makedirs(run_dir, exist_ok=True)
    gpu = free_gpu() if a.gpu is None else a.gpu

    n, status = c.last_iteration(run_dir)
    if a.force or status != "PASS":
        if p.get("scene"):                                   # a scene already written for this prompt
            src = os.path.join(c.REPO, p["scene"])
            n = json.load(open(src)).get("meta", {}).get("iteration", 1)
            os.makedirs(os.path.join(run_dir, f"it{n}"), exist_ok=True)
            shutil.copyfile(src, os.path.join(run_dir, f"it{n}", "scene.json"))
            iterate(run_dir, n, gpu)
        else:                                                # the SceneAgent writes it
            task = c.task_text(p, run_dir, gpu)
            open(os.path.join(run_dir, "task.md"), "w").write(task)
            if a.task_only:
                print(f"[p2s] task written -> {os.path.join(run_dir, 'task.md')}")
                return
            cmd = shlex.split(a.agent_cmd.format(run_dir=run_dir))
            print(f"[p2s] SceneAgent: {' '.join(cmd)}  (log: {os.path.join(run_dir, 'agent.log')})", flush=True)
            with open(os.path.join(run_dir, "agent.log"), "w") as log:
                rc = subprocess.run(cmd, input=task, text=True, stdout=log, stderr=subprocess.STDOUT, cwd=c.REPO,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
            print(f"[p2s] SceneAgent exited with {rc}")
        n, status = c.last_iteration(run_dir)
    if status != "PASS":
        sys.exit(f"[p2s] {p['name']}: not accepted, the last iteration is {'it%d: %s' % (n, status) if n else 'missing'}. "
                 f"See {run_dir}")

    it = os.path.join(run_dir, f"it{n}")
    x = json.load(open(os.path.join(it, "x.json")))
    bad = c.check_format(p, x)
    if bad:
        sys.exit(f"[p2s] {p['name']}: it{n} passes the checker but not the prompt file:\n  " + "\n  ".join(bad))
    spec = c.describe(p, x)
    if os.path.isdir(world):
        shutil.rmtree(world)
    os.makedirs(world)
    for f in ("scene.json", "x.json", "r.md", "r.json"):
        shutil.copyfile(os.path.join(it, f), os.path.join(world, f))
    shutil.copytree(os.path.join(it, "streams"), os.path.join(world, "streams"))
    yaml.safe_dump(spec, open(spec_path, "w"), sort_keys=False, default_flow_style=None, width=200)
    report = json.load(open(os.path.join(it, "r.json")))
    recorded = x["scene"].get("meta", {}).get("prompt", "")
    json.dump({"name": p["name"], "rev": p["rev"], "source": "p2s", "prompt_file": p["path"], "prompt": c.full_prompt(p),
               "scene_prompt": recorded, "accepted_iteration": n, "fps": spec["fps"], "frames": spec["frames"],
               "times_s": [f / spec["fps"] for f in range(spec["frames"])], "resolution": spec["resolution"],
               "cameras": spec["cameras"], "events": x["events"],
               "checks": [{"type": "p2s_report", "ok": True, "detail": {"status": report["status"], "counts": report["counts"]}}]},
              open(os.path.join(world, "world.json"), "w"), indent=1)
    if " ".join(recorded.split()) != c.full_prompt(p):
        print(f"[p2s] note: the scene's own meta.prompt differs from the prompt file's text (both are in {world}/world.json)")
    print(f"[p2s] {p['name']} r{p['rev']}: it{n} accepted, {len(spec['objects'])} objects, {len(spec['static'])} scenery, "
          f"{len(spec['cameras'])} cameras -> {root}")


if __name__ == "__main__":
    main()
