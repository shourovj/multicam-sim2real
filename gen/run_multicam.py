"""Run every job of a multicam experiment (configs/experiments/*.yaml) on whichever GPUs have room.

  source ~/.sim2real.env; $S2R_ROOT/envs/eval/.venv/bin/python gen/run_multicam.py --exp configs/experiments/multicam_v1.yaml \
      [--status] [--dry-run] [--only scene=..,view=..,cond=..,seed=..] [--rerun-stale]

A job is done when its run.json says done AND its input recipe (multicam_common.recipe: hashes of the spec, the
simulated world, the source frames, the prompt, the condition, the model and sampler settings, the input code)
still matches. A finished job whose inputs changed is reported as stale and only rerun with --rerun-stale; its old
output is kept as out_video.stale_<time>.mp4. GPU memory is read right before each launch; a GPU already running one
of our jobs is skipped. Each launch and finish is appended to $S2R_DATA/<out_root>/manifest.jsonl.
To stop: touch $S2R_DATA/<out_root>/STOP (no new jobs start; running ones finish).
"""

import argparse
import json
import os
import re
import subprocess
import time

import multicam_common as mc
from run_matrix import free_gib, my_gpus

REPO = mc.REPO


def gpu_names():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {int(i): n.strip() for i, n in (l.split(",", 1) for l in out.strip().splitlines())}


def our_mem_gib():
    """Memory (GiB) of this user's compute processes, per GPU index."""
    q = lambda what, fields: subprocess.run(["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True).stdout.strip().splitlines()
    index = dict(l.split(", ")[::-1] for l in q("gpu", "index,uuid"))
    mem = {}
    for l in q("compute-apps", "gpu_uuid,pid,used_memory"):
        uuid, pid, used = l.split(", ")
        try:
            if os.stat(f"/proc/{pid}").st_uid == os.getuid():
                g = int(index[uuid])
                mem[g] = mem.get(g, 0) + int(used) / 1024
        except (OSError, KeyError, ValueError):
            pass
    return mem


def state(exp, j):
    d = mc.run_dir(exp, j)
    root = mc.scene_root(exp, j["scene"])
    need = [os.path.join(root, "world", "world.npz")] + (mc.cut_sources(exp, j) if j["kind"] == "cut" else [mc.source_rgb(exp, j)])
    if not all(os.path.exists(p) for p in need):
        return "blocked"
    try:
        pid = int(open(os.path.join(d, ".running")).read())
        os.kill(pid, 0)
        return "running"
    except (OSError, ValueError):
        pass
    rj = os.path.join(d, "run.json")
    if os.path.exists(os.path.join(d, "out_video.mp4")) and os.path.exists(rj):
        r = json.load(open(rj))
        if r.get("status") == "done":
            return "done" if mc.same_recipe(r.get("recipe", {}), mc.recipe(exp, j)) else "stale"
    return "pending"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="filters, e.g. scene=intersection_4way,seed=0")
    ap.add_argument("--rerun-stale", action="store_true")
    a = ap.parse_args()
    exp_path = os.path.join(REPO, a.exp) if not os.path.isabs(a.exp) else a.exp
    exp = mc.load_exp(exp_path)
    flt = dict(kv.split("=") for kv in a.only.split(",") if kv)
    jobs = [j for j in mc.jobs(exp) if all(str(j[k]) == v for k, v in flt.items())]
    base = os.path.join(os.environ["S2R_DATA"], exp["out_root"])
    os.makedirs(base, exist_ok=True)
    states = {mc.run_id(exp, j): state(exp, j) for j in jobs}
    if a.status or a.dry_run:
        for j in jobs:
            print(f"{states[mc.run_id(exp, j)]:8s} {mc.run_id(exp, j)}")
        from collections import Counter
        print(dict(Counter(states.values())), f"of {len(jobs)}")
        return
    want = ("pending", "stale") if a.rerun_stale else ("pending",)
    queue = [j for j in jobs if states[mc.run_id(exp, j)] in want]
    blocked = [j for j in jobs if states[mc.run_id(exp, j)] == "blocked"]     # rechecked every loop: start once bundles exist
    names, g = gpu_names(), exp["gpu"]
    running, tries = {}, {}
    log = lambda row: open(os.path.join(base, "manifest.jsonl"), "a").write(json.dumps(row) + "\n")
    print(f"[multicam] {len(queue)} of {len(jobs)} jobs to run ({sum(s == 'blocked' for s in states.values())} blocked on missing bundles)", flush=True)
    while queue or running or blocked:
        for j in list(blocked):
            if state(exp, j) != "blocked":
                blocked.remove(j)
                queue.append(j)
                print(f"[multicam] unblocked {mc.run_id(exp, j)}", flush=True)
        mem = our_mem_gib()
        for gi, r in list(running.items()):
            r["peak"] = max(r["peak"], mem.get(gi, 0.0))
            if r["proc"].poll() is None:
                continue
            del running[gi]
            j, d = r["job"], mc.run_dir(exp, r["job"])
            if os.path.exists(os.path.join(d, ".running")):
                os.remove(os.path.join(d, ".running"))
            ok = r["proc"].returncode == 0 and os.path.exists(os.path.join(d, "out_video.mp4"))
            info = json.load(open(os.path.join(d, "inputs.json"))) if os.path.exists(os.path.join(d, "inputs.json")) else {}
            from world_git import git_state
            snap = None
            if os.path.exists(os.path.join(d, "run.log")):
                m = re.search(r"models--nvidia--Cosmos3-Nano/snapshots/([0-9a-f]{40})", open(os.path.join(d, "run.log"), errors="ignore").read())
                snap = m.group(1) if m else None
            row = {**info, "model_snapshot": snap, "model_snapshot_matches_pin": snap == exp["model"]["revision"], "status": "done" if ok else "failed", "returncode": r["proc"].returncode, "gpu": gi, "gpu_name": names.get(gi),
                   "started": r["started"], "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - r["t0"]),
                   "peak_gpu_gib": round(r["peak"], 1), "git": git_state(REPO), "experiment_file": os.path.relpath(exp_path, REPO),
                   "experiment_file_sha256": mc.sha256_file(exp_path)}
            json.dump(row, open(os.path.join(d, "run.json"), "w"), indent=1)
            log({k: row[k] for k in ("run_id", "status", "gpu", "started", "finished", "wall_s", "peak_gpu_gib") if k in row})
            print(f"[multicam] {row['status']}: {mc.run_id(exp, j)} on GPU {gi}, {row['wall_s']} s, {row['peak_gpu_gib']} GiB", flush=True)
            if not ok and tries.get(mc.run_id(exp, j), 0) < g.get("retries", 1):
                tries[mc.run_id(exp, j)] = tries.get(mc.run_id(exp, j), 0) + 1
                queue.append(j)
        stop = os.path.exists(os.path.join(base, "STOP"))
        if stop and not running:
            print("[multicam] STOP file found, exiting", flush=True)
            return
        if not queue and not running and blocked:
            time.sleep(60)
            continue
        if queue and not stop and len(running) < g["max_parallel"]:
            free, busy = free_gib(), my_gpus() | set(running)
            pick = next((gi for gi, f in sorted(free.items(), key=lambda x: -x[1]) if gi not in busy and f >= g["min_free_gib"]), None)
            if pick is not None:
                j = queue.pop(0)
                cp = os.path.join(REPO, exp["scenes"][j["scene"]]["prompts"], "composed", f"{j['view']}.json")
                if not os.path.exists(cp) or json.load(open(cp)) != mc.compose_prompt(exp, j):
                    print(f"[multicam] SKIP {mc.run_id(exp, j)}: {os.path.relpath(cp, REPO)} is missing or out of date; "
                          f"run gen/compose_prompts.py and commit it", flush=True)
                    continue
                d = mc.run_dir(exp, j)
                os.makedirs(d, exist_ok=True)
                if os.path.exists(os.path.join(d, "out_video.mp4")):              # a stale rerun: keep the old output
                    os.replace(os.path.join(d, "out_video.mp4"), os.path.join(d, f"out_video.stale_{time.strftime('%Y%m%d_%H%M%S')}.mp4"))
                env = dict(os.environ, EXP=os.path.relpath(exp_path, REPO), SCENE=j["scene"], VIEW=j["view"], COND=j["cond"],
                           SEED=str(j["seed"]), CUDA_VISIBLE_DEVICES=str(pick))
                proc = subprocess.Popen(["bash", os.path.join(REPO, "gen", "run_cosmos_multicam.sh")], env=env,
                                        stdout=open(os.path.join(d, "launcher.log"), "w"), stderr=subprocess.STDOUT)
                open(os.path.join(d, ".running"), "w").write(str(proc.pid))
                running[pick] = {"job": j, "proc": proc, "t0": time.time(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "peak": 0.0}
                log({"run_id": mc.run_id(exp, j), "status": "started", "gpu": pick, "free_gib_at_launch": round(free[pick], 1),
                     "started": running[pick]["started"]})
                print(f"[multicam] started {mc.run_id(exp, j)} on GPU {pick} ({free[pick]:.0f} GiB free)", flush=True)
                time.sleep(g.get("launch_gap_s", 120))
                continue
        time.sleep(30)
    print("[multicam] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
