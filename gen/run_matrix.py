"""Single-camera rounds 1 and 2 (the multi-camera runner is gen/run_multicam.py).
Run every scene x condition x seed in configs/matrix.yaml, a few at a time, on whichever GPUs have room.

  source env.sh; $S2R_ROOT/envs/eval/.venv/bin/python gen/run_matrix.py [--status] [--only-seed 0] [--dry-run]

A job is done when its out_video.mp4 exists, so a rerun picks up where the last one stopped.
A running job leaves a .running file with its process id in its output folder, so a restarted runner does not start it twice.
Each launch is appended to $S2R_DATA/poc/manifest.jsonl, and again when it ends, with wall time and peak memory.
GPU memory is read right before every launch; a GPU already running one of our jobs is not used again.
To stop: touch $S2R_DATA/poc/STOP (no new jobs start; running ones finish).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from gpus import free_gib, my_gpus  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.environ["S2R_DATA"], "poc")
SCENES = os.path.join(DATA, "single-cam")   # one folder per scene; the manifest, logs and STOP stay in DATA
# model name in the matrix -> (run script, extra environment). The name is also the output folder prefix.
RUNNERS = {"vacefun-a14b": ("gen/wan22_vace_fun/singlecam_run.sh", {}),                         # Wan2.2-VACE-Fun-A14B (VideoX-Fun)
           "vace21-14B": ("gen/wan21_vace/singlecam_run_diffusers.sh", {"MODEL": "14B"}),       # Wan2.1-VACE-14B (diffusers)
           "vace21-1.3B": ("gen/wan21_vace/singlecam_run_diffusers.sh", {"MODEL": "1.3B"}),     # Wan2.1-VACE-1.3B (diffusers)
           "cosmos3-nano": ("gen/cosmos3_nano/singlecam_run.sh", {})}                           # Cosmos3-Nano base; keeps its own step counts


def out_dir(tag, scene, cond, seed):
    return os.path.join(SCENES, scene, cond, tag, f"seed{seed}")


def log(row):
    with open(os.path.join(DATA, "manifest.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(REPO, "configs/matrix.yaml"))
    ap.add_argument("--status", action="store_true", help="list every job and whether it is done, then exit")
    ap.add_argument("--only-seed", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print the launch order, then exit")
    a = ap.parse_args()
    m = yaml.safe_load(open(a.matrix))
    model, gpu, steps = m["model"], m["gpu"], m.get("steps")   # no steps: the model's own defaults, folder named after the model
    assert model in RUNNERS, f"unknown model {model!r}; known: {sorted(RUNNERS)}"
    tag = f"{model}-s{steps}" if steps else model
    jobs = [(scene, cond, seed) for seed in m["seeds"] for scene in m["scenes"] for cond in m["conditions"]
            if seed in m.get("seeds_override", {}).get(cond, m["seeds"]) and (a.only_seed is None or seed == a.only_seed)]
    done = lambda j: os.path.exists(os.path.join(out_dir(tag, *j), "out_video.mp4"))
    if a.status or a.dry_run:
        for j in jobs:
            print(f"{'done   ' if done(j) else 'pending'}  {j[0]:16s} {j[1]:10s} seed {j[2]}")
        print(f"{sum(map(done, jobs))} of {len(jobs)} done")
        return

    for scene, cond, _ in jobs:        # fail before launching anything if an input is missing
        if model != "cosmos3-nano" and cond not in ("A", "Aplus"):     # Nano builds its own controls from the bundle
            for f in ("src_video.mp4", "src_mask.mp4"):
                assert os.path.exists(os.path.join(SCENES, scene, cond, "inputs", f)), f"missing {scene}/{cond}/inputs/{f}"

    def in_progress(j):                # launched by an earlier runner and still alive
        try:
            os.kill(int(open(os.path.join(out_dir(tag, *j), ".running")).read()), 0)
            return True
        except (OSError, ValueError):
            return False

    queue = [j for j in jobs if not done(j) and not in_progress(j)]
    running = {}                       # gpu index -> (job, process, start time, mode)
    tries = {}                         # job -> times it was requeued after failing
    print(f"[matrix] {len(queue)} of {len(jobs)} jobs to run", flush=True)
    while queue or running:
        for g, (job, proc, t0, mode) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[g]
            if os.path.exists(os.path.join(out_dir(tag, *job), ".running")):
                os.remove(os.path.join(out_dir(tag, *job), ".running"))
            row = {"scene": job[0], "cond": job[1], "seed": job[2], "model": model, "steps": steps, "gpu": g, "mode": mode,
                   "status": "done" if done(job) else "failed", "returncode": proc.returncode,
                   "launcher_s": round(time.time() - t0), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            info = os.path.join(out_dir(tag, *job), "run.json")
            if os.path.exists(info):
                r = json.load(open(info))
                row.update(wall_s=r["wall_s"], peak_gpu_gib=r["peak_gpu_gib"])
            log(row)
            print(f"[matrix] {row['status']}: {job} on GPU {g} ({mode}), {row['launcher_s']} s", flush=True)
            if row["status"] == "failed" and tries.get(job, 0) < gpu.get("retries", 2):
                tries[job] = tries.get(job, 0) + 1      # usually another user's job took the GPU while ours was loading
                queue.append(job)
                print(f"[matrix] requeued {job}, retry {tries[job]}", flush=True)
        stop = os.path.exists(os.path.join(DATA, "STOP"))
        if stop and not running:
            print("[matrix] STOP file found, exiting", flush=True)
            return
        if queue and not stop and len(running) < gpu["max_parallel"]:
            free, busy = free_gib(), my_gpus() | set(running)          # read right before launching
            pick = next(((g, md["name"]) for md in gpu["modes"] for g, f in sorted(free.items(), key=lambda x: -x[1])
                         if g not in busy and f >= md["min_free_gib"]), None)
            if pick:
                g, mode = pick
                job = queue.pop(0)
                os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)
                script, extra = RUNNERS[model]
                env = dict(os.environ, SCENE=job[0], COND=job[1], SEED=str(job[2]), STEPS=str(steps or ""), MEMORY_MODE=mode,
                           CUDA_VISIBLE_DEVICES=str(g), **extra)
                logf = open(os.path.join(DATA, "logs", f"{job[0]}__{job[1]}__s{job[2]}.log"), "w")
                proc = subprocess.Popen(["bash", os.path.join(REPO, script)], env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[g] = (job, proc, time.time(), mode)
                os.makedirs(out_dir(tag, *job), exist_ok=True)
                open(os.path.join(out_dir(tag, *job), ".running"), "w").write(str(proc.pid))
                log({"scene": job[0], "cond": job[1], "seed": job[2], "model": model, "steps": steps, "gpu": g, "mode": mode,
                     "status": "started", "free_gib_at_launch": round(free[g], 1), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
                print(f"[matrix] started {job} on GPU {g} ({mode}, {free[g]:.0f} GiB free)", flush=True)
                time.sleep(gpu.get("launch_gap_s", 240))   # let it load and claim its memory before judging the GPUs again
                continue
        time.sleep(30)
    print("[matrix] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
