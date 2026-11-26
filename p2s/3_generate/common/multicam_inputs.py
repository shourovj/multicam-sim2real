"""Build one multicam run's inputs. The parts every model shares are here; a model adds its own request on top.

  <python> p2s/3_generate/common/multicam_inputs.py <experiment.yaml> <scene> <view> <cond> <seed>

Run directly (eval env: numpy, OpenCV, yaml, ffmpeg), it prepares a run for the models that take one paragraph of
text and at most one control video (Wan2.2-VACE-Fun, LTX-2.5) and writes a model-independent request.json that their
launchers read. The Cosmos models write their own request with p2s/3_generate/models/cosmos3_nano/request.py, from the functions here.

Writes (shared across seeds and models, built once):
  cut/<cut>/{rgb.npy, instance.npy, depth.npy, rgb.mp4, schedule.json}   for a cut view: frames taken from each camera's
  cut/<cut>/controls/<hint>.mp4                                          bundle by the experiment's schedule, one clock
  inputs/<view>/prompt.json                                              the composed caption (multicam_common.compose_prompt)
and in the run folder: the prompt as the model gets it (prompt.json or prompt.txt), the negative prompt if the
experiment has one, request.json, and inputs.json (the recipe hashes + file hashes).
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "2_render"))
import multicam_common as mc  # noqa: E402
from control_videos import CONTROLS, read_mp4  # noqa: E402


def write_mp4(path, frames, fps, crf=12):
    T, H, W, _ = frames.shape
    tmp = path + f".{os.getpid()}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {path}"
    os.replace(tmp, path)                       # atomic: parallel jobs never read a half-written file


def build_cut(exp, j):
    out = os.path.join(mc.scene_root(exp, j["scene"]), "cut", j["view"])
    if os.path.exists(os.path.join(out, "schedule.json")):
        return out
    os.makedirs(out, exist_ok=True)
    sched = exp["cuts"][j["scene"]][j["view"]]
    root = mc.scene_root(exp, j["scene"])
    arrays = {}
    for name in ("rgb", "instance", "depth"):
        parts = []
        for s in sched:
            a = np.load(os.path.join(root, "cams", s["cam"], "bundle", f"{name}.npy"), mmap_mode="r")
            parts.append(np.asarray(a[s["frames"][0]:s["frames"][1] + 1]))
        arrays[name] = np.concatenate(parts)
        tmp = os.path.join(out, f".{name}.{os.getpid()}.npy")       # per-process name: models launched together don't collide
        np.save(tmp, arrays[name])
        os.replace(tmp, os.path.join(out, f"{name}.npy"))
    n = exp["video"]["num_frames"]
    assert len(arrays["rgb"]) == n, f"cut has {len(arrays['rgb'])} frames, expected {n}"
    for s in sched[1:]:
        assert s["frames"][0] % 4 == 1, f"shot starting at {s['frames'][0]} would share a latent frame with the shot before"
    write_mp4(os.path.join(out, "rgb.mp4"), arrays["rgb"], exp["video"]["fps"])
    # control videos: each shot's frames from that camera's own controls (same clock), joined
    os.makedirs(os.path.join(out, "controls"), exist_ok=True)
    for name in CONTROLS:
        srcs = [os.path.join(root, "cams", s["cam"], "bundle", "controls", f"{name}.mp4") for s in sched]
        if not all(os.path.exists(p) for p in srcs):
            continue
        parts = [read_mp4(p)[s["frames"][0]:s["frames"][1] + 1] for p, s in zip(srcs, sched)]
        write_mp4(os.path.join(out, "controls", f"{name}.mp4"), np.concatenate(parts), exp["video"]["fps"])
    per_frame = [s["cam"] for s in sched for _ in range(s["frames"][1] - s["frames"][0] + 1)]
    json.dump({"schedule": sched, "camera_per_frame": per_frame,
               "sources": {s["cam"]: mc.sha256_file(os.path.join(root, "cams", s["cam"], "bundle", "rgb.npy")) for s in sched}},
              open(os.path.join(out, "schedule.json"), "w"), indent=1)
    return out


def start(exp_path, scene, view, cond, seed):
    """Everything before the model's own request: the job, the cut (if the view is one), the prompt and the negative
    prompt in the run folder. Returns (exp, job, run folder, {file name: sha256} of what was written there)."""
    exp = mc.load_exp(exp_path)
    j = next(x for x in mc.jobs(exp) if (x["scene"], x["view"], x["cond"], x["seed"]) == (scene, view, cond, int(seed)))
    if j["kind"] == "cut":
        build_cut(exp, j)
    inp = os.path.join(mc.scene_root(exp, scene), "inputs", view)
    os.makedirs(inp, exist_ok=True)
    json.dump(mc.compose_prompt(exp, j), open(os.path.join(inp, "prompt.json"), "w"), indent=1)
    out = mc.run_dir(exp, j)
    os.makedirs(out, exist_ok=True)
    prompt = mc.model_prompt(exp, j)
    if isinstance(prompt, str):
        name = "prompt.txt"
        open(os.path.join(out, name), "w").write(prompt + "\n")
    else:
        name = "prompt.json"
        json.dump(prompt, open(os.path.join(out, name), "w"), indent=1)
    files = {name: mc.sha256_file(os.path.join(out, name))}
    neg = mc.negative_prompt_path(exp)          # a frozen copy in the repo, or none
    if neg:
        name = "negative_prompt" + os.path.splitext(neg)[1]
        open(os.path.join(out, name), "w").write(open(neg).read())
        files[name] = mc.sha256_file(os.path.join(out, name))
    return exp, j, out, files


def simulator_control(exp, j, hint):
    """An exact control video exported with the bundle (p2s/2_render/control_videos.py): depth, seg or edge."""
    ctrl = mc.control_path(exp, j, hint)
    assert os.path.exists(ctrl), f"missing {ctrl}; export the bundle again (p2s/build_scene.sh; legacy/sim/export_world.py for a spec v2 scene)"
    return ctrl


def finish(exp, j, out, req, files):
    """Write request.json and inputs.json, and print the line the run scripts read the run folder from."""
    json.dump(req, open(os.path.join(out, "request.json"), "w"), indent=1)
    files = {**files, "request.json": mc.sha256_file(os.path.join(out, "request.json"))}
    json.dump({"run_id": mc.run_id(exp, j), "job": {k: j[k] for k in ("scene", "view", "kind", "cond", "cond_spec", "seed", "probe")},
               "recipe": mc.recipe(exp, j), "files": files}, open(os.path.join(out, "inputs.json"), "w"), indent=1)
    print(f"[multicam_inputs] {mc.run_id(exp, j)} -> {out}")


def prepare(exp_path, scene, view, cond, seed):
    """A run of a prose-prompt model: the paragraph, at most one simulator control video, and the clip's format."""
    exp, j, out, files = start(exp_path, scene, view, cond, seed)
    assert mc.prompt_kind(exp) == "prose", f"{mc.family(exp)} builds its request with gen/{mc.FAMILIES[mc.family(exp)]['dir']}/request.py"
    assert len(j["cond_spec"]) <= 1, f"{cond}: {mc.family(exp)} takes at most one control video, got {sorted(j['cond_spec'])}"
    control = None
    for hint, cfg in j["cond_spec"].items():
        cfg = cfg or {}
        if hint not in ("depth", "edge") or cfg.get("from") == "rgb":
            raise SystemExit(f"{cond}: {mc.family(exp)} takes the simulator's depth or edge video, not {hint} {cfg}")
        control = {"hint": hint, "path": simulator_control(exp, j, hint), "strength": cfg.get("weight", 1.0)}
        files[f"control_{hint}"] = mc.sha256_file(control["path"])
    W, H = mc.scene_spec(exp, scene)["resolution"]
    v = exp["video"]
    req = dict(run_id=mc.run_id(exp, j), seed=int(seed), width=W, height=H, num_frames=v["num_frames"], fps=v["fps"],
               prompt_file="prompt.txt", negative_prompt_file="negative_prompt.txt" if "negative_prompt.txt" in files else None,
               control=control, sampler=exp["sampler"], model=exp["model"])
    finish(exp, j, out, req, files)


if __name__ == "__main__":
    prepare(*sys.argv[1:6])
