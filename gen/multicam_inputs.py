"""Build one multicam run's inputs and its Cosmos3-Nano request (runs in the cosmos-framework venv: it uses Cosmos's
own blur code, so a control made here is exactly what Cosmos would compute from the same video).

  cd $S2R_ROOT/cosmos-framework && .venv/bin/python <repo>/gen/multicam_inputs.py <experiment.yaml> <scene> <view> <cond> <seed>

Writes (shared across seeds, built once):
  cut/<cut>/{rgb.npy, instance.npy, depth.npy, rgb.mp4, schedule.json}   for a cut view: frames taken from each camera's
                                                                          bundle by the experiment's schedule, one clock
  inputs/<view>/prompt.json                                               the composed caption (multicam_common.compose_prompt)
  inputs/<view>/control_<cond>.mp4                                        the blur control (preset none = the video as rendered)
and in the run folder: request.json, prompt.json, negative_prompt.json, inputs.json (the recipe hashes + file hashes).
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
import multicam_common as mc  # noqa: E402
from world.controls import CONTROLS, read_mp4  # noqa: E402

NEG = os.path.join(mc.REPO, "configs/prompts/cosmos3_negative_prompt.json")   # frozen copy of the cookbook negative prompt


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
        np.save(os.path.join(out, f".{name}.npy"), arrays[name])
        os.replace(os.path.join(out, f".{name}.npy"), os.path.join(out, f"{name}.npy"))
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


def blur_control(rgb, preset):
    if preset == "none":
        return rgb
    import torch
    from cosmos_framework.inference.args import PresetBlurStrength, PresetEdgeThreshold, TransferHintKey
    from cosmos_framework.inference.transfer import apply_transfer_control_augmentor
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2)                   # C T H W uint8
    y = apply_transfer_control_augmentor(x, hint_key=TransferHintKey.BLUR, preset_edge_threshold=PresetEdgeThreshold("medium"),
                                         preset_blur_strength=PresetBlurStrength(preset))
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return np.clip(y, 0, 255).astype(np.uint8).transpose(1, 2, 3, 0)


def prepare(exp_path, scene, view, cond, seed):
    exp = mc.load_exp(exp_path)
    j = next(x for x in mc.jobs(exp) if (x["scene"], x["view"], x["cond"], x["seed"]) == (scene, view, cond, int(seed)))
    if j["kind"] == "cut":
        build_cut(exp, j)
    inp = os.path.join(mc.scene_root(exp, scene), "inputs", view)
    os.makedirs(inp, exist_ok=True)
    cap = mc.compose_prompt(exp, j)
    json.dump(cap, open(os.path.join(inp, "prompt.json"), "w"), indent=1)
    hints = {}
    for hint, cfg in j["cond_spec"].items():
        if hint == "blur":                     # built here with Cosmos's own blur code, from the render
            ctrl = os.path.join(inp, f"control_{cond}.mp4")
            if not os.path.exists(ctrl):
                rgb = np.load(mc.source_rgb(exp, j))
                assert len(rgb) == exp["video"]["num_frames"], f"{view}: {len(rgb)} frames, expected {exp['video']['num_frames']}"
                write_mp4(ctrl, blur_control(rgb, cfg.get("preset_blur_strength", "none")), exp["video"]["fps"])
        elif hint in ("depth", "seg", "edge"):  # exact simulator controls, exported with the bundle (sim/world/controls.py)
            ctrl = mc.control_path(exp, j, hint)
            assert os.path.exists(ctrl), f"missing {ctrl}; export the bundle again (sim/export_world.py)"
        else:
            raise SystemExit(f"{cond}: unknown hint {hint}")
        hints[hint] = {"control_path": ctrl, **({"weight": cfg["weight"]} if "weight" in cfg else {})}
    out = mc.run_dir(exp, j)
    os.makedirs(out, exist_ok=True)
    json.dump(cap, open(os.path.join(out, "prompt.json"), "w"), indent=1)
    open(os.path.join(out, "negative_prompt.json"), "w").write(open(NEG).read())
    v, s = exp["video"], exp["sampler"]
    req = dict(name="out", seed=int(seed), resolution=v["resolution"], aspect_ratio=v["aspect_ratio"], fps=v["fps"], num_frames=v["num_frames"],
               model_mode="video2video", shift=s["shift"], num_steps=s["num_steps"], guidance=s["guidance"],
               control_guidance=s["control_guidance"], num_video_frames_per_chunk=121, num_conditional_frames=1,
               num_first_chunk_conditional_frames=0, share_vision_temporal_positions=True, negative_metadata_mode="none",
               negative_prompt_keep_metadata=False, negative_prompt_file="negative_prompt.json", emphasize_control_in_prompt=False,
               prompt_path="prompt.json", **hints)
    json.dump(req, open(os.path.join(out, "request.json"), "w"), indent=1)
    files = {"prompt.json": mc.sha256_file(os.path.join(out, "prompt.json")), "request.json": mc.sha256_file(os.path.join(out, "request.json"))}
    files.update({f"control_{h}": mc.sha256_file(c["control_path"]) for h, c in hints.items()})
    json.dump({"run_id": mc.run_id(exp, j), "job": {k: j[k] for k in ("scene", "view", "kind", "cond", "cond_spec", "seed", "probe")},
               "recipe": mc.recipe(exp, j), "files": files}, open(os.path.join(out, "inputs.json"), "w"), indent=1)
    print(f"[multicam_inputs] {mc.run_id(exp, j)} -> {out}")


if __name__ == "__main__":
    prepare(*sys.argv[1:6])
