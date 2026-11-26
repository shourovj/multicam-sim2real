"""Build one multicam run's inputs and its Cosmos request, for Cosmos3-Nano and Cosmos3-Nano-Sim-Transfer (both run on
cosmos-framework and take the same request format). Runs in the cosmos-framework venv: it uses Cosmos's own blur code,
so a blur control made here is exactly what Cosmos would compute from the same video.

  cd $S2R_ROOT/cosmos-framework && .venv/bin/python <repo>/gen/cosmos3_nano/request.py <experiment.yaml> <scene> <view> <cond> <seed>

The shared inputs (cut, prompt, negative prompt, inputs.json) come from p2s/3_generate/common/multicam_inputs.py. Added here:
  inputs/<view>/control_<cond>.mp4   the blur control (preset none = the video as rendered)
  <run folder>/request.json          the Cosmos request: mode, sampler, and one entry per control
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))
import multicam_common as mc  # noqa: E402
from multicam_inputs import finish, simulator_control, start, write_mp4  # noqa: E402


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
    exp, j, out, files = start(exp_path, scene, view, cond, seed)
    inp = os.path.join(mc.scene_root(exp, scene), "inputs", view)
    hints, vision = {}, None
    for hint, cfg in j["cond_spec"].items():
        cfg = cfg or {}
        if hint == "blur":                     # built here with Cosmos's own blur code, from the render
            ctrl = os.path.join(inp, f"control_{cond}.mp4")
            if not os.path.exists(ctrl):
                rgb = np.load(mc.source_rgb(exp, j))
                assert len(rgb) == exp["video"]["num_frames"], f"{view}: {len(rgb)} frames, expected {exp['video']['num_frames']}"
                write_mp4(ctrl, blur_control(rgb, cfg.get("preset_blur_strength", "none")), exp["video"]["fps"])
        elif hint == "edge" and cfg.get("from") == "rgb":   # Cosmos computes Canny itself from the render, after resizing
            vision = simulator_control(exp, j, "rgb")
            hints[hint] = {"preset_edge_threshold": cfg.get("preset_edge_threshold", "medium"),
                           **({"weight": cfg["weight"]} if "weight" in cfg else {})}
            continue
        elif hint in ("depth", "seg", "edge"):  # exact simulator controls, exported with the bundle (p2s/2_render/control_videos.py)
            ctrl = simulator_control(exp, j, hint)
        else:
            raise SystemExit(f"{cond}: unknown hint {hint}")
        hints[hint] = {"control_path": ctrl, **({"weight": cfg["weight"]} if "weight" in cfg else {})}
    neg = mc.negative_prompt_path(exp)          # frozen copy in the repo (transfer: the cookbook's; text2video: the default)
    v, s = exp["video"], exp["sampler"]
    if exp.get("mode") == "text2video":
        # Text only: no control, no first frame. Sampler = Nano's text-to-video defaults, written out so the request
        # says what ran. The negative prompt is passed as the same compact JSON string the defaults would load.
        assert not hints and vision is None, f"{cond}: a text2video run takes no control"
        req = dict(name="out", seed=int(seed), resolution=v["resolution"], aspect_ratio=v["aspect_ratio"], fps=v["fps"],
                   num_frames=v["num_frames"], model_mode="text2video", num_steps=s["num_steps"], guidance=s["guidance"],
                   shift=s["shift"], negative_prompt=json.dumps(json.load(open(neg))), prompt_path="prompt.json")
    elif mc.family(exp) == "sim-transfer":
        # Cosmos3-Nano-Sim-Transfer: the rest comes from NVIDIA's defaults file (sample_args.json), passed by the launcher.
        # One control; guidance 1 means no classifier-free guidance, so the negative prompt is unused (defaults: "").
        assert len(hints) == 1, f"{cond}: Sim-Transfer takes exactly one control"
        req = dict(name="out", seed=int(seed), resolution=v["resolution"], aspect_ratio=v["aspect_ratio"], fps=v["fps"],
                   num_frames=v["num_frames"], max_frames=v["num_frames"], model_mode="video2video", num_steps=s["num_steps"],
                   guidance=s["guidance"], control_guidance=s["control_guidance"], shift=s["shift"],
                   num_first_chunk_conditional_frames=0, num_conditional_frames=0, transfer_resize_mode="direct_resize",
                   transfer_output_size="model", prompt_path="prompt.json", **hints)
    else:
        req = dict(name="out", seed=int(seed), resolution=v["resolution"], aspect_ratio=v["aspect_ratio"], fps=v["fps"], num_frames=v["num_frames"],
                   model_mode="video2video", shift=s["shift"], num_steps=s["num_steps"], guidance=s["guidance"],
                   control_guidance=s["control_guidance"], num_video_frames_per_chunk=121, num_conditional_frames=1,
                   num_first_chunk_conditional_frames=0, share_vision_temporal_positions=True, negative_metadata_mode="none",
                   negative_prompt_keep_metadata=False, negative_prompt_file="negative_prompt.json", emphasize_control_in_prompt=False,
                   prompt_path="prompt.json", **hints)
    if vision:
        req["vision_path"] = vision      # source of the edge map only; with no conditional frames it is not a first frame
        files["vision_rgb"] = mc.sha256_file(vision)
    files.update({f"control_{h}": mc.sha256_file(c["control_path"]) for h, c in hints.items() if "control_path" in c})
    finish(exp, j, out, req, files)


if __name__ == "__main__":
    prepare(*sys.argv[1:6])
