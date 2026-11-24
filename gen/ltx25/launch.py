"""Run LTX-2.5 (distilled, 22B) on one multi-camera run. Runs in the LTX-2 repo's venv; called by gen/ltx25/run.sh:

  $LTX_DIR/.venv/bin/python gen/ltx25/launch.py --request <run folder>/request.json --hub $HF_HUB_CACHE [--offload cpu]

Everything comes from request.json (gen/common/multicam_inputs.py): the prompt paragraph, the clip's size, length
and rate, the seed, and at most one simulator control video (depth or edges). The experiment file's model block
names the weight files, so the request says exactly which were used.

How the control goes in. LTX-2.5 has no control input of its own: it follows a reference video through an IC-LoRA,
and Lightricks' LTX-2.5 workflow uses the LTX-2.3 Union Control adapter (depth, canny, pose) for that. This is
ltx_pipelines' ICLoraPipeline with ONE stage at the full size: the distilled model's 8 fixed steps with the adapter
and the control as its reference. The command-line pipeline is two stages with the first at half size, which our
832x480 does not allow (240 is not a multiple of 32), and its second stage runs without the control.
Without a control the same single stage runs with no adapter: text to video.

Size. The frame must fit the model's 32 px grid, and so must the adapter's reference, which is the control at
1 / reference_downscale_factor of the frame (2 for the Union Control adapter). So the model generates at the nearest
size that is a multiple of 32 x that factor (832x512 for 832x480); the control is stretched to it first and the result
is stretched back, so every pixel stays over the simulator pixel it came from.

Audio. LTX-2.5 generates sound with the picture; it is dropped, and out_video.mp4 has video only.

Writes <run folder>/out_video.mp4 and launch.json (settings, files, wall time, peak GPU memory). guidance is 1: the
distilled model runs without classifier-free guidance, so there is no negative prompt.
"""

import argparse
import json
import math
import os
import subprocess
import time

import torch


def snapshot(hub, repo, revision):
    p = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", revision)
    assert os.path.isdir(p), f"missing {p}: download {repo} at revision {revision} (setup/README.md)"
    return p


def scale_video(src, dst, width, height):
    """Stretch a video to width x height (no crop, no padding), same frames and rate."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an", dst], check=True)


def count_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return int(out.strip().split(",")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--hub", required=True, help="the Hugging Face cache that holds the weights")
    ap.add_argument("--offload", default="none", choices=["none", "cpu", "disk"],
                    help="where the transformer's weights wait: none = all on the GPU (fastest), cpu / disk = streamed")
    a = ap.parse_args()
    t0 = time.time()
    req, d = json.load(open(a.request)), os.path.dirname(os.path.abspath(a.request))
    m, smp, ctl = req["model"], req["sampler"], req["control"]
    prompt = open(os.path.join(d, req["prompt_file"])).read().strip()
    W, H, n, fps = req["width"], req["height"], req["num_frames"], req["fps"]
    assert n % 8 == 1, f"LTX takes 1 + 8k frames, got {n}"

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ic_lora import ICLoraPipeline, ICLoraStageConfig
    from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.media_io.encode import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    assert smp["num_steps"] == len(DISTILLED_SIGMA_VALUES) - 1 and smp["guidance"] == 1.0, \
        f"the distilled model runs {len(DISTILLED_SIGMA_VALUES) - 1} fixed steps without guidance; the experiment file says {smp}"
    base, f = snapshot(a.hub, m["hf"], m["revision"]), m["files"]
    files = {k: os.path.join(base, v) for k, v in f.items()}
    loras, factor = [], 1
    if ctl:
        ad = m["control_adapter"]
        files["control_adapter"] = os.path.join(snapshot(a.hub, ad["hf"], ad["revision"]), ad["file"])
        factor = read_lora_reference_downscale_factor(files["control_adapter"])
        loras = [LoraPathStrengthAndSDOps(files["control_adapter"], 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
    for k, p in files.items():
        assert os.path.exists(p), f"missing {k}: {p} (setup/README.md)"
        print(f"[ltx25] {k}: {p}", flush=True)                     # the snapshot folders, for the run record

    grid = 32 * factor
    gw, gh = math.ceil(W / grid) * grid, math.ceil(H / grid) * grid
    reference = []
    if ctl:
        assert count_frames(ctl["path"]) == n, f"{ctl['path']} does not have the clip's {n} frames"
        ref = ctl["path"]
        if (gw, gh) != (W, H):
            ref = os.path.join(d, f"control_{ctl['hint']}_{gw}x{gh}.mp4")
            scale_video(ctl["path"], ref, gw, gh)
        reference = [(ref, float(ctl.get("strength", 1.0)))]
    print(f"[ltx25] {W}x{H}, {n} frames at {fps} fps; generating at {gw}x{gh} (grid {grid}); control: {reference or 'none'}", flush=True)

    stages = (ICLoraStageConfig(resolution_divisor=1, apply_ic_lora=bool(ctl), sigmas=tuple(DISTILLED_SIGMA_VALUES)),)
    pipe = ICLoraPipeline(
        model_paths=ModelPaths.from_split(transformer_path=files["transformer"], text_encoder_path=files["text_encoder"],
                                          video_vae_path=files["video_vae"], audio_vae_path=files["audio_vae"]),
        spatial_upsampler_path=files["spatial_upsampler"],          # required by the class; one stage never runs it
        loras=loras, offload_mode=OffloadMode(a.offload) if a.offload != "none" else OffloadMode.NONE, stages=stages)
    raw = os.path.join(d, "raw_video.mp4") if (gw, gh) != (W, H) else os.path.join(d, "out_video.mp4")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():                                    # prompt encoding included (LTX-2 docs)
        out = pipe(prompt=prompt, seed=req["seed"], height=gh, width=gw, num_frames=n, frame_rate=float(fps), images=[],
                   video_conditioning=reference)
        encode_video(out.video, float(fps), None, raw, video_chunks_number=None, crf=12)      # audio=None: video only
    if raw.endswith("raw_video.mp4"):
        scale_video(raw, os.path.join(d, "out_video.mp4"), W, H)
    got = count_frames(os.path.join(d, "out_video.mp4"))
    assert got == n, f"the model returned {got} frames, expected {n}"
    info = dict(model="LTX-2.5 22B distilled", files=files, size=[W, H], generated_size=[gw, gh], reference_downscale_factor=factor,
                frames=n, fps=fps, seed=req["seed"], steps=smp["num_steps"], sigmas=list(DISTILLED_SIGMA_VALUES), guidance=1.0,
                control=ctl, reference=reference, offload=a.offload, dtype="bf16", audio="generated by the model, not kept",
                wall_s=round(time.time() - t0, 1), peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    json.dump(info, open(os.path.join(d, "launch.json"), "w"), indent=1)
    print(f"[ltx25] saved {d}/out_video.mp4  wall {info['wall_s']} s  peak {info['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
