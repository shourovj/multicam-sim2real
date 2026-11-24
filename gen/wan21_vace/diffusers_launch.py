"""Run Wan2.1-VACE (14B or 1.3B) through diffusers on one prompt and, optionally, one VACE-style source video + mask.

Same inputs and outputs as gen/wan22_vace_fun/launch.py, so the two models can be compared on identical files.
This route needs far less GPU memory than the original VACE code (gen/wan21_vace/singlecam_run.sh): each part of the model is on
the GPU only while it is in use. Called by gen/wan21_vace/singlecam_run_diffusers.sh, in an env with torch and diffusers >= 0.34:

  python gen/wan21_vace/diffusers_launch.py --ckpt Wan-AI/Wan2.1-VACE-14B-diffusers --prompt "..." --seed 0 \
      --save_dir <dir> [--src_video src_video.mp4 --src_mask src_mask.mp4]

  no source             text only (A, A+)
  src_video + src_mask  mask black = keep this frame as given (frame 0), white = generate it; the source frames
                        under a white mask are the control (gray = none, depth, box outlines)
Writes <save_dir>/out_video.mp4 (81 frames, 16 fps) and run.json (settings, wall time, peak GPU memory).
"""

import argparse
import json
import os
import time

import torch

# Wan's default negative prompt, the same string gen/wan22_vace_fun/launch.py uses.
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
H, W, FRAMES, FPS = 480, 832, 81, 16


def read_frames(path):
    import imageio.v3 as iio
    from PIL import Image
    frames = [Image.fromarray(f).convert("RGB").resize((W, H)) for f in iio.imiter(path)]
    assert len(frames) == FRAMES, f"{path} has {len(frames)} frames, expected {FRAMES}"
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="diffusers repo id or local folder")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--src_video")
    ap.add_argument("--src_mask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--flow_shift", type=float, default=3.0)   # 3.0 for 480p, 5.0 for 720p
    ap.add_argument("--save_dir", required=True)
    a = ap.parse_args()
    if bool(a.src_video) != bool(a.src_mask):
        ap.error("--src_video and --src_mask go together")

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    from diffusers.utils import export_to_video

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(a.ckpt, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(a.ckpt, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=a.flow_shift)
    pipe.enable_model_cpu_offload()

    video = read_frames(a.src_video) if a.src_video else None
    mask = read_frames(a.src_mask) if a.src_mask else None
    torch.cuda.reset_peak_memory_stats()
    out = pipe(prompt=a.prompt, negative_prompt=a.negative, video=video, mask=mask, height=H, width=W, num_frames=FRAMES,
               num_inference_steps=a.steps, guidance_scale=a.guidance,
               generator=torch.Generator().manual_seed(a.seed)).frames[0]

    os.makedirs(a.save_dir, exist_ok=True)
    export_to_video(out, os.path.join(a.save_dir, "out_video.mp4"), fps=FPS)
    info = dict(vars(a), model=os.path.basename(a.ckpt.rstrip("/")), size=[W, H], frames=FRAMES, fps=FPS, dtype="bf16",
                wall_s=round(time.time() - t0, 1), peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                gpu=torch.cuda.get_device_name(0))
    json.dump(info, open(os.path.join(a.save_dir, "run.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[vace21] saved {a.save_dir}/out_video.mp4  wall {info['wall_s']} s  peak {info['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
