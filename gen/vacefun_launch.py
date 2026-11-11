"""Run Wan2.2-VACE-Fun-A14B (VideoX-Fun) on one prompt and, optionally, one VACE-style source video + mask.

Runs in VideoX-Fun's venv; called by gen/run_vacefun.sh:

  $VIDEOX_FUN_DIR/.venv/bin/python gen/vacefun_launch.py --videox_dir $VIDEOX_FUN_DIR --ckpt_dir <model dir> \
      --prompt "..." --seed 0 --save_dir <dir> [--src_video src_video.mp4 --src_mask src_mask.mp4]

The inputs are the same files sim/controls.py writes for Wan2.1 VACE, so every condition is a choice of inputs:
  no source             text only (A, A+)
  src_video + src_mask  mask black = keep this frame as given (frame 0), white = generate it; the source frames
                        under a white mask are the control (gray = none, depth, box outlines)
Writes <save_dir>/out_video.mp4 (81 frames, 16 fps) and run.json (settings, wall time, peak GPU memory).
"""

import argparse
import json
import os
import sys
import time

import torch

# Wan's default negative prompt, as shipped in VideoX-Fun's examples/wan2.2_vace_fun/*.py.
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
H, W, FRAMES, FPS = 480, 832, 81, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videox_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--src_video")
    ap.add_argument("--src_mask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--memory_mode", default="model_cpu_offload")   # one transformer on the GPU at a time
    ap.add_argument("--save_dir", required=True)
    a = ap.parse_args()
    if bool(a.src_video) != bool(a.src_mask):
        ap.error("--src_video and --src_mask go together")

    sys.path.insert(0, a.videox_dir)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan, AutoTokenizer, VaceWanTransformer3DModel, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2VaceFunPipeline
    from videox_fun.utils import (apply_gpu_memory_mode, filter_kwargs, get_image_to_video_latent,
                                  get_video_to_video_latent, save_videos_grid)

    t0 = time.time()
    dtype, device = torch.bfloat16, "cuda"
    cfg = OmegaConf.load(os.path.join(a.videox_dir, "config/wan2.2/wan_civitai_t2v.yaml"))
    tkw = OmegaConf.to_container(cfg["transformer_additional_kwargs"])

    def transformer(sub):   # low-noise and high-noise experts; the pipeline switches between them at `boundary`
        return VaceWanTransformer3DModel.from_pretrained(os.path.join(a.ckpt_dir, tkw[sub]), transformer_additional_kwargs=tkw,
                                                         low_cpu_mem_usage=True, torch_dtype=dtype)

    vae = AutoencoderKLWan.from_pretrained(os.path.join(a.ckpt_dir, cfg["vae_kwargs"]["vae_subpath"]),
                                           additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(dtype)
    tek = cfg["text_encoder_kwargs"]
    pipe = Wan2_2VaceFunPipeline(
        transformer=transformer("transformer_low_noise_model_subpath"),
        transformer_2=transformer("transformer_high_noise_model_subpath"),
        vae=vae,
        tokenizer=AutoTokenizer.from_pretrained(os.path.join(a.ckpt_dir, tek["tokenizer_subpath"])),
        text_encoder=WanT5EncoderModel.from_pretrained(os.path.join(a.ckpt_dir, tek["text_encoder_subpath"]),
                                                       additional_kwargs=OmegaConf.to_container(tek),
                                                       low_cpu_mem_usage=True, torch_dtype=dtype).eval(),
        scheduler=FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler,
                                                                  OmegaConf.to_container(cfg["scheduler_kwargs"]))),
    )
    apply_gpu_memory_mode(pipe, a.memory_mode, device, dtype)

    if a.src_video:
        # The pipeline takes the source through `control_video` and splits it by the mask itself, as Wan2.1 VACE does.
        src, _, _, _ = get_video_to_video_latent(a.src_video, video_length=FRAMES, sample_size=[H, W])
        mask, _, _, _ = get_video_to_video_latent(a.src_mask, video_length=FRAMES, sample_size=[H, W])
        assert src.shape[2] == mask.shape[2] == FRAMES, (src.shape, mask.shape)
        video, mask, control = src, mask[:, :1], src
    else:
        # Text only: VideoX-Fun's own no-image path, an empty source with every frame to be generated.
        video, mask, _ = get_image_to_video_latent(None, None, video_length=FRAMES, sample_size=[H, W])
        control = None

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = pipe(a.prompt, negative_prompt=a.negative, num_frames=FRAMES, height=H, width=W,
                   generator=torch.Generator(device=device).manual_seed(a.seed),
                   guidance_scale=a.guidance, num_inference_steps=a.steps,
                   video=video, mask_video=mask, control_video=control, subject_ref_images=None,
                   boundary=tkw.get("boundary", 0.875), shift=cfg["scheduler_kwargs"]["shift"], vace_context_scale=1.0).videos

    os.makedirs(a.save_dir, exist_ok=True)
    save_videos_grid(out, os.path.join(a.save_dir, "out_video.mp4"), fps=FPS)
    info = dict(vars(a), model="Wan2.2-VACE-Fun-A14B", size=[W, H], frames=FRAMES, fps=FPS, dtype="bf16",
                wall_s=round(time.time() - t0, 1), peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                gpu=torch.cuda.get_device_name(0))
    json.dump(info, open(os.path.join(a.save_dir, "run.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[vacefun] saved {a.save_dir}/out_video.mp4  wall {info['wall_s']} s  peak {info['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
