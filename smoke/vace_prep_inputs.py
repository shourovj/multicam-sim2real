"""Prepare the inputs for the VACE smoke tests (firstframe and ffdepth cases).

Run through smoke/vace_prep_inputs.sh, which uses VACE's venv and runs from the VACE repo root
(VACE's annotator configs use paths relative to it, e.g. models/VACE-Annotators/...).

Steps:
  1. source: VACE's assets/videos/test.mp4 (1280x720, 16 fps, 81 frames), resized and center
     cropped to 832x480 the same way VACE's video processor does it. Frame 0 saved as PNG.
  2. firstframe: VACE's own preprocessing CLI,
     `vace/vace_preproccess.py --task frameref --mode firstframe --expand_num 80`, giving
     frame 0 + 80 gray (127) frames and a mask that is black on frame 0, white after.
  3. ffdepth: VACE's DepthV2VideoAnnotator (Depth Anything V2 ViT-L) on all 81 source frames,
     then VACE's CompositionAnnotator with ('frameref', 'depth') = ('extension', 'control'),
     fed the same way VACE's Gradio composition tab feeds it (mask videos read back and
     thresholded at 127, control mask all white).

Works on CPU (CUDA_VISIBLE_DEVICES=) or GPU; the depth annotator uses cuda if visible.
"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.environ.get("S2R_ROOT", os.path.expanduser("~/code/sim2real-videogen"))
DATA = os.environ.get("S2R_DATA", os.path.join(ROOT, "data"))
VACE_DIR = os.path.join(ROOT, "VACE")
sys.path.insert(0, os.path.join(VACE_DIR, "vace"))

from annotators.utils import read_video_frames, save_one_video  # noqa: E402

W, H, N, FPS = 832, 480, 81, 16


def resize_crop(frame, oh, ow):
    """Scale so the frame covers (oh, ow), then center crop (as VaceVideoProcessor.resize_crop)."""
    h, w = frame.shape[:2]
    scale = max(oh / h, ow / w)
    nh, nw = round(h * scale), round(w * scale)
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (nh - oh) // 2, (nw - ow) // 2
    return frame[y:y + oh, x:x + ow]


def binarize(masks):
    # Same as vace/gradios/vace_preprocess_demo.py process_composition_data.
    return [np.where(m > 127, 1, 0).astype(np.uint8) for m in masks]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(VACE_DIR, "assets/videos/test.mp4"))
    p.add_argument("--out", default=os.path.join(DATA, "smoke/vace"))
    p.add_argument("--skip_depth", action="store_true")
    args = p.parse_args()

    os.chdir(VACE_DIR)
    src_dir = os.path.join(args.out, "inputs")
    ff_dir = os.path.join(args.out, "firstframe", "inputs")
    fd_dir = os.path.join(args.out, "ffdepth", "inputs")
    for d in (src_dir, ff_dir, fd_dir):
        os.makedirs(d, exist_ok=True)

    # 1. source clip at 832x480
    frames, fps, w, h, n = read_video_frames(args.src, use_type="cv2", info=True)
    assert n >= N, f"{args.src} has {n} frames, need {N}"
    assert round(fps) == FPS, f"{args.src} is {fps} fps; VACE samples at {FPS} fps"
    frames = [resize_crop(np.ascontiguousarray(f), H, W) for f in frames[:N]]
    source_mp4 = os.path.join(src_dir, "source_832x480.mp4")
    frame0_png = os.path.join(src_dir, "frame0.png")
    save_one_video(source_mp4, frames, fps=FPS)
    cv2.imwrite(frame0_png, frames[0][..., ::-1])
    print(f"source: {source_mp4} ({len(frames)} frames from {w}x{h}), frame0: {frame0_png}")

    # 2. firstframe via VACE's own preprocessing CLI
    subprocess.run([sys.executable, "vace/vace_preproccess.py", "--task", "frameref",
                    "--mode", "firstframe", "--expand_num", str(N - 1), "--image", frame0_png,
                    "--save_fps", str(FPS), "--pre_save_dir", ff_dir], check=True)
    ff_video = os.path.join(ff_dir, "src_video-frameref.mp4")
    ff_mask = os.path.join(ff_dir, "src_mask-frameref.mp4")

    if args.skip_depth:
        return

    # 3a. depth control video with VACE's DepthV2 video annotator
    from configs import VACE_PREPROCCESS_CONFIGS
    from annotators.depth import DepthV2VideoAnnotator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(32, os.cpu_count() or 1))
    cfg = dict(VACE_PREPROCCESS_CONFIGS["depthv2"])
    t0 = time.time()
    anno = DepthV2VideoAnnotator(cfg, device=device)
    depth = anno.forward(frames)
    print(f"depthv2 on {device}: {len(depth)} frames in {time.time() - t0:.1f}s")
    depth_mp4 = os.path.join(fd_dir, "src_video-depthv2.mp4")
    save_one_video(depth_mp4, depth, fps=FPS)
    ctrl_mask_mp4 = os.path.join(fd_dir, "src_mask-depthv2.mp4")  # control tasks: mask all white
    save_one_video(ctrl_mask_mp4, [np.full((H, W, 3), 255, np.uint8)] * N, fps=FPS)

    # 3b. composition: process_type_1='frameref' (extension), process_type_2='depth' (control)
    from annotators.composition import CompositionAnnotator
    v1 = read_video_frames(ff_video, use_type="cv2")
    v2 = read_video_frames(depth_mp4, use_type="cv2")
    m1 = binarize(read_video_frames(ff_mask, use_type="cv2"))
    m2 = binarize(read_video_frames(ctrl_mask_mp4, use_type="cv2"))
    assert v1[0].shape == v2[0].shape == m1[0].shape == m2[0].shape, "size mismatch"
    out_video, out_mask = CompositionAnnotator({}).forward("frameref", "depth", v1, v2, m1, m2)
    out_video = [np.asarray(f, np.uint8) for f in out_video]
    out_mask = [np.asarray(m, np.uint8) for m in out_mask]
    save_one_video(os.path.join(fd_dir, "src_video.mp4"), out_video, fps=FPS)
    save_one_video(os.path.join(fd_dir, "src_mask.mp4"), out_mask, fps=FPS)
    print(f"composition: {fd_dir}/src_video.mp4, src_mask.mp4 ({len(out_video)} frames)")


if __name__ == "__main__":
    main()
