"""Write one Cosmos3-Nano (base) request for a scene, condition and seed, and build the control videos it needs.

  python gen/cosmos_nano_request.py <scene> <cond> <seed> <out_dir>      -> <out_dir>/request.json

Each of our conditions maps to one Nano mode:
  A, Aplus          text2video
  B_edit, B         image2video from the edited / simulator frame 0
  Bbox_edit, Bbox   transfer with an edge control: the simulator boxes as white outlines on black, plus frame 0
  P1_edit           as Bbox_edit, but the outlines stop after the first second (frames 0..16), then black
  C                 transfer with a depth control, plus the simulator frame 0
  C0                transfer with a depth control only
Nano has no box input, so the box conditions use its edge control. Controls are 81 frames at 16 fps, 832x480,
cached in <scene>/bundle/controls/nano_*.mp4. Sampler settings are Nano's own defaults for each mode.
"""
import json, os, shutil, sys

import cv2, numpy as np, imageio.v3 as iio, yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, FPS, PAD = 81, 16, 6
scene, cond, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B = os.path.join(os.environ["S2R_DATA"], "poc", scene, "bundle")
C = os.path.join(B, "controls")
IDX = [round(t * 30 / FPS) for t in range(N)]            # simulator frame (30 fps) for each output frame


def cached(name, make):
    """Build controls/<name> once; written under a temporary name first so parallel jobs never read a partial file."""
    path = os.path.join(C, name)
    if not os.path.exists(path):
        tmp = os.path.join(C, f".{os.getpid()}_{name}")
        iio.imwrite(tmp, np.stack(make()), fps=FPS, codec="libx264", output_params=["-crf", "12"])
        os.replace(tmp, path)
    return path


def depth():                                              # cosmos_depth.mp4: 1/z, one scale for the whole clip
    d = np.stack(list(iio.imiter(os.path.join(C, "cosmos_depth.mp4"))))
    return [d[min(i, len(d) - 1)] for i in IDX]


def box_edges(until=None):
    boxes = json.load(open(os.path.join(B, "boxes.json")))
    frames = []
    for t, i in enumerate(IDX):
        f = np.zeros((480, 832, 3), np.uint8)
        for b in boxes.values():
            if (until is None or t <= until) and i < len(b) and b[i]:
                x0, y0, x1, y1 = [int(round(v)) for v in b[i]]
                cv2.rectangle(f, (x0 - PAD, y0 - PAD), (x1 + PAD, y1 + PAD), (255, 255, 255), 3)
        frames.append(f)
    return frames


def frame0(edited):
    im = iio.imread(os.path.join(B, "frame0_edited.png" if edited else "frame0.png"))[..., :3]
    return cv2.resize(im, (832, 480), interpolation=cv2.INTER_AREA)


def rgb(edited):                                          # transfer reads frame 0 from a video; only its first frame is used
    return cached(f"nano_rgb_{'edit' if edited else 'sim'}.mp4", lambda: [frame0(edited)] * N)


prompts = yaml.safe_load(open(os.path.join(REPO, "configs/prompts", f"{scene}.yaml")))
os.makedirs(out, exist_ok=True)
r = dict(name="out", seed=seed, resolution="480", aspect_ratio="16,9", fps=FPS, num_frames=N,
         prompt=prompts["detailed" if cond == "Aplus" else "base"])
edited = cond.endswith("_edit")
if cond in ("A", "Aplus"):
    r.update(model_mode="text2video")
elif cond in ("B", "B_edit"):
    iio.imwrite(os.path.join(out, "frame0.png"), frame0(edited))
    r.update(model_mode="image2video", vision_path="frame0.png")
else:                                                     # transfer; settings from Nano's cookbook depth / edge specs
    neg = os.path.join(os.environ["S2R_ROOT"], "cosmos-framework/.agents/skills/cookbooks/cosmos3/generator/transfer/assets/negative_prompt.json")
    shutil.copy(neg, os.path.join(out, "negative_prompt.json"))
    r.update(model_mode="video2video", shift=10.0, num_steps=50, guidance=3.0, control_guidance=1.5, num_video_frames_per_chunk=121,
             num_conditional_frames=1, num_first_chunk_conditional_frames=0, share_vision_temporal_positions=True,
             negative_metadata_mode="none", negative_prompt_keep_metadata=False, negative_prompt_file="negative_prompt.json",
             emphasize_control_in_prompt=False)
    if cond in ("C", "C0"):
        r["depth"] = {"control_path": cached("nano_depth.mp4", depth)}
    elif cond in ("Bbox", "Bbox_edit"):
        r["edge"] = {"control_path": cached("nano_boxedge.mp4", box_edges)}
    elif cond == "P1_edit":
        r["edge"] = {"control_path": cached("nano_boxedge_p1.mp4", lambda: box_edges(until=16))}
    else:
        sys.exit(f"unknown condition {cond}")
    if cond != "C0":
        r.update(vision_path=rgb(edited), num_first_chunk_conditional_frames=1)
json.dump(r, open(os.path.join(out, "request.json"), "w"), indent=1)
