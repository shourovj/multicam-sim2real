# Step 1 Environments: Results and Gotchas (Nov 10, 2026)

**Owner:** SP Sharan · **Status:** record of the Step 1 smoke tests in the [v0 engineering plan](2026-11-10-0012-v0-engineering-plan.md) (§2), run on swarm5 on Nov 10. It lists what works, where everything is installed, and every problem we hit, so the runners in Step 5 don't rediscover them. The smoke scripts are in [`smoke/`](../smoke/README.md).

## 1. Results

One smoke clip per tool, one A100 80 GB each. Peak memory is what nvidia-smi shows for the job's processes, so it includes the CUDA context and allocator cache.

| Tool | Result | Wall time | Peak GPU | Installed at (`~/code/sim2real-videogen/`) |
|---|---|---|---|---|
| BlenderProc 2.8 / Blender 4.2.1, Cycles CUDA | Cube with RGB, depth, segmentation | 4 s (render 1.9 s) | 3.0 GB | `uv tool install blenderproc`; Blender in `blender/` |
| Cosmos3-Nano-Sim-Transfer, 153 frames | 153 frames, 832×480, 30 fps; follows the control | 131 s | 42.5 GB | `cosmos-framework/` |
| VACE-14B, text-only | 81 frames, 832×480, 16 fps | 27.1 min | 74.1 GB | `VACE/.venv` |
| VACE-14B, first frame | Frame 0 matches the input | 26.9 min | 73.7 GB | same |
| VACE-14B, first frame + depth | Poses follow the depth control | 26.9 min | 73.7 GB | same |
| Wan-Move-14B, shipped example | Dragged region follows the track | 24.6 min (shared card) | 40.8 GB | `Wan-Move/.venv` |
| SAM 2.1-L + CoTracker3 | 50-frame clip, tight mask on every frame, 100 tracks | 16 s | 6.5 GB | `envs/eval/.venv`; weights in `checkpoints/` |

All VACE-14B numbers are with the bf16 cast (§3.3). VACE-1.3B was downloaded but not run.

## 2. Shared server

- **VACE-14B needs nearly a whole 80 GB card.** It peaks at 74 GB.
- **CPU is shared too.** The load average was about 30, and the CPU-side steps (§3.3, §3.4) are slower when other jobs are busy.

## 3. Per tool

### 3.1 BlenderProc

- **The first GPU render takes about 6 minutes.** Blender 4.2.1 ships CUDA kernels for sm_75 and sm_86 but not sm_80 (A100), and an sm_86 binary won't run on sm_80, so the driver JIT-compiles the PTX kernel on one CPU thread. The result is cached in `~/.nv/ComputeCache` (53 MB); set `CUDA_CACHE_MAXSIZE=4294967296` so the cache never evicts it. Later renders start in seconds. A long stretch at 100% of one CPU core with about 400 MiB on the GPU means it's compiling, not hung.
- **`import blenderproc as bproc` must be the first line of the script**, above the docstring, or `blenderproc run` refuses to start.
- **Pass `--blender-install-path ~/code/sim2real-videogen/blender`**, or BlenderProc downloads another copy of Blender.
- **Cycles lists OptiX, CUDA and CPU devices**, all enabled. `smoke/blender_cube.py` exits with an error if no CUDA device is in use.
- **EGL is present** (`libEGL_nvidia.so.0`), contrary to the plan. EEVEE wasn't tried.
- **Sky depth is effectively infinite.** Mask it before scaling depth for display or for controls.

### 3.2 Cosmos3-Nano-Sim-Transfer

- **The checkpoints were gone.** `~/.cache/huggingface/hub` was emptied around 00:10 on Nov 10, probably by a disk cleanup. We re-downloaded them (about 32 GB). Check the cache exists before relying on it.
- **153 frames works with the `example2` profile.** Full history, no attention sink (`--kv-cache-inference-size 0 --attention-sink-size 0`). Set `num_frames`, `max_frames` and `num_video_frames_per_chunk` all to 153. Memory rises from about 35 GB at 121 frames to 42.5 GB at 153.
- **The test control was a stand-in.** It was NVIDIA's 121-frame example depth with a 32-frame reversed tail appended (`data/smoke/cosmos/example2_depth_153.mp4`). Our own 153-frame bundles are still untested.

### 3.3 VACE

- **The 14B checkpoint runs out of memory on an empty 80 GB A100 as shipped.** It stores its 14.3B base weights in fp32 (57 GB); only the 3.05B VACE blocks are bf16. VACE loads them as stored, so 63 GB of weights are on the GPU before sampling, even with `--offload_model True --t5_cpu`.
  - **Fix:** `VACE_DIT_BF16=1` (in `smoke/vace_launch.py`) casts the DiT to bf16 after loading, giving 32.3 GiB of weights and a 74 GB peak.
  - **What the cast changes:** sampling already runs under bf16 autocast, so mostly memory. The few ops Wan keeps in fp32 (time embedding, modulation) now use bf16 weights.
  - **Choice:** this or VACE-1.3B is the open checkpoint decision for A–C.
- **Each clip spends about 4 minutes encoding the prompt on CPU while the GPU sits idle.** With `--t5_cpu`, the 5.7B umt5-xxl encoder runs in bf16 on an EPYC 7413, which has no bf16 instructions. The log stops at "Generating video..." and GPU utilization is 0% during this step; that's normal.
  - **For Step 5:** prompts are frozen, so encode each prompt once and cache the embeddings.
- **Sampling speed:** 23.5 s/step × 50 steps, about 20 min. Load plus prompt encoding takes about 6 min, so about 27 min per clip.
- **flash-attn is required.** Wan's model calls `flash_attention` directly and has no SDPA fallback. The env uses the prebuilt flash-attn 2.7.4 wheel with torch 2.6 / cu124.
- **`import wan` fails with no GPU visible.** Wan's T5 module calls `torch.cuda.current_device()` at import time. CPU-only checks need a shim.
- **Pass `--size 832*480`.** VACE's size table is (W, H), and `480p` maps to 480×832, which is portrait for text-only runs. With a `src_video`, the output follows the source's aspect ratio.
- **`vace_preproccess.py --task composition` is broken from the command line.** It never fills in the second input or the masks. `smoke/vace_prep_inputs.py` calls `CompositionAnnotator` directly with ("frameref", "depth"), as VACE's Gradio composition tab does. The result matches the plan: the real frame 0, then the depth of frames 1–80; the mask is black on frame 0 and white after.
- **Gray filler frames come out at 126, not 127.** VACE fills with 127.5, and the cast to 8-bit plus H.264 shifts it down. VACE's own inputs go through the same path.
- **Annotators:** only `depth/depth_anything_v2_vitl.pth` and `scribble/anime_style/netG_A_latest.pth` were downloaded (1.3 GB). The depth annotator took 8.9 min on CPU for 81 frames.
- **Checkpoints are symlinks** in `VACE/models/` into the HF cache, so they show as untracked in the VACE clone.
- **A harmless warning** at startup: "failed to importing 'ltx'".

### 3.4 Wan-Move

- **flash-attn is required**, for the same reason as VACE (`attention.py:112`). The env uses flash-attn 2.8.3 with torch 2.7.1 / cu126.
- **`requirements.txt` is incomplete.** `decord`, `einops` and `regex` had to be added.
- **Importing `wan/modules/trajectory.py` sets `HF_ENDPOINT=hf-mirror.com`** (line 2). Anything that downloads from HF inside that process goes through the mirror.
- **Use `--offload_model True --t5_cpu --dtype bf16`.** The default dtype is fp32. Prompt encoding takes about 4.5 min on CPU, as with VACE.
- **Output is 832×464, not 832×480.** The size comes from `max_area` and the image's aspect ratio. Resample for eval.
- **Track format:**
  - **Files:** tracks are float32 `[1, T, N, 2]` or `[T, N, 2]`, in pixels of the input image, x (column) first, then y (row). Visibility is bool `[1, T, N]` or `[T, N]`. Both are `.npy` files. Tracks must be float; integer arrays crash on the in-place rescale.
  - **Frames:** T must be exactly 81. The first-frame mask is hard-coded to 81 frames; T=97 raises an IndexError, and T=41 leaves the last 10 latent frames without control. Only frames 0, 4, …, 80 are used, giving 21 latent frames.
  - **Visibility:** hidden or out-of-frame points are simply skipped for that latent frame. **Every point must be visible at frame 0**: otherwise its source cell defaults to the bottom-right corner of the latent.
  - **Points:** all N are used, with no limit and no object IDs. The shipped example has one point.
  - **How tracks steer the model:** the frame-0 latent feature at each point's cell (pixel ÷ 8) is copied to the point's cell in each later latent frame. Only the conditional branch gets the edited latent, so classifier-free guidance strengthens the motion.
- **Points less than 8 px apart overwrite each other in random order.** The order isn't tied to `--base_seed`, because the code shuffles with torch's unseeded global RNG (`trajectory.py:106`, `:140`). A 32 px drone covers only 4×4 latent cells, so sample at most one point per 8×8 cell per object rather than 16 surface points.

### 3.5 SAM 2 and CoTracker3

- **SAM 2's CUDA extension isn't built**, because swarm5 has no nvcc (`SAM2_BUILD_CUDA=0`). SAM 2 skips its mask hole-filling step and prints a harmless `_C` warning.
- **Rebuilding the env takes two steps:** `uv sync --no-install-package sam-2`, then `uv sync`. SAM 2 builds against the env's own torch, so torch has to be installed first.
- **`torch.cuda.reset_peak_memory_stats` raises "Invalid device argument" before CUDA is initialized.** Call `torch.cuda.set_device` and `torch.cuda.init()` first.
- **The default SAM 2 prompt in `smoke/eval_tools.py` only fits `apple.mp4`.** Pass `--points` or `--box` for any other clip.

## 4. Disk and downloads

- **Free space:** the root disk had 606 GB free after the downloads (96% used).
- **HF cache:** the project's models are in a shared Hugging Face cache, `~/code/sim2real-videogen/hf-hub` (moved out of `~/.cache/huggingface/hub` on Nov 10 so other swarm5 users can read it; the old path links to it). It is 187 GB. It holds VACE-14B (59 GB on disk), Wan-Move-14B (82 GB), VACE-1.3B (19 GB), the Cosmos Sim-Transfer checkpoint, its tokenizer and the Wan2.2 VAE.
- **The cache was deleted once**, so check it's there before a long run.
- **Other caches** from the VACE scripts go to `~/code/sim2real-videogen/.cache`.

## 5. Scene bundle and first PoC runs (drones_air, Nov 10)

Problems found while building the `drones_air` proof of concept (`sim/`, `gen/`, `eval/grid.py`).

- **Blender's Python has no PyYAML.** Install it once with `blenderproc pip install --blender-install-path ~/code/sim2real-videogen/blender pyyaml`.
- **A noise texture on a huge ground plane renders flat.** With no vector input, the noise uses Generated coordinates, which span 0–1 across the whole 4 km plane, so any normal scale gives one smooth gradient. Feed it Object coordinates so the scale is per metre.
- **The default colour management washes the sky out** to a pale grey-white. `sim/render_scene.py` sets `view_transform = "Standard"`.
- **Render time:** 153 frames at 832×480, 48 samples + OpenImageDenoise, 304 s on a shared A100 (about 2 s per frame), 1.8 GB of GPU memory.
- **Output piped through `grep` is buffered.** `blenderproc run ... | grep -v '^Fra:' | tee log` writes nothing until Blender exits. Check progress by counting the frames in `/dev/shm/blender_proc_*/`.
- **The Cosmos data guide doesn't spell out the depth encoding.** It only says "matching normalization and near/far polarity". NVIDIA's `example2_depth.mp4` (1920×1080, 121 frames) shows it: three equal channels, near = bright, sky = 0, and a scale shared across the clip (the per-frame maximum drifts from 253 to 230), which looks like Video Depth Anything's relative inverse depth. `sim/controls.py` writes `cosmos_depth` as 1/z with the sky set to 0, min–max scaled over the whole clip. VACE's `vace_depth` is the same but scaled per frame.
- **Cosmos writes its output to `<OUT>/<name>/vision.mp4`**, with the control copy as `control_depth.mp4`, not `<OUT>/<name>.mp4`.
- **Cosmos with a plain-text prompt:** Sim-Transfer is trained 93% on structured JSON captions and 7% on dense text. The PoC uses the frozen plain `base` prompt for parity with the other conditions. The D clip came out fine, but this is a difference from NVIDIA's examples.
- **Wan-Move's 8×8 cells are in its resized frame.** It resizes an 832×480 image to 832×464, so `sim/controls.py` bins points with `y × 464/480`. Binning on the input's own 8 px rows could put two points in one latent cell.
- **Thin parts give few full cells.** The far drone (about 75×16 px) has only about 6 cells that are at least half covered by the mask; at 35% coverage it gets 10. Drone frames, arms and rotor discs are mostly 1–3 px thick.
- **VACE's layout colour list wasn't downloaded.** Only `depth/` and `scribble/` of VACE-Annotators are on disk, so `layout/ram_tag_color_list.txt` (4585 RAM tags → RGB) is missing and the layout annotators fall back to black boxes. `uvx hf@latest download ali-vilab/VACE-Annotators layout/ram_tag_color_list.txt --local-dir <dir>` fetches just that file. The colour for `drone` is `[201, 103, 188]` (line 1417); `sim/controls.py` hard-codes it.
- **VACE's two layout annotators disagree on channel order.** `LayoutBboxAnnotator` (used by move_anything) draws 2 px boxes and then flips to `[..., ::-1]`, so the tag colour comes out channel-reversed. `LayoutTrackAnnotator` / `LayoutMaskAnnotator` draw 3 px contours and don't flip. B-box follows layout_track: 3 px, no flip, white background.
- **layout_track takes one label per call.** It draws one colour for all tracked objects, and in label mode it merges their masks, so overlapping objects become one outline. B-box draws each drone separately and merges the canvases, so both boxes stay visible at the crossing. Same-category objects get the same colour, so the boxes don't say which drone is which; only frame 0 does.
