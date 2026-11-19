# Quick guide: branch `shourov/pilot-v0` and its outputs (Nov 11, 2026)

This branch is round 2 of the pilot: three scenes whose motion comes from a physics model, run under several control conditions on Wan2.2-VACE-Fun-A14B. This guide covers what the branch adds, how to run it and where the outputs are. Results are in the [round 2 findings doc](2026-11-11-0845-round2-findings.md).

## 1. Status

- **Round 2 is complete: 51 clips.** Three scenes × seven conditions, seeds 0–2 (text-only conditions have seed 0 only).
- **A follow-up batch of 18 clips is running** as of 13:04 on Nov 11: conditions `Bbox` and `B`, which use the simulator's own frame 0 in place of the edited one. None has finished yet.
- The follow-up conditions are a change to `configs/matrix.yaml` that is **not committed yet**.

## 2. What the branch adds to `main`

One commit, `3177669`, 34 files.

| Area | Files | What it does |
|---|---|---|
| Physics | `sim/physics/` (`quadrotor.py`, `ground_vehicle.py`, `rigid_body.py`), `sim/simulate.py`, `tests/test_physics.py` | Objects with a `motion` block in the scene file are moved by a physics model, not by scripted keyframes. `simulate.py` prints the motion without rendering. |
| Scenes | `configs/scenes/*.yaml`, `configs/prompts/*.yaml`, `sim/assets.py`, `sim/motion.py`, `sim/render_scene.py`, `sim/export.py` | Two new scenes (`drone_takeoff`, `convoy_overhead`) and a `_phys` version of all three. Tank, truck, bridge, tree canopy and aerial-photo ground for the overhead scene. |
| Control inputs | `sim/controls.py`, `frame0_edits/` | New conditions: `P1` (boxes for the first second only), a hand-edited frame 0 (`--frame0`, `--suffix`) and padded boxes (`--box_pad`). |
| Generation | `gen/run_vacefun.sh`, `gen/vacefun_launch.py`, `gen/run_vace_diffusers.sh`, `gen/vace_diffusers_launch.py` | Runners for Wan2.2-VACE-Fun-A14B (VideoX-Fun) and for Wan2.1-VACE through diffusers. Both read the same input files. |
| Batch runner | `gen/run_matrix.py`, `configs/matrix.yaml` | Runs every scene × condition × seed, a few at a time, on GPUs that have room. |
| Write-up | `docs/2026-11-11-0845-round2-findings.md`, two rows in the README `Artifacts` table | Per-clip results. |

## 3. Setup

Follow [`setup/README.md`](../setup/README.md) first (Option A on swarm5). This branch needs one more thing that the setup guide does not cover yet: a VideoX-Fun clone with its own `.venv`, and the Wan2.2-VACE-Fun-A14B weights linked inside it. `gen/run_vacefun.sh` finds the clone through `VIDEOX_FUN_DIR` (default `$S2R_ROOT/VideoX-Fun`).

### Shortcut on swarm5: reuse the existing copy

A working copy is at `/mnt/Data/sj37434/sim2real-local/VideoX-Fun`. It is world-readable, so if you can open `/mnt/Data` this one line replaces steps 1 to 5 below:

```
export VIDEOX_FUN_DIR=/mnt/Data/sj37434/sim2real-local/VideoX-Fun
```

### Build your own, step by step

These are the commands that built that copy on Nov 10, with the paths turned into variables. Two things are added here so the result is pinned: the `git checkout` (the original was a `--depth 1` clone that landed on this commit) and `--revision` on the download (the original took the latest, which was this snapshot). It needs `uv`, `git`, about 7 GB for the env and 76 GB for the weights. On swarm5 put both under `/mnt/Data`; the root disk is nearly full.

**1. Pick where things go.**

```
source env.sh
ME=$(basename $HOME)                    # your EID; $USER on swarm5 also carries the domain, so it is not used here
L=/mnt/Data/$ME/sim2real-local          # the clone and its env
HUB=/mnt/Data/$ME/hf/hub                # a Hugging Face cache you can write to
export UV_CACHE_DIR=$S2R_CACHE/uv       # keep uv's cache off the home disk
mkdir -p $L $HUB
```

**2. Clone VideoX-Fun at the commit this round used.**

```
git clone https://github.com/aigc-apps/VideoX-Fun.git $L/VideoX-Fun
git -C $L/VideoX-Fun checkout e8c314fbfa494d0429bbef605c9f6112b79a7ead
```

**3. Build its Python env.** Python 3.10, torch 2.6.0 for CUDA 12.4, the clone's own `requirements.txt`, and the prebuilt flash-attn wheel that matches that torch.

```
cd $L/VideoX-Fun
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 numpy==1.26.4 -r requirements.txt pyyaml \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
```

**4. Check the env.** This should print the torch version, `True`, and `videox_fun imports ok`.

```
.venv/bin/python -c "
import sys, torch, flash_attn; print(torch.__version__, torch.cuda.is_available())
sys.path.insert(0, '.')
from videox_fun.pipeline import Wan2_2VaceFunPipeline
from videox_fun.utils import apply_gpu_memory_mode, get_video_to_video_latent
print('videox_fun imports ok')"
cd -
```

**5. Download the weights and link them into the clone.** The download needs the network, so `HF_HUB_OFFLINE` is switched off for this one command. Its last line of output is the snapshot folder.

```
SNAP=$(HF_HUB_OFFLINE=0 HF_HUB_CACHE=$HUB uvx hf@2.4.0 download alibaba-pai/Wan2.2-VACE-Fun-A14B \
         --revision 4438caabbfdd7437bbb0283d86cb200d1b7223ba -q | tail -1)
mkdir -p $L/VideoX-Fun/models/Diffusion_Transformer
ln -sfn "$SNAP" $L/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B
ls $L/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B    # high_noise_model, low_noise_model, Wan2.1_VAE.pth, ...
```

**6. Tell the repo where the clone is.** Add this next to your other workspace variables.

```
export VIDEOX_FUN_DIR=$L/VideoX-Fun
```

**7. Smoke test: a 2-step, text-only clip.** `sequential_cpu_offload` loads the model layer by layer, so it fits beside other jobs on a shared card. Check `nvidia-smi` right before launching.

```
SCENE=drones_air_phys COND=A SEED=0 STEPS=2 MEMORY_MODE=sequential_cpu_offload CUDA_VISIBLE_DEVICES=<gpu> bash gen/run_vacefun.sh
```

It passes if `$S2R_DATA/poc/single-cam/drones_air_phys/A/vacefun-a14b-s2/seed0/out_video.mp4` exists; the clip itself will be noise at 2 steps. The Nov 10 check was the same idea but not this exact command: 2 steps on the box inputs, through `gen/vacefun_launch.py` directly.

### Only for Wan2.1-VACE

`gen/run_vace_diffusers.sh` is not used by round 2. It needs `VACE_DIFFUSERS_PY` (a Python with torch and diffusers) and `VACE_DIFFUSERS_HUB` (the HF cache with `Wan-AI/Wan2.1-VACE-14B-diffusers` or the 1.3B one).

## 4. How to run

All commands run from the repo root after `source env.sh` with your workspace variables set.

### Check the physics (seconds, no GPU)

```
$S2R_ROOT/envs/eval/.venv/bin/python tests/test_physics.py                                  # 6 tests, all pass
$S2R_ROOT/envs/eval/.venv/bin/python sim/simulate.py configs/scenes/drones_air_phys.yaml     # positions, speeds, events
```

### Build one scene's inputs

```
D=$S2R_DATA/poc/single-cam/drones_air_phys; PY=$S2R_ROOT/VACE/.venv/bin/python
bash sim/build_bundle.sh configs/scenes/drones_air_phys.yaml $D/bundle                       # render + export
$PY sim/controls.py $D/bundle $D --only vace C C0                                            # depth conditions
$PY sim/controls.py $D/bundle $D --only B Bbox P1 --frame0 $D/bundle/frame0_edited.png --suffix _edit --box_pad 6
$PY sim/controls.py $D/bundle $D --only B Bbox --box_pad 6                                   # simulator frame 0
```

The three `controls.py` lines are rebuilt from the script's options and the folders it produced; the exact commands of the run were not logged.

The edited frame 0 is made by hand: take `bundle/frame0.png`, make it realistic in an image editor, and save it as `bundle/frame0_edited.png`. The full-size originals for this round are in [`frame0_edits/`](../frame0_edits/); the bundle copies are the same images at 832×480.

### Run the matrix

```
$S2R_ROOT/envs/eval/.venv/bin/python gen/run_matrix.py --status      # every job, done or pending
$S2R_ROOT/envs/eval/.venv/bin/python gen/run_matrix.py --dry-run     # launch order only
nohup setsid $S2R_ROOT/envs/eval/.venv/bin/python gen/run_matrix.py >> $S2R_DATA/poc/run_matrix.log 2>&1 < /dev/null &
touch $S2R_DATA/poc/STOP                                             # no new jobs start; running ones finish
```

- A job is done when its `out_video.mp4` exists, so a rerun continues where the last one stopped.
- A GPU is used only if it has 55 GiB free right before launch, at most four at once. A failed job is retried twice.
- One clip takes about 13.5 min on a free A100 and up to 28 min on a shared one. Peak GPU memory is 40.8 GiB.

### Run one clip

```
SCENE=drones_air_phys COND=Bbox_edit SEED=0 STEPS=30 CUDA_VISIBLE_DEVICES=<gpu> bash gen/run_vacefun.sh
```

Set `STEPS=30`. The script's default is 50, and the step count is part of the output folder name.

## 5. Outputs

Everything is under `$S2R_DATA/poc/`. For this round that is `/mnt/Data/sj37434/sim2real-data/poc/` on swarm5 (3.3 GB, world-readable). Single-camera scenes are in `single-cam/`; the two-camera follow-up of Nov 13 is in `multi-cam/`.

```
poc/
  manifest.jsonl                  one line per job start and end: GPU, mode, status, wall time, peak memory
  run_matrix.log                  the batch runner's log
  logs/<scene>__<cond>__s<k>.log  output of each job
  single-cam/<scene>/
    bundle/                       what the simulator produced (about 500 MB per scene)
    <cond>/inputs/                what the model gets: src_video.mp4, src_mask.mp4
    <cond>/vacefun-a14b-s30/seed<k>/
      out_video.mp4               the generated clip: 81 frames, 16 fps, 832×480
      run.json                    prompt, seed, steps, guidance, wall time, peak GPU memory
      run.log
  multi-cam/
    drones_air_phys_cam2/         drones_air_phys from a second camera: scene.yaml, bundle, clips, scripts
    drones_air_phys_mv/           both cameras in one video (cut and split screen)
```

### Scenes

| Folder | Scene |
|---|---|
| `drones_air_phys` | Two drones, orange and white, pass each other in the air |
| `drone_takeoff_phys` | One drone lifts off and passes behind a tree |
| `convoy_overhead_phys` | A tank and two trucks on a bend, seen from straight above |

### Conditions

| Folder | What the model gets | Clips per scene |
|---|---|---|
| `A` | Text | 1 |
| `Aplus` | Detailed text describing the path | 1 |
| `B_edit` | Text + edited frame 0 | 3 |
| `Bbox_edit` | Edited frame 0 + box outlines for the whole clip | 3 |
| `P1_edit` | Edited frame 0 + box outlines for the first second only | 3 |
| `C` | Simulator frame 0 + depth | 3 |
| `C0` | Depth, no frame 0 | 3 |
| `B` | Text + simulator frame 0 | running |
| `Bbox` | Simulator frame 0 + box outlines for the whole clip | running |

In `src_mask.mp4`, a black frame means "keep this frame as given" and a white frame means "generate it". Under a white mask, the matching `src_video.mp4` frame is the control: gray for none, a depth image, or box outlines on white.

### The scene bundle

| File | What it is |
|---|---|
| `rgb.mp4`, `rgb/0000.png` … | The simulator render: 153 frames at 30 fps |
| `frame0.png`, `frame0_edited.png` | Simulator frame 0 and its hand-edited realistic version |
| `overlay.mp4` | The render with boxes and track points drawn on it. Watch this first to see what the scene is meant to do. |
| `boxes.json` | Per object, per frame: `[x0, y0, x1, y1]`, or `null` when the object is fully hidden |
| `tracks.npz` | Point tracks with visibility |
| `depth.npy`, `instance.npy` | Depth in metres and the object-id mask, per frame |
| `timeline.json` | Per-frame object poses, the camera and events (lift-off, crossing, hidden ranges) |
| `checks.json` | In-frame and size checks, the crossing time, the frames where each object is hidden |
| `controls/vace_depth.mp4` | The depth video given to `C` and `C0` |
| `meta.json` | Scene file hash, Blender version, render time |

### Folders to ignore

`drones_air`, `drone_takeoff` and `convoy_overhead` (no `_phys`) are the scripted-motion versions and earlier trials, including single clips on Wan2.1-VACE-14B and 1.3B under `drones_air/Bbox_edit/`. They are not part of the round 2 results.

## 6. Results in short

Clips were read by eye from contact sheets; no tool scored them.

- **Box control works on the two drone scenes.** All 6 `Bbox_edit` clips keep the right number of drones and follow the simulator's path. Two flaws: the drones swap colours in 1 of 3 crossing clips, and the drone flies in front of the tree in 1 of 3 take-off clips.
- **Without control for the whole clip, the event does not happen** in 15 of 16 drone clips (`A`, `Aplus`, `B_edit`, `P1_edit`).
- **Depth follows the path but does not hold colours** through the crossing in 4 of 6 two-drone clips.
- **The convoy scene fails in every condition** and needs rework.

More detail:

- [Round 2 findings doc](2026-11-11-0845-round2-findings.md): per-clip tables, caveats, next steps
- [Sim2Real Round 2 page](https://claude.ai/artifact/DMwkj9covsGwnoHo46j3hD): all 51 clips with playback
- [Sim2Real Round 2 Results deck](https://claude.ai/artifact/S8mjMoxtDY3y6cFJrnEcsM)
