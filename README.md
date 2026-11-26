# Simulator-controlled video generation

From a text prompt to a realistic video whose motion comes from a physics simulator.

1. A **prompt file** names the scene, the camera views and the clip format. It is the only input.
2. An LLM, the **SceneAgent**, writes a physics scene from the prompt. **PhysX** simulates it, a checker reports what is wrong (physics, timing, what each camera sees), and the agent fixes the scene until the report passes.
3. **Blender** renders that one simulated world from every camera and exports control videos: depth, segmentation, edges.
4. A **video-to-video model** gets a control video and a text prompt for each view. The control carries the structure and motion; the text carries how things look. Four models are set up: Cosmos3-Nano, Cosmos3-Nano-Sim-Transfer, Wan2.2-VACE-Fun-A14B and LTX-2.5.
5. Every generated clip is **scored against the simulator**.

Two parts of this pipeline get evaluated:

- **Text → sim.** Does the simulated world do what the prompt asked for?
- **Sim → video.** Does the generated video follow the simulator (same objects, same events, same colours across cameras), and does it look real?

This repo started as a copy of branch `shourov/pilot-v0` of [Sim2Real-VideoGen](https://github.com/UTAustin-SwarmLab/Sim2Real-VideoGen) at commit `0b3e23c` (2026-11-22), with its history kept.

## The pipeline

Everything for it is in [`p2s/`](p2s/), one folder per stage, in the order the stages run:

```
p2s/prompts/<scene>.yaml           INPUT   the prompt, the camera views, the clip format
p2s/experiments/v1/<model>.yaml    INPUT   which scenes, controls, model and seeds
   │
   │  1 scene     p2s/1_scene/      SceneAgent writes the scene; PhysX simulates; the checker reports; repeat until PASS
   ▼
world/                             scene.json, x.json + streams/ (every pose per frame), r.md (the report), spec.yaml
   │  2 render    p2s/2_render/     Blender renders each camera; the exporter writes its bundle
   ▼
cams/<cam>/bundle/                 rgb, depth, masks, tracks; controls/ depth.mp4, seg.mp4, edge.mp4, normals.mp4, rgb.mp4
   │  3 generate  p2s/3_generate/   captions + one control video → one clip per view, control and seed
   ▼
gen/<cam>/<control>/<model>/seed<k>/out_video.mp4
   │  4 score     p2s/4_score/      depth, masks and tracks read from the clip, compared with the simulator's
   ▼
scores/<experiment>/, tables/<experiment>.csv
```

Stages 1 and the captions of stage 3 are the LLM jobs. Everything else is deterministic code. Each stage reads and writes files, so any stage can be run alone. [`p2s/README.md`](p2s/README.md) has the details.

## Run it

```bash
source ~/.sim2real.env
bash p2s/run.sh p2s/experiments/v1/smoke_ltx25.yaml        # smallest end-to-end run: one scene, three clips, a few minutes
bash p2s/run.sh p2s/experiments/v1/ltx25.yaml 5,6,7        # a full experiment; the GPU list is for the Blender renders
```

`p2s/run.sh` runs all four stages for every scene the experiment file names and skips what is already done. The server is shared: read `nvidia-smi` first. Measured on an A100:

| Stage | Time | Peak GPU memory |
| --- | --- | --- |
| 1 scene: the SceneAgent's iterations (`ball_ramp`: three) | about 5 minutes | not measured |
| 2 render and export, per camera | about 3.5 minutes | not measured |
| 3 generate, LTX-2.5 distilled, per clip | 49 s | 36.5 GiB |
| 3 generate, LTX-2.5 full model at 50 steps, per clip | about 6.5 minutes | 36.6 GiB |
| 4 score, per clip | about 45 s | not measured |

## Repo map

```
p2s/                  ★ the pipeline
├── run.sh            one command for an experiment file
├── prompts/          INPUT: one prompt file per scene
├── experiments/v1/   INPUT: one file per model
├── captions/         what each view's video prompt says, per scene
├── 1_scene/          prompt file → accepted PhysX world
├── 2_render/         world → camera bundles and control videos
├── 3_generate/       control + prompt → video; models/<model>/ holds each model's scripts
└── 4_score/          video against the simulator → scores, tables, figures
sim/physx/            the PhysX simulator and checker (C++), and the SceneAgent's manual
sim/blender/          builds the Blender scene from the physics output
legacy/               the earlier route: hand-written scene specs, a numpy simulator, their experiments
setup/                clone script, env specs, the full setup guide
docs/                 design docs, the evaluation report, the literature review
tests/                unit tests for the metrics and the older motion models
```

| Folder | What it holds |
| --- | --- |
| [`p2s/`](p2s/) | The pipeline. [`p2s/README.md`](p2s/README.md) has the prompt file's format, what each stage reads and writes, and how to add a scene, a control or a model. |
| [`p2s/3_generate/`](p2s/3_generate/) | The GPU job runner, prompt composing, and one folder per model. [README](p2s/3_generate/README.md). |
| [`p2s/4_score/`](p2s/4_score/) | Scoring against the simulator: five scene-independent scores, the metric functions, one folder per reader model. [README](p2s/4_score/README.md). |
| [`sim/physx/`](sim/physx/) | GPU physics on PhysX 5.9 from a JSON scene, and the checker whose report drives the SceneAgent's loop. [README](sim/physx/README.md). |
| [`sim/blender/`](sim/blender/) | `import_x.py`: builds the Blender scene from the physics output, with no per-scene code. |
| [`legacy/`](legacy/) | The earlier route and its experiments, still runnable. [README](legacy/README.md). |
| [`setup/`](setup/) | Clone script for third-party code, env specs, asset fetcher, and the [full setup guide](setup/README.md). |

## Where outputs go

```
$S2R_DATA/cv-project/p2s/               the experiments' out_root
  manifest.jsonl                        one line per launch and finish
  STOP                                  if present, the runner starts no new jobs
  tables/<experiment>.csv               one score table per experiment
  <scene>/r<rev>/
    agent/                              the SceneAgent's task and iterations: it<N>/scene.json, x.json, r.md
    world/                              the accepted iteration, spec.yaml, world.json
    cams/<cam>/bundle/                  rgb, depth, instance, boxes, tracks, visibility, controls/
    cut/<cut>/                          several cameras joined into one clip by the experiment's schedule
    gen/<view>/<control>/<model>/seed<k>/   out_video.mp4, the prompt, request.json, run.log, run.json
    gen_smoke/...                       smoke-test runs
    gen/.../seed<k>/est/                reader outputs for scoring: depth, masks, point tracks
    scores/<experiment>/                eval2.csv, eval3.csv, eval_summary.json
```

`run.json` records the git commit, whether the tree was dirty, hashes of every input, the model revision, the GPU and the peak memory. A finished clip is redone only if one of those inputs changes.

## Evaluation

### Prompt → sim

The checker (`sim/physx`, `p2s_check`) runs on every iteration and its report decides whether the scene is accepted:

| Check | Status |
| --- | --- |
| Physics: finite values, no interpenetration, nothing below ground or flung away, no energy gained beyond what actuators delivered | Runs on every iteration |
| The scene's own expectations: event timing and order, speeds, gaps, what each camera must see | Runs on every iteration |
| What each camera sees, from geometry alone: share of frames in view, occlusion, screen size | Written into every report |
| Does the scene's expectation list cover the prompt? | **Not built yet** |

### Sim → video

[`p2s/4_score/run_eval.py`](p2s/4_score/run_eval.py) scores every finished run of an experiment. The five scores are the same for every scene, because every reference comes from the camera bundles. All are in [0, 1], 1 best. Definitions, files and limits: [`p2s/4_score/README.md`](p2s/4_score/README.md). Equations, figures and first results: the [evaluation report](docs/2026-11-24-1158-eval-report.md).

| Score | Question | Simulator reference | Read from the video with |
| --- | --- | --- | --- |
| `depth` | Is the scene's shape where the simulator put it? | Rendered depth | Depth Anything 3 |
| `mask_iou` | Is each object where the simulator put it, in every frame? | Rendered object masks | SAM 2.1, started from the simulator's box |
| `track` | Does each surface point move as in the simulation? | Surface points on every moving part | CoTracker3 |
| `object_colour` | Is the same object the same colour in every camera? | Object masks of two cameras at one instant | Colour inside the masks |
| `surface_colour` | Is the same surface point the same colour in every camera? | Depth and camera poses, which give exact pixel pairs | Colour at the pairs |

- **Eval2, per camera.** `eval2` is the harmonic mean of `depth`, `mask_iou` and `track`, each averaged over the cameras first.
- **Eval3, across cameras.** `object_colour` and `surface_colour` compare the separate camera views of one condition and seed. They count only objects and surfaces that are where the simulator puts them in both cameras, and report that share (`object_in_place`, `surface_in_place`).
- **One table across experiments.** [`p2s/4_score/summarize.py`](p2s/4_score/summarize.py) lists every model and condition from the densest control to text only.
- **Ceilings.** `--on-render` scores the simulator's own render with the same readers. Read the generated clips' scores against it: the readers are not perfect even on a clip that follows the simulator exactly.

Also in the repo: [`legacy/eval/score_multicam.py`](legacy/eval/score_multicam.py), the earlier scorer, names each object's colour and reads the traffic-light state per frame (`colour.csv`, `lights.csv`). The light check is specific to scenes with signals and belongs with text → sim.

Not built: optical flow (the renderer does not write motion vectors; `track` covers motion), a realism score, and a judge model for object identity beyond colour.


## Setup

Pick the path for where you are. Both end with the same [check](#check-the-install-both-paths), then [Run it](#run-it). The PhysX simulator is built separately, once: [`sim/physx/README.md`](sim/physx/README.md).

| Where you are | Path | What you install | Time |
| --- | --- | --- | --- |
| The lab server (swarm5, hostname `ece-a52150`) | [A. Use the shared workspace](#path-a-on-the-lab-server-use-the-shared-workspace) | BlenderProc and 35 MB of scene assets | about 15 minutes |
| Any other Linux machine with an NVIDIA GPU | [B. Build from scratch](#path-b-on-another-machine-build-from-scratch) | Code, Python envs, Blender and about 40 GB of weights | an hour or more, mostly downloads |

The repo is private. Ask Shourov Joarder to add your GitHub account as a collaborator before you clone it.

Nothing big lives in the repo. Every script reads its paths from environment variables:

| Variable | Holds |
| --- | --- |
| `S2R_ROOT` | Third-party code, Python envs, Blender, the robot model |
| `HF_HUB_CACHE`, `COSMOS_HF_CACHE` | Model weights |
| `S2R_DATA` | Every output, and the scene assets |
| `S2R_CACHE`, `UV_CACHE_DIR` | Caches |

They go in `~/.sim2real.env`. Run `source ~/.sim2real.env` in every new shell before you use the repo. Keep them out of `~/.bashrc`: the Hugging Face lines would change Hugging Face in your other projects.

### Path A: on the lab server, use the shared workspace

Shourov's workspace on the data disk already has the third-party code, the Python envs, Blender and the model weights, and every lab account can read it. You use it read-only. Your clone, outputs and caches are your own, so nothing you run touches anyone else's files.

| What | Where | Whose |
| --- | --- | --- |
| Third-party code, Python envs, Blender, robot model | `/mnt/Data/sj37434/sim2real-local` | Shared, read-only |
| Model weights | `/mnt/Data/sj37434/hf/hub` | Shared, read-only |
| Your outputs and scene assets | `/mnt/Data/<your EID>/sim2real-data` | Yours |
| Your caches | `/mnt/Data/<your EID>/.cache/sim2real` | Yours |
| This repo | Any folder you like, e.g. `~/projects/` | Yours |

This path was tested on 2026-11-22 with the shared folders mounted read-only and a fresh clone: the physics tests, simulating, rendering and exporting all three cameras of `robot_pour`, and a Cosmos3-Nano run.

**A1. Clone the repo into your own folder.**

```bash
mkdir -p ~/projects && cd ~/projects
git clone git@github.com:shourovj/multicam-sim2real.git
cd multicam-sim2real
```

**A2. Check that you can read the shared workspace.**

```bash
ls /mnt/Data/sj37434/sim2real-local/envs/eval/.venv/bin/python /mnt/Data/sj37434/hf/hub/models--nvidia--Cosmos3-Nano/refs/main
/mnt/Data/sj37434/sim2real-local/envs/eval/.venv/bin/python -c "import numpy, cv2, yaml; print('shared env ok')"
```

You should see both paths, then `shared env ok`. `Permission denied` means your account can't read the workspace: ask Shourov, or use path B.

**A3. Write your paths file.** Paste it as it is: `${USER%@*}` is your login without `@austin.utexas.edu`, which is your EID. It replaces any `~/.sim2real.env` you already have.

```bash
cat > ~/.sim2real.env <<'EOF'
export S2R_ROOT=/mnt/Data/sj37434/sim2real-local          # shared, read-only
export HF_HUB_CACHE=/mnt/Data/sj37434/hf/hub              # shared, read-only
export COSMOS_HF_CACHE=$HF_HUB_CACHE
export HF_HUB_OFFLINE=1                                   # never try to download into the shared cache
export S2R_DATA=/mnt/Data/${USER%@*}/sim2real-data        # yours: outputs and scene assets
export S2R_CACHE=/mnt/Data/${USER%@*}/.cache/sim2real     # yours: caches
export UV_CACHE_DIR=$S2R_CACHE/uv
export CUDA_CACHE_MAXSIZE=4294967296                      # keep Blender's compiled CUDA kernels
EOF
source ~/.sim2real.env
mkdir -p $S2R_DATA $S2R_CACHE
```

Your files go on `/mnt/Data` because the root disk is nearly full.

**A4. Install uv and BlenderProc for your account.** BlenderProc is the one tool each person installs for themselves.

```bash
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }
uv tool install blenderproc==2.8.0
```

Don't run `blenderproc pip install`. The shared Blender already has what it needs, and you can't write to it.

**A5. Fetch the scene assets into your data folder.** About 35 MB of CC0 Poly Haven textures and models, listed in [`legacy/configs/assets/polyhaven.yaml`](legacy/configs/assets/polyhaven.yaml).

```bash
$S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py
```

Then run the [check](#check-the-install-both-paths).

**What not to run on path A.** `setup/clone.sh`, `uv sync`, `uv pip install`, `blenderproc pip install` and `hf download` all write into `$S2R_ROOT` or `$HF_HUB_CACHE`, so with the shared workspace they fail with `Permission denied`. If you need another model or package, ask Shourov to add it, or build your own workspace with path B.

### Path B: on another machine, build from scratch

**Requirements**

| What | Needs | Tested with |
| --- | --- | --- |
| OS | Linux x86_64 | Ubuntu 22.04 |
| GPU | NVIDIA, about 40 GB free per Cosmos job (peak 38.8 GiB); a few GB per Blender render | A100 80 GB |
| Driver | 580 or newer, because the Cosmos env is the CUDA 13.0 build. No CUDA toolkit needed. | 580.159 |
| Disk | About 70 GB to start: code 1 GB, Python envs 19 GB, Blender 2 GB, Cosmos3-Nano 37 GB, plus uv's cache. Another 33 GB for Cosmos3-Nano-Sim-Transfer, 76 GB for Wan2.2-VACE-Fun and its env, and about 75 GB for LTX-2.5 and its env. About 3 GB of outputs per scene. | |
| System packages | `git`, `curl`, `xz-utils`, `ffmpeg` (the pipeline calls `ffmpeg` and `ffprobe` from your `PATH`), and the X11 and OpenGL libraries Blender loads even when it renders headless | |
| Network | GitHub, PyPI, Hugging Face, download.blender.org, polyhaven.com | |
| Accounts | A Hugging Face account, only for the gated Cosmos3-Nano-Sim-Transfer and LTX-2.5 | |

You need root only for the system packages. No Docker. With a driver from 570 to 579, cosmos-framework also has a CUDA 12.8 build (`--group=cu128-train` in B6); it has not been tried here.

**B1. System packages** (Ubuntu or Debian; skip any you already have).

```bash
sudo apt-get install -y git curl xz-utils ffmpeg libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1
```

The packages after `ffmpeg` are the X11 and OpenGL libraries Blender 4.2 loaded on the lab server.

**B2. uv.**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

**B3. Clone the repo.**

```bash
git clone git@github.com:shourovj/multicam-sim2real.git && cd multicam-sim2real
```

**B4. Write your paths file.** Set `BIG` to a folder on a disk with about 100 GB free; everything large goes under it.

```bash
BIG=$HOME/sim2real        # change this if your home disk is small
cat > ~/.sim2real.env <<EOF
export S2R_ROOT=$BIG/local          # third-party code, Python envs, Blender
export HF_HUB_CACHE=$BIG/hf/hub     # model weights
export COSMOS_HF_CACHE=$BIG/hf/hub
export HF_HUB_OFFLINE=1             # never download by accident; B8 turns it off for the download
export S2R_DATA=$BIG/data           # outputs and scene assets
export S2R_CACHE=$BIG/cache         # caches
export UV_CACHE_DIR=$BIG/cache/uv
export CUDA_CACHE_MAXSIZE=4294967296
EOF
source ~/.sim2real.env
```

**B5. Third-party code.**

```bash
bash setup/clone.sh
```

This clones cosmos-framework, VACE, Wan-Move, VideoX-Fun, LTX-2, CoTracker, SAM 2 and the Franka Panda model from MuJoCo Menagerie into `$S2R_ROOT`, each at a pinned commit (about 1.1 GB). Source `~/.sim2real.env` first, or the script falls back to `~/code/sim2real-videogen`.

**B6. Python environments.** The multi-camera pipeline needs the first two. The other two are only for their model. uv downloads the Python versions they need (3.11 and 3.13); VideoX-Fun uses the system's Python 3.10.

| Env | Used for |
| --- | --- |
| `$S2R_ROOT/envs/eval` | Simulating, exporting, prompts, the job runner, scoring |
| `$S2R_ROOT/cosmos-framework/.venv` | Building Cosmos requests and running both Cosmos models |
| `$S2R_ROOT/VideoX-Fun/.venv` | Running Wan2.2-VACE-Fun |
| `$S2R_ROOT/LTX-2/.venv` | Running LTX-2.5 |

```bash
mkdir -p $S2R_ROOT/envs/eval && cp setup/envs/eval/{pyproject.toml,uv.lock} $S2R_ROOT/envs/eval/
(cd $S2R_ROOT/envs/eval && uv sync --no-install-package sam-2 && uv sync)                  # SAM 2 builds against the env's torch: two steps
(cd $S2R_ROOT/cosmos-framework && uvx uv@0.11.29 sync --all-extras --group=cu130-train)   # the clone pins uv 0.11.29
# only for Wan2.2-VACE-Fun (torch 2.6 / cu124 with a prebuilt flash-attn wheel)
(cd $S2R_ROOT/VideoX-Fun && uv venv --clear --python /usr/bin/python3.10 && uv pip install -r $OLDPWD/setup/envs/videox-fun-requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match)
# only for LTX-2.5 (torch 2.13 / CUDA 13.2 wheels; the clone has no lockfile, so ours goes in first)
(cd $S2R_ROOT/LTX-2 && cp $OLDPWD/setup/envs/ltx-2/uv.lock . && uv sync --frozen --extra natten)
```

**B7. Blender and BlenderProc.**

```bash
uv tool install blenderproc==2.8.0
mkdir -p $S2R_ROOT/blender && curl -L https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar -xJ -C $S2R_ROOT/blender
blenderproc pip install pyyaml --blender-install-path $S2R_ROOT/blender
```

Always pass `--blender-install-path $S2R_ROOT/blender`, or BlenderProc downloads a second Blender. Put the package name before the flag: with the flag first, `blenderproc pip install` silently installs nothing.

**B8. Model weights.** Cosmos3-Nano and Wan2.2-VACE-Fun are open. Cosmos3-Nano-Sim-Transfer and LTX-2.5 are gated: accept the terms on each model's Hugging Face page and run `uvx hf@2.4.0 auth login` first. LTX-2.5 is under the LTX-2.x Community License; read its acceptable-use list before you use the model.

```bash
export HF_HUB_OFFLINE=0
hfdl() { uvx hf@2.4.0 download "$@" -q | tail -1; }
hfdl nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2               # 37 GB, every Cosmos3-Nano experiment
# Only for Cosmos3-Nano-Sim-Transfer (multicam_v1.2): the model, the text tokenizer and the VAE its script loads
hfdl nvidia/Cosmos3-Nano-Sim-Transfer --revision 45510d5b8cefa338b7abc01eb702cf1e9cdf6742   # 30 GB
hfdl nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*'
hfdl Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --include Wan2.2_VAE.pth   # 2.7 GB
# Only for Wan2.2-VACE-Fun: 76 GB, linked into the VideoX-Fun clone
mkdir -p $S2R_ROOT/VideoX-Fun/models/Diffusion_Transformer
ln -sfn "$(hfdl alibaba-pai/Wan2.2-VACE-Fun-A14B --revision 4438caabbfdd7437bbb0283d86cb200d1b7223ba)" \
    $S2R_ROOT/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B
# Only for LTX-2.5: the distilled model (about 66 GiB) and the Union Control adapter that carries the depth control
hfdl Lightricks/LTX-2.5 --revision 49b093a39c19169a8a81721dcc541e07e945232f \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
hfdl Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control --revision 20a580c8606c06a0988d40375a61309e0fe5893d
export HF_HUB_OFFLINE=1
```

On 2026-11-22 each of the Cosmos commands resolved on Hugging Face (`--dry-run`) to exactly the files the run scripts load. The two LTX commands have not been run yet: the lab server is not logged in to Hugging Face.

**B9. Scene assets.**

```bash
$S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py
```

The Wan2.1 VACE and Wan-Move models of the earlier rounds, and the SAM 2 and CoTracker checkpoints, are optional; [`setup/README.md`](setup/README.md) has them.

### Check the install (both paths)

```bash
source ~/.sim2real.env
PY=$S2R_ROOT/envs/eval/.venv/bin/python
$PY tests/test_physics.py                                                                   # physics unit tests, no GPU
$PY legacy/sim/simulate_world.py legacy/configs/scenes/robot_pour.yaml $S2R_DATA/check/robot_pour/world   # simulate + checks, seconds, no GPU
nvidia-smi                                                                                  # find a GPU with a few GB free
GPU=0                                                                                       # set to that GPU's index
CUDA_VISIBLE_DEVICES=$GPU blenderproc run --blender-install-path $S2R_ROOT/blender legacy/sim/render_world.py \
    legacy/configs/scenes/robot_pour.yaml $S2R_DATA/check/robot_pour/world side $S2R_DATA/check/preview --frames 0 --samples 8
```

- The tests print `ok` lines.
- The simulation ends with `[simulate] robot_pour r2: 113 frames, 5 objects, 3 cameras, 2 events`.
- The render writes `rgb.npy`, `depth.npy` and `instance.npy` to `$S2R_DATA/check/preview/`.

The first render on a new account or machine spends about 6 minutes compiling CUDA kernels on one CPU core, with about 400 MiB on the GPU: it is not hung. The kernels are cached in `~/.nv/` for later renders.

Known problems and fixes for the tools: [Step 1 gotchas](docs/2026-11-10-0945-step1-environment-gotchas.md).

## The earlier route

Scenes written by hand as spec v2 files, a numpy simulator, and the experiments run on them (`multicam_v1*`, `cvp_v1`) are in [`legacy/`](legacy/), with their own [README](legacy/README.md). They run with the same generation and scoring code as the pipeline. Documents in [`docs/`](docs/) dated before 2026-11-25 name files by their old paths; the table below maps them.

| Before | Now |
| --- | --- |
| `gen/` (runner, `common/`, model folders) | `p2s/3_generate/` (model folders under `models/`) |
| `eval/` | `p2s/4_score/` |
| `sim/world/export.py`, `sim/world/controls.py` | `p2s/2_render/bundle.py`, `p2s/2_render/control_videos.py` |
| `sim/` (everything but `physx/` and `blender/`), `configs/`, `smoke/` | `legacy/sim/`, `legacy/configs/`, `legacy/smoke/` |
| `gen/run_matrix.py`, `gen/camera_facts.py`, `singlecam_*`, `eval/score_multicam.py` | `legacy/gen/`, `legacy/eval/` |

## Known problems

- `p2s/3_generate/models/cosmos3_nano/run.sh`, `p2s/3_generate/models/cosmos3_sim_transfer/run.sh` and `legacy/gen/cosmos3_nano/singlecam_run.sh` default to `/mnt/Data/sj37434/hf/hub` for weights. Both setup paths set `COSMOS_HF_CACHE`, which overrides it.
- If you run `setup/clone.sh` without sourcing `~/.sim2real.env`, it clones into `~/code/sim2real-videogen`, which on the lab server is the nearly full root disk.
- Cosmos3-Nano runs (`p2s/3_generate/models/cosmos3_nano/run.sh`, `legacy/gen/cosmos3_nano/singlecam_run.sh`) turn offline mode off and ask Hugging Face which commit `main` of `nvidia/Cosmos3-Nano` is. On path A, `run.log` then shows `Ignored error while writing commit hash ... Read-only file system`. That line is harmless: the run loads the cached snapshot `45666784`, the revision the experiment files pin, which is `main` as of 2026-11-22. If NVIDIA moves `main`, a run will try to download the new version. On path A that fails; on path B it silently uses a different model from the one the experiment file names.
- The shared workspace of path A depends on its folders staying readable. An earlier shared workspace on this server became private overnight. If yours stops working with `Permission denied`, use path B.
- Every Wan model needs flash-attn; there is no fallback. Cosmos and LTX do not.
- More: [Step 1 gotchas](docs/2026-11-10-0945-step1-environment-gotchas.md).

## Docs

The [evaluation report](docs/2026-11-24-1158-eval-report.md) explains Eval2 with its equations, figures and first results. Design docs and findings carried over from Sim2Real-VideoGen are in [`docs/`](docs/). The most relevant here: the [multi-camera v1 design](docs/2026-11-18-1739-multicam-v1-design.md), [round 2 findings](docs/2026-11-11-0845-round2-findings.md) and the [literature review synthesis](docs/lit-review/2026-11-09-1750-lit-review-synthesis.md). The original repo's README is kept as [docs/2026-11-22-1956-sim2real-videogen-readme.md](docs/2026-11-22-1956-sim2real-videogen-readme.md).

| Local draft | Google Doc tab |
| --- | --- |

## Artifacts

claude.ai artifacts made for this project, oldest first.

| Artifact | Kind | Author | Created | What it covers |
| --- | --- | --- | --- | --- |
| [Simulator-Referenced Metrics](https://claude.ai/artifact/DYB7nimVQjprhcwhgP875p) | Page | Shourov Joarder | 2026-11-23 | The evaluation metrics whose reference is already in the bundles (depth, masks, point tracks, edges, object look, cross-camera pixel pairs, light timing), each shown on the three intersection cameras with a generated clip |
