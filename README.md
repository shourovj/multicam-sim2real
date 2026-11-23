# Simulator-controlled video generation

From a text prompt to a realistic video whose motion comes from a physics simulator.

1. An LLM turns the prompt into a **scene spec**: the world, the objects, what each object does, and the cameras.
2. The **simulator** moves every object once, on one clock, and checks the result against rules in the spec.
3. **Blender** renders that one simulated world from every camera as plain grey "clay", and exports control videos: depth, segmentation, edges.
4. A **video-to-video model** (Cosmos3-Nano) gets the control videos and a text prompt. The controls carry the structure and motion; the text carries how things look.

Two parts of this pipeline get evaluated:

- **Text → sim.** Does the simulated world do what the prompt asked for?
- **Sim → video.** Does the generated video follow the simulator (same objects, same events, same colours across cameras), and does it look real?

This repo started as a copy of branch `shourov/pilot-v0` of [Sim2Real-VideoGen](https://github.com/UTAustin-SwarmLab/Sim2Real-VideoGen) at commit `0b3e23c` (2026-11-22), with its history kept.

## Pipeline

```
text prompt                                   configs/scenes/<scene>.source.md
   │  [1] LLM writes the scene spec           (by hand-in-the-loop for now, recorded in the .source.md)
   ▼
scene spec                                    configs/scenes/<scene>.yaml  (spec v2: world, signals, objects, cameras, checks)
   │  [2] simulate once, run the checks        sim/simulate_world.py  (numpy only, no GPU)
   ▼
world/                                        world.npz (every part's pose per frame), world.json (signals, events, check results)
   │  [3] render each camera in Blender        sim/render_world.py
   │  [4] export a bundle per camera           sim/export_world.py
   ▼
cams/<cam>/bundle/                            rgb, depth, instance masks, boxes, tracks, visibility, checks
cams/<cam>/bundle/controls/                   depth.mp4, seg.mp4, edge.mp4, normals.mp4, rgb.mp4 (clay), controls_sheet.jpg
   │  [5] compose the prompt                   gen/compose_prompts.py  (world block + camera block → composed/<view>.json)
   │  [6] generate on free GPUs                gen/run_multicam.py  (Cosmos3-Nano or Cosmos3-Nano-Sim-Transfer)
   ▼
gen/<view>/<cond>/<model>/seed<k>/            out_video.mp4, prompt.json, request.json, run.json (hashes of every input)
   │  [7] score against the simulator          eval/score_multicam.py
   ▼
scores/<experiment>/                          colour.csv, lights.csv, summary.json
```

Steps 1 and 5 are the LLM jobs. Everything else is deterministic code. Each step reads and writes files, so any step can be re-run alone.

`sim/build_world.sh` runs steps 2 to 4 for every camera of a scene.

## What is in the repo

| Folder | What it holds |
| --- | --- |
| [`configs/scenes/`](configs/scenes/) | Scene specs (`<scene>.yaml`) and the prompt each was written from (`<scene>.source.md`). `history/` keeps old revisions byte for byte. |
| [`configs/prompts/`](configs/prompts/) | Prompt blocks per experiment and scene: `world.json`, `cams/<cam>.json`, `cut3.json`, the exact captions in `composed/`, and for text-only runs the simulator facts in `facts/`. |
| [`configs/experiments/`](configs/experiments/) | One file per experiment: scenes, model and revision, sampler, conditions, seeds, GPU limits. |
| [`sim/world/`](sim/world/) | The multi-camera simulator: spec loading, layouts, kinematics, cameras, checks, controls, and Blender mesh builders. Read [`sim/world/__init__.py`](sim/world/__init__.py) first. |
| [`sim/physics/`](sim/physics/) | Motion models: quadrotor, ground and lane vehicles, pedestrian, robot arm, pour, rigid body. |
| [`gen/`](gen/) | Prompt composing, model inputs, the GPU job runner, and one shell script per model. |
| [`eval/`](eval/) | Scoring against the simulator. |
| [`setup/`](setup/) | Clone script for third-party code, env specs, asset fetcher, and the full setup guide. |
| [`smoke/`](smoke/), [`tests/`](tests/) | Tool smoke tests and physics unit tests. |
| [`docs/`](docs/) | Design docs, findings and the literature review from Sim2Real-VideoGen. |

Single-camera scripts from the earlier rounds (`sim/build_bundle.sh`, `sim/controls.py`, `gen/run_matrix.py`, the Wan/VACE scripts) still work. See [`setup/README.md`](setup/README.md) and the [round 2 guide](docs/2026-11-11-1304-pilot-v0-branch-guide.md).

## 1. Setup

Pick the path for where you are. Both end with the same [check](#check-the-install-both-paths), then section 2.

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

**A5. Fetch the scene assets into your data folder.** About 35 MB of CC0 Poly Haven textures and models, listed in [`configs/assets/polyhaven.yaml`](configs/assets/polyhaven.yaml).

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
| Disk | About 70 GB to start: code 1 GB, Python envs 19 GB, Blender 2 GB, Cosmos3-Nano 37 GB, plus uv's cache. Another 33 GB for Cosmos3-Nano-Sim-Transfer. About 3 GB of outputs per scene. | |
| System packages | `git`, `curl`, `xz-utils`, `ffmpeg` (the pipeline calls `ffmpeg` and `ffprobe` from your `PATH`), and the X11 and OpenGL libraries Blender loads even when it renders headless | |
| Network | GitHub, PyPI, Hugging Face, download.blender.org, polyhaven.com | |
| Accounts | A Hugging Face account, only for the gated Cosmos3-Nano-Sim-Transfer | |

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

This clones cosmos-framework, VACE, Wan-Move, VideoX-Fun, CoTracker, SAM 2 and the Franka Panda model from MuJoCo Menagerie into `$S2R_ROOT`, each at a pinned commit (about 1.1 GB). Source `~/.sim2real.env` first, or the script falls back to `~/code/sim2real-videogen`.

**B6. Python environments.** The multi-camera pipeline needs two. uv downloads the Python versions they need (3.11 and 3.13).

| Env | Used for |
| --- | --- |
| `$S2R_ROOT/envs/eval` | Simulating, exporting, prompts, the job runner, scoring |
| `$S2R_ROOT/cosmos-framework/.venv` | Building model inputs and running Cosmos |

```bash
mkdir -p $S2R_ROOT/envs/eval && cp setup/envs/eval/{pyproject.toml,uv.lock} $S2R_ROOT/envs/eval/
(cd $S2R_ROOT/envs/eval && uv sync --no-install-package sam-2 && uv sync)                  # SAM 2 builds against the env's torch: two steps
(cd $S2R_ROOT/cosmos-framework && uvx uv@0.11.29 sync --all-extras --group=cu130-train)   # the clone pins uv 0.11.29
```

**B7. Blender and BlenderProc.**

```bash
uv tool install blenderproc==2.8.0
mkdir -p $S2R_ROOT/blender && curl -L https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar -xJ -C $S2R_ROOT/blender
blenderproc pip install pyyaml --blender-install-path $S2R_ROOT/blender
```

Always pass `--blender-install-path $S2R_ROOT/blender`, or BlenderProc downloads a second Blender. Put the package name before the flag: with the flag first, `blenderproc pip install` silently installs nothing.

**B8. Model weights.** Cosmos3-Nano is open. Cosmos3-Nano-Sim-Transfer is gated: accept the licence on its Hugging Face page and run `uvx hf@2.4.0 auth login` first.

```bash
export HF_HUB_OFFLINE=0
hfdl() { uvx hf@2.4.0 download "$@" -q | tail -1; }
hfdl nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2               # 37 GB, every Cosmos3-Nano experiment
# Only for Cosmos3-Nano-Sim-Transfer (multicam_v1.2): the model, the text tokenizer and the VAE its script loads
hfdl nvidia/Cosmos3-Nano-Sim-Transfer --revision 45510d5b8cefa338b7abc01eb702cf1e9cdf6742   # 30 GB
hfdl nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*'
hfdl Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --include Wan2.2_VAE.pth   # 2.7 GB
export HF_HUB_OFFLINE=1
```

On 2026-11-22 each of these commands resolved on Hugging Face (`--dry-run`) to exactly the files the run scripts load.

**B9. Scene assets.**

```bash
$S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py
```

The Wan and VACE models of the earlier rounds, and the SAM 2 and CoTracker checkpoints, are optional; [`setup/README.md`](setup/README.md) has them.

### Check the install (both paths)

```bash
source ~/.sim2real.env
PY=$S2R_ROOT/envs/eval/.venv/bin/python
$PY tests/test_physics.py                                                                   # physics unit tests, no GPU
$PY sim/simulate_world.py configs/scenes/robot_pour.yaml $S2R_DATA/check/robot_pour/world   # simulate + checks, seconds, no GPU
nvidia-smi                                                                                  # find a GPU with a few GB free
GPU=0                                                                                       # set to that GPU's index
CUDA_VISIBLE_DEVICES=$GPU blenderproc run --blender-install-path $S2R_ROOT/blender sim/render_world.py \
    configs/scenes/robot_pour.yaml $S2R_DATA/check/robot_pour/world side $S2R_DATA/check/preview --frames 0 --samples 8
```

- The tests print `ok` lines.
- The simulation ends with `[simulate] robot_pour r2: 113 frames, 5 objects, 3 cameras, 2 events`.
- The render writes `rgb.npy`, `depth.npy` and `instance.npy` to `$S2R_DATA/check/preview/`.

The first render on a new account or machine spends about 6 minutes compiling CUDA kernels on one CPU core, with about 400 MiB on the GPU: it is not hung. The kernels are cached in `~/.nv/` for later renders.

Known problems and fixes for the tools: [Step 1 gotchas](docs/2026-11-10-0945-step1-environment-gotchas.md).

## 2. Run the pipeline on one scene

The shared server has other users. Read `nvidia-smi` right before each GPU step.

| Step | Time on an A100 | Peak GPU memory |
| --- | --- | --- |
| Simulate and check | Seconds | None |
| Render and export, per camera | About 3 minutes (plus 6 minutes of kernel compiling on your first render) | A few GB |
| Cosmos3-Nano with controls, 50 steps, 113 frames (`multicam_v1.1`) | About 28 minutes | 38.8 GiB |
| The same, 4-step smoke test | About 5 minutes | 38.8 GiB |
| Cosmos3-Nano-Sim-Transfer, 4 steps (`multicam_v1.2`) | About 2 minutes | 39.0 GiB |
| Cosmos3-Nano text-to-video, 35 steps (`multicam_v1.3`) | About 3 minutes | 38.7 GiB |

A full `multicam_v1.1` is 16 runs, about 7.5 GPU-hours. The runner runs up to 4 at once, on whichever GPUs have room.

The commands use the scene `intersection_4way` and the experiment `multicam_v1.1`, which is Cosmos3-Nano with depth and segmentation controls. Outputs go to `$S2R_DATA/<out_root>/<scene>/r<rev>/`, where `out_root` and `rev` come from the experiment file.

```bash
source ~/.sim2real.env
PY=$S2R_ROOT/envs/eval/.venv/bin/python
S=intersection_4way
OUT=$S2R_DATA/poc/multi-cam/$S/r2
EXP=configs/experiments/multicam_v1_1.yaml
```

**Step 1. Simulate and check (no GPU, seconds).** This writes `world/world.npz` and `world/world.json`. It exits with an error if any check in the spec's `checks:` list fails.

```bash
$PY sim/simulate_world.py configs/scenes/$S.yaml $OUT/world
```

**Step 2. Preview one camera (three frames, low samples).** Look at the framing before a full render.

```bash
CUDA_VISIBLE_DEVICES=$GPU blenderproc run --blender-install-path $S2R_ROOT/blender sim/render_world.py configs/scenes/$S.yaml \
    $OUT/world cctv $S2R_DATA/preview/$S --frames 0,56,112 --samples 16
```

**Step 3. Build every camera.** This simulates again (same result), renders each camera, and exports its bundle and control videos. The third argument lists GPUs; cameras render in parallel, one per GPU. One GPU works too, one camera after another.

```bash
bash sim/build_world.sh configs/scenes/$S.yaml $OUT 0,2,7        # GPUs that nvidia-smi shows as free
```

Look at `$OUT/cams/<cam>/bundle/controls/controls_sheet.jpg` and `overlay.mp4`. The `visible check` lines at the end say whether each camera sees the objects the spec requires.

**Step 4. Check the prompts.** The runner refuses to start a job whose caption file is missing or out of date. `--check` only reports. If you edited prompt blocks, run it without `--check` to rewrite the captions, and commit them.

```bash
$PY gen/compose_prompts.py --exp $EXP --check
```

**Step 5. Generate.** Look at what would run, try a 4-step smoke test (it writes to `gen_smoke/`, never over real runs), then run the experiment. The runner picks GPUs with enough free memory and re-checks before each launch.

```bash
$PY gen/run_multicam.py --exp $EXP --status
$PY gen/run_multicam.py --exp configs/experiments/multicam_v1_1_smoke.yaml --only scene=$S
$PY gen/run_multicam.py --exp $EXP --only scene=$S
```

`--status` gives each job one of these states:

| State | Meaning |
| --- | --- |
| `blocked` | The scene's renders are not there yet. Run step 3. |
| `pending` | Ready to run, or a past attempt failed. |
| `running` | Running now. |
| `done` | Finished, and none of its inputs changed since. It won't run again. |
| `stale` | Finished, but an input (spec, world, prompt, settings, code) changed since. It reruns only with `--rerun-stale`, which keeps the old video as `out_video.stale_<time>.mp4`. |

To stop starting new jobs, `touch $S2R_DATA/poc/multi-cam/STOP`; running jobs finish.

**Step 6. Score.**

```bash
$PY eval/score_multicam.py --exp $EXP --only scene=$S
```

**Your own experiments.** Copy an experiment file under a new name and change `experiment:` (the name in run IDs and the scores folder) and, to keep your runs apart from earlier ones, `out_root:`. Don't edit the existing files: each is the record of runs that already exist.

## 3. Experiments so far

| File | Model | What the model gets |
| --- | --- | --- |
| [`multicam_v1.yaml`](configs/experiments/multicam_v1.yaml) | Cosmos3-Nano | The textured render as a blur control (V). The model copied the render; kept as a baseline. Uses the rev-1 specs in `configs/scenes/history/`. |
| [`multicam_v1_1.yaml`](configs/experiments/multicam_v1_1.yaml) | Cosmos3-Nano, 50 steps | Clay render; depth + segmentation (G). Appearance only from the prompt. |
| [`multicam_v1_2.yaml`](configs/experiments/multicam_v1_2.yaml) | Cosmos3-Nano-Sim-Transfer, 4 steps | One control per clip: depth (D), segmentation (S) or edges (E). |
| [`multicam_v1_3.yaml`](configs/experiments/multicam_v1_3.yaml) | Cosmos3-Nano text-to-video | Text only, no control. A baseline for how much the controls add. |

Each experiment covers two scenes, `intersection_4way` (cars, pedestrians, traffic lights; CCTV, dashcam, phone) and `robot_pour` (a Panda arm pours water; side, wrist, CCTV), plus `cut3`, one video that cuts between the three cameras. The design and the findings are in the [multi-camera v1 design doc](docs/2026-11-18-1739-multicam-v1-design.md), §14 for the clay revision.

### Text-only runs (multicam_v1.3)

With no control, the prompt has to say where each object is in each camera's frame and when. Those claims come from the simulator:

```bash
$PY gen/camera_facts.py --exp configs/experiments/multicam_v1_3.yaml     # needs the bundles from step 3
```

This writes `configs/prompts/multicam_v1_3/<scene>/facts/<camera>.{json,txt}`. The camera blocks in `cams/<camera>.json` are written from those facts. Then run steps 4 to 6 with `multicam_v1_3.yaml`.

## 4. Evaluation

### 4.1 Text → sim

| Check | Where | Status |
| --- | --- | --- |
| Rules in the spec: minimum gap between objects, cars obey signals and yield to pedestrians, events in order, robot joint limits | `checks:` in the spec, [`sim/world/checks.py`](sim/world/checks.py); results in `world/world.json` | Runs on every simulation; a failed check stops it |
| Each camera sees the objects it should | `visible` checks, evaluated after rendering into `cams/<cam>/bundle/checks.json` | Runs on every build |
| What each camera sees, written as text (position in frame, size, direction, on-screen times) | [`gen/camera_facts.py`](gen/camera_facts.py) → `facts/<camera>.txt` | Runs on demand |
| Provenance: the frozen prompt, who wrote the spec, every revision | `configs/scenes/<scene>.source.md` | Kept by hand |
| Does the spec do what the prompt says (each claim in the prompt checked against `world.json` and the facts) | | **Not built yet** |

### 4.2 Sim → video

| Check | Where | Status |
| --- | --- | --- |
| Colour of each object inside its simulator mask vs the colour the prompt and the render say | `scores/<exp>/colour.csv` | Built |
| Traffic-light state read from the video vs the simulator, per frame | `scores/<exp>/lights.csv` | Built (intersection) |
| The same object gets the same colour in every separate view, and before and after each cut | `scores/<exp>/summary.json` | Built |
| Edge F1 against the simulator, object counts, realism rubric | Design doc §7 | **Not built yet**; read by eye |

These checks are coarse. Findings have been read by eye first, with the scores as support.

## 5. Add a new scene

1. Write the user prompt and the spec's provenance in `configs/scenes/<name>.source.md`.
2. Write the spec, `configs/scenes/<name>.yaml` (`spec_version: 2`). Copy `intersection_4way.yaml` or `robot_pour.yaml` and keep the conventions in [`sim/world/__init__.py`](sim/world/__init__.py): Z up, metres, +X east, +Y north.
3. Iterate on step 1 of section 2 until every check passes, then on step 2 until the framing is right.
4. Write the prompt blocks in `configs/prompts/<experiment>/<name>/`: `world.json`, `cams/<cam>.json`, and a cut file if you want one.
5. Add the scene to an experiment file, then run steps 3 to 6.

A new kind of object needs a motion model in `sim/physics/`, a kinematics function in `sim/world/kinematics.py` and a mesh builder in `sim/world/blender/`. A new camera kind is one function in `sim/world/cameras.py`.

## 6. Where outputs go

```
$S2R_DATA/<out_root>/                   poc/multi-cam for the experiments so far
  manifest.jsonl                        one line per launch and finish
  STOP                                  if present, the runner starts no new jobs
  <scene>/r<rev>/
    world/                              world.npz, world.json, spec.yaml
    cams/<cam>/bundle/                  rgb, depth, instance, boxes, tracks, visibility, checks, controls/
    cut/cut3/                           the same, joined by the experiment's schedule
    gen/<view>/<cond>/<model>/seed<k>/  out_video.mp4, prompt.json, request.json, run.log, run.json
    gen_smoke/...                       smoke-test runs
    scores/<experiment>/                colour.csv, lights.csv, summary.json
```

`run.json` records the git commit, whether the tree was dirty, hashes of the spec, prompt, experiment file and controls, the model revision, the GPU and the peak memory.

## 7. Known problems

- `gen/run_cosmos_multicam.sh`, `gen/run_cosmos_simtransfer_multicam.sh` and `gen/run_cosmos_nano.sh` default to `/mnt/Data/sj37434/hf/hub` for weights. Both setup paths set `COSMOS_HF_CACHE`, which overrides it.
- If you run `setup/clone.sh` without sourcing `~/.sim2real.env`, it clones into `~/code/sim2real-videogen`, which on the lab server is the nearly full root disk.
- Cosmos3-Nano runs (`gen/run_cosmos_multicam.sh`, `gen/run_cosmos_nano.sh`) turn offline mode off and ask Hugging Face which commit `main` of `nvidia/Cosmos3-Nano` is. On path A, `run.log` then shows `Ignored error while writing commit hash ... Read-only file system`. That line is harmless: the run loads the cached snapshot `45666784`, the revision the experiment files pin, which is `main` as of 2026-11-22. If NVIDIA moves `main`, a run will try to download the new version. On path A that fails; on path B it silently uses a different model from the one the experiment file names.
- The shared workspace of path A depends on its folders staying readable. An earlier shared workspace on this server became private overnight. If yours stops working with `Permission denied`, use path B.
- Every Wan model needs flash-attn; there is no fallback. Cosmos does not.
- More: [Step 1 gotchas](docs/2026-11-10-0945-step1-environment-gotchas.md).

## Docs

Design docs and findings carried over from Sim2Real-VideoGen are in [`docs/`](docs/). The most relevant here: the [multi-camera v1 design](docs/2026-11-18-1739-multicam-v1-design.md), [round 2 findings](docs/2026-11-11-0845-round2-findings.md) and the [literature review synthesis](docs/lit-review/2026-11-09-1750-lit-review-synthesis.md). The original repo's README is kept as [docs/2026-11-22-1956-sim2real-videogen-readme.md](docs/2026-11-22-1956-sim2real-videogen-readme.md).

| Local draft | Google Doc tab |
| --- | --- |

## Artifacts

claude.ai artifacts made for this project, oldest first.

| Artifact | Kind | Author | Created | What it covers |
| --- | --- | --- | --- | --- |
