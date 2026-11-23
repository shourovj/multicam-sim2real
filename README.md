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

## 1. Setup (once per machine)

Tested on Ubuntu 22.04 with A100 80 GB GPUs, NVIDIA driver 580, no root and no Docker. You need `git` and [`uv`](https://docs.astral.sh/uv/). The full guide, including the Wan and VACE models, is [`setup/README.md`](setup/README.md). Known problems and fixes: [Step 1 gotchas](docs/2026-11-10-0945-step1-environment-gotchas.md).

### 1.1 Paths

Nothing big lives in the repo. Every script reads three paths from [`env.sh`](env.sh). Put them in `~/.sim2real.env`, not `~/.bashrc`: the Hugging Face lines would change Hugging Face in your other projects. Replace `<you>` with your folder on the data disk.

```bash
cat > ~/.sim2real.env <<'EOF'
export S2R_ROOT=/mnt/Data/<you>/sim2real-local       # third-party code, Python envs, Blender, checkpoints
export S2R_DATA=/mnt/Data/<you>/sim2real-data        # every output
export S2R_CACHE=/mnt/Data/<you>/.cache/sim2real     # torch, matplotlib and uv caches
export UV_CACHE_DIR=$S2R_CACHE/uv
export HF_HUB_CACHE=/mnt/Data/<you>/hf/hub           # model weights
export COSMOS_HF_CACHE=$HF_HUB_CACHE                 # the Cosmos scripts read this; without it they use the original author's folder
export HF_HUB_OFFLINE=1                              # never download by accident
export CUDA_CACHE_MAXSIZE=4294967296                 # keep Blender's compiled CUDA kernels
EOF
source ~/.sim2real.env
```

Run `source ~/.sim2real.env` in every new shell before you use the repo.

### 1.2 Third-party code

```bash
bash setup/clone.sh
```

This clones cosmos-framework, VACE, Wan-Move, VideoX-Fun, CoTracker, SAM 2 and the Franka Panda model from MuJoCo Menagerie into `$S2R_ROOT`, each at a pinned commit.

### 1.3 Python environments

The multi-camera pipeline needs two envs:

| Env | Used for |
| --- | --- |
| `$S2R_ROOT/envs/eval` | simulating, exporting, prompts, the job runner, scoring |
| `$S2R_ROOT/cosmos-framework/.venv` | building model inputs and running Cosmos |

```bash
mkdir -p $S2R_ROOT/envs/eval && cp setup/envs/eval/{pyproject.toml,uv.lock} $S2R_ROOT/envs/eval/
(cd $S2R_ROOT/envs/eval && uv sync --no-install-package sam-2 && uv sync)        # SAM 2 builds against the env's torch: two steps
(cd $S2R_ROOT/cosmos-framework && uvx uv@0.11.29 sync --all-extras --group=cu130-train)   # the clone pins uv 0.11.29
```

### 1.4 Blender and BlenderProc

```bash
uv tool install blenderproc==2.8.0
mkdir -p $S2R_ROOT/blender && curl -L https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar -xJ -C $S2R_ROOT/blender
blenderproc pip install pyyaml --blender-install-path $S2R_ROOT/blender
```

Always pass `--blender-install-path $S2R_ROOT/blender`, or BlenderProc downloads a second Blender.

### 1.5 Model weights

Gated models need you to accept the licence on the model's Hugging Face page and run `uvx hf@2.4.0 auth login` first. Turn offline mode off for the download only.

```bash
export HF_HUB_OFFLINE=0
hfdl() { uvx hf@2.4.0 download "$@" -q | tail -1; }
hfdl nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2               # 37 GB, all Cosmos3-Nano experiments
# Only for Cosmos3-Nano-Sim-Transfer (multicam_v1.2). Paths and revisions match gen/run_cosmos_simtransfer_multicam.sh;
# these three lines were not run as written, so check that script's "missing" message if one fails.
hfdl nvidia/Cosmos3-Nano-Sim-Transfer --revision 45510d5b8cefa338b7abc01eb702cf1e9cdf6742
hfdl nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*'
hfdl Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --include Wan2.2_VAE.pth
export HF_HUB_OFFLINE=1
```

### 1.6 Scene assets

```bash
$S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py
```

This fetches about 35 MB of CC0 Poly Haven textures and models (list: [`configs/assets/polyhaven.yaml`](configs/assets/polyhaven.yaml)) into `$S2R_DATA/assets/polyhaven/`.

### 1.7 Check the install

```bash
PY=$S2R_ROOT/envs/eval/.venv/bin/python
$PY tests/test_physics.py                                              # physics unit tests, no GPU
$PY sim/simulate_world.py configs/scenes/robot_pour.yaml $S2R_DATA/check/robot_pour/world   # simulate + checks, seconds, no GPU
```

## 2. Run the pipeline on one scene

The shared server has other users. Read `nvidia-smi` right before each GPU step. A render needs a few GB per card; Cosmos needs about 50 GB.

The commands use the scene `intersection_4way` and the experiment `multicam_v1.1`, which is Cosmos3-Nano with depth and segmentation controls. Outputs go to `$S2R_DATA/poc/multi-cam/<scene>/r<rev>/`, where `rev` comes from the experiment file.

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
blenderproc run --blender-install-path $S2R_ROOT/blender sim/render_world.py configs/scenes/$S.yaml \
    $OUT/world cctv $S2R_DATA/preview/$S --frames 0,56,112 --samples 16
```

**Step 3. Build every camera.** This simulates again (same result), renders each camera, and exports its bundle and control videos. The third argument lists GPUs; cameras render in parallel, one per GPU.

```bash
bash sim/build_world.sh configs/scenes/$S.yaml $OUT 0,1,2
```

Look at `$OUT/cams/<cam>/bundle/controls/controls_sheet.jpg` and `overlay.mp4`. The `visible check` lines at the end say whether each camera sees the objects the spec requires.

**Step 4. Compose the prompts.** The runner refuses to start a job whose caption file is missing or out of date. `--check` only reports; without it, the files are rewritten. Commit the result.

```bash
$PY gen/compose_prompts.py --exp $EXP --check
```

**Step 5. Generate.** Look at what would run, try a 4-step smoke test (it writes to `gen_smoke/`, never over real runs), then run the experiment. The runner picks GPUs with enough free memory and re-checks before each launch.

```bash
$PY gen/run_multicam.py --exp $EXP --status
$PY gen/run_multicam.py --exp configs/experiments/multicam_v1_1_smoke.yaml --only scene=$S
$PY gen/run_multicam.py --exp $EXP --only scene=$S
```

To stop starting new jobs, `touch $S2R_DATA/poc/multi-cam/STOP`; running jobs finish. A job is skipped when its `run.json` says done and its inputs have not changed. If an input changed, the job shows as stale; `--rerun-stale` reruns it and keeps the old video.

**Step 6. Score.**

```bash
$PY eval/score_multicam.py --exp $EXP --only scene=$S
```

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
$S2R_DATA/poc/multi-cam/
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

- `gen/run_cosmos_multicam.sh`, `gen/run_cosmos_simtransfer_multicam.sh` and `gen/run_cosmos_nano.sh` default to the original author's weight folder. Set `COSMOS_HF_CACHE` (section 1.1).
- If you run `setup/clone.sh` without sourcing `~/.sim2real.env`, it clones into `~/code/sim2real-videogen` on the root disk.
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
