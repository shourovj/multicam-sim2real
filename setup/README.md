# Setup

How to get a working copy of the v0 pipeline: third-party code, environments, Blender and model weights, plus the steps to re-run the `drones_air` proof of concept.

Known problems and their fixes are in the [Step 1 gotchas doc](../docs/2026-11-10-0945-step1-environment-gotchas.md). Read it before debugging.

## Workspace paths

Everything big lives outside this repo. [`env.sh`](../env.sh) defines three paths, and every script reads them:

| Variable | Default | Holds |
| --- | --- | --- |
| `S2R_ROOT` | `~/code/sim2real-videogen` | Third-party clones, Python envs, Blender, weights. Can be someone else's workspace that you only read. |
| `S2R_DATA` | `$S2R_ROOT/data` | Outputs (scene bundles, generated videos, grids). Must be writable by you. |
| `S2R_CACHE` | `$S2R_ROOT/.cache` | torch, matplotlib and uv caches. Must be writable by you. |

Set them in your shell (e.g. `~/.bashrc`) before running anything. The defaults reproduce the original layout on swarm5.

## Option A: reuse SP's workspace (no longer available)

S P Sharan's workspace at `/home/ss99569/code/sim2real-videogen` was readable by every user on swarm5 until mid-November 2026. On 2026-11-18 it was not (`drwx------`), so use Option B.

## Option B: build your own workspace

Tested end to end on swarm5 on 2026-11-18: Ubuntu 22.04, A100 80 GB, driver 580, no CUDA toolkit, no root, no Docker. Needs [`uv`](https://docs.astral.sh/uv/) and `git`. Run every block from the repo root. With uv's cache already filled, the environments take about two minutes; from an empty cache, expect several more minutes of downloads.

### 0. Paths

Put these in `~/.sim2real.env` and `source ~/.sim2real.env` before running anything. Keep them out of `~/.bashrc`: the two Hugging Face lines would change Hugging Face for your other projects. Replace `<you>` with your folder on `/mnt/Data`.

```
export S2R_ROOT=/mnt/Data/<you>/sim2real-local          # third-party code, envs, Blender
export S2R_DATA=/mnt/Data/<you>/sim2real-data           # outputs
export S2R_CACHE=/mnt/Data/<you>/.cache/sim2real        # torch / matplotlib caches
export UV_CACHE_DIR=$S2R_CACHE/uv                       # uv's download cache; the root disk is nearly full
export HF_HUB_CACHE=/mnt/Data/<you>/hf/hub              # model weights
export HF_HUB_OFFLINE=1                                 # never download; unset it for a deliberate download
export CUDA_CACHE_MAXSIZE=4294967296                    # keep Blender's compiled CUDA kernels (gotchas §3.1)
export VIDEOX_FUN_DIR=$S2R_ROOT/VideoX-Fun
```

### 1. Third-party code

```
source ~/.sim2real.env
bash setup/clone.sh          # VACE, Wan-Move, co-tracker, sam2, cosmos-framework, VideoX-Fun at the pinned commits,
                             # and a sparse clone of MuJoCo Menagerie (franka_emika_panda only, Apache-2.0)
```

Source `~/.sim2real.env` first: without it `env.sh` falls back to `~/code/sim2real-videogen` on the root disk.

### 2. Python envs

| Env | Python | Spec in this repo | Used by |
| --- | --- | --- | --- |
| `$S2R_ROOT/envs/eval` | 3.11 | [`setup/envs/eval/`](envs/eval/) (`pyproject.toml` + `uv.lock`) | `sim/export.py`, `gen/run_matrix.py`, `gen/cosmos_nano_request.py`, SAM 2 + CoTracker3 |
| `$S2R_ROOT/VACE/.venv` | 3.10 (system) | [`setup/envs/vace-requirements.txt`](envs/vace-requirements.txt) | `sim/controls.py`, VACE runs, prompt loading |
| `$S2R_ROOT/VideoX-Fun/.venv` | 3.10 (system) | [`setup/envs/videox-fun-requirements.txt`](envs/videox-fun-requirements.txt) | Wan2.2-VACE-Fun runs (`gen/run_vacefun.sh`) |
| `$S2R_ROOT/cosmos-framework/.venv` | 3.13 | the clone's own `uv.lock` | Cosmos3-Nano runs (`gen/run_cosmos_nano.sh`) |
| `$S2R_ROOT/Wan-Move/.venv` | 3.10 (system) | [`setup/envs/wanmove-requirements.txt`](envs/wanmove-requirements.txt) | Wan-Move runs (not re-tested on 2026-11-18) |

```
source ~/.sim2real.env; REPO=$PWD

# eval: SAM 2 builds against the env's own torch, so sync in two steps (gotchas §3.5)
mkdir -p $S2R_ROOT/envs/eval && cp setup/envs/eval/{pyproject.toml,uv.lock} $S2R_ROOT/envs/eval/
(cd $S2R_ROOT/envs/eval && uv sync --no-install-package sam-2 && uv sync)

# VACE and VideoX-Fun: torch 2.6 / cu124 with the prebuilt flash-attn 2.7.4 wheel
(cd $S2R_ROOT/VACE && uv venv --clear --python /usr/bin/python3.10 && \
  uv pip install -r $REPO/setup/envs/vace-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match)
(cd $S2R_ROOT/VideoX-Fun && uv venv --clear --python /usr/bin/python3.10 && \
  uv pip install -r $REPO/setup/envs/videox-fun-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match)

# Cosmos: the clone pins uv to exactly 0.11.29 (`required-version`), so run that uv through uvx. CUDA 13.0 build (driver 580).
(cd $S2R_ROOT/cosmos-framework && uvx uv@0.11.29 sync --all-extras --group=cu130-train)

# Wan-Move (torch 2.7.1 / cu126, flash-attn 2.8.3 wheel), only if you need it
(cd $S2R_ROOT/Wan-Move && uv venv --clear --python /usr/bin/python3.10 && \
  uv pip install -r $REPO/setup/envs/wanmove-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)
```

These commands rebuild an env from scratch: `uv venv --clear` deletes an existing `.venv` without asking, and the install refills it from uv's cache in seconds. Without `--clear`, uv stops to ask, and in a pasted block the next line can be taken as the answer. Paste one command at a time if you are unsure.

The requirements files are `uv pip freeze` output from working envs. flash-attn is required by every Wan model; there is no SDPA fallback (gotchas §3.3, §3.4). The cosmos-framework env needs no overlay: `COSMOS_OVERLAY` in `gen/run_cosmos_nano.sh` is only for a shared env whose packages you can't read.

### 3. Blender and BlenderProc

```
uv tool install blenderproc==2.8.0
mkdir -p $S2R_ROOT/blender && curl -L https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar -xJ -C $S2R_ROOT/blender
blenderproc pip install pyyaml --blender-install-path $S2R_ROOT/blender
```

Put the package name before `--blender-install-path`. With the flag first, `blenderproc pip install` prints the Blender path and silently installs nothing. Always pass `--blender-install-path $S2R_ROOT/blender`, or BlenderProc downloads another Blender.

### 4. Model weights

Downloads go to `HF_HUB_CACHE`. Turn offline mode off for the download only. Cosmos models other than the base Nano are gated: accept the licence on each model page and `uvx hf@2.4.0 auth login` first.

```
source ~/.sim2real.env; export HF_HUB_OFFLINE=0
hfdl() { uvx hf@2.4.0 download "$@" -q | tail -1; }
hfdl nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2                         # 37 GB, used by gen/run_cosmos_nano.sh
mkdir -p $S2R_ROOT/VideoX-Fun/models/Diffusion_Transformer
ln -sfn "$(hfdl alibaba-pai/Wan2.2-VACE-Fun-A14B --revision 4438caabbfdd7437bbb0283d86cb200d1b7223ba)" \
  $S2R_ROOT/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B                         # 76 GB
```

These two are what Round 2 and the two-camera test used. They were already on disk on 2026-11-18, so the download lines above were not re-run then. The models of the first proof of concept (VACE-14B, its annotators, Wan-Move, Cosmos Sim-Transfer) are optional:

| Model | HF repo | Revision | Size | Link into the workspace |
| --- | --- | --- | --- | --- |
| VACE-14B | `Wan-AI/Wan2.1-VACE-14B` | `539c162b1387eac9dc4c20bd3f74671309e76a4c` | 59 GB | `VACE/models/Wan2.1-VACE-14B` |
| VACE annotators | `ali-vilab/VACE-Annotators` | `63512cbbe4e3a98e1636356ef2230da84649f617` | 1.3 GB (`depth/`, `scribble/` only) | `VACE/models/VACE-Annotators` |
| Wan-Move-14B | `Ruihang/Wan-Move-14B-480P` | `63a21fd3f4f1e6c6104e29f6b8d9b8ce3d635456` | 82 GB | `Wan-Move/Wan-Move-14B-480P` |
| Cosmos Sim-Transfer | `nvidia/Cosmos3-Nano-Sim-Transfer` | `main` (`45510d5b…` when the PoC ran) | | fetched by `smoke/cosmos_simtransfer_153.sh` |

Eval checkpoints go in `$S2R_ROOT/checkpoints/`:

```
mkdir -p $S2R_ROOT/checkpoints/{sam2,cotracker}
curl -L -o $S2R_ROOT/checkpoints/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
curl -L -o $S2R_ROOT/checkpoints/cotracker/scaled_offline.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

### 4b. Scene assets (multi-camera scenes)

```
$S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py   # CC0 Poly Haven textures and models, ~35 MB
```

The list is [`configs/assets/polyhaven.yaml`](../configs/assets/polyhaven.yaml). Files go to `$S2R_DATA/assets/polyhaven/<id>/`, each folder with a `SOURCE.md` (URLs, licence, sha256). Re-running keeps files whose md5 matches.

### 5. Check it

Three checks, each on its own free GPU (read `nvidia-smi` just before). These are the ones run on 2026-11-18; each passed.

```
source ~/.sim2real.env
SM=$S2R_DATA/smoke/setup/drones_air_phys

# simulator: render + export, then controls (about 5 min). The poses must match the Round 2 bundle exactly.
CUDA_VISIBLE_DEVICES=<gpu> bash sim/build_bundle.sh configs/scenes/drones_air_phys.yaml $SM/bundle
$S2R_ROOT/VACE/.venv/bin/python sim/controls.py $SM/bundle $SM --only vace cosmos C0

# Wan2.2-VACE-Fun at 2 steps (about 4.5 min, 40.8 GiB peak). Writes A/vacefun-a14b-s2/ next to the real clips; delete it after.
SCENE=drones_air_phys COND=A SEED=0 STEPS=2 CUDA_VISIBLE_DEVICES=<gpu> bash gen/run_vacefun.sh

# Cosmos3-Nano, depth only, full 50 steps (about 9 min). Seed 9 is unused, so no real clip is overwritten; delete C0/cosmos3-nano/seed9/ after.
SCENE=drones_air_phys COND=C0 SEED=9 CUDA_VISIBLE_DEVICES=<gpu> bash gen/run_cosmos_nano.sh
```

Each clip should be 832×480 with 81 frames: `ffprobe -v error -show_entries stream=width,height,nb_frames -of csv=p=0 <clip>`.

## Re-run the drones_air proof of concept

VACE-14B peaks at 74 GB, so it needs nearly a whole 80 GB card. Results and per-clip cost are in the [PoC findings doc](../docs/2026-11-10-1251-poc-drones-air-findings.md).

```
source env.sh; D=$S2R_DATA/poc/single-cam/drones_air; mkdir -p $D
bash sim/build_bundle.sh configs/scenes/drones_air.yaml $D/bundle           # render + bundle, ~5 min
$S2R_ROOT/VACE/.venv/bin/python sim/controls.py $D/bundle $D                  # control videos for every condition
for c in A Aplus B C C0 Bbox; do SCENE=drones_air COND=$c SEED=0 bash gen/run_vace.sh; done   # ~27 min each
SCENE=drones_air SEED=0 bash gen/run_cosmos.sh                               # condition D, ~3 min
SCENE=drones_air SEED=0 bash gen/run_wanmove.sh                              # condition E, ~31 min
$S2R_ROOT/envs/eval/.venv/bin/python eval/grid.py $D                          # grid.mp4 + grid_frames.png
```

Wrap any of these in `smoke/measure.py --out <dir>/measure.json --` to record wall time and peak memory.
