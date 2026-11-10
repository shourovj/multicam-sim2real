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

## Option A (recommended on swarm5): reuse the existing workspace

S P Sharan's workspace at `/home/ss99569/code/sim2real-videogen` is readable by every user on swarm5: third-party clones, Python envs, Blender, model weights and the PoC outputs. Use it in place and keep your own outputs and caches in your space. Add this to your `~/.bashrc`:

```
export S2R_ROOT=/home/ss99569/code/sim2real-videogen
export S2R_DATA=$HOME/sim2real-data                                    # your outputs
export S2R_CACHE=$HOME/.cache/sim2real                                 # your torch / matplotlib / uv caches
export HF_HUB_CACHE=/home/ss99569/code/sim2real-videogen/hf-hub        # shared model weights (read-only for you)
export HF_HUB_OFFLINE=1                                                # use the cached weights, never download
```

Then install BlenderProc for yourself (`uv tool install blenderproc==2.8.0`) and run the smoke tests in [`smoke/README.md`](../smoke/README.md) to check everything resolves.

- **Weights:** `$S2R_ROOT/hf-hub` holds every model the PoC uses (VACE-14B and 1.3B, the VACE annotators, Wan-Move-14B, Cosmos Sim-Transfer, its tokenizer and the Wan2.2 VAE). VACE and Wan-Move read them through links in their clones; the Cosmos script finds them through `HF_HUB_CACHE`. Cosmos models are gated, so accept the licence on their Hugging Face pages before using them.
- **Existing outputs:** read them from `$S2R_ROOT/data/poc/drones_air/` (grid video, every clip, the scene bundle, `notes.md`). Copy what you need into `$S2R_DATA`; you can't write there.
- **First GPU render** takes about 6 minutes, because the Blender CUDA kernel cache is per user (gotchas §3.1).
- **Shared envs break for you if the owner changes them.** If you need to change one, or work on another machine, build your own with Option B.

## Option B: build your own workspace

Tested piece by piece on swarm5 (Ubuntu, A100 80 GB, driver 580, no CUDA toolkit), but not yet run end to end from scratch. Needs [`uv`](https://docs.astral.sh/uv/) and `git`.

### 1. Third-party code

```
source env.sh
bash setup/clone.sh          # clones VACE, Wan-Move, co-tracker, sam2, cosmos-framework at the pinned commits
```

### 2. Python envs

| Env | Python | Spec in this repo | Used by |
| --- | --- | --- | --- |
| `$S2R_ROOT/envs/eval` | 3.11 | [`setup/envs/eval/`](envs/eval/) (`pyproject.toml` + `uv.lock`) | `sim/export.py`, `eval/grid.py`, SAM 2 + CoTracker3 |
| `$S2R_ROOT/VACE/.venv` | 3.10 (system) | [`setup/envs/vace-requirements.txt`](envs/vace-requirements.txt) | VACE runs, `sim/controls.py`, prompt loading |
| `$S2R_ROOT/Wan-Move/.venv` | 3.10 (system) | [`setup/envs/wanmove-requirements.txt`](envs/wanmove-requirements.txt) | Wan-Move runs |
| `$S2R_ROOT/cosmos-framework/.venv` | 3.13 | the clone's own `uv.lock` | Cosmos Sim-Transfer runs |

```
# eval: SAM 2 builds against the env's own torch, so sync in two steps (gotchas §3.5)
mkdir -p $S2R_ROOT/envs/eval && cp setup/envs/eval/{pyproject.toml,uv.lock} $S2R_ROOT/envs/eval/
(cd $S2R_ROOT/envs/eval && uv sync --no-install-package sam-2 && uv sync)

# VACE (torch 2.6 / cu124, flash-attn 2.7.4 wheel) and Wan-Move (torch 2.7.1 / cu126, flash-attn 2.8.3 wheel)
(cd $S2R_ROOT/VACE && uv venv --python /usr/bin/python3.10 && \
  uv pip install -r $OLDPWD/setup/envs/vace-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match)
(cd $S2R_ROOT/Wan-Move && uv venv --python /usr/bin/python3.10 && \
  uv pip install -r $OLDPWD/setup/envs/wanmove-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)

# Cosmos
(cd $S2R_ROOT/cosmos-framework && uv sync)
```

The two requirements files are `uv pip freeze` output from the working envs. flash-attn is required by both Wan models; there is no SDPA fallback (gotchas §3.3, §3.4).

### 3. Blender and BlenderProc

```
uv tool install blenderproc==2.8.0
mkdir -p $S2R_ROOT/blender && curl -L https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar -xJ -C $S2R_ROOT/blender
blenderproc pip install --blender-install-path $S2R_ROOT/blender pyyaml
```

Always pass `--blender-install-path $S2R_ROOT/blender`, or BlenderProc downloads another Blender.

### 4. Model weights

Hugging Face downloads go to your own cache (`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`). Cosmos models are gated: accept the licence on each model page and `hf auth login` first.

| Model | HF repo | Revision | Size | Link into the workspace |
| --- | --- | --- | --- | --- |
| VACE-14B | `Wan-AI/Wan2.1-VACE-14B` | `539c162b1387eac9dc4c20bd3f74671309e76a4c` | 59 GB | `VACE/models/Wan2.1-VACE-14B` |
| VACE-1.3B (optional) | `Wan-AI/Wan2.1-VACE-1.3B` | `574e6a744642ce3bee319afc31496b88bde8aac4` | 19 GB | `VACE/models/Wan2.1-VACE-1.3B` |
| VACE annotators | `ali-vilab/VACE-Annotators` | `63512cbbe4e3a98e1636356ef2230da84649f617` | 1.3 GB (`depth/`, `scribble/` only) | `VACE/models/VACE-Annotators` |
| Wan-Move-14B | `Ruihang/Wan-Move-14B-480P` | `63a21fd3f4f1e6c6104e29f6b8d9b8ce3d635456` | 82 GB | `Wan-Move/Wan-Move-14B-480P` |
| Cosmos Sim-Transfer | `nvidia/Cosmos3-Nano-Sim-Transfer` | `main` (`45510d5b…` when the PoC ran) | | fetched by the run script |
| Cosmos text tokenizer | `nvidia/Cosmos3-Nano` | `e59a53c25979a090fa8706c9acc0c254a6e89b92` (`text_tokenizer/*`) | | fetched by the run script |
| Wan2.2 VAE | `Wan-AI/Wan2.2-TI2V-5B` | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` (`Wan2.2_VAE.pth`) | | fetched by the run script |

```
hfdl() { uvx hf@2.4.0 download "$@" -q | tail -1; }
mkdir -p $S2R_ROOT/VACE/models
ln -sfn "$(hfdl Wan-AI/Wan2.1-VACE-14B --revision 539c162b1387eac9dc4c20bd3f74671309e76a4c)" $S2R_ROOT/VACE/models/Wan2.1-VACE-14B
ln -sfn "$(hfdl ali-vilab/VACE-Annotators --revision 63512cbbe4e3a98e1636356ef2230da84649f617 --include 'depth/*' 'scribble/*')" $S2R_ROOT/VACE/models/VACE-Annotators
ln -sfn "$(hfdl Ruihang/Wan-Move-14B-480P --revision 63a21fd3f4f1e6c6104e29f6b8d9b8ce3d635456)" $S2R_ROOT/Wan-Move/Wan-Move-14B-480P
```

Eval checkpoints go in `$S2R_ROOT/checkpoints/`:

```
mkdir -p $S2R_ROOT/checkpoints/{sam2,cotracker}
curl -L -o $S2R_ROOT/checkpoints/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
curl -L -o $S2R_ROOT/checkpoints/cotracker/scaled_offline.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

### 5. Check it

Run the smoke tests in [`smoke/README.md`](../smoke/README.md). Each one writes a `measure.json` with wall time and peak GPU memory to compare against the gotchas doc's table.

## Re-run the drones_air proof of concept

VACE-14B peaks at 74 GB, so it needs nearly a whole 80 GB card. Results and per-clip cost are in the [PoC findings doc](../docs/2026-11-10-1251-poc-drones-air-findings.md).

```
source env.sh; D=$S2R_DATA/poc/drones_air; mkdir -p $D
bash sim/build_bundle.sh configs/scenes/drones_air.yaml $D/bundle           # render + bundle, ~5 min
$S2R_ROOT/VACE/.venv/bin/python sim/controls.py $D/bundle $D                  # control videos for every condition
for c in A Aplus B C C0 Bbox; do SCENE=drones_air COND=$c SEED=0 bash gen/run_vace.sh; done   # ~27 min each
SCENE=drones_air SEED=0 bash gen/run_cosmos.sh                               # condition D, ~3 min
SCENE=drones_air SEED=0 bash gen/run_wanmove.sh                              # condition E, ~31 min
$S2R_ROOT/envs/eval/.venv/bin/python eval/grid.py $D                          # grid.mp4 + grid_frames.png
```

Wrap any of these in `smoke/measure.py --out <dir>/measure.json --` to record wall time and peak memory.
