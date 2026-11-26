# gen/: from simulator bundles to generated video

One folder per model, plus the code they share. The three commands you run are at the top level.

```
gen/
  run_multicam.py          run every job of an experiment file on free GPUs
  compose_prompts.py       write the exact prompt each view gets into configs/prompts/.../composed/
  camera_facts.py          what each camera sees, as text (for text-only prompts)
  run_matrix.py            the single-camera rounds 1 and 2 (configs/matrix*.yaml)
  common/                  shared by every model
    multicam_common.py       job list, folder layout, model families, prompt composing, input hashes
    multicam_inputs.py       a run's inputs: cut videos, prompt, negative prompt, request.json, inputs.json
    gpus.py                  free memory and our own jobs per GPU
    world_git.py             git commit and dirty flag for run records
  cosmos3_nano/            Cosmos3-Nano (base)
  cosmos3_sim_transfer/    Cosmos3-Nano-Sim-Transfer
  wan22_vace_fun/          Wan2.2-VACE-Fun-A14B (VideoX-Fun)
  ltx25/                   LTX-2.5, distilled
  wan21_vace/              Wan2.1-VACE (single-camera rounds only)
  wan_move/                Wan-Move (single-camera rounds only)
```

In a model's folder, `run.sh` does one multi-camera job, and anything named `singlecam_*` belongs to the single-camera rounds.

## The four benchmark models

| Folder | Model | `model.family` | Python env | Prompt it takes | Controls it takes | Experiment file |
| --- | --- | --- | --- | --- | --- | --- |
| [`cosmos3_nano/`](models/cosmos3_nano/) | Cosmos3-Nano | `cosmos3-nano` (the default) | `cosmos-framework/.venv` | Structured JSON caption | Depth, segmentation, edges, blur; several at once | [`cvp_v1/cosmos3_nano.yaml`](../../legacy/configs/experiments/cvp_v1/cosmos3_nano.yaml) |
| [`cosmos3_sim_transfer/`](models/cosmos3_sim_transfer/) | Cosmos3-Nano-Sim-Transfer | `sim-transfer` | `cosmos-framework/.venv` | Structured JSON caption | Exactly one: depth, segmentation or edges | [`cvp_v1/cosmos3_sim_transfer.yaml`](../../legacy/configs/experiments/cvp_v1/cosmos3_sim_transfer.yaml) |
| [`wan22_vace_fun/`](models/wan22_vace_fun/) | Wan2.2-VACE-Fun-A14B | `wan22-vace-fun` | `VideoX-Fun/.venv` | One paragraph | At most one: depth or edges | [`cvp_v1/wan22_vace_fun.yaml`](../../legacy/configs/experiments/cvp_v1/wan22_vace_fun.yaml) |
| [`ltx25/`](models/ltx25/) | LTX-2.5 (22B, distilled) | `ltx25` | `LTX-2/.venv` | One paragraph | At most one: depth or edges | [`cvp_v1/ltx25.yaml`](../../legacy/configs/experiments/cvp_v1/ltx25.yaml) |

Both prompt formats are composed from the same blocks in `configs/prompts/<set>/<scene>/` (`multicam_common.compose_prompt` and `prose_prompt`), so every model is told the same things about a view. The two Cosmos models share `cosmos3_nano/request.py`: they run on the same framework and take the same request format.

## What happens in one job

1. `run_multicam.py` reads the experiment file, lists its jobs (scene × view × condition × seed) and picks a GPU with enough free memory.
2. It starts `gen/<model folder>/run.sh` with `EXP`, `SCENE`, `VIEW`, `COND`, `SEED` and `CUDA_VISIBLE_DEVICES` set. The folder comes from the experiment's `model.family` (`multicam_common.FAMILIES`).
3. `run.sh` builds the inputs: `common/multicam_inputs.py` for Wan and LTX, `cosmos3_nano/request.py` for the Cosmos models. Both write the prompt, `request.json` and `inputs.json` into the run folder.
4. `run.sh` runs the model in its own Python env and leaves `out_video.mp4` and `run.log` in the run folder.
5. `run_multicam.py` writes `run.json` (status, GPU, wall time, peak memory, input hashes, git state).

To run one job by hand, without the runner:

```bash
source ~/.sim2real.env
EXP=legacy/configs/experiments/cvp_v1/wan22_vace_fun.yaml SCENE=robot_pour VIEW=side COND=D SEED=0 CUDA_VISIBLE_DEVICES=2 bash p2s/3_generate/models/wan22_vace_fun/run.sh
```

## Add a model

1. Make `gen/<model>/run.sh`. It gets `EXP`, `SCENE`, `VIEW`, `COND`, `SEED` and the GPU from its caller, and must leave `out_video.mp4` (the experiment's frame count, rate and size) and `run.log` in the run folder.
2. If the model takes a paragraph and at most one control video, call `common/multicam_inputs.py` and read its `request.json` in a `launch.py`, as `wan22_vace_fun/` and `ltx25/` do. Otherwise write a `request.py` on the functions of `common/multicam_inputs.py`, as `cosmos3_nano/` does.
3. Add a line to `FAMILIES` in `common/multicam_common.py`: the folder and the prompt format (`caption` or `prose`).
4. Write an experiment file with `model.family` set, run `compose_prompts.py` on it, and commit the composed prompts.

## Where the scripts were before 2026-11-23

Dated docs in `docs/` and the comments in older experiment files use the old names.

| Old path | New path |
| --- | --- |
| `gen/multicam_common.py`, `gen/world_git.py` | `p2s/3_generate/common/` |
| `gen/multicam_inputs.py` | `p2s/3_generate/common/multicam_inputs.py` (shared part) and `p2s/3_generate/models/cosmos3_nano/request.py` (Cosmos request) |
| `gen/run_cosmos_multicam.sh` | `p2s/3_generate/models/cosmos3_nano/run.sh` |
| `gen/run_cosmos_simtransfer_multicam.sh` | `p2s/3_generate/models/cosmos3_sim_transfer/run.sh` |
| `gen/run_cosmos_nano.sh`, `gen/cosmos_nano_request.py` | `legacy/gen/cosmos3_nano/singlecam_run.sh`, `singlecam_request.py` |
| `gen/run_cosmos.sh` | `legacy/gen/cosmos3_sim_transfer/singlecam_run.sh` |
| `gen/run_vacefun.sh`, `gen/vacefun_launch.py` | `legacy/gen/wan22_vace_fun/singlecam_run.sh`, `launch.py` |
| `gen/run_vace.sh`, `gen/run_vace_diffusers.sh`, `gen/vace_diffusers_launch.py` | `legacy/gen/wan21_vace/singlecam_run.sh`, `singlecam_run_diffusers.sh`, `diffusers_launch.py` |
| `gen/run_wanmove.sh` | `legacy/gen/wan_move/singlecam_run.sh` |
