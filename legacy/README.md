# legacy: the earlier route

Before the prompt-file pipeline in [`p2s/`](../p2s/), a scene was a hand-written **scene spec** (spec v2, YAML) moved by a
small numpy simulator. That route, its experiments and the single-camera scripts of the first rounds live here. They
still run, and their results are the ones in the [evaluation report](../docs/2026-11-24-1158-eval-report.md).

What is shared with `p2s/`: generation and scoring. An experiment file here is run with the same
`p2s/3_generate/run_multicam.py` and `p2s/4_score/run_eval.py`; it names its scenes with `spec:` where a prompt-file
scene has `p2s:`. The bundle exporter and the control videos are shared too (`p2s/2_render/bundle.py`,
`control_videos.py`). `legacy/` keeps the layout the old scripts expect (`configs/`, `sim/`, `gen/`, `eval/`, `smoke/`).

| Folder | What it holds |
| --- | --- |
| [`configs/scenes/`](configs/scenes/) | Scene specs (`<scene>.yaml`) and the prompt each was written from (`<scene>.source.md`). `history/` keeps old revisions byte for byte. |
| [`configs/prompts/`](configs/prompts/) | Caption blocks per experiment and scene, and the first rounds' prompt files. |
| [`configs/experiments/`](configs/experiments/) | `multicam_v1*.yaml` (the control comparison) and [`cvp_v1/`](configs/experiments/cvp_v1/) (the four-model benchmark on the spec v2 scenes). |
| [`sim/`](sim/) | `simulate_world.py`, `render_world.py`, `export_world.py`, `build_world.sh`; [`sim/world/`](sim/world/) (spec loading, layouts, kinematics, cameras, checks, Blender mesh builders); [`sim/physics/`](sim/physics/) (motion models). |
| [`gen/`](gen/) | `camera_facts.py`, and the single-camera scripts: `run_matrix.py`, `wan21_vace/`, `wan_move/`, the `singlecam_*` scripts. |
| [`eval/`](eval/) | `score_multicam.py` (the earlier scorer: colour names, traffic-light state), `grid.py`. |
| [`smoke/`](smoke/), `frame0_edits/` | Tool smoke tests and the edited first frames of round 2. |

Paths in the commands below are from the repository root.

## The route

```
text prompt                                   legacy/configs/scenes/<scene>.source.md
   │  [1] LLM writes the scene spec           (by hand-in-the-loop for now, recorded in the .source.md)
   ▼
scene spec                                    legacy/configs/scenes/<scene>.yaml  (spec v2: world, signals, objects, cameras, checks)
   │  [2] simulate once, run the checks        legacy/sim/simulate_world.py  (numpy only, no GPU)
   ▼
world/                                        world.npz (every part's pose per frame), world.json (signals, events, check results)
   │  [3] render each camera in Blender        legacy/sim/render_world.py
   │  [4] export a bundle per camera           legacy/sim/export_world.py
   ▼
cams/<cam>/bundle/                            rgb, depth, instance masks, boxes, tracks, visibility, checks
cams/<cam>/bundle/controls/                   depth.mp4, seg.mp4, edge.mp4, normals.mp4, rgb.mp4 (clay), controls_sheet.jpg
   │  [5] compose the prompt                   p2s/3_generate/compose_prompts.py  (world block + camera block → composed/<view>.json or .txt)
   │  [6] generate on free GPUs                p2s/3_generate/run_multicam.py  (one experiment file per model; gen/<model>/run.sh does one job)
   ▼
gen/<view>/<cond>/<model>/seed<k>/            out_video.mp4, the prompt, request.json, run.json (hashes of every input)
   │  [7] score against the simulator          p2s/4_score/run_eval.py  (three scores per camera, two across cameras)
   ▼
scores/<experiment>/                          eval2.csv, eval3.csv, eval_summary.json
```

Steps 1 and 5 are the LLM jobs. Everything else is deterministic code. Each step reads and writes files, so any step can be re-run alone.

`legacy/sim/build_world.sh` runs steps 2 to 4 for every camera of a scene.


## Run the route on one scene

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
EXP=legacy/configs/experiments/multicam_v1_1.yaml
```

**Step 1. Simulate and check (no GPU, seconds).** This writes `world/world.npz` and `world/world.json`. It exits with an error if any check in the spec's `checks:` list fails.

```bash
$PY legacy/sim/simulate_world.py legacy/configs/scenes/$S.yaml $OUT/world
```

**Step 2. Preview one camera (three frames, low samples).** Look at the framing before a full render.

```bash
CUDA_VISIBLE_DEVICES=$GPU blenderproc run --blender-install-path $S2R_ROOT/blender legacy/sim/render_world.py legacy/configs/scenes/$S.yaml \
    $OUT/world cctv $S2R_DATA/preview/$S --frames 0,56,112 --samples 16
```

**Step 3. Build every camera.** This simulates again (same result), renders each camera, and exports its bundle and control videos. The third argument lists GPUs; cameras render in parallel, one per GPU. One GPU works too, one camera after another.

```bash
bash legacy/sim/build_world.sh legacy/configs/scenes/$S.yaml $OUT 0,2,7        # GPUs that nvidia-smi shows as free
```

Look at `$OUT/cams/<cam>/bundle/controls/controls_sheet.jpg` and `overlay.mp4`. The `visible check` lines at the end say whether each camera sees the objects the spec requires.

**Step 4. Check the prompts.** The runner refuses to start a job whose caption file is missing or out of date. `--check` only reports. If you edited prompt blocks, run it without `--check` to rewrite the captions, and commit them.

```bash
$PY p2s/3_generate/compose_prompts.py --exp $EXP --check
```

**Step 5. Generate.** Look at what would run, try a 4-step smoke test (it writes to `gen_smoke/`, never over real runs), then run the experiment. The runner picks GPUs with enough free memory and re-checks before each launch.

```bash
$PY p2s/3_generate/run_multicam.py --exp $EXP --status
$PY p2s/3_generate/run_multicam.py --exp legacy/configs/experiments/multicam_v1_1_smoke.yaml --only scene=$S
$PY p2s/3_generate/run_multicam.py --exp $EXP --only scene=$S
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

**Step 6. Score.** Three scores per camera and two across cameras ([Sim → video](../README.md#sim--video)). It needs a GPU with a few GB free.

```bash
$PY p2s/4_score/run_eval.py --exp $EXP --only scene=$S
```

**Your own experiments.** Copy an experiment file under a new name and change `experiment:` (the name in run IDs and the scores folder) and, to keep your runs apart from earlier ones, `out_root:`. Don't edit the existing files: each is the record of runs that already exist.


## Experiments

| File | Model | What the model gets |
| --- | --- | --- |
| [`multicam_v1.yaml`](configs/experiments/multicam_v1.yaml) | Cosmos3-Nano | The textured render as a blur control (V). The model copied the render; kept as a baseline. Uses the rev-1 specs in `legacy/configs/scenes/history/`. |
| [`multicam_v1_1.yaml`](configs/experiments/multicam_v1_1.yaml) | Cosmos3-Nano, 50 steps | Clay render; depth + segmentation (G). Appearance only from the prompt. |
| [`multicam_v1_2.yaml`](configs/experiments/multicam_v1_2.yaml) | Cosmos3-Nano-Sim-Transfer, 4 steps | One control per clip: depth (D), segmentation (S) or edges (E). |
| [`multicam_v1_3.yaml`](configs/experiments/multicam_v1_3.yaml) | Cosmos3-Nano text-to-video | Text only, no control. A baseline for how much the controls add. |

Each experiment covers two scenes, `intersection_4way` (cars, pedestrians, traffic lights; CCTV, dashcam, phone) and `robot_pour` (a Panda arm pours water; side, wrist, CCTV), plus `cut3`, one video that cuts between the three cameras. The design and the findings are in the [multi-camera v1 design doc](../docs/2026-11-18-1739-multicam-v1-design.md), §14 for the clay revision.

### The four-model benchmark (cvp_v1)

The same clips on four models. The files in [`legacy/configs/experiments/cvp_v1/`](configs/experiments/cvp_v1/) differ only in the model:

| File | Model | Steps | Prompt format |
| --- | --- | --- | --- |
| [`cosmos3_nano.yaml`](configs/experiments/cvp_v1/cosmos3_nano.yaml) | Cosmos3-Nano | 50 | Structured caption |
| [`cosmos3_sim_transfer.yaml`](configs/experiments/cvp_v1/cosmos3_sim_transfer.yaml) | Cosmos3-Nano-Sim-Transfer | 4, distilled | Structured caption |
| [`wan22_vace_fun.yaml`](configs/experiments/cvp_v1/wan22_vace_fun.yaml) | Wan2.2-VACE-Fun-A14B | 50 | One paragraph |
| [`ltx25.yaml`](configs/experiments/cvp_v1/ltx25.yaml) | LTX-2.5, 22B distilled | 8, distilled | One paragraph |

What is held equal:

- **Scenes, cameras, cut, seeds.** Both scenes at rev 2 (clay), three cameras plus `cut3`, seeds 0 and 1: 16 clips per model, 64 in all.
- **Clip.** 113 frames at 16 fps, 832×480.
- **Control.** One condition, `D`: the simulator's depth video, the same file for every model, and no first frame. Depth is the one control all four take. Sim-Transfer takes a single control, and Wan and LTX have no segmentation input.
- **Prompt content.** Each view's prompt is composed from the same world and camera blocks, as a structured caption for the Cosmos models and as one paragraph for Wan and LTX (`composed/<view>.json` and `.txt`).
- **Output root.** All four write under `$S2R_DATA/cv-project/multi-cam/`, so they share one simulated world and one set of bundles, and a view's four results sit side by side in `gen/<view>/D/<model name>/`.

What is not equal, and belongs in any comparison:

- **Step count.** Sim-Transfer and LTX-2.5 are distilled few-step models with no guidance; the other two run 50 steps with guidance. Each model uses its own recommended settings.
- **Clip length for Wan.** Wan2.2-VACE-Fun was trained on 81-frame clips; 113 frames is longer than that.
- **Size for LTX.** LTX-2.5 generates at 832×512 and the result is resized to 832×480 (its control adapter needs a height that is a multiple of 64). It also generates audio, which is dropped.
- **Depth convention.** The depth video is inverse depth with one scale per clip, the Cosmos convention. VACE's own depth annotator scales each frame separately.

Run it, one model at a time or several runners at once:

```bash
for M in cosmos3_sim_transfer ltx25 cosmos3_nano wan22_vace_fun; do
  $PY p2s/3_generate/compose_prompts.py --exp legacy/configs/experiments/cvp_v1/$M.yaml --check
  $PY p2s/3_generate/run_multicam.py --exp legacy/configs/experiments/cvp_v1/$M.yaml
  $PY p2s/4_score/run_eval.py --exp legacy/configs/experiments/cvp_v1/$M.yaml
done
```

`smoke_cosmos3_nano.yaml` and `smoke_wan22_vace_fun.yaml` are 4-step smoke tests that write to `gen_smoke/`. The two distilled models are quick enough to try with `--only scene=robot_pour,view=side,seed=0` on the real file.

Cost per clip on an A100 80 GB, at the benchmark's clip (113 frames, depth control):

| Model | Time | Peak GPU memory | Measured? |
| --- | --- | --- | --- |
| Cosmos3-Nano, 50 steps | About 10 minutes | 37.4 GiB | Estimate: a 4-step run took 150 s, about 9.5 s per step (2026-11-23) |
| Cosmos3-Nano-Sim-Transfer, 4 steps | About 2 minutes | 39.0 GiB | Yes, in `multicam_v1.2` |
| Wan2.2-VACE-Fun-A14B, 50 steps | About 33 minutes | 58.5 GiB | Estimate: a 4-step run took 331 s, about 36 s per step (2026-11-23) |
| LTX-2.5, 8 steps | Not known | Not known | Not run yet: the weights are not on the server |

**Status on 2026-11-23.** The bundles for both scenes are built under `cv-project/multi-cam`. Cosmos3-Nano and Wan2.2-VACE-Fun each passed a 4-step smoke test through the runner. The Sim-Transfer request was checked against `multicam_v1.2`'s and is identical. LTX-2.5 has its code, environment and experiment file, and its launcher's imports were checked in that environment, but it has not generated a clip: the server is not logged in to Hugging Face, so the gated weights are not downloaded (setup step B8).

### Text-only runs (multicam_v1.3)

With no control, the prompt has to say where each object is in each camera's frame and when. Those claims come from the simulator:

```bash
$PY legacy/gen/camera_facts.py --exp legacy/configs/experiments/multicam_v1_3.yaml     # needs the bundles from step 3
```

This writes `legacy/configs/prompts/multicam_v1_3/<scene>/facts/<camera>.{json,txt}`. The camera blocks in `cams/<camera>.json` are written from those facts. Then run steps 4 to 6 with `multicam_v1_3.yaml`.


## Checks on the simulated world


| Check | Where | Status |
| --- | --- | --- |
| Rules in the spec: minimum gap between objects, cars obey signals and yield to pedestrians, events in order, robot joint limits | `checks:` in the spec, [`legacy/sim/world/checks.py`](sim/world/checks.py); results in `world/world.json` | Runs on every simulation; a failed check stops it |
| Each camera sees the objects it should | `visible` checks, evaluated after rendering into `cams/<cam>/bundle/checks.json` | Runs on every build |
| What each camera sees, written as text (position in frame, size, direction, on-screen times) | [`legacy/gen/camera_facts.py`](gen/camera_facts.py) → `facts/<camera>.txt` | Runs on demand |
| Provenance: the frozen prompt, who wrote the spec, every revision | `legacy/configs/scenes/<scene>.source.md` | Kept by hand |
| Does the spec do what the prompt says (each claim in the prompt checked against `world.json` and the facts) | | **Not built yet** |


## Add a scene

1. Write the user prompt and the spec's provenance in `legacy/configs/scenes/<name>.source.md`.
2. Write the spec, `legacy/configs/scenes/<name>.yaml` (`spec_version: 2`). Copy `intersection_4way.yaml` or `robot_pour.yaml` and keep the conventions in [`legacy/sim/world/__init__.py`](sim/world/__init__.py): Z up, metres, +X east, +Y north.
3. Iterate on step 1 of [Run the route on one scene](#run-the-route-on-one-scene) until every check passes, then on step 2 until the framing is right.
4. Write the prompt blocks in `configs/prompts/<experiment>/<name>/`: `world.json`, `cams/<cam>.json`, and a cut file if you want one.
5. Add the scene to an experiment file, then run steps 3 to 6.

A new kind of object needs a motion model in `legacy/sim/physics/`, a kinematics function in `legacy/sim/world/kinematics.py` and a mesh builder in `legacy/sim/world/blender/`. A new camera kind is one function in `legacy/sim/world/cameras.py`.


## Where outputs go

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
    gen/.../seed<k>/est/                reader outputs for scoring: depth, masks, point tracks
    scores/<experiment>/                eval2.csv, eval3.csv, eval_summary.json (and colour.csv, lights.csv)
```

`run.json` records the git commit, whether the tree was dirty, hashes of the spec, prompt, experiment file and controls, the model revision, the GPU and the peak memory.

