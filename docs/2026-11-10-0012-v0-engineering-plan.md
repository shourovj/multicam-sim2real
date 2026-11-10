# v0 Engineering Plan: Simulator → Video Models → Eval

**Owner:** SP Sharan · **Drafted:** Nov 10, 2026 · **Status:** build plan for the week-1 experiment. The experimental design and the reasons behind it are in the [project understanding doc](2026-11-09-1626-project-understanding.md) (§7). This doc covers only what we build and run, in order.

## 0. What the pipeline does

```
scene spec (YAML) ──► Blender/BlenderProc render ──► scene bundle
                                                      (RGB, depth, masks, tracks, timeline)   
                                                              │
                         ┌──────────────┬───────────────┬─────┴────────┬──────────────┐
                         ▼              ▼               ▼              ▼              ▼
                    VACE A / A+      VACE B          VACE C       Cosmos Sim-T D   Wan-Move E
                    (text)       (text+frame 0)  (frame 0+control)  (control+text)  (frame 0+tracks)
                         └──────────────┴───────┬───────┴──────────────┴──────────────┘
                                                ▼
                               outputs/<scene>/<cond>/seed<k>.mp4 + run.json
                                                ▼
                     eval: resample → track (SAM 2 / CoTracker) → human audit → metrics
                                                ▼
                              score tables, claimed vs. observed, video grid
```

3 scenes × 6 generated conditions (A, A+, B, C, D, E) × 3 seeds = **54 clips**, plus 3 raw renders (R). The minimum run is 3 scenes × {A, B, C} × 3 seeds = 27 clips.

## 1. Layout

**Code goes in this repo:**

```
configs/
  scenes/{drones_takeoff,drones_air,convoy_overhead}.yaml   # scene specs
  prompts/<scene>.yaml                                      # frozen prompts: base, detailed (A+), negative
  matrix.yaml                                               # scenes × conditions × seeds, model settings
sim/
  render_scene.py        # BlenderProc entry point: spec → scene bundle
  assets.py              # drone / vehicle / ground / background loading
  motion.py              # trajectories: take-off spline, crossing paths, kinematic bicycle model
  export.py              # tracks, visibility, masks, timeline → files
  controls.py            # per-model control videos (VACE / Cosmos formats) from the bundle
gen/
  common.py              # resampling, video IO, run.json logging (time, peak mem, cmd, git hash)
  run_vace.py            # conditions A, A+, B, C
  run_cosmos.py          # condition D (wraps cosmos-framework)
  run_wanmove.py         # condition E
  launch.py              # expand matrix.yaml into jobs, skip finished outputs
eval/
  resample.py            # all outputs → common fps / length / resolution for scoring
  track.py               # SAM 2 masks + CoTracker points on generated clips
  annotate/              # keyframe export + import of corrected labels
  metrics.py             # count, events, position error, heading, ID switches, stability
  specs.py               # NeuS-V-style temporal specs built from tracks
  realism_pairs.py       # blinded side-by-side pairs vs. the raw render
  report.py              # score tables, claimed vs. observed, video grid
```

**Data and third-party code stay out of the repo,** in `~/code/sim2real-videogen/` on swarm5 next to the existing `cosmos-framework/`:

```
~/code/sim2real-videogen/
  cosmos-framework/   (already set up)
  VACE/  Wan-Move/  sam2/  co-tracker/
  data/scenes/<scene>/          # scene bundles
  data/outputs/<scene>/<cond>/  # generated clips
  data/eval/                    # tracks, annotations, metrics
```

The swarm5 root disk is 97% full (about 500 GB left). Checkpoints already take about 65 GB for Cosmos; VACE-14B and Wan-Move-14B add roughly another 150 GB. Check `df -h` before each download, and point `HF_HOME` at a disk with room if needed.

## 2. Step 1: environments

One uv environment per tool, since their torch and CUDA pins conflict.

1. **Blender + BlenderProc.** `uv tool install blenderproc` (or a uv project in `sim/`). BlenderProc downloads its own Blender on first run. Render with Cycles on CUDA (`bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"])`), headless. EEVEE needs EGL, so don't use it unless EGL turns out to be present. Done when a test cube renders RGB, depth and segmentation on a swarm5 GPU.
2. **VACE.** Clone `ali-vilab/VACE`, set up its environment with uv, download `Wan2.1-VACE-1.3B` first for interface debugging, then `Wan2.1-VACE-14B`, plus the `VACE-Annotators` weights (depth, scribble). Done when one clip each for text-only, first-frame, and first-frame + depth control (the extension + control composition, §6) comes out of 14B. Record wall time and peak memory. If 14B doesn't fit or takes over ~30 min per clip, use 1.3B for all of A–C.
3. **Wan-Move.** Clone `ali-vilab/Wan-Move`, set it up with uv, run its own example tracks. Done when the example reproduces. Then read its loader to pin down the track file format (array shape, coordinate convention, visibility encoding, number of points). Budget half a day; this is the least documented piece.
4. **Cosmos3-Nano-Sim-Transfer.** Already working in `cosmos-framework/`, using `cosmos-logs/run-simtransfer-depth.sh` (see the [Cosmos hardware check](2026-11-09-2309-cosmos-hardware-check.md)). Nothing to install. Run with `--no-guardrails` until Guardrail access is approved. One new check: the tested profile (`example2`, full history, no attention sink) only ran 121 frames. Run it once on a 153-frame control (set `num_frames` / `max_frames` to 153) and record memory and time. If it fails, fall back to 121 frames and make every condition 4 s (see §4).
5. **Eval tools.** SAM 2 (`facebookresearch/sam2`) and CoTracker3 (`facebookresearch/co-tracker`) in one uv env.

## 3. Step 2: scene specs

Each scene is one YAML file, so a scene can be re-rendered exactly and the same spec drives every model.

```yaml
name: drones_air
duration_s: 5.1                 # 153 frames at 30 fps; frame counts must be 1 + 4n (§4)
fps: 30
resolution: [832, 480]          # Wan 480p native; matches Cosmos Sim-Transfer output
seed: 0
camera: {type: perspective, position: [...], look_at: [...], fov_deg: 50, motion: static}
background: {type: hdri | ground_plane | image_tile, path: ...}
objects:
  - id: drone_1
    asset: assets/quadcopter.glb
    scale_m: 0.5
    trajectory: {type: spline, keyframes: [[t, x, y, z, yaw], ...]}
  - id: drone_2
    ...
events:                         # ground-truth event times, used by eval
  - {name: drone_1_liftoff, t: 0.8}
  - {name: crossing, t: 2.6}
```

**The three scenes:**

| Scene | Camera | Objects and motion | Notes |
|---|---|---|---|
| `drones_takeoff` | Static, ground level, a few metres back | 2–3 quadcopters at rest on the ground → vertical lift → translate off to one side | Events: lift-off per drone, leave-ground, start-translate. Drones ≥ 32 px across |
| `drones_air` | Static, slight upward angle, sky + horizon | 2–3 quadcopters crossing or converging | Event: crossing time. Keep every drone in frame for all 5 s (Wan-Move copies frame-0 features, and re-entry is out of scope) |
| `convoy_overhead` | Near-orthographic top-down, static | 4–6 vehicles on a road; a right turn onto a field. Kinematic bicycle model, fixed speed | Background: an image tile on a ground plane (NAIP-style), road and field drawn on it. Vehicles ≥ 40 px long. Events: each vehicle's turn-start and on-field time |

Motion is scripted (kinematic). gym-pybullet-drones for take-off is optional and comes only after everything else works.

## 4. Step 3: render the scene bundle

`sim/render_scene.py configs/scenes/<scene>.yaml` writes `data/scenes/<scene>/`:

| File | Contents |
|---|---|
| `rgb.mp4`, `rgb/%04d.png` | Raw render, 153 frames at 30 fps (5.1 s). This is condition R |
| `frame0.png` | First RGB frame, the image input for B, C, E |
| `depth.npy` | Metric depth, float32 `[T, H, W]` |
| `instance.npy` | Per-pixel object ID, uint8 `[T, H, W]`, 0 = background |
| `tracks.npz` | `points [N, T, 2]` (pixels), `visible [N, T]` (bool), `object_id [N]`. N surface points per object (start with ~16 sampled on the mesh), projected through the camera with occlusion from the depth buffer |
| `boxes.json` | Per-frame 2D box per object, derived from `instance.npy` |
| `timeline.json` | Per-frame object 3D pose, camera intrinsics/extrinsics, and the `events` list |
| `meta.json` | Spec hash, Blender version, render time |

Check: overlay `tracks.npz` and `boxes.json` on `rgb.mp4` and look at it. Every later metric depends on these being right.

**Why 153 frames.** Both Wan (VACE, Wan-Move) and Cosmos encode video with 4× temporal compression and need `F = 1 + 4n` frames. 153 frames at 30 fps is 5.1 s and is a valid count, so Cosmos can take the whole bundle at its native 30 fps without retiming. Wan's 81 frames at 16 fps cover 5.06 s, so it is resampled from the same 5.1 s window. Squeezing 150 frames into 121 and playing them at 30 fps would speed the motion up by about 1.24× and shift every event time.

## 5. Step 4: control videos

**Both models take exactly one control type per sample, and each expects it in the format it was trained on.** Inventing our own control (raw Canny, our own depth scaling, an arbitrary blur) hands the model an input it has never seen. So controls are built per model, not once for both.

What each model accepts (checked in the VACE repo at `48eb44f` and in `cosmos-framework/.agents/skills/cosmos3-nano-sim-transfer/shared/model-contract.md`):

| Model | Control types | Training format |
|---|---|---|
| VACE (C) | depth, scribble, gray, flow, pose, layout (box / track). **No edge type** | **Depth:** MiDaS / Depth Anything V2 output, i.e. relative *inverse* depth, min–max scaled **per frame**, near = bright, 3-channel. **Scribble:** VACE's learned line-drawing annotator, not Canny. Two controls can only be combined region by region with a mask, not overlaid |
| Cosmos Sim-Transfer (D) | edge, blur, depth, segmentation | **Edge / blur:** computed by the model from the resized RGB (Canny 100/200 and its reference blur), so we pass the raw render. **Depth:** 3-channel, encoding / polarity / range per its data guide. **Segmentation:** colour-coded region masks with a fixed palette |

`sim/controls.py <scene> --model {vace,cosmos} --control <type>` writes `data/scenes/<scene>/controls/<model>_<control>.mp4`:

- **`vace_depth`**: simulator depth → inverse depth → per-frame min–max → 3-channel uint8. Sky goes black, as in VACE's training data. Also write `vace_depth_annot` by running VACE's own `DepthV2VideoAnnotator` on `rgb.mp4`, and compare the two in the smoke test, since the estimator may miss a small drone.
- **`vace_scribble`**: VACE's `ScribbleVideoAnnotator` on `rgb.mp4`.
- **`vace_gray`**: grayscale of `rgb.mp4` (a fallback if scribble loses the road).
- **`cosmos_depth`**: simulator depth encoded per the Sim-Transfer data guide (`shared/data-contract.md`).
- **`cosmos_seg`**: `instance.npy` rendered with a fixed colour palette. Gives a clean drone-against-sky outline.
- **`cosmos_edge`, `cosmos_blur`**: nothing to build; the request points at `rgb.mp4` and selects the control type.
- **`*_static`**: frame 0's control repeated for all frames. Only needed if we run the static-control ablation.

**Per-scene choice**, set in `matrix.yaml` and fixed after the Step 1 smoke outputs:

| Scene | VACE C | Cosmos D |
|---|---|---|
| `drones_takeoff` | depth | depth (or seg) |
| `drones_air` | depth | depth (or seg) |
| `convoy_overhead` | scribble (or gray) | edge (or blur) |

Overhead depth is nearly flat, so the convoy uses line or appearance controls. For the convoy, C and D get different control types; the report must say so rather than present C vs. D as a like-for-like comparison.

## 6. Step 5: model runners

Every runner has the same CLI and output contract:

```
uv run gen/run_<model>.py --scene <scene> --cond <A|A+|B|C|D|E> --seed <k> --out data/outputs/<scene>/<cond>/seed<k>.mp4
```

Each run writes the video plus `run.json` with the prompt, all model arguments, seed, input file hashes, wall time, peak GPU memory (`torch.cuda.max_memory_allocated` in-process, or sampled from `nvidia-smi` for subprocess wrappers), retries, and the git commit.

**Native formats differ, so each runner resamples its inputs.** The bundle is 153 frames at 30 fps (5.1 s). Resampling always picks source frames by timestamp over the same 5.1 s window; never retime one stream on its own.

| Runner | Native output | Inputs it builds from the bundle |
|---|---|---|
| `run_vace.py` A | 81 frames, 16 fps, 832×480 | Base prompt only (VACE `txt2vid`) |
| `run_vace.py` A+ | same | Detailed prompt describing the trajectory and events in words |
| `run_vace.py` B | same | Base prompt + `frame0.png` via `frameref` `firstframe`: `src_video` = frame 0 then 80 gray (127) frames; `src_mask` = black on frame 0, white elsewhere |
| `run_vace.py` C | same | Extension + control composition: `src_video` = `frame0.png` at frame 0, then the scene's VACE control (§5) resampled to frames 1–80; `src_mask` = black on frame 0, white elsewhere |
| `run_cosmos.py` D | 153 frames, 30 fps, 832×480 (121 / 4 s if the Step 1 check fails) | Base prompt + one control (§5) at full length, no first frame (`num_conditional_frames: 0`). Wraps `run-simtransfer-depth.sh` with a generated request JSON |
| `run_wanmove.py` E | 81 frames, 16 fps, 480p | Base prompt + `frame0.png` + `tracks.npz` resampled in time and converted to Wan-Move's format (from Step 1.3) |

Prompts come only from `configs/prompts/<scene>.yaml`, and are frozen before the full run. Use neutral wording ("quadcopters", "trucks").

**Side test (Step 1 or 5, optional):** VACE box-layout mode with our own boxes. Draw `boxes.json` as outlines on white frames, put the real `frame0.png` at frame 0, and run through VACE's `layout_track` control (`bboxtrack` mode) composed with `firstframe`, as in its `move_anything` example. If it works with several objects, it becomes a same-checkpoint sparse condition next to C. If it doesn't, drop it; Wan-Move (E) stays the sparse condition.

## 7. Step 6: run the matrix

`gen/launch.py configs/matrix.yaml` expands the matrix into jobs and runs them.

- Skip a job if its `.mp4` and `run.json` already exist, so the launcher can be re-run after crashes.
- Logs go to `data/outputs/logs/`.
- Order: the 27-clip minimum (A, B, C) first, then D, A+, E.
- Set `NCCL_P2P_DISABLE=1` for anything multi-GPU on swarm5.

Rough budget: VACE/Wan-Move 14B at 15–30 min per clip and Cosmos at ~2 min per clip puts the full 54 clips at about 15–25 GPU-hours, which spreads across 3–4 cards in under a day if the cards are free.

## 8. Step 7: evaluation

1. **Resample** (`eval/resample.py`): every output and every R render goes to 16 fps, 832×480, over the common 5.06 s window (81 frames). Frames are picked by timestamp, and scene-bundle labels are resampled to the same timestamps.
2. **Track** (`eval/track.py`):
   - SAM 2 for per-object masks. For B, C and E, seed it on frame 0 with the simulator's masks, since those conditions get the real first frame. For A, A+ and D, seed it from manual clicks on the first frame where each object is visible: D gets no first frame, so its frame 0 is generated, and seeding from simulator masks would pull the tracker toward the correct positions and hide errors. For D, also log the IoU between the simulator's frame-0 masks and the clicked masks, as a measure of how well D follows the layout.
   - CoTracker3 for points, seeded from the same initial masks.
   - Output: per-clip `tracks_pred.npz` and `boxes_pred.json`, in the same format as the scene bundle.
3. **Human audit** (`eval/annotate/`): export ~8 keyframes per clip plus frames around each event, with predicted boxes drawn on. Annotators correct boxes, mark missing or extra objects and ID swaps, and log event frames. A spreadsheet or JSON per clip is enough at this scale. Audited labels are the ground truth for the generated clip; tracker output is logged next to them for agreement.
4. **Metrics** (`eval/metrics.py`), per clip:
   - Count: objects present vs. expected, and extra or missing objects per frame.
   - Events: was each event observed, and its time error vs. `timeline.json`. Scored for every condition.
   - Spec checks (`eval/specs.py`): temporal specs such as "lift-off before translate" or "on road until turn, then on field", evaluated on the audited tracks. Scored for every condition.
   - Position error (controlled conditions only): center distance divided by object size, matched by ID.
   - Heading through the convoy turn, and ID switches.
   - Background and camera stability: drift of fixed landmarks (tracked with CoTracker on background points).
   - Cost: read from `run.json`.
   - Tolerances (e.g. ≤ 0.5 body length, ≤ 0.25 s) are set in a config before scoring, and first run on R. R must pass all of them, or the tolerance or the label pipeline is wrong.
5. **Realism** (`eval/realism_pairs.py`): randomized, unlabeled side-by-side pairs (generated vs. R), shown as an HTML page. Raters pick the more realistic one and tick an artifact checklist. Results go to a CSV.

## 9. Step 8: outputs

`eval/report.py` produces:

- Per scene, a score table: one row per condition (R, A, A+, B, C, D, E), one column per metric, with mean and per-seed values.
- A claimed vs. observed table per controlled model: the paper's reported control accuracy next to ours.
- A video grid per scene: prompt, frame 0, control input, every output and seed, and a trajectory overlay (simulator path vs. tracked path). Failures included.

## 10. Order of work and dependencies

| # | Step | Needs | Done when |
|---|---|---|---|
| 1 | Environments (Blender, VACE, Wan-Move, eval tools) | — | Each tool's smoke clip runs; VACE 14B vs. 1.3B decided; Wan-Move track format written down; Cosmos 153-frame run passes or the 4 s fallback is chosen |
| 2 | Scene specs | — | Three YAML files reviewed |
| 3 | Renderer + bundle export | 1, 2 | Three bundles with overlays checked by eye |
| 4 | Control videos | 3 | Each model's controls built in its own format; per-scene control choice for C and D fixed |
| 5 | Model runners | 1, 4 | One clip per condition on one scene, with `run.json` |
| 6 | Launcher + full matrix | 5 | 27 then 54 clips on disk |
| 7 | Eval harness | 3 (can be built on R before 6 finishes) | Metrics pass on R; then all clips scored |
| 8 | Report | 6, 7 | Tables and grids generated |

Steps 1–2 run in parallel, and 7 can be built on the raw renders while 6 is running. A natural split across people is renderer (2–4), generation (1, 5, 6) and eval (7, 8).

## 11. Technical unknowns, in the order they block

1. Whether VACE-14B fits and runs fast enough on a shared A100 (Step 1). Sets the checkpoint for A–C.
2. Wan-Move's track format and whether it handles ~3 objects × 16 points (Step 1). Blocks E.
3. Whether Cycles on CUDA works headless on swarm5 without EGL (Step 1). Blocks everything; if it fails, try swarmllm.
4. Whether Cosmos Sim-Transfer runs 153 frames with the full-history profile (Step 1). Sets the clip length for every condition.
5. How a small drone against the sky reads in each model's depth format, and whether VACE's own depth annotator catches it (Step 4).
6. Whether scribble (VACE) and edge (Cosmos) keep the convoy road and vehicles readable from overhead (Step 4).
7. Whether VACE's layout mode takes several boxes (side test). Optional.
