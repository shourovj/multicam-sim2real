# eval/: scoring generated clips against the simulator

Five scores, the same for every scene. Three are per camera (Eval2: does the clip follow the simulation?) and two are across cameras (Eval3: do the cameras show one world?). Every reference comes from the camera bundles, so a new scene needs no new evaluation code.

```
eval/
  run_eval.py              the five scores for an experiment's finished runs
  summarize.py             one table across experiments, ordered from the densest control to text only
  make_figures.py          pictures of the per-camera scores for one scene, and a chart of a summary table
  score_multicam.py        earlier scorer: colour names per object and traffic-light state (kept; scene-specific)
  grid.py                  side-by-side grid video for the single-camera rounds
  common/
    reference.py             a view's simulator reference: depth, masks, cameras, surface points, which objects count
    metrics.py               the metric functions: numpy only, no model, no files
    video_io.py              read a clip; reader cache stamps
  readers/                 one folder per model that turns a video into arrays
    depth_anything3/         depth (its own Python env)
    sam2/                    object masks
    cotracker3/              point tracks
    dinov2/                  appearance features (diagnostic)
```

## The five scores

All are in [0, 1], 1 best.

| Score | Question | Simulator reference | Read from the video with |
| --- | --- | --- | --- |
| `depth` | Is the scene's shape where the simulator put it? | Rendered depth | Depth Anything 3 (metric model), brought to scale by a median ratio per frame; a pixel passes within 1.25× |
| `mask_iou` | Is each object where the simulator put it, in every frame? | Rendered object masks | SAM 2.1, started once per object from its simulator box and three clicks at its clearest frame |
| `track` | Does each surface point move as in the simulation? | Surface points on every moving part, with visibility | CoTracker3; TAP-Vid's position accuracy averaged over 1, 2, 4, 8 and 16 px at 256 px scale |
| `object_colour` | Is the same object the same colour in every camera? | Object masks of two cameras at the same instant | The median colour inside each mask; a pair agrees under a CIELAB distance of 20 |
| `surface_colour` | Is the same surface point the same colour in every camera? | Rendered depth and camera poses, which give exact pixel pairs | The two videos' colours at each pair |

`eval2` is the harmonic mean of the first three, so a clip cannot make up for failing one by doing well on another. For a scene, condition and seed, each score is averaged over the camera views first and the harmonic mean is taken after. Cut videos are scored like camera views (per shot for masks and tracks) and reported separately.

The two cross-camera scores count only what is in place: an object whose mask in the video overlaps its rendered mask (IoU of 0.5 or more) in both cameras, and a pixel pair whose estimated depth passes in both. Where a clip does not follow the simulator, the simulator's masks and pixel pairs do not point at the same thing in the video, so comparing colours there says nothing. `object_in_place` and `surface_in_place` give the share that was counted. Read them together: a high colour score from a small share describes little of the scene.

An object is not scored in the camera that it carries (a car's own dashcam shows only its hood) or that hides it. Add `eval: false` to an object in the scene spec to leave it out everywhere.

## Run it

```bash
source ~/.sim2real.env
PY=$S2R_ROOT/envs/eval/.venv/bin/python
$PY p2s/4_score/run_eval.py --exp legacy/configs/experiments/cvp_v1/cosmos3_sim_transfer.yaml                  # every finished run
$PY p2s/4_score/run_eval.py --exp legacy/configs/experiments/cvp_v1/cosmos3_sim_transfer.yaml --only scene=robot_pour,seed=0
$PY tests/test_eval_metrics.py                                                                   # the metric functions, no GPU
$PY p2s/4_score/summarize.py --exp legacy/configs/experiments/cvp_v1/*.yaml --reference legacy/configs/experiments/multicam_v1.yaml   # one table
```

It needs one GPU with a few GB free and picks the one with the most free memory unless `CUDA_VISIBLE_DEVICES` is set. Readers run once per clip and cache their output in `<run folder>/est/`; a reader runs again only when the video or the reader changes, or with `--force`. Scoring a clip takes well under a minute.

Results go to `<scene>/r<rev>/scores/<experiment>/`:

| File | Rows |
| --- | --- |
| `eval2.csv` | One per run: `depth`, `depth_objects`, `mask_iou`, `track`, `track_occlusion`, `eval2` |
| `eval2_objects.csv` | One per run and object: its mask IoU |
| `eval3.csv` | One per scene, condition and seed: `object_colour`, `surface_colour`, the share in place for each, mean colour distances, feature similarities |
| `eval3_objects.csv` | One per object: colour agreement across cameras |
| `eval_summary.json` | Per condition and seed: the camera-averaged scores, `eval2`, the cross-camera scores, and the cuts |

## Read the scores against the render

The readers are models too, and they are not perfect even on a clip that follows the simulator exactly. `--on-render` scores the simulator's own render in place of the generated videos (files end in `_render`):

```bash
$PY p2s/4_score/run_eval.py --exp legacy/configs/experiments/multicam_v1.yaml --on-render       # the textured render (scene rev 1)
```

Use the textured render, not the clay one. Its cameras truly show one world with real surfaces, so:

- **Per camera,** what it scores is the most a generated clip can be expected to score with these readers. A generated clip at that level follows the simulation as closely as the readers can tell.
- **Across cameras,** its `object_colour` and `surface_colour` are what consistent views score under these viewpoint and lighting changes.

The clay render is one grey everywhere. The depth and track readers do worse on it than on the textured render, and across cameras it scores 1.0 trivially, so it is not a reference for either.

Measured on 2026-11-23, camera views averaged, seed 0 (one seed: an illustration, not a result):

| Scene | Clips | `depth` | `mask_iou` | `track` | `object_colour` | `surface_colour` |
| --- | --- | --- | --- | --- | --- | --- |
| intersection_4way | Textured render | 0.83 | 0.85 | 0.63 | 0.92 | 0.97 |
| intersection_4way | Sim-Transfer, depth control | 0.87 | 0.78 | 0.59 | 0.38 | 0.62 |
| robot_pour | Textured render | 0.88 | 0.89 | 0.73 | 0.95 | 0.91 |
| robot_pour | Sim-Transfer, depth control | 0.88 | 0.84 | 0.77 | 0.37 | 0.54 |

Per camera the generated clips sit at the render's level; across cameras they fall far below it.

## Why colour, not features, across cameras

Both are computed. On the textured render the DINOv2 similarity of the same object in two cameras is barely above that of two different objects, because the crops differ more by viewing angle (overhead against street level) than by identity. Colour inside the exact mask separates the consistent render from the generated clips clearly, so the two cross-camera scores use colour and the feature similarities stay in `eval3.csv` as diagnostics. Colour cannot see a change of shape or markings with the same paint; a judge model on the same crops would add that.

## Readers and their envs

| Reader | Env | Weights | Licence |
| --- | --- | --- | --- |
| `depth_anything3` | `$S2R_ROOT/envs/da3` ([`setup/envs/da3-requirements.txt`](../../setup/envs/da3-requirements.txt)) | `depth-anything/DA3METRIC-LARGE` in the Hugging Face cache | Apache-2.0 (DA3-Large and Giant are CC BY-NC 4.0) |
| `sam2` | eval env | `$S2R_ROOT/checkpoints/sam2/sam2.1_hiera_large.pt` | Apache-2.0 |
| `cotracker3` | eval env | `$S2R_ROOT/checkpoints/cotracker/scaled_offline.pth` | CC BY-NC 4.0 (model card) |
| `dinov2` | eval env | torch.hub, first use downloads into `$TORCH_HOME/hub` | Apache-2.0 |

To swap a reader, add a folder under `readers/` that writes the same arrays (`depth.npy` with a `kind`, masks per instance id, points per track) and point `run_eval.py` at it. The proposal's SAM 3 prompted by class name would replace `sam2/` this way once its gated weights are available.

## Known limits

- The mask reader is told where each object is at one frame, so `mask_iou` measures whether one coherent thing stays where the simulator puts the object. It does not test that the thing is recognisable as its class.
- `track` is low for small, distant or fast-spinning parts even on the render. Compare with the render's score.
- Bundles rendered before 2026-11-23 store track points relative to each part's last pose; `reference.py` corrects for it, and their `tracks.npz` files are wrong.
- Optical flow is not scored: the renderer does not write motion vectors yet, and `track` covers motion.
