# p2s: from a prompt to scored multi-camera video

One prompt file goes in. Out come a physics simulation, a render of it from every camera, a generated video per camera
and control, and scores. Each stage is one folder, in the order it runs, and reads only files the stage before wrote.

```
p2s/prompts/<scene>.yaml          INPUT   the prompt, the camera views, the clip format
p2s/experiments/v1/<model>.yaml   INPUT   which scenes, which controls, which model, which seeds
      │
      ▼
1_scene/      write_scene.py    an LLM (the SceneAgent) writes a PhysX scene from the prompt, simulates it,
              iterate.sh        reads the checker's report and fixes the scene, until the report says PASS
      │       → world/          scene.json, x.json + streams/ (every pose per frame), r.md, spec.yaml
      ▼
2_render/     render_x.py       Blender renders the world from each camera
              export_x.py       and writes that camera's bundle
      │       → cams/<cam>/bundle/     rgb, depth, masks, tracks; controls/ depth.mp4, seg.mp4, edge.mp4, ...
      ▼
3_generate/   write_captions.py, compose_prompts.py     what each view's prompt says
              run_multicam.py   one clip per view, control and seed, on free GPUs
      │       → gen/<cam>/<control>/<model>/seed<k>/out_video.mp4
      ▼
4_score/      run_eval.py       each clip against the simulator's depth, masks and tracks
              summarize.py      one table per experiment
              → scores/<experiment>/, tables/<experiment>.csv
```

Everything is written under `$S2R_DATA/cv-project/p2s/<scene>/r<rev>/`.

## Run it

```bash
source ~/.sim2real.env                                    # not sim/physx/env.sh: that one points at another workspace
bash p2s/run.sh p2s/experiments/v1/smoke_ltx25.yaml       # the smallest end-to-end run: one scene, three clips
bash p2s/run.sh p2s/experiments/v1/ltx25.yaml 5,6,7       # a full experiment; the GPU list is for the renders
```

`run.sh` does every stage for every scene the experiment file names, and skips whatever is already done. `--until
scene`, `--until render` or `--until generate` stops early. The PhysX programs must be built once first
([`sim/physx/README.md`](../sim/physx/README.md)).

The stages by hand, for one scene and one experiment:

```bash
PY=$S2R_ROOT/envs/eval/.venv/bin/python
bash p2s/build_scene.sh p2s/prompts/ball_ramp.yaml 5,6,7                      # stages 1 and 2
$PY p2s/3_generate/write_captions.py p2s/prompts/ball_ramp.yaml               # stage 3, only for a scene with no captions yet
$PY p2s/3_generate/compose_prompts.py --exp p2s/experiments/v1/ltx25.yaml     # stage 3: the exact prompt per view, tracked in the repo
$PY p2s/3_generate/run_multicam.py   --exp p2s/experiments/v1/ltx25.yaml      # stage 3: generate   (--status lists the jobs)
$PY p2s/4_score/run_eval.py          --exp p2s/experiments/v1/ltx25.yaml      # stage 4
```

## What is where

| Path | What |
| --- | --- |
| [`prompts/`](prompts/) | One prompt file per scene. The only input a scene has. |
| [`experiments/v1/`](experiments/v1/) | One file per model: scenes, controls, seeds, sampler, GPU limits. |
| [`captions/<scene>/`](captions/) | What each view's video prompt says: `world.json`, `cams/<cam>.json`, and the exact prompts in `composed/`. |
| [`promptfile.py`](promptfile.py) | Reads a prompt file, knows the folder layout, and describes a PhysX world to the later stages. |
| [`run.sh`](run.sh), [`build_scene.sh`](build_scene.sh) | Everything for an experiment file; stages 1 and 2 for one prompt file. |
| [`1_scene/`](1_scene/) | `write_scene.py` (prompt file to accepted world), `iterate.sh` (one simulate-and-check iteration). The simulator and checker themselves are in [`sim/physx/`](../sim/physx/). |
| [`2_render/`](2_render/) | `render_x.py` (Blender, with [`sim/blender/import_x.py`](../sim/blender/import_x.py)), `export_x.py`, `bundle.py` (the exporter), `control_videos.py` (depth, seg, edge, normals). |
| [`3_generate/`](3_generate/) | The job runner, prompt composing, one folder per model under `models/`, and the negative prompts. [README](3_generate/README.md). |
| [`4_score/`](4_score/) | `run_eval.py`, `summarize.py`, `make_figures.py`, the metric functions and one folder per reader model. [README](4_score/README.md). |

## The prompt file

```yaml
name: ball_ramp
rev: 1                                    # the folder is <name>/r<rev>; raise it to keep an old world beside a new one
prompt: >
  In a workshop, a red rubber ball rolls down a wooden ramp onto a concrete floor and knocks over a stack ...
cameras:                                  # the views; the id names the camera everywhere downstream
  side: a tripod camera to one side, seeing the ramp and the boxes
  overhead: a ceiling camera looking straight down at the floor
clip: {fps: 16, frames: 113, resolution: [832, 480]}
# scene: sim/physx/scenes/<file>.json   # optional: a scene already written for this prompt; then no LLM runs
score_static: [glass]                     # optional: things that never move but should be scored as objects
```

- **Without `scene:`**, the SceneAgent writes the scene. It is given `agent/task.md`: where its manual is
  ([`sim/physx/docs/`](../sim/physx/docs/)), the prompt with its views, what is fixed (the name, the clip format, the
  camera ids), and the one command it may run, `p2s/1_scene/iterate.sh <run_dir> <N>`.
- **The agent is a command** that reads the task on standard input. The default is Claude Code run headless, allowed
  to read and write files and run `iterate.sh` only. Change it with `--agent-cmd` or `P2S_AGENT_CMD`. `--task-only`
  writes `task.md` and stops, so the task can be handed to another tool; run `write_scene.py` again to accept the result.
- **The agent may change the scene's `expect` list between iterations.** Its manual allows that only when an
  expectation was wrong about the prompt, with the reason in `meta.notes`. Read the notes of an accepted scene.

## What "accepted" means

`write_scene.py` accepts the last iteration only if the checker's status is PASS and the simulated clip matches the
prompt file: the frame count, the frame rate, the camera ids and every camera's resolution. It then copies that
iteration to `world/` and writes two files:

| File | What it holds | Read by |
| --- | --- | --- |
| `world/spec.yaml` | clip format, cameras (with the object each rides on), an instance id for everything drawn | `3_generate/common/multicam_common.py`, `4_score/common/reference.py`, `2_render/render_x.py` |
| `world/world.json` | the prompt, the accepted iteration, the events, the report's counts | `4_score/common/reference.py`, `2_render/export_x.py` |

Everything that can move is a scored object (instance ids 1 to 99, in the order of `x.json`): vehicles, walkers, robot
arms, dynamic and kinematic bodies, particles, cloth. The ground is 100; static bodies and traffic signals are scenery
(101 and up). A soft body mounted on a host takes its host's id. A static body listed under `score_static` is scored
as an object.

## Add a scene, a control, a model

- **A scene:** write `prompts/<name>.yaml`, add one line under `scenes:` in each experiment file that should run it
  (`<name>: {rev: 1, p2s: p2s/prompts/<name>.yaml, prompts: p2s/captions/<name>}`), and run `p2s/run.sh`.
- **A control:** add it under `conditions:` in an experiment file. The controls a bundle offers are `depth`, `seg`,
  `edge`, `normals` and `rgb`; which of them a model takes is in [`3_generate/README.md`](3_generate/README.md).
- **A model:** a folder under `3_generate/models/` with a `run.sh` that does one job, one entry in `FAMILIES`
  (`3_generate/common/multicam_common.py`), and an experiment file.

## Limits

- **Captions for a new scene are only a starter.** `write_captions.py` repeats the prompt for every view with that
  view's camera sentence in front. `intersection` and `robot_pour` use hand-written blocks; `ball_ramp` uses starter blocks.
- **The render is plain clay.** Shapes are drawn exactly as PhysX collided them: no textures, road markings or sky.
- **Particles, cloth and soft bodies have masks and depth but no track points.**
- **The checker confirms the scene's `expect` list, not the prompt.** Whether that list covers the prompt is for a
  person, or Eval1, to judge.
