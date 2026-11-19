# Multi-camera v1: one simulation, three cameras, Cosmos3-Nano with no first frame

Design doc for this week's toy experiment. Author: Shourov Joarder. Drafted 2026-11-18 with Claude Code (Opus 5.5). Status: built on 2026-11-18 (§10 lists the files, §13 how to run them); the intersection is rendered and the robot scene is rendering.

Decided on 2026-11-18:

- Clips are 7 s (113 frames at 16 fps), separate views and cut videos alike, so each shot of a cut video is about 2.3 s. This was changed from 5 s the same day.
- The control is the simulator's own colour video, unblurred, in Cosmos's blur slot: one condition, `V`. Depth and segmentation conditions come later. They are still rendered, because scoring needs them.
- Water is the analytic stream; other liquid models (Mantaflow) plug in later.
- Claude Code writes the specs in session.
- Assets: CC0 and Apache-2.0 are permissive, pending the PI's confirmation (§8).

## 1. What this week tests

Two scenes are simulated once in Blender and rendered from three cameras each. Cosmos3-Nano gets only the simulator's colour video as its control and a text prompt, with no first frame, and makes it look like real footage. Two kinds of output:

1. **Separate views.** Each camera's controls go into Cosmos alone: three generations per scene.
2. **Cut video.** The three cameras' controls are joined into one 7 s video with hard cuts (camera 1, then 2, then 3, on one shared clock), and Cosmos generates it in one pass.

The question: when the world, the motion and the cameras all come from one simulation, do the generated views agree with each other? Do the cars keep their colours, is the background the same street, and does a traffic light that turns green in one view turn green in the others? And does the cut video agree better across its cuts than three separate generations do?

The two scenes:

| Scene | What happens | Cameras |
| --- | --- | --- |
| `intersection_4way` | Four-way intersection with traffic lights. Two cars cross north–south on green while two wait on red on the east–west road. Two pedestrians cross in front of the waiting cars; a third stands on a corner filming. The lights change and the waiting cars drive off. | Dashcam of one waiting car; the bystander's phone; a CCTV camera on a pole |
| `robot_pour` | A robot arm holding a pitcher moves it over a glass, tilts it, pours water until the glass is about two-thirds full, then tilts back. Fallback if pouring is not ready in time: `robot_pickdrop`, where the arm picks up a block and drops it into a bin. | Wrist camera on the arm (egocentric); side camera on a tripod; CCTV in a corner of the room |

Round 2's two-camera drone test is the starting point ([Round 2 findings](2026-11-11-0845-round2-findings.md), Nov 13 follow-up on the [Round 2 page](https://claude.ai/artifact/DMwkj9covsGwnoHo46j3hD)). With depth only, the drone colours matched the simulator in 0 of 6 views. With the whole simulator video as the control (Vraw), colours matched in 6 of 6, but the backgrounds disagreed between views and one camera kept the simulator look. In the cut and split-screen videos each part followed its own control, so a shot's look changed at the cut. This week keeps that Vraw control, the unblurred simulator video with no first frame. It adds moving cameras (dashcam, phone, wrist), more objects, a signal state that must agree across views, and a three-camera cut. A small probe tries Cosmos's own blur presets on one view, to see whether a light blur drops the simulator look and keeps the colours.

## 2. Is this the VideoCoCo pipeline?

[VideoCoCo](https://arxiv.org/abs/2607.27380) (arXiv 2607.27380; read from its abstract, summaries and Figure 2, not the full text) has two stages. A coding agent writes a self-contained Blender program from the prompt and runs it to get a low-fidelity "white-clay" draft video. An instruction agent then writes an editing instruction, and a video editing model turns the draft into the final video. The draft supplies the dynamics; the editor supplies the appearance.

Our pipeline already has that shape: stage 1 is `sim/` running inside Blender, and stage 2 is Cosmos3-Nano transfer. We keep the idea and change four things:

| | VideoCoCo | Ours (this week) | Why |
| --- | --- | --- | --- |
| What the LLM writes | A free Blender program | A scene spec (YAML) that is checked against a schema; our fixed compiler in `sim/` turns it into Blender | Cameras must be named entities that can ride on objects (dashcam on `car_ego`, phone on `ped_phone`). The world is simulated once and rendered N times. A spec diff is readable and its hash is stable, and the simulator can check it (no car runs a red). A free program has its camera baked in and differs on every run. |
| Draft given to the generator | RGB "white clay" render | The simulator's colour render per camera, in Cosmos's blur slot. Depth and segmentation are rendered too, and become controls later. | The colour render is the closest thing to VideoCoCo's draft, and it carries colour, which is what Round 2's views disagreed on. Depth and segmentation are exact in a simulator, so they are the next step. |
| Generator | Closed editing models, and an open editor (OmniWeaving) adapted on 3,000 draft–instruction–target triplets whose targets come from a teacher editor (reported as Seedance 2.0) | Cosmos3-Nano, open weights, on our A100s | Runs locally; settings and seeds are recorded |
| Views | One | Three cameras per scene, separately and joined by cuts | This is the experiment |

A free program can describe scenes our library has no behaviour for, and that is a real loss. The middle path, later and not this week, is for the LLM to write a new behaviour module as code against the library's interface when the spec has nothing that fits. The module is reviewed, versioned and then reused, so specs stay declarative.

On scaling: once the world is simulated, each extra camera costs one render (minutes) and one generation (≈12–16 min, §6.4). The views are not guaranteed to agree when they are generated separately, and that is what this test measures.

## 3. Can Blender be the simulator for these two scenes?

Blender ships a rigid-body solver (Bullet), a liquid solver (Mantaflow FLIP), keyframes, armatures with inverse kinematics, and per-frame Python. Both are present in our Blender 4.2.1 build (checked: `bpy.app.build_options.fluid` and `.bullet` are True). Blender has no vehicle dynamics, traffic rules, pedestrian behaviour or robot control. Those come from small numpy controllers in `sim/physics/`, as the drones and the convoy already do. The controllers produce poses that Blender keyframes, and Blender's own solvers run where they fit. Everything runs in one `render_scene.py` call with a fixed seed.

| Part | How | Built in or ours | Risk |
| --- | --- | --- | --- |
| Car motion: speed, braking, stopping at a red, keeping a gap | `sim/physics/ground_vehicle.py` (IDM) extended to lane paths through the junction. A red signal, or a pedestrian on the car's part of the crosswalk, acts as a stopped car at the stop line (standard IDM trick). | Ours | Low; the model exists |
| Wheels turning, small body pitch when braking | Wheel spin from speed; pitch from acceleration | Ours | Low |
| Traffic lights | One signal plan per approach; lamp materials switch emission per frame | Ours, plus Blender materials | Low |
| Pedestrians walking | Path follower at about 1.4 m/s, playing an animated walk cycle from the asset with its phase set by distance walked | Ours, plus asset animation | Medium: needs rigged, animated people (§8) |
| Road, crosswalks, kerbs, buildings | Procedural meshes in `sim/assets.py` (like `build_ground`) | Ours | Low |
| Robot arm motion | End-effector waypoints → damped-least-squares IK on the Franka chain in numpy → joint angles → link poses keyframed in Blender | Ours | Medium: a new controller |
| Grasp | The pitcher (or block) is parented to the hand between `grasp` and `release` events. No contact physics. | Blender parenting | Low |
| Drop (pick-and-drop fallback) | At release the block becomes an active Bullet rigid body and falls into the bin | Blender (Bullet) | Low |
| Pouring water, level 1 (this week) | Analytic stream: exit speed from the tilt and the water head, a ballistic path, radius from flow continuity; the glass level rises by the integrated flow | Ours: a simple fluid model, not a fluid simulation | Low |
| Pouring water, level 2 (later, not this week) | Mantaflow FLIP liquid with the pitcher and glass as animated collision objects | Blender (Mantaflow) | High: thin moving walls leak at low resolution, a 1 cm stream is only about 2 cells wide in a 0.6 m domain at resolution 128, and the CPU bake time on our 96 cores is unmeasured |
| Cameras on moving things | Camera pose each frame = parent object's pose × fixed offset (dashcam, wrist), plus handheld shake for the phone | Ours | Low |

Both scenes are feasible in Blender this week. The intersection is mostly asset and controller work. The pour is feasible with the level-1 stream, which is scripted fluid kinematics, so the doc and the results will call it that, not a fluid simulation. The liquid sits behind one interface (`sim/physics/pour.py`: pitcher pose in, stream and level meshes out), so Mantaflow or another solver can replace it later without touching the rest. If the arm controller is not working by mid-week, the robot scene becomes `robot_pickdrop`.

## 4. Pipeline

```
user prompt (one paragraph, in configs/scenes/<scene>.source.md)
   │  [1] spec author: LLM (this week Claude Code, claude-opus-5-5, in session; recorded in the .source.md)
   ▼
scene spec  configs/scenes/<scene>.yaml   (schema v2: world, signals, objects, cameras, checks)
   │  [2] sim/render_scene.py in Blender: controllers + Blender solvers, simulated ONCE
   ▼
world/      per-frame poses, signal states, events, check results
   │  [3] render each camera from the same poses (one Blender pass per camera)
   ▼
cams/<cam>/bundle/   rgb, depth, instance, timeline (camera pose per frame), boxes, visibility
   │  [4] controls per camera                    [4b] cut assembler (schedule from experiment.yaml)
   ▼                                              ▼
cams/<cam>/controls/  rgb.mp4 (this week's control), depth.mp4, seg.mp4 (later)      cut/<cut>/  the same files, joined frame-exact
   │  [5] caption composer: shared world block + camera block → prompt.json
   ▼
gen/<view>/<cond>/cosmos3-nano/seed<k>/   request.json, prompt.json, out_video.mp4, run.json
   │  [6] scoring against the simulator + a by-eye rubric
   ▼
scores/
```

Steps 1 and 5 are the two LLM jobs, matching VideoCoCo's coding agent and instruction agent. Everything else is deterministic code. Each step reads and writes files, so any step can be re-run alone, as in Round 2.

## 5. Scenes

All positions are in metres in the world frame: Z up, +Y north, +X east. Times are world seconds. The clip is 7 s. The renderer runs at 16 fps for 113 frames (7.06 s, 1 + 4n), the rate Cosmos3-Nano takes, so no frames are resampled and every rendered frame is used. The simulation itself steps at 1 ms.

### 5.1 `intersection_4way`

**User prompt** (frozen in `configs/scenes/intersection_4way.source.md`):

> A four-way city intersection with traffic lights on a clear afternoon. Two cars drive through north–south on a green light while two cars wait at the red light on the east–west road. Two pedestrians cross the street in front of the waiting cars, and a third stands on the corner filming with a phone. The lights change and the waiting cars drive off. Seen from the dashcam of one waiting car, the bystander's phone, and a CCTV camera on a pole.

**World.** Two roads crossing at the origin, one lane each way. Lanes are 3.5 m, so each road is 7 m wide. Sidewalks are 3 m with 0.15 m kerbs. Crosswalks are 3 m wide with 0.5 m stripes, and stop lines sit 8 m from the centre. Each leg is 60 m long. Procedural box buildings with window textures stand 4 m behind the sidewalks, 8–20 m tall, with a fixed seed. Nishita sky, sun at 40° elevation from the south-east.

**Signals.** One head faces each approach, on a pole at the far right corner, US style.

| Signal | 0–3.0 s | 3.0–4.0 s | 4.0–4.5 s | 4.5–7.1 s |
| --- | --- | --- | --- | --- |
| `tl_NS` (north and south approaches) | green | yellow | red | red |
| `tl_EW` (east and west approaches) | red | red | red (all-red) | green |

**Objects.** The colours are distinct on purpose, so each object can be told apart in every view and the colour scorer can check identity.

| id | inst | Asset | Colour | Motion (target; exact numbers are tuned in the spec until the checks pass) |
| --- | --- | --- | --- | --- |
| `car_red` | 1 | sedan | red | Southbound, 10 m/s from 26 m north; past the stop line on green at ≈1.8 s, at the centre ≈2.6 s, clear of the junction ≈3.4 s |
| `car_white` | 2 | SUV | white | Northbound, 9 m/s from 20 m south; at the centre ≈2.2 s |
| `car_ego` | 3 | hatchback | dark grey | Westbound, stopped at the east stop line; starts on `tl_EW` green at 4.5 s (later if a pedestrian is still in its lane), ≈2.5 m/s², about 8 m covered by the end, into the junction |
| `car_blue` | 4 | sedan | blue | Eastbound, stopped at the west stop line; starts on green like `car_ego` |
| `ped_red` | 5 | pedestrian | red jacket, dark trousers | Steps off the north kerb onto the east crosswalk at t = 0, walking south at 1.4 m/s; out of `car_ego`'s (north) lane by ≈2.6 s, at the south kerb ≈5.2 s |
| `ped_white` | 6 | pedestrian | white shirt, jeans | On the east crosswalk at t = 0, 0.6 m from the south kerb, walking north at 1.5 m/s; clears `car_ego`'s lane by ≈4.5 s, as its light turns green |
| `ped_phone` | 7 | pedestrian | green hoodie | Stands on the south-east corner holding a phone up at the junction; slight sway. Hidden from its own phone camera. |

The east crosswalk is just in front of `car_ego`, so the dashcam sees both crossing pedestrians close up, and the phone sees them from the corner.

**Cameras.** All three are 832×480 landscape. The phone is held sideways, so the cut video keeps one aspect ratio.

| id | Kind | Pose | FOV (horizontal) | Sees |
| --- | --- | --- | --- | --- |
| `cctv` | fixed | Pole on the north-east corner, (9, 9, 7), looking at (−2, −2, 0) | 80° | All four legs, all cars, all pedestrians, `ped_phone` filming |
| `dashcam` | attached to `car_ego` | Behind the windscreen, 0.6 m ahead of the car's centre, 1.25 m up, 4° down; the hood is in frame | 100° | `car_red` and `car_white` crossing left↔right, the pedestrians in front, `tl_EW` across the junction turning green, `car_blue` facing it |
| `phone` | handheld by `ped_phone` | 1.45 m up at the hand, aimed at (−3, 3, 1); shake of 0.6° and 4 mm at 1–3 Hz, fixed seed | 66° (a phone main camera) | Cars crossing, `car_ego` waiting on the right, the pedestrians close on the right |

**Checks the simulator runs** (the spec fails if any check fails):

- No two objects come closer than 1.0 m.
- No car crosses its stop line on red.
- No car enters a crosswalk lane segment while a pedestrian is in it.
- The event order is `tl_NS` yellow → `tl_EW` green → `car_ego` moves.
- Visibility: in the dashcam, `car_red` and `car_white` are each visible for ≥ 20 output frames. In the CCTV view, all 7 objects are visible for ≥ 60 frames. In the phone view, `car_ego` and both crossing pedestrians are visible for ≥ 20 frames.
- Each traffic-light head that a camera shows is at least N pixels tall, recorded per camera. This decides whether the light-state score (§7) is used for that view.

### 5.2 `robot_pour`

**User prompt:**

> A white robot arm on a table in a bright lab kitchen pours water from a clear pitcher into an empty glass until the glass is about two-thirds full, then tilts the pitcher back. Seen from a camera on the robot's wrist, a camera on a tripod beside the table, and a CCTV camera in the corner of the room.

**World.** A 5 × 5 × 2.8 m room with a light floor, pale walls, a shelf and a window on the back wall, and a ceiling light. A 1.2 × 0.8 m table, 0.75 m high. A Franka Panda arm is mounted at the table's back edge, using the Apache-2.0 meshes from `franka_description` or the MuJoCo Menagerie Panda.

**Objects.**

| id | inst | What |
| --- | --- | --- |
| `robot` | 1 | Franka Panda; each link has its own mesh, and all links share instance 1 |
| `pitcher` | 2 | Clear plastic pitcher with a handle, already grasped at t = 0 |
| `glass` | 3 | Clear tumbler on the table, 8 cm diameter |
| `water_stream` | 4 | Level 1: a tapered tube mesh from the spout to the glass, rebuilt every frame |
| `water_glass` | 5 | Level 1: a cylinder inside the glass whose height rises with the integrated flow |
| `table`, room | 0 | Background |

**Timeline.**

| Time | Event |
| --- | --- |
| 0–1.4 s | Arm carries the upright pitcher from its start pose to 8 cm above and to the side of the glass |
| 1.4–2.3 s | Tilt 0 → 45°; the stream starts once the tilt passes the angle at which water reaches the spout |
| 2.3–4.7 s | Tilt 45 → 75°, steady stream; the level reaches about two-thirds |
| 4.7–5.6 s | Tilt back to 0°; the stream thins and stops |
| 5.6–7.1 s | Arm moves the pitcher back toward the start |

**Cameras.**

| id | Kind | Pose | FOV | Sees |
| --- | --- | --- | --- | --- |
| `wrist` | attached to `panda_hand` | 5 cm off the flange, looking along the gripper with a 20° tilt toward the fingers (like a wrist RealSense); fingers and pitcher in frame | 87° | Pitcher close up, the stream, the glass from above |
| `side` | fixed tripod | (0.0, −1.3, 1.1), looking at (0.45, 0, 0.85) | 55° | Whole arm, pitcher, stream, glass in profile |
| `cctv` | fixed | Ceiling corner (−1.8, 2.2, 2.6), looking at the table | 90° | The room, arm, table; the stream is small |

**Checks:**

- Arm joints stay within the Panda's limits.
- The pitcher never passes through the glass or the table.
- The stream lands inside the glass rim on every frame where it exists.
- The final level is 60–75 % of the glass height.
- The wrist view shows the glass for ≥ 30 output frames.

**Fallback `robot_pickdrop`.**

| Time | Event |
| --- | --- |
| 0–1.8 s | Reach a 5 cm red block |
| 1.8–2.2 s | Close the gripper (attach) |
| 2.2–4.2 s | Lift and move over a bin |
| 4.2 s | Release; Bullet drops the block into the bin |
| 5.6–7.1 s | Retract |

It uses the same three cameras.

## 6. Cosmos3-Nano settings

Read from our pinned cosmos-framework checkout (commit `5950e7b4`): `cosmos_framework/inference/args.py`, `inference/transfer.py`, `data/generator/augmentors/transfer_control_input/control_input.py`, and the transfer cookbook specs.

### 6.1 The four controls and what each one carries

| Control | Input we give | What it tells the model | Carries colour? |
| --- | --- | --- | --- |
| `depth` | Our rendered depth as 1/z with one scale per clip (as Round 2) | Shape and distance | No |
| `seg` | Our instance masks drawn as flat colours. Cosmos's training augmentor draws a random colour per mask by default, so the colours should mean nothing to the model. It resizes seg with nearest-neighbour. | Which pixels belong to one object, and the object boundaries | No; a colour is only an ID |
| `edge` | Canny on the simulator's colour render, computed by Cosmos | Outlines and fine structure: lane paint, crosswalk stripes, window frames, gripper fingers | No |
| `blur` | The simulator's colour render, blurred by Cosmos at a preset strength | Coarse colour and layout | **Yes**; the only control that does |

So with no first frame, colour comes only from `blur` or from the prompt. In Round 2, depth alone got the drone colours wrong in every view.

### 6.2 How the blur and edge presets work

Cosmos builds `edge` and `blur` itself from `vision_path` (the source video) when a hint has no `control_path`. **If a hint has a `control_path`, that file is used exactly as given and the preset is ignored.** Round 2's Vraw passed the raw render as `blur.control_path`, which is why it was unblurred. This week we pass the simulator render as `vision_path`, give `edge` and `blur` no `control_path`, and choose a preset.

| `preset_blur_strength` | What Cosmos does to the source frames |
| --- | --- |
| `none` | Nothing: the raw render, the same as Round 2's Vraw, and outside the training range |
| `very_low` | Blur filter, then down-up resize by 4 |
| `low` | Downscale by 4, blur, upscale, then down-up by 4 |
| `medium` (Cosmos default) | Downscale by 2, blur, upscale, then down-up by 10 |
| `high` | Blur, then down-up by 16 |
| `very_high` | Downscale by 4, blur, upscale, then down-up by 16 |

Training drew the down-up factor at random from 4 to 17, so `very_low` to `very_high` are all inside what the model has seen and `none` is not.

| `preset_edge_threshold` | Canny low / high |
| --- | --- |
| `very_low` | 20 / 50 |
| `low` | 50 / 100 |
| `medium` (Cosmos default) | 100 / 200 |
| `high` | 200 / 300 |
| `very_high` | 300 / 400 |

**What this means for our scenes.** A traffic-light lamp is about 0.3 m across. From the CCTV pole at about 25 m with an 80° FOV, a pixel covers about 5 cm, so a lamp is about 6 px. A down-up by 10 (`medium`) smears it into the housing, so its colour will not survive. At `low` (factor 4) it may. Lane paint and crosswalk stripes are larger and survive in `edge`. The main runs use the unblurred video (`none`), so the lamp colours are in the control; the prompt also states the light changes. The sweep (§6.5) measures what each preset keeps.

Because the simulator video is the control, its look reaches the output. Round 2's Vraw kept the simulator look in one of the two drone views. The renders should therefore use plausible materials and lighting: road and building textures from CC0 sources, a sky with sun, and soft shadows. Untextured grey or flat-colour surfaces should be avoided.

### 6.3 Request settings

These are the Round 2 transfer settings, except for the clip length and the controls.

| Field | Value | Note |
| --- | --- | --- |
| `model_mode` | `video2video` | |
| `resolution`, `aspect_ratio` | `"480"`, `"16,9"` | 832×480 |
| `num_frames`, `fps` | 113, 16 | 7 s. 113 = 1 + 4·28 fits in one chunk (`num_video_frames_per_chunk` = 121), so there is no chunk-to-chunk carry-over. Untested at 113 on our setup (§6.4). |
| `num_first_chunk_conditional_frames` | 0 | **No first frame.** `vision_path` is set only as the source of the blur control. The cookbook's multi-control example does the same. The smoke test confirms that output frame 0 is not the simulator's frame 0. |
| `num_conditional_frames` | 1 | Only matters between chunks; unused at 113 frames |
| `num_steps`, `guidance`, `control_guidance`, `shift` | 50, 3.0, 1.5, 10.0 | Cookbook values. The seg cookbook uses `control_guidance` 2.0; we keep 1.5 across conditions so that it is not a variable. |
| `share_vision_temporal_positions` | true | Cookbook |
| `negative_prompt_file` | cookbook `negative_prompt.json` | Copied into each run folder |
| `prompt_path` | `prompt.json` (§6.6) | Structured caption, the format Cosmos was trained on |
| `blur` | `{preset_blur_strength: none}`, no `control_path` | Cosmos takes the simulator video from `vision_path` and leaves it unblurred. This is the same control as the Nov 13 Vraw runs, which passed the raw video as `control_path`. |
| Per-hint `weight` | 1.0 | One control, so the weight has no effect. With several controls, Cosmos runs one attention pass per control and averages them with the weights normalised to sum to 1. |
| Guardrails | off (`--no-guardrails`) | As Round 2 |

Cut frames: shots start at output frames 0, 37 and 77, so the shots are 37, 40 and 36 frames (about 2.3, 2.5 and 2.25 s). The video tokenizer packs frame 0 alone and then groups of 4 (1–4, 5–8, …), so a cut at frame 1 + 4k never puts two cameras in one latent frame. Round 2's Vcut cut at 41 for the same reason.

Cosmos3 also has a multiview mode (`multiview.views` with one `camera_key` per view). It is built for a fixed car camera rig with world-scenario-map controls, and our CCTV, phone and wrist cameras are not in that rig. It is out of scope this week; it would be worth checking later whether it can take arbitrary cameras.

### 6.4 Cost, to be measured before the matrix

Round 2: 81 frames with one control took about 8.5 min and 40–44 GB on one A100 80 GB. This week is also one control. Going from 81 to 113 frames adds 1.4× the tokens, and attention grows a little faster than that. That gives an estimate of about 12–16 min per clip and perhaps 45–55 GB.

Memory should fit comfortably. Our sequence has fewer video tokens than the cookbook's single-control run (121 frames at 1280×720), which runs on one GPU. Even so, a 4-step smoke run at 113 frames measures time and peak memory first. If it does not fit, the fallback is 81 frames (5 s).

When depth and segmentation come back as extra controls, each adds its own tokens and attention pass. Three controls would roughly double the cost again.

### 6.5 Conditions

| Condition | Controls | Colour comes from | Purpose |
| --- | --- | --- | --- |
| `V` (this week) | blur slot, preset `none`: the simulator's colour video as rendered | The simulator, plus the prompt | Main condition; the same control as the Nov 13 Vraw test, now with three cameras and a three-camera cut |
| `G` (later) | depth + seg | Prompt only | Geometry-only; successor of Round 2's C0 |
| `GA` (later) | depth + seg + blur (`low`) | Simulator colours, blurred, plus the prompt | Does a blurred colour hint plus exact geometry keep the views consistent without the simulator look? |

Probe, run on the intersection CCTV view with seed 0 only:

- **Blur sweep:** `V` with blur `very_low` / `low` / `medium` / `high`. `none` is the main run. Answers how much blur drops the simulator look while keeping the car colours and the light state.

Probes on edge and segmentation wait for the depth and segmentation conditions.

Notes for when `G` and `GA` come back:

- **Separate videos, not a composite.** Cosmos gets each control as its own video. It does not merge them into one picture: each control gets its own attention pass, and the results are averaged with the hint weights.
- **Background segments.** Our seg today marks only the moving objects, on black. Cosmos trained on segmentations that cover the whole frame, with every region a mask, so the static parts should get their own instance IDs too: road, sidewalks, each building, poles, sky, table, walls.
- **Examples.** One Round 2 frame shown as each control (depth, seg, and blur at `low`, `medium` and `high`, made with Cosmos's own blur code) is in `$S2R_DATA/poc/multi-cam/control_examples/`.

### 6.6 Prompts

Cosmos3 trained on structured JSON captions with these fields: `subjects` (`description`, `appearance_details`, `location`, `action`, …), `background_setting`, `lighting`, `aesthetics`, `cinematography`, `style_medium` and `artistic_style`. The cookbook's multi-control example is a robot arm pouring from a pitcher. Each prompt is composed from two frozen parts:

- **World block**, one per scene and shared by every view: subject descriptions with fixed appearance wording ("a red four-door sedan"), the background, the lighting and the overall action. Every view and the cut video use the same wording, so prompt differences between views come only from the camera block.
- **Camera block**, one per camera: what this camera sees and when (from the simulator's visibility table), `cinematography` (motion, framing, angle) and `style_medium`.

The camera blocks for the intersection:

| Camera | `cinematography` | `style_medium` | View-specific action sentence |
| --- | --- | --- | --- |
| `cctv` | Static, high wide shot from a pole, looking down at the intersection | "CCTV security camera footage" | All four cars and three pedestrians; lights change at about 3 s and 4.5 s |
| `dashcam` | Forward-facing, mounted behind the windscreen of a stopped car, hood at the bottom, moves forward at the end | "Dashcam footage" | Two cars cross in front left↔right, two pedestrians cross just ahead, the light ahead turns green, the car drives forward |
| `phone` | Handheld smartphone at eye height, slight natural shake, held landscape | "Smartphone video" | Cars crossing ahead, pedestrians close on the right, waiting cars drive off |

The cut video's prompt has the same world block. Its cinematography reads: "Three shots joined by hard cuts: first about 2.3 s from a fixed CCTV camera high on a pole, then about 2.5 s from the dashcam of a car waiting at the red light, then about 2.2 s from a bystander's handheld smartphone." Its style is "real-world footage edited from three cameras". Whether Cosmos respects a described cut schedule is untested; the Round 2 drone cut suggests the controls set the cuts whatever the text says.

Both scenes end every prompt with the same realism sentence: "Realistic real-world video, natural lighting and motion, not a rendering." The full composed `prompt.json` is saved in each run folder.

The action sentences are written by hand (or by the LLM in session) from the simulator's timeline and frozen. Generating them automatically from `timeline.json` + `visibility.json`, as VideoCoCo's instruction agent does from the draft, is a later step.

### 6.7 Matrix

| Set | Count |
| --- | --- |
| Separate views: 2 scenes × 3 cameras × 2 seeds (0, 1), condition `V` | 12 |
| Cut videos: 2 scenes × 1 cut × 2 seeds, condition `V` | 4 |
| Probe: blur sweep on the intersection CCTV view | 4 |
| **Total** | **20 clips** |

At 12–16 min per clip, that is about 4–5.5 GPU-hours, or 1–1.5 h of wall time on four free cards. As always, check `nvidia-smi` right before launching and run one job per card.

The seeds are the same across views, so a seed's initial noise is the same for all three cameras (as in Round 2). This is recorded rather than varied. Its effect on cross-view agreement would need a per-view seed offset, which is a later probe.

## 7. Scoring

The simulator knows the truth for every view, so most checks need no labelling. All numbers are written to `scores/` with the run IDs they came from.

| Check | How | Applies to |
| --- | --- | --- |
| Object colour per view | HSV label inside each object's eroded simulator mask (from Round 2's `score_colour.py`, extended to more colour names) | Every object visible ≥ 10 frames |
| Cross-view identity | The same object gets the same colour label in all three separate views | Separate views |
| Across-cut identity | The same object gets the same label before and after each cut | Cut videos |
| Light state | Lit-lamp colour at the lamp's projected pixels, compared with the simulator's state per frame | Views where the lamp ≥ N px (§5.1) |
| Count | Number of cars and pedestrians in the generated video vs the simulator | By eye; SAM2 seeded with simulator boxes later |
| Outline agreement | Edge F1 between Canny of the output and Canny of the simulator render | Every clip; a weak proxy for "followed the control" |
| Rubric (by eye) | Realism 1–5; the event happens (y/n); count right; colours right; background matches the other views; artefacts (free text) | Every clip, in `scores/rubric.csv` |

The automatic checks are coarse. As in Round 2, the findings are read by eye first and the scores back them up.

## 8. Assets

**Why assets at all.** Cosmos follows the silhouettes in the depth and segmentation controls. A car built from three boxes comes back as a boxy car, and a pedestrian needs a rigged body whose legs actually swing. A robot arm only reads as a Franka if its links have a Franka's shapes. Cars could be procedural at some cost in realism. Walking people and a recognisable arm are hard to make procedurally.

**Licences.** CC0 is a public-domain dedication: any use, no attribution, and no restriction on government or military use. Apache-2.0 allows any use too. If we redistribute the files, its licence and NOTICE must go with them, and it includes a patent grant. MuJoCo Menagerie releases its Panda model under Apache-2.0; the visual meshes are converted from the `franka_ros` meshes, which are also Apache-2.0. Neither licence has the clauses that ruled out BeamNG or Unreal (military or governmental use excluded). Whether the AFRL contract needs third-party assets listed or approved is a question for the PI, not the licences. The assets stay on `/mnt/Data` with a `SOURCE.md` and are not committed to the repo; the generated videos are our renders.

| Need | Plan | Licence |
| --- | --- | --- |
| Cars | Low-poly CC0 car packs (Kenney Car Kit, Quaternius cars), with separate wheels so they can spin. Fallback: procedural car (body, cabin, 4 wheels) in `sim/assets.py` like `build_truck`. | CC0 |
| Pedestrians | Rigged, animated CC0 characters (Quaternius animated packs), or MakeHuman/MPFB2 bodies (assets CC0) with CMU mocap walk cycles | CC0; CMU mocap may be used freely but not resold |
| Robot | Franka Panda visual meshes from `franka_description` or MuJoCo Menagerie | Apache-2.0 |
| Pitcher, glass, bin, block, table, room | Procedural in `sim/assets.py` | Ours |
| Road, kerbs, crosswalks, buildings, traffic lights | Procedural | Ours |

Not used:

- BlenderProc's AMASS loader for walking people. SMPL bodies are licensed for non-commercial research only; check that before any AFRL use.
- BlenderProc's URDF loader. It installs `urdfpy` from GitHub inside Blender's Python, and our numpy IK plus per-link meshes avoids that dependency.

Downloaded assets go under `$S2R_DATA/assets/<pack>/` with a `SOURCE.md` (URL, version or date, licence, sha256). Specs refer to them by pack and file name.

## 9. Making everything trackable

Rule: **text that decides an output lives in git; large files live on `/mnt/Data`; every output folder records the hashes of everything that made it.**

### 9.1 In the repo (git)

```
configs/scenes/intersection_4way.yaml          scene spec, schema v2 (rev in the file)
configs/scenes/intersection_4way.source.md     user prompt; who wrote the spec (this week: Claude Code, claude-opus-5-5,
                                               with the date and the instruction it was given); change log, one line per rev
configs/scenes/robot_pour.yaml, .source.md
configs/scenes/robot_pickdrop.yaml, .source.md (fallback)
configs/prompts/multicam_v1/<scene>/world.json        shared caption block
configs/prompts/multicam_v1/<scene>/cams/<cam>.json   per-camera caption block
configs/prompts/multicam_v1/<scene>/cut3.json         cut-video caption block
configs/experiments/multicam_v1.yaml           scenes + revs, model + revision, video and sampler settings,
                                               conditions, cut schedules, seeds, probes
docs/2026-11-18-1739-multicam-v1-design.md     this doc
```

Excerpt of schema v2, the parts that are new beyond the Round 2 specs:

```yaml
spec_version: 2
name: intersection_4way
rev: 1                                   # bump on any change; outputs of rev 1 are never overwritten
duration_s: 7.1                          # 213 frames at 30 fps (1 + 4n); Cosmos gets 113 at 16 fps
fps: 30
frames: 213
resolution: [832, 480]
seed: 0

world: {type: intersection, lanes_per_direction: 1, lane_width_m: 3.5, leg_length_m: 60,
        sidewalk_m: 3.0, crosswalk_m: 3.0, stop_line_m: 8.0, buildings: {height_m: [8, 20], seed: 3}}

signals:
  - {id: tl_NS, approaches: [north, south],
     plan: [{state: green, until_s: 3.0}, {state: yellow, until_s: 4.0}, {state: red, until_s: 7.2}]}
  - {id: tl_EW, approaches: [east, west], plan: [{state: red, until_s: 4.5}, {state: green, until_s: 7.2}]}

objects:
  - id: car_ego
    instance_id: 3
    asset: {type: car, pack: kenney_car_kit, model: hatchback, color: [0.12, 0.12, 0.13]}
    motion: {model: ground_vehicle, lane: east_in, through: true, start_m: 49.7, start_mps: 0.0,   # front on the stop line
             signal: tl_EW, yield_to: [ped_red, ped_white], params: {cruise_mps: 9.0, accel_max: 2.5}}
  - id: ped_phone
    instance_id: 7
    asset: {type: pedestrian, pack: quaternius_people, model: casual_01,
            colors: {top: [0.10, 0.45, 0.15], bottom: [0.08, 0.08, 0.12]}}
    motion: {model: pedestrian, idle_at: [6.5, -6.5], facing_deg: 135, sway_m: 0.02}
    hide_from: [phone]                   # not rendered by its own phone camera

cameras:
  - {id: cctv, type: fixed, position: [9.0, 9.0, 7.0], look_at: [-2.0, -2.0, 0.0], fov_deg: 80}
  - {id: dashcam, type: attached, parent: car_ego, offset: {pos: [0.6, 0.0, 1.25], pitch_deg: -4}, fov_deg: 100}
  - {id: phone, type: handheld, parent: ped_phone, offset: {pos: [0.25, 0.0, 1.45]}, look_at: [-3.0, 3.0, 1.0],
     shake: {rot_deg: 0.6, pos_m: 0.004, hz: [1.0, 3.0], seed: 11}, fov_deg: 66}

checks:
  - {type: min_gap, metres: 1.0}
  - {type: obeys_signals}
  - {type: yields_to_pedestrians}
  - {type: event_order, events: [tl_NS_yellow, tl_EW_green, car_ego_moves]}
  - {type: visible, camera: dashcam, objects: [car_red, car_white], min_frames: 20}
```

Excerpt of the experiment file:

```yaml
experiment: multicam_v1
scenes: {intersection_4way: {rev: 1, cut: cut3}, robot_pour: {rev: 1, cut: cut3}}
model: {name: cosmos3-nano, hf: nvidia/Cosmos3-Nano, revision: 45666784803b3a7ad577e0def405f9afcc3bddb2,
        framework_commit: 5950e7b4ea3677b99a2de279a7ed7058b5446e42}
video: {num_frames: 113, fps: 16, resolution: "480", aspect_ratio: "16,9"}
sampler: {num_steps: 50, guidance: 3.0, control_guidance: 1.5, shift: 10.0}
first_frame: none                        # num_first_chunk_conditional_frames: 0
conditions:
  V:  {blur: {preset_blur_strength: none}}          # vision_path = the camera's (or cut's) simulator rgb.mp4
  # later: G:  {depth: {weight: 1.0}, seg: {weight: 1.0}}
  # later: GA: {depth: {weight: 1.0}, seg: {weight: 1.0}, blur: {weight: 1.0, preset_blur_strength: low}}
cuts:                                    # output-frame ranges (16 fps); shots start at 1 + 4k
  intersection_4way: {cut3: [{cam: cctv, frames: [0, 36]}, {cam: dashcam, frames: [37, 76]}, {cam: phone, frames: [77, 112]}]}
  robot_pour:        {cut3: [{cam: side, frames: [0, 36]}, {cam: wrist, frames: [37, 76]}, {cam: cctv, frames: [77, 112]}]}
seeds: [0, 1]
probes:
  blur_sweep: {scene: intersection_4way, view: cctv, cond: V, seed: 0, preset_blur_strength: [very_low, low, medium, high]}
```

Shot order: the intersection opens on the wide CCTV shot (0–2.25 s), while the north–south cars approach and enter the junction. It goes to the dashcam (2.3–4.75 s) as they cross in front, the pedestrians cross just ahead, the north–south light turns yellow and red, and the light ahead turns green. It ends on the phone (4.8–7 s) as the waiting cars drive off. The robot opens on the side view, cuts to the wrist for the stream, and ends on CCTV as the arm returns.

### 9.2 On the data disk

```
$S2R_DATA/poc/multi-cam/<scene>/r<rev>/
  world/                 poses.npz, signals.json, events.json, checks.json, meta.json
  cams/<cam>/bundle/     rgb.npy/mp4, depth.npy, instance.npy, timeline.json (camera pose per frame),
                         boxes.json, visibility.json, meta.json
  cams/<cam>/controls/   depth.mp4, seg.mp4 (one fixed colour per instance id, the same in every camera), rgb.mp4
  cut/cut3/              the same controls and rgb/instance joined by the schedule, plus schedule.json
  gen/<cam | cut3>/<cond>/cosmos3-nano/seed<k>/
                         request.json, prompt.json, negative_prompt.json, out_video.mp4, run.log, run.json
  scores/                colour.csv, lights.csv, edge_f1.csv, rubric.csv
$S2R_DATA/poc/multi-cam/manifest.jsonl    one line per run, appended when the run finishes
```

`run.json` records:

- the run ID (`multicam_v1/<scene>/r<rev>/<view>/<cond>/seed<k>`)
- the git commit and whether the tree was dirty (with a hash of the diff if so)
- the sha256 of the spec, the composed `prompt.json`, the experiment file and every control video
- the model ID and revision, and the cosmos-framework commit
- every request field
- the GPU index and name, start and end times, peak memory, and exit status

The renderer already writes the spec's sha256 into `meta.json`; this extends that to every step.

### 9.3 Rules

- A scene spec change bumps `rev` and writes to a new `r<rev>/`. Old outputs stay.
- Changing a prompt or the experiment file after the matrix has started makes a new experiment version (`multicam_v1.1`), not an edit.
- The runner skips a run only if `run.json` exists **and** every input hash still matches. A mismatch is reported as stale, never silently reused. The current `run_matrix.py` checks only that the output exists.
- The commit used to run the matrix is tagged `multicam_v1-run`.

## 10. What was built

Everything below exists as of 2026-11-18. The Round 2 pipeline (`sim/render_scene.py`, `gen/run_matrix.py`) is untouched, so Round 2 stays reproducible.

**Simulator: `sim/world/`, spec v2.** One numpy simulation computes, for every frame, the pose of every part of every object and every camera. Blender only renders it, one camera per run.

| File | What it does |
| --- | --- |
| `sim/world/spec.py` | Loads and checks a spec v2: frames 1 + 4n, unique ids, cameras, `hide_from` |
| `sim/world/layout.py` | World layouts in numbers: the intersection (lanes, stop lines, crosswalks, signal head positions, kerb heights) and the room (table) |
| `sim/world/signals.py` | Fixed-time signal plans |
| `sim/physics/lane_vehicle.py` | Cars on a lane: IDM; a red light or a pedestrian on the crosswalk acts as a stopped virtual car; stop-or-go on yellow |
| `sim/physics/pedestrian.py` | Walking a path at a set pace, or standing / holding a phone |
| `sim/physics/arm.py` | Robot arm: keyframes for a point on the held object (the pitcher's lip, yaw, tilt), the tool point from those, damped-least-squares IK on the 7 joints, joint limits, null-space pull to home |
| `sim/physics/pour.py` | The analytic pour: weir flow over the lip, ballistic stream, glass level |
| `sim/world/mjcf.py` | Reads a MuJoCo Menagerie MJCF (body tree, joints, visual meshes); forward kinematics |
| `sim/world/kinematics.py` | Part poses from motion state: car (body pitch, rolling wheels), person (articulated walk cycle, feet kept on the ground, phone pose), robot (one part per link plus the tool point), rigid |
| `sim/world/cameras.py` | Camera poses: fixed (look-at), attached (rides a part; forward and up axes in the part's frame), handheld (rides a part, aims at a point, deterministic shake) |
| `sim/world/checks.py` | min_gap, obeys_signals, yields_to_pedestrians, event_order, plus robot checks from the models (ik_error, joint_limits, stream_in_glass, final_level); `visible` is checked after rendering |
| `sim/world/simulate.py`, `sim/simulate_world.py` | Runs it all and writes `world.npz` + `world.json` (events, checks, a framing report per camera, git state, spec hash) |
| `sim/world/blender/` | Blender builders: materials (Poly Haven PBR, paint, glass, lamps), street (road, markings, sidewalks, buildings with windows, signals, props, skyline), car, human, room, robot (Menagerie meshes), vessels (pitcher, glass, water column, per-frame stream) |
| `sim/render_world.py` | Renders one camera from `world.npz`: RGB, depth, instance masks (0 sky, 1–99 objects, 100+ static parts, named in `static_ids.json`), `timeline.json` |
| `sim/world/export.py`, `sim/export_world.py` | Per camera: mp4, boxes, visibility, point tracks with moving cameras, overlay, and the `visible` checks |
| `sim/build_world.sh` | Simulate once, then render and export each camera, in parallel on the GPUs given |

**Generation and scoring**

| File | What it does |
| --- | --- |
| `gen/multicam_common.py` | Job list (cameras + cuts + probes), folder layout, prompt composition (world block + camera block), and the input recipe that decides whether a run is current |
| `gen/multicam_inputs.py` | In the Cosmos venv: joins a cut's frames by its schedule, builds the blur control with Cosmos's own blur code (`preset none` = the video as rendered), and writes `request.json` with `blur.control_path` and no `vision_path`, so no first frame can be used |
| `gen/run_cosmos_multicam.sh` | One run: inputs, then Cosmos3-Nano inference |
| `gen/run_multicam.py` | Scheduler: GPU free-memory checks, one job per card, `run.json` (recipe hashes, git state, GPU, wall time, peak memory), `manifest.jsonl`, stale detection, STOP file |
| `eval/score_multicam.py` | Colour per object vs the simulator, cross-view and across-cut agreement, traffic-light state |
| `setup/fetch_assets.py`, `configs/assets/polyhaven.yaml` | The CC0 Poly Haven assets, each with a `SOURCE.md` (URLs, sha256) |
| `setup/clone.sh` | Now also a sparse clone of MuJoCo Menagerie (`franka_emika_panda` only), pinned |

**Changes from the plan above, made while building:**
- The renders run at 16 fps directly (113 frames), not 30 fps resampled.
- The blur control is always a file built with Cosmos's code and passed as `control_path`, rather than `vision_path` plus a preset. It's the same control either way, but this way there is no source video that could act as a first frame, and the control file is saved and hashed.
- The CCTV camera sits 15 m up on the north-east building's corner: at 9 m the corner signal pole blocked the junction.
- The dashcam sits at the top of the windscreen, 4° up, so the hood is a strip at the bottom.
- The robot grips the pitcher 3 cm below its rim, so the wrist camera looks over the rim at the glass.

## 11. Order of work and gates

1. **Specs and prompts.** Write the two specs, the `.source.md` files, the prompt blocks and the experiment file. Review them before any rendering.
2. **Intersection world.** Assets, signals, cars, pedestrians, then a low-sample preview render from all three cameras. **Gate A:** the checks pass, and in the three previews (output frames 0, 60, 120) the same objects are in the same places at the same times.
3. **Cosmos smoke test.** 113 frames, `V`, 4 steps, on the intersection CCTV view. **Gate B:** it fits in memory, the time is measured, and output frame 0 is not the simulator frame 0. If it fails, use the §6.4 fallback.
4. **Intersection matrix.** 8 clips plus the 4 blur-sweep probes. This can run while step 5 is being built.
5. **Robot world.** The arm controller and the level-1 pour. **Gate C** (mid-week): if the arm IK or the pour does not pass its checks, build `robot_pickdrop` instead.
6. **Robot matrix.** 8 clips.
7. **Scoring and write-up.** Scores, a findings doc in `docs/`, and a section on the Round 2 page or a new page. A new page gets a row in the README Artifacts table.

## 12. Decisions

| Question | Decision (2026-11-18) |
| --- | --- |
| Control | The simulator's colour video, unblurred (`V`); depth + segmentation conditions later |
| Clip length | 7 s for every clip: 113 frames at 16 fps, rendered at 16 fps; cut shots ≈2.3 s each |
| Pour | Level-1 analytic stream; other liquid models plug in later behind `sim/physics/pour.py` |
| Spec author | Claude Code (claude-opus-5-5) in session, recorded in each `.source.md`; a re-runnable API script comes later |
| Assets | CC0 packs and the Apache-2.0 Panda meshes, as their licences allow (§8); PI to confirm the contract has no third-party-asset rule. SMPL/AMASS not used. |

## 13. How to run

Every command assumes `source ~/.sim2real.env` and the repo as the working directory. Check `nvidia-smi` first; renders need a few GB per card, Cosmos needs about 50 GB.

```bash
PY=$S2R_ROOT/envs/eval/.venv/bin/python
# once: assets and the robot model
$PY setup/fetch_assets.py
bash setup/clone.sh
# 1. simulate a scene and look at its checks and framing (seconds, no GPU)
$PY sim/simulate_world.py configs/scenes/intersection_4way.yaml $S2R_DATA/poc/multi-cam/intersection_4way/r1/world
# 2. preview a camera: three frames, few samples
blenderproc run --blender-install-path $S2R_ROOT/blender sim/render_world.py configs/scenes/intersection_4way.yaml \
    $S2R_DATA/poc/multi-cam/intersection_4way/r1/world cctv /tmp/preview --frames 0,56,112 --samples 16
# 3. build: simulate once, render + export every camera (one GPU per camera)
bash sim/build_world.sh configs/scenes/intersection_4way.yaml $S2R_DATA/poc/multi-cam/intersection_4way/r1 0,3,5
# 4. generate: status, smoke test (4 steps, gen_smoke/), then the experiment
(cd gen && $PY run_multicam.py --exp configs/experiments/multicam_v1.yaml --status)
(cd gen && $PY run_multicam.py --exp configs/experiments/multicam_v1_smoke.yaml)
(cd gen && $PY run_multicam.py --exp configs/experiments/multicam_v1.yaml)
# 5. score
$PY eval/score_multicam.py --exp configs/experiments/multicam_v1.yaml
```

A new scene is a new spec (`configs/scenes/<name>.yaml` with a `.source.md`) plus prompt blocks (`configs/prompts/<experiment>/<name>/`) and a line in an experiment file. A new kind of object, camera or world type is one model, kinematics function, mesh builder or layout (see `sim/world/__init__.py`).

## 14. Revision after the first results (2026-11-18): clay renders, structure controls

**What the V runs showed.** With the textured simulator video as the control (blur slot, preset `none`), Cosmos3-Nano copied the render almost pixel for pixel at 50 steps. The blur presets up to `high` changed textures a little but kept the simulator look. The blur slot is trained as a blurred copy of the target video, so an unblurred control tells the model "the answer looks like this". V is a dense appearance control, not a physics control. The 20 V clips are kept as the baseline (`r1/gen/`).

**What changed.** The simulator video now carries only structure and motion, and the appearance comes from the prompt:

| | Before (multicam_v1, scene rev 1) | Now (multicam_v1.1, scene rev 2) |
| --- | --- | --- |
| Render | Textured (Poly Haven asphalt, brick, plaster; car paint; glass) | Clay: one matte grey on every surface; only the signal lamps keep their colour |
| Controls exported per camera | rgb only (depth and instance masks were saved as arrays, not as control videos) | `controls/depth.mp4`, `seg.mp4`, `edge.mp4`, `normals.mp4`, `rgb.mp4` (clay), plus `controls_sheet.jpg` (sim/world/controls.py); a cut gets the same, joined by its schedule |
| Conditions | V: the render in the blur slot | G: depth + seg. GE: depth + seg + edge. No colour from the simulator. |
| Where appearance comes from | The render | The prompt (world block: paint, clothes, buildings, lighting) |

The controls in detail:

- **depth:** inverse depth with one scale per clip, near bright, sky black.
- **seg:** one fixed colour per instance id. It covers the static parts too (road 100, sidewalk 101, markings 102, each building, each signal, props), so the segmentation covers the whole frame as in Cosmos's training data, and an object has the same colour in every camera.
- **edge:** only object boundaries, part and paint boundaries (from the ids), depth steps and creases (from the normals). There are no texture edges, unlike Canny on a textured render.

**What the model gets, and where it is written down.** The exact caption for each view is in `configs/prompts/multicam_v1/<scene>/composed/<view>.json`, written by `gen/compose_prompts.py` from the world and camera blocks. The negative prompt is `configs/prompts/cosmos3_negative_prompt.json`, a frozen copy of the Cosmos cookbook's. The runner refuses to start a job whose composed file is missing or out of date. Each run folder also keeps its `prompt.json`, `negative_prompt.json` and `request.json`, with their hashes in `run.json`.

**Known limit.** With structure-only controls, the traffic-light state reaches the model only through the prompt ("the north-south lights turn amber at 0:03"). The lamps stay coloured in the clay render, so a later sparse colour control can carry them: for example, a heavily blurred render with only the lamps and the objects' intended colours.

**Bookkeeping.** The rev-1 specs that the V runs used are kept, byte for byte, in `configs/scenes/history/<scene>.r1.yaml` (their hashes match the r1 `world.json`), and `multicam_v1.yaml` points at them. From rev 2 on, every simulated world also keeps its spec as `world/spec.yaml`.
