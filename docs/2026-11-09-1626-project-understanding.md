# Project Understanding and Scoping (as of Nov 9, 2026)

**Owner:** SP Sharan · **Status:** working notes. This is how the project reads to us before the literature survey. It builds on the [v0 design doc](2026-11-09-1416-design-doc-v0.md) and the notes added to the end of its Google Doc tab.

## 1. The question

Can text-to-video (T2V) models, video generators, and world models make more *faithful* videos if a simulator runs first?

```
Baseline A:  text                                  -> video model -> video
Baseline B:  text + first frame                    -> video model -> video
Ours:        text -> simulator (layout + motion)   -> video model -> video
```

"Faithful" means the right number of objects, in the right places, moving the right way, with events in the right order, and with plausible physics. Current models get these wrong when they work from text alone, especially when an object leaves the frame or comes back into it. We already have one-line experiments that show this; they are our motivating result.

The simulator has two jobs:

1. **Planner.** It fixes the motion and layout explicitly, so the video model doesn't have to work them out.
2. **Ground truth.** It gives us counts, poses, tracks, and event order for free, so we can score any generated video against it. This is why the dataset and evals come first.

## 2. Scoping experiment: three videos

We start with three test videos, not the full factor grid.

| # | Scene | What we test |
|---|---|---|
| 1 | Drones take off from the ground and fly somewhere | Take-off and climb, counts, heading, ground contact before launch |
| 2 | Drones moving in the air (ground-to-air or air-to-air view) | Relative motion, crossing and converging paths. The drones are the main subject and clearly visible, not tiny |
| 3 | Satellite view of a convoy that drives along a road, then turns right onto a field | Count, consistent heading, staying on the road and then leaving it at the right moment, fixed background |

For each video we run every conditioning setting (text only, text + first frame, simulator-conditioned) and score all of them against the simulator's ground truth.

## 3. An open design question: what does the video model receive from the simulator?

There is a spectrum of what the simulator can pass to the video model:

```
dense <--------------------------------------------------------------------> sparse
rendered RGB video   depth / seg / edges   3D boxes / tracks   trajectories / poses   text only
(Cosmos-Transfer,    (ControlNet-style     (driving world      (motion-control        (Baseline A)
 VACE)                video models)         models)             video models)
```

- **Dense end (rendered video).** The simulator does nearly all the work, and the video model mostly makes it look real. This is close to video-to-video *style transfer*. It should be the most faithful, but it adds the least new content or physics, and it is only as good as the render.
- **Sparse end (scene state).** The model gets the motion (boxes, tracks, trajectories, camera path) but not the pixels, so it has to actually *generate* the appearance, lighting, and secondary physics such as dust, shadows, and rotor blur. This is closer to "world model + planner" and is the more interesting research claim, but it is harder to follow.

We don't know yet which open models accept which level, or how faithful each level is. The literature survey will answer that. Ideally the scoping run includes at least one dense and one sparse setting, so we can show where the sweet spot is.

## 4. What we want by Tue 3 PM

1. **Best pairing of simulator and model** (world model, T2V, or X2V) for each of the three videos. RM and SJ are asking Guandou.
2. **Main prior work** in the field, starting from [CoDeR](https://becauseimbatman0.github.io/CoDeR) ("Code Plans, Diffusion Renders: Open-Ended Generative World Modeling"; Fang et al., USTC + Alibaba). In CoDeR, LLM coding agents build the world as code and video models render it.
3. **Metrics** that can judge sim-faithfulness, prompt fidelity, realism, and physics.
4. Start writing the paper or report doc by Wed 3 PM.

## 5. Literature survey plan (SP)

General domain first, then narrow down to drones and satellite.

| Bucket | Question it answers |
|---|---|
| Simulator / code / physics-in-the-loop video generation (CoDeR, PhysGen, GPT4Motion, ...) | Has this been done? What is closest to us? |
| Dense-control video generation and sim2real rendering (Cosmos-Transfer, VACE, ...) | The "style transfer" end. Which open models take rendered passes? |
| Sparse scene-state conditioning (trajectories, boxes, layouts, actions) | Can a model take the sim's state and motion directly, with no render? |
| Evals for motion and physics fidelity | Evidence that T2V fails at motion; metrics we can reuse |
| Drone and satellite generation, simulators, and pairings | Which tools to use for the three videos |

Output: [lit survey v0](lit-review/2026-11-09-1650-lit-survey-claude-opus.md), which will be pasted into a new Google Doc tab.

## 6. Assumptions and risks

- **Training:** none in v0. We use off-the-shelf open models on the swarm A100s (80 GB each).
- **Isaac Sim:** it does not run on A100s, which have no RT cores, so it is ruled out on swarm unless we find an RTX machine. BlenderProc and CARLA-Air are the A100-friendly options. *Update (Nov 9):* swarmllm (4× RTX 6000 Ada) is an RTX machine, so Isaac Sim may run there. Untested; see the [Cosmos hardware check](2026-11-09-2309-cosmos-hardware-check.md).
- **Drone size:** in all three videos the objects are the main subject and clearly visible (drones at least about 30 px). Tiny targets (<3 px) are out of scope for v0: they are smaller than one VAE latent cell, and off-the-shelf detectors would make the count and track metrics unreliable. They can come back in the full factor grid later.
- **Rendered-video conditioning:** if Cosmos-Transfer or VACE simply reproduce the render, high faithfulness scores are trivial. We need the realism axis (and the sparse setting) to show added value.
- **Baseline B fairness:** the first frame should be the simulator's first frame (or a real frame of the same layout), so that every setting starts from the same scene.

## 7. This week's experiments (by Fri Nov 13)

Revised after the [literature review synthesis](lit-review/2026-11-09-1750-lit-review-synthesis.md) (§5–6 there). The reasoning is in that synthesis.

**Framing.** This is an *adherence audit*, not a claim that "a simulator makes video generation better." When a model is handed the full future motion and then scored against that same motion, the score is close to circular. The real question is how well released video models follow what they are told, compared with what their papers advertise, for drones and overhead convoys. Every controlled model gets a **claimed vs. observed** row: the control and accuracy its paper reports next to what we measure. Treat the output as a *renderer* of the simulator's motion. v0 makes no world-model, physics, or memory claims.

**Videos.** Each one is a deterministic BlenderProc scene with its timeline, camera, object IDs, visibility, projected tracks, RGB, depth, edges, and masks saved. Motion is scripted, so call it "kinematically controlled" unless a physics engine drives it (gym-pybullet-drones is optional for take-off).

1. Drones take off from the ground and fly somewhere.
2. Drones move in the air, clearly visible, crossing or converging.
3. Satellite view of a convoy on a road that turns right onto a field. Fixed near-orthographic camera.

**Conditions, run on each video.** A–C use the same Wan2.1-VACE checkpoint (14B if it passes the smoke test, otherwise 1.3B for all of them). Only the input changes.

| | What the model gets | Model | Purpose |
|---|---|---|---|
| R | The raw simulator render | — | Reference for motion and realism |
| A | Text only | VACE | No-information baseline |
| A+ | Text + a detailed description of the trajectory | VACE | No-simulator planner: does the simulator add anything beyond writing the plan down? |
| B | Text + the simulator's first frame | VACE | Effect of the starting layout and appearance |
| C | B + the simulator's control video (depth + edges for drones; edges/blur for the convoy, where overhead depth is nearly flat) | VACE | Effect of dense control on the same model |
| D | Cosmos3-Nano-Sim-Transfer with the same scene's control video plus text | Cosmos | Practical sim-to-real pipeline (main candidate) |
| E | B + projected surface-point tracks and visibility | Wan-Move | Sparse simulator state, no rendered video |

**Why Sim-Transfer for D** (changed from Cosmos-Transfer2.5 on Nov 9). It is the newer model (Cosmos 3) and is trained on simulator renders, while Transfer2.5 is reported to degrade on them. It ran on one shared swarm5 A100 in about 2 min per clip with 4 distilled steps. Its OpenMDW licence has no guardrail clause, whereas Transfer2.5's licence ends if the guardrail is disabled. It takes only one control video plus text per sample. Details are in the [Cosmos hardware check](2026-11-09-2309-cosmos-hardware-check.md).

**Candidate: VACE box layout as a same-model sparse condition (needs a test).** VACE can take a control video of box outlines drawn on white frames, with the real first frame at frame 0 (its `move_anything` mode). We can draw the simulator's per-drone boxes ourselves instead of running VACE's detector. That would put dense (C) and sparse on the same checkpoint, which the lit review flags as the missing comparison. This comes from reading the repo and docs only. Nothing has been run, multiple objects per control video aren't confirmed, and the Fable survey cites open issues saying VACE's box-track mode is broken. A short run with our own box video on swarm5 will settle it. Until then, Wan-Move (E) stays the sparse condition.

Each video gets **3 seeds** per condition: 3 videos × 6 generated conditions × 3 seeds = 54 clips, plus 3 raw renders. Every sample is reported, failures included. Resolution, length (about 5 s), and frame rate are matched, and prompts are frozen. If time allows, add Wan2.2-TI2V-5B text-only and first-frame runs as an outside reference.

**Metrics, for each clip:**

- **What the prompt asks for (all conditions):** count, direction, and events. For example, takeoff happens before the drones move off; the convoy stays on the road before the right turn, then is on the field. Checked with NeuS-V-style specs over audited observations. Text-only is *not* penalized for taking a different valid path.
- **Adherence (controlled conditions only):** position error normalized by object size, heading through the convoy turn, and identity switches. Missing or extra objects are reported separately, so they don't disappear from the averages.
- **Background and camera stability:** drift of fixed landmarks and any unintended camera motion. This matters most for the overhead scene.
- **Realism:** blinded side-by-side preference against the raw render, plus an artifact checklist. FVD/FID are not headline numbers at this sample size.
- **Cost:** wall time, peak GPU memory, retries, and setup effort.

Human annotation is the primary measurement, assisted by SAM 2 / CoTracker. Each clip gets about 8 keyframes plus event boundaries, and every failure is audited. Tolerances are fixed in advance (e.g., within half a body length and 0.25 s of the event time) and checked on the raw render before any generator is judged.

**Deliverable:** for each video, one score table (one row per condition, one column per metric) plus a claimed vs. observed table. Also a video grid showing prompt, first frame, control input, outputs, and a trajectory overlay, with failures included, and a short findings memo.

**If time is short, cut in this order:**

1. The Wan2.2 outside reference.
2. Wan-Move (E).
3. Cosmos (D).

The aim is to run every condition, since we have several people on this, so the order is only a fallback. One alternative: cut Cosmos (D) before Wan-Move (E). D is a second dense condition, while E is the only sparse one, and dense vs. sparse is the comparison the lit review says is new.

The minimum experiment is 3 videos × VACE {A, B, C} × 3 seeds = 27 clips, plus raw renders and audited measurements.

**Schedule:**

| By | Task |
|---|---|
| Tue 10, 3 PM | One end-to-end smoke test (Blender → VACE and Cosmos on a custom scene). Lock pairings, the three scene specs, the condition matrix, and metric definitions. Assign an owner to each video. Check which swarm GPUs have the ~65 GB free that Cosmos needs. *Update (Nov 9):* both Cosmos 3 models run on swarm5's A100s, and Sim-Transfer needs only about 35 GB ([Cosmos hardware check](2026-11-09-2309-cosmos-hardware-check.md)). What's left is the run on our own Blender scene, plus the VACE box-layout test. |
| Wed 11, 3 PM | Simulator exports for all 3 videos. Annotation templates and the eval harness. Start the report: related work, hypotheses, condition table, and empty result tables. |
| Thu 12 | Run the frozen matrix. Audit tracks and event labels. Blinded realism comparisons. |
| Fri 13 | Score tables, claimed vs. observed, video grid, and findings memo. |

**Later (not v0):** objects leaving and re-entering the frame, and testing memory and world-model prediction (past frames + actions → future). If the simulator supplies the unseen interval, a correct re-entry shows the simulator's state, not the model's own memory. Also later: physics-engine vs. scripted motion, tiny targets, and downstream training on real data.
