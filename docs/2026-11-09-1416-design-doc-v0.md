# Sim-Conditioned Video Synthesis for Drone and Satellite Scenarios: Design Doc v0

**Owner:** SP Sharan (co-lead with Shourov) · **Thread:** AFRL Thread 1 (video generation, evals) · **Drafted:** Nov 9, 2026 · **v0 target:** Fri Nov 13, 2026

## 1. Problem

Simulators can make any scenario we want, but the footage looks synthetic and misses physics the simulator doesn't model. Real footage looks right, but we can't film the scenario we need (for example, three drones converging at under 3 px). We want a generator that combines the two:

```
text prompt (new scenario) + real images of the target domain (optional) + simulator video  ->  realistic synthetic video
```

**Why:** it can make new scenarios that stay faithful to the sim layout, runs faster than a full-fidelity render, covers physics the simulator doesn't model, and closes the sim2real gap.

**Principle (Sandeep):** we can't call a video "good" without a testbench, so **the dataset and evals come first**. v0 is a small end-to-end version of the whole pipeline, not a new model.

## 2. Scope for v0: two narrow use cases

| | **A. Drones (ground-to-air / air-to-air)** | **B. Satellite (overhead)** |
|---|---|---|
| Scene | 1–3 small multirotor drones against sky, treeline, or urban backgrounds | Ground vehicles or convoys moving through fields or roads, seen from orbit |
| Example prompt | "3 drones; 2 far off (<3 px) heading towards each other, 1 close crossing left-to-right; overcast sky" | "This field, but with a convoy of 3 tanks moving along the dirt road from north to south" |
| What makes it hard | Tiny objects, motion direction, counts, relative distance | Object count, consistent heading, staying on the road, keeping the background fixed |

Out of scope for v0: e-commerce, audio, long videos (more than about 5 s), and training our own model.

## 3. Pipeline

```
Scenario template ──► Simulator render ──► Conditioned generators ──► Evaluators ──► Scores + leaderboard
 (factor grid)        (RGB + depth/seg     (K open models,            (fidelity, quality,
                       + GT boxes/tracks)   varied conditioning)       realism, physics)
```

The simulator gives us **ground truth for free**: object count, positions, trajectories, and the order of events. The evaluators check generated videos against that ground truth, so they don't have to rely on a VLM's opinion. This is where the neuro-symbolic verifiers (NeuS-V/NeuS-E, temporal-order checks) fit in.

## 4. Scenario templates (dSprites-style factor grid)

Each prompt is a templated cross product, so results can be broken down factor by factor.

| Factor | Drones | Satellite |
|---|---|---|
| Background | clear sky, overcast, treeline, urban, desert | field, desert, urban grid, forest edge, snow |
| # objects | 1, 2, 3 | 1, 3, 5 (convoy) |
| Size / distance | far (<3 px), mid, near | fixed GSD (ground sample distance); vehicle length in px |
| Motion | approaching, crossing, converging, hovering | straight, turning, stopping |
| Lighting / time | noon, dusk, low sun, haze | noon, oblique sun, cloud shadow |

v0 target: **about 30 prompts per domain** (a sampled subset of the grid), each with a sim render. The full grid comes later and becomes the "DoD Video Bench" deliverable.

## 5. Simulator (TBD: search in progress)

Any simulator that can export RGB together with depth, segmentation, and object poses will work. It does not need to be Unreal.

| Option | Good for | Notes |
|---|---|---|
| AirSim forks (e.g. Cosys-AirSim), CARLA-Air | Drones | Unreal-based; multirotor physics and sensor export. CARLA-Air (2026) combines CARLA and AirSim APIs |
| NVIDIA Isaac Sim / Omniverse | Both | Fits Cosmos Transfer best (same company); needs a heavy GPU |
| Blender (scripted, procedural) | Satellite, simple drone shots | Easiest to script and render passes; top-down orthographic camera is trivial |
| 2D compositing of sprites over real backgrounds | Satellite (quick fallback) | dSprites-like; perfect ground truth, lowest realism. Good for a Friday baseline |

**Leaning:** Blender or 2D compositing for satellite, and AirSim-family or Blender for drones. Decide by Tue.

## 6. Real reference data (verify licenses and access)

Real data is used as image conditioning, as the reference set for realism metrics, and as the test set for the detector check (train on synthetic, test on real).

- **Drones as targets:** Anti-UAV (RGB+IR), DUT Anti-UAV, MAV-VID, Drone-vs-Bird. Note: VisDrone is footage *from* drones, not *of* drones.
- **Satellite video:** VISO (Jilin-1 videos with moving vehicles; detection and tracking labels), SatSOT (cars, planes, ships, trains). For still images: xView, DOTA.

## 7. Generators: conditioning matrix

The goal is to cover each conditioning combination with at least one open model.

| Model | Text | Image | Sim video / control | Role in v0 |
|---|:-:|:-:|:-:|---|
| Wan2.2 T2V / TI2V-5B | ✓ | ✓ | – | Text-only and text+image baselines |
| Wan2.1-VACE | ✓ | ✓ | ✓ | Uses all three inputs |
| NVIDIA Cosmos-Transfer2.5 | ✓ | – | ✓ (depth/seg/edge) | Built for sim→real; main candidate |
| (stretch) a closed model | ✓ | ✓ | – | Upper-bound reference |

Fixed settings: same seeds, resolution, and length (about 5 s) for every model.

## 8. Evals: separate the kinds of fidelity

| Axis | Question | v0 metric |
|---|---|---|
| Prompt fidelity | Right count, size, motion, order? | NeuS-E/NeuS-V score; detector counts and tracks compared against sim ground truth |
| Sim faithfulness | Does it keep the sim layout and trajectories? | Track IoU / position error against sim poses |
| Visual quality (prompt-agnostic) | Flicker, blur, temporal coherence | VBench subset |
| Realism / sim2real | Does it look like real data? | FVD/FID against the real reference set; a detector trained on synthetic and tested on real (TRSTR) |
| Physical plausibility | Do objects teleport, merge, or move impossibly? | Track smoothness heuristics plus a VLM-judge question |
| Baseline judge | Does a frontier VLM judge already solve this? | GPT-6 or similar as judge; compare its agreement against the checks above |

Small sanity check: about 20 videos per domain hand-labeled by us, to see which metric agrees with humans.

## 9. v0 deliverable (Fri Nov 13) and plan

**Deliverable:** a dataset page and a slide that show, for each domain, about 30 prompts → sim render → outputs from 2–3 generators → a score table with one row per model and one column per eval axis, plus 3–4 qualitative video examples (including failures).

| Day | Task | Who |
|---|---|---|
| Mon 9 | AFRL meeting; this doc (≤1 page version) | SP |
| Tue 10 | Pick a simulator; finalize the factor grid; download VISO and Anti-UAV | SP, Shourov |
| Wed 11 | Design doc final (datasets, metrics, prompts); first sim renders | SP, course team |
| Thu 12 | Run generators on swarm GPUs; detector-based evals | SP, Shourov, course team |
| Fri 13 | Score table, qualitative examples, dataset page | All |

Course project team (Raghav, Shourov, Shourjo): prompt templating and the dataset page. They own the dataset deliverable.

## 10. Open questions

1. Simulator choice: is one simulator for both domains worth the extra setup?
2. Is the conditioning image a real frame of the *same* site (style and location anchor) or just any frame from the domain?
3. Which detector should we use on tiny drones (<3 px)? Off-the-shelf detectors may fail, which would make the counting metric unreliable.
4. Compute: Cosmos-Transfer and Wan-14B need large GPUs. Check what is free on swarm5 and whether to use Minkyu's AWS setup.
