# Sim2Real-VideoGen

Sim-conditioned video synthesis for drone and satellite scenarios (AFRL Thread 1).

Input: a text prompt, optional real images of the domain, and a simulator video. Output: a realistic synthetic video, scored by a testbench of verifiers and metrics.

## Getting started

[`setup/README.md`](setup/README.md) covers the workspace (third-party code, envs, Blender, weights) and how to re-run the `drones_air` proof of concept. On swarm5 you can reuse the existing workspace and read its outputs directly.

The [round 2 guide](docs/2026-11-11-1304-pilot-v0-branch-guide.md) covers the current pipeline: how to run it and where the outputs are. Round 2 also needs a VideoX-Fun clone with the Wan2.2-VACE-Fun-A14B weights; the setup guide does not cover it yet, so the steps are in §3 of the round 2 guide.

## Pipeline

Each step reads and writes files under `$S2R_DATA/poc/single-cam/<scene>/`, so any step can be re-run on its own. Multi-camera tests are under `$S2R_DATA/poc/multi-cam/`.

Multi-camera scenes (spec v2, from Nov 18) use a second pipeline that simulates a world once and renders it from any number of cameras: [`sim/world/`](sim/world/) (simulator, kinematics, cameras, Blender builders), [`sim/build_world.sh`](sim/build_world.sh), [`gen/run_multicam.py`](gen/run_multicam.py) with an experiment file in [`configs/experiments/`](configs/experiments/), and [`eval/score_multicam.py`](eval/score_multicam.py). The [multi-camera v1 design doc](docs/2026-11-18-1739-multicam-v1-design.md) covers the scenes, the Cosmos settings, the tracking rules and how to run it (§13).

| Step | Code | Output |
| --- | --- | --- |
| 1. Describe a scene: camera, background, objects and what each object tries to do | [`configs/scenes/`](configs/scenes/), [`configs/prompts/`](configs/prompts/) | |
| 2. Move the objects with a physics model (quadrotor, ground vehicle, rigid body) | [`sim/physics/`](sim/physics/), [`sim/simulate.py`](sim/simulate.py) | |
| 3. Render and export the scene bundle: RGB, depth, object masks, boxes, point tracks, events | [`sim/build_bundle.sh`](sim/build_bundle.sh) | `bundle/` |
| 4. Make frame 0 realistic by hand in an image editor | [`frame0_edits/`](frame0_edits/) | `bundle/frame0_edited.png` |
| 5. Build each condition's model inputs | [`sim/controls.py`](sim/controls.py) | `<condition>/inputs/` |
| 6. Generate every scene × condition × seed on the shared GPUs | [`gen/run_matrix.py`](gen/run_matrix.py), [`configs/matrix.yaml`](configs/matrix.yaml) | `<condition>/<model>-s<steps>/seed<k>/out_video.mp4` |

A condition is only a choice of inputs to the same model: text, a first frame, box outlines, depth, or a mix. The conditions are listed in [`configs/matrix.yaml`](configs/matrix.yaml).

## Results so far

| Round | Setup | Findings |
| --- | --- | --- |
| Proof of concept (Nov 10) | `drones_air` with scripted motion, Wan2.1-VACE-14B, Wan-Move and Cosmos Sim-Transfer | [PoC findings](docs/2026-11-10-1251-poc-drones-air-findings.md) |
| Round 2 (Nov 11) | Three physics scenes × seven conditions on Wan2.2-VACE-Fun-A14B, 51 clips | [Round 2 findings](docs/2026-11-11-0845-round2-findings.md) |

Round 2 in short, read by eye and not scored by a tool: box outlines for the whole clip keep the drones on the simulator's path in all 6 clips of the two drone scenes; with text, a first frame, or boxes for the first second only, the event does not happen in 15 of 16 drone clips; the overhead convoy scene fails in every condition and needs rework.

## Links

- [Google Drive folder](https://drive.google.com/drive/folders/19sR4ksrWNhh8aFPyejz8-W5eYPbB3Epu?usp=sharing): shared project files
- [Design doc (Google Doc)](https://docs.google.com/document/d/1cAho8uc2gT9nnzAg9GN1NU0pHQ4cggaQcGlL9v-IBb4/edit?usp=sharing): the living version, one tab per version (`v0`, ...). Edits, annotations, and comments happen here.
- [NSVS meeting notes](https://docs.google.com/document/d/1ifhAz0vf8T1qxOz1hjouDQRohipBHqKDNB8rFTXzELA/edit?usp=sharing): Sandeep's meeting jottings and deadlines

## Docs

Each local draft and the Google Doc tab it was pasted into. The Google Doc may have changed since.

| Local draft | Google Doc tab |
| --- | --- |
| [docs/2026-11-09-1416-design-doc-v0.md](docs/2026-11-09-1416-design-doc-v0.md) | [v0](https://docs.google.com/document/d/1cAho8uc2gT9nnzAg9GN1NU0pHQ4cggaQcGlL9v-IBb4/edit?tab=t.0) |
| [docs/2026-11-09-1626-project-understanding.md](docs/2026-11-09-1626-project-understanding.md) | not pasted yet |
| [docs/lit-review/2026-11-09-1750-lit-review-synthesis.md](docs/lit-review/2026-11-09-1750-lit-review-synthesis.md) | not pasted yet |
| [docs/2026-11-09-2309-cosmos-hardware-check.md](docs/2026-11-09-2309-cosmos-hardware-check.md) | not pasted yet |
| [docs/2026-11-10-0012-v0-engineering-plan.md](docs/2026-11-10-0012-v0-engineering-plan.md) | not pasted yet |
| [docs/2026-11-10-1251-poc-drones-air-findings.md](docs/2026-11-10-1251-poc-drones-air-findings.md) | not pasted yet |
| [docs/2026-11-11-0845-round2-findings.md](docs/2026-11-11-0845-round2-findings.md) | not pasted yet |
| [docs/2026-11-11-1304-pilot-v0-branch-guide.md](docs/2026-11-11-1304-pilot-v0-branch-guide.md) | not pasted yet |
| [docs/2026-11-18-1739-multicam-v1-design.md](docs/2026-11-18-1739-multicam-v1-design.md) | not pasted yet |

### Literature review

[`docs/lit-review/`](docs/lit-review/) holds the [synthesis](docs/lit-review/2026-11-09-1750-lit-review-synthesis.md) and its four source surveys, written independently: [Claude Opus](docs/lit-review/2026-11-09-1650-lit-survey-claude-opus.md), [Codex GPT-6.1 Sol](docs/lit-review/2026-11-09-1741-lit-survey-codex-sol.md), [Codex GPT-6 Astra](docs/lit-review/2026-11-09-1744-lit-survey-codex-astra.md), and [Claude Fable 5.2](docs/lit-review/2026-11-09-1935-lit-survey-fable.md). The [shared prompt](docs/lit-review/2026-11-09-1735-codex-review-prompt.md) given to the Codex and Fable surveys is saved alongside them. A [paper-direction review by Fable 5.2](docs/lit-review/2026-11-09-1918-paper-direction-fable.md) covers how this could become a top-tier paper.

## Artifacts

claude.ai artifacts made for this project, oldest first. Links open for people in the UT Austin org.

| Artifact | Kind | Author | Created | What it covers |
| --- | --- | --- | --- | --- |
| [Sim2Real v0 Walkthrough](https://claude.ai/artifact/WnryWm5K5uvxH6kC9oTVoj) | Page | Shourov Joarder | 2026-11-09 | v0 smoke test on the lab server: Blender scene gives sparse motion (boxes or point tracks), Wan2.1 1.3B generates the clip |
| [Sim2Real-VideoGen: Week 1](https://claude.ai/artifact/Xnc3VJyqozBpXNLK3LrQkB) | Slides | S P Sharan | 2026-11-10 | Week 1 deck: literature review, problem formulation, v0 architecture, drones_air proof of concept, next steps |
| [Sim2Real Round 2](https://claude.ai/artifact/DMwkj9covsGwnoHo46j3hD) | Page | Shourov Joarder | 2026-11-11 | Round 2 from prompt to results: three physics scenes, seven conditions plus B / B-box repeated on the simulator's frame 0, run on Wan2.2-VACE-Fun-A14B and Cosmos3-Nano, every setting, a dense reference where Cosmos3-Nano gets the whole simulator video, and all 147 clips with playback. A two-camera follow-up (Nov 13) renders the two-drone scene from a second camera and generates both views on Cosmos3-Nano, separately and in one video (cut and split screen). Multi-camera v1 (Nov 18–19) simulates an intersection and a robot pour once each, renders each from three cameras plus a three-camera cut, and compares the textured render against depth + segmentation as the Cosmos3-Nano control, then segmentation, depth or edges alone on Cosmos3-Nano-Sim-Transfer, with the pipeline, scores and the exact prompts |
| [Sim2Real Round 2 Results](https://claude.ai/artifact/S8mjMoxtDY3y6cFJrnEcsM) | Slides | Shourov Joarder | 2026-11-11 | Round 2 deck: pipeline, scenes, prompts, conditions, settings, Wan results per scene, edited vs simulator frame 0, the same conditions on Cosmos3-Nano, Nano with the whole simulator video, Wan vs Nano, cost, caveats, next steps; a last slide (Nov 20) on the multi-camera scenes generated from text alone |
