# Literature Review Synthesis: Simulator-Conditioned Video Generation

**Owner:** SP Sharan · **Drafted:** Nov 9, 2026 · **Updated:** Nov 9, 2026, adding the Fable 5.2 survey · **Status:** main highlights from four independent surveys and one advisor review. Details and full paper tables are in the source files.

| Source | Model | Notes |
|---|---|---|
| [Claude survey](2026-11-09-1650-lit-survey-claude-opus.md) | Claude Opus 5.5, with five topic subagents | Saw the whole conversation with the lead |
| [Codex Sol survey](2026-11-09-1741-lit-survey-codex-sol.md) | GPT-6.1 Sol, with three subagents | Saw only the [prompt](2026-11-09-1735-codex-review-prompt.md) and the design doc |
| [Codex Astra survey](2026-11-09-1744-lit-survey-codex-astra.md) | GPT-6 Astra | Same input as Sol |
| [Fable survey](2026-11-09-1935-lit-survey-fable.md) | Claude Fable 5.2, high effort, with seven subagents | Same input as Sol, through headless Claude Code |
| [Fable paper-direction review](2026-11-09-1918-paper-direction-fable.md) | Claude Fable 5.2, high effort | Advisor view on turning this into a top-tier paper. Saw the repo docs, including this synthesis |

The key arXiv IDs from all four surveys were checked against their arXiv pages. Reported numbers are the authors' own and come from different protocols, so they should not be compared across papers.

**Coverage warning.** Each survey found a partly different set of 2026 papers. For example, the Fable survey found VideoCoCo, PhysAgent and NeuralSRender but missed Interactive World Simulation with Cosmos, State World Models and VeriPhy. No single survey is complete; the union below is the working literature set.

## 1. Bottom line

- **The general idea is already established.** Several lines of work run prompt → simulator, code or structured state → generative video. These include GPT4Motion, PhysGen, PSIVG, CoDeR, State World Models, Video-Coder, VideoCoCo, PhysAgent, the Cosmos-Transfer family and NVIDIA's new Cosmos simulation work. A method paper claiming "a simulator helps video generation" would not be novel. **All four surveys agree.**
- **What is still open is measurement in our setting.** No published work does simulator-conditioned generation of drones as subjects or of overhead convoys. No work reports counts, identity, trajectory and event order together across text-only, first-frame, sparse-state and dense-render inputs on one model.
  - **Framing for v0:** Astra suggested the title "Motion and Label Preservation in Simulator-Conditioned Video Generation: Three Case Studies." This also fits the lead's goal of comparing how these models work in practice with how they are advertised.
  - **Framing for the paper:** the Fable advisor review suggests "Do Simulator Labels Survive Generative Sim-to-Real?", with *label validity* as the headline metric.
- **All four surveys converge on the v0 pipeline:**
  - Blender or BlenderProc with Cycles on CUDA renders the scenes.
  - The Wan2.1 family runs the same-model conditions: text, first frame, VACE dense control, and Wan-Move sparse tracks.
  - Cosmos-Transfer2.5 is a second generator. The Codex runs ranked it as the main candidate; the Fable survey ranked it optional.
  - Isaac Sim is ruled out because it does not run on A100s.

## 2. Closest prior work (must-cite)

| Paper | What it does | Why it matters to us | Found by |
|---|---|---|---|
| **Interactive World Simulation with Cosmos** (NVIDIA, Nov 4 2026) · [2611.06627](https://arxiv.org/abs/2611.06627) | An external simulator holds physics and state and streams depth, segmentation, edge or blur controls into Cosmos | Essentially our architecture, released last week. The main paper to position against | Sol |
| **State World Models** · [2610.26811](https://arxiv.org/abs/2610.26811) | Executable rules plus a state ledger feed a frozen MiniMax H3. Rule-following goes 32% → 64% → 92% as more structure is added | A same-model ablation of how much structure to supply, the comparison we want. Code released | Sol |
| **CoDeR** · [2609.26458](https://arxiv.org/abs/2609.26458) | Coding agents build a Three.js world. Whitebox RGB, depth and text go to a trained "Artist" | Closest to the lead's starting point. Compares against world models, not T2V or I2V. Code not released. Its backbone (MiniMax H3) may have licence problems for US use (§5) | All |
| **VideoCoCo** · [2607.27380](https://arxiv.org/abs/2607.27380) | An agent writes Blender Python, which renders a "white-clay" draft video for a video editor | Cleanest same-backbone evidence that a simulator draft helps: PhyGenBench 0.475 → 0.506 training-free → 0.558 with a LoRA | Fable |
| **PhysAgent** · [2607.16355](https://arxiv.org/abs/2607.16355) | A VLM writes a physics program with a verify-and-repair loop; simulated flow drives the generator | 0.714 vs Wan2.2's 0.633, but **without the repair loop it ties plain Wan2.2 (0.631)**. A one-shot simulator pass may not help, which supports "verify, then repair" | Fable |
| **PSIVG** (CVPR 2026) · [2603.06408](https://arxiv.org/abs/2603.06408) | Reconstruct, simulate with MPM, render, then guide CogVideoX, Hunyuan or Go-with-the-Flow | Very close published precedent. The authors list vehicles and small or thin objects as limitations | Sol, Astra, Fable |
| **NeuralSRender** · [2610.29200](https://arxiv.org/abs/2610.29200) | CARLA/UE5 G-buffers feed a LoRA on Wan2.2-TI2V-5B | The closest analogue of our pipeline, in driving. Beats Cosmos-Transfer2.5 on FVD, and detector AP50 rises from 35.1 to 40.6 | Fable |
| **VLIPP** (ICCV 2025) · [2503.23368](https://arxiv.org/abs/2503.23368) | An LLM plans box trajectories, which become flow and then warped noise | The only paper found that beats both T2V and I2V baselines with a structured motion step | Claude |
| **Programmable World Model** · [2609.10540](https://arxiv.org/abs/2609.10540) | An LLM-written program yields 3D boxes and direction maps that drive a ControlNet world model | Count/state accuracy 94/98 vs 41/8. A template for checking counts and direction | Claude |
| **VeriPhy** · [2609.03153](https://arxiv.org/abs/2609.03153) | MuJoCo depth and silhouettes feed Wan2.2-VACE, plus an agent-based physics evaluator | Overlaps both our pipeline and our evals. Its self-reported score improved while an independent physics score did not | Sol |
| **Video-Coder** (Nov 6 2026) · [2611.09894](https://arxiv.org/abs/2611.09894) | An RL-trained coding agent writes Blender scenes that guide video generation | Further limits any novelty claim for "an agent codes a simulation" | Astra |
| **Code World Model** · [2608.25927](https://arxiv.org/abs/2608.25927); **ChronoAgentic** · [2605.14398](https://arxiv.org/abs/2605.14398) | Coding agent → game-engine proxy video → LoRA renderer; agents writing PyChrono vehicle-dynamics programs | Coarse proxies are enough for entity control. Chrono is a natural later simulator for the convoy | Fable |
| **Cosmos-Transfer1 / 2.5** · [2503.14492](https://arxiv.org/abs/2503.14492), [2511.00062](https://arxiv.org/abs/2511.00062) | Depth, segmentation, edge and blur controls with adaptive weights | The main off-the-shelf sim-to-real model. Even under edge-only control, edge F1 is only 0.28–0.49 | All |
| **GPT4Motion**, **PhysGen**, **PhysCtrl**, **WonderPlay**, **NewtonGen**, **DriveDreamer-2**, **LVD** | Earlier simulator, physics or LLM-planner pipelines into video | The older lineage | All |

**Out-of-frame failure is already quantified.** Three 2026 benchmarks measure it: WRBench ("Current World Models Lack a Persistent State Core", [2606.20545](https://arxiv.org/abs/2606.20545)), MemoBench ([2606.27537](https://arxiv.org/abs/2606.27537)) and "Can 4D Foundation Models Remember?" ([2609.20819](https://arxiv.org/abs/2609.20819)). All three hide the target by **moving the camera**. Exit and re-entry driven by the object's own motion, under a known camera, is still new.

**Domain gap.** Aerial video work is first-person and action-conditioned (AirScape, Aero-World, MotionScape). Satellite generation produces still images (DiffusionSat, GeoSynth, AeroGen). The closest overhead moving-target work is a small study that guides SG-I2V with boat GPS trajectories ([2605.16420](https://arxiv.org/abs/2605.16420)). For drones as subjects, SynDroneVision gives a simulator-only baseline.

## 3. The lead's question: scene state or rendered video?

**All four surveys agree that models can take structured state without a rendered RGB video, but only through the specific interfaces they were trained on.** No model accepts an arbitrary simulator state dump, and the practical open models all need a first frame.

| What the simulator hands over | Open examples | Notes |
|---|---|---|
| Tokens (no rasterizing) | 3DTrajMaster | Closest to "state in". Open weights cover animals only, at most 3 entities |
| Numeric boxes, poses, camera | MagicDrive-V2, MotionCtrl, CameraCtrl, SG-I2V, VerseCrafter | Tied to each model's coordinate conventions |
| Point tracks or flow on the latent grid | **Wan-Move**, ATI, Tora, Go-with-the-Flow, Time-to-Move, WorldCanvas | Best practical non-RGB option this week. Use several surface points per object with visibility flags. Objects must be visible in frame 1, because Wan-Move copies first-frame features |
| Rendered structural maps | Cosmos-Transfer, VACE, Diffusion as Shader, 4Director | The most practical overall. Still rasterized into the image, view by view |

**Reframing:**
- All three outside reviewers pushed back on "it's just style transfer." Depth or track conditioning still leaves appearance for the model to generate, so both are legitimate experimental conditions. The Fable survey adds three points:
  - Under dense control, "style transfer" is a fair description.
  - Even dense control is not faithful: edge F1 stays low, and third parties report spurious objects.
  - Appearance leaks in through the first frame anyway.
- The useful distinction is **what information is supplied and what the model has to predict**:
  - **Full future controls supplied:** the model is a *renderer*.
  - **Past frames plus actions, with the future predicted:** this tests a *world model*.
  - **Persistent state across unseen intervals:** this tests *memory*.
- v0 tests the first case, so the write-up should not claim world-model or physics gains.
- Sol's operational version of the question: *which input preserves the required behavior with the least rendering effort: sparse tracks, structural maps, or RGB?* The Fable survey recommends making this gradient the experiment itself.

## 4. Recommended pairings for the three v0 videos

| Video | Simulator | Same-model conditions | Second generator | Main risk |
|---|---|---|---|---|
| 1. Drones take off and fly | Blender; optional dynamics from RotorPy or gym-pybullet-drones | Wan2.1 T2V/I2V, VACE (depth + edges), Wan-Move | Cosmos-Transfer2.5 | Ground contact, drone shape, extra or missing drones |
| 2. Drones in the air | Blender, static or chase camera | Same; **Wan-Move with projected surface tracks and visibility** | Cosmos-Transfer2.5 | Identity swaps at crossings. Edge detection fails on a drone against the sky |
| 3. Overhead convoy, road → right turn onto field | Blender with a near-orthographic camera over a real (e.g. NAIP), GeoSynth, or sprite background; kinematic bicycle model for the vehicles | Same; VACE with **edges/blur** (overhead depth is nearly flat) | Cosmos-Transfer2.5 | Wrong heading, merged vehicles, invented camera motion. No evidence any open model has top-down priors. Keep vehicles at least ~40 px and call it "overhead", not "satellite", in v0 |

**Hardware and setup notes:**
- **Isaac Sim** does not support A100s, which have no RT cores. *Update (Nov 9):* swarmllm's RTX 6000 Ada cards do have RT cores, so Isaac Sim may run there. Untested; see the [Cosmos hardware check](../2026-11-09-2309-cosmos-hardware-check.md).
- **Unreal-based simulators** (CARLA, CARLA-Air, AirSim) need Vulkan, which may be missing and cannot be installed without root. EEVEE needs EGL. Cycles on CUDA is the safe choice. Run a 15-minute pre-flight check (driver, glibc, EGL/Vulkan) first.
- **Cosmos-Transfer2.5** supports A100 but needs about 65 GB on one GPU, plus driver ≥570.124 and glibc ≥2.35. An NVIDIA collaborator noted on GitHub that it is trained on real scenes and degrades on simulator input.
- **Cosmos3-Nano-Sim-Transfer** is trained on simulator renders but lists only Hopper and Blackwell GPUs. *Update (Nov 9):* it ran on a shared swarm5 A100 anyway, in about 2 min per clip and about 35 GB ([Cosmos hardware check](../2026-11-09-2309-cosmos-hardware-check.md)). It is now condition D in the plan.
- **Wan-Move** runs on 40 GB with offloading. Its track format is undocumented; budget half a day for it. Estimated runtime for Wan-14B is 15–30 minutes per 5-second clip on an A100.

## 5. Licences and guardrails (new, from the Fable survey; needs a compliance check)

This is urgent for AFRL work. Treat it as a reading of the licences, not legal advice.

- **HunyuanVideo-1.5:** the licence bans use "for military purposes" (verified by the survey). Avoid.
- **MiniMax H3:** the licence reportedly excludes the USA. This is the backbone of CoDeR and State World Models. Avoid.
- **Cosmos:** ships a prompt and output guardrail that may refuse "military convoy" prompts. Disabling it without a substitute reportedly ends the licence. *Update (Nov 9):* that clause is in the NVIDIA Open Model License, which covers Transfer2.5, Predict2.5 and Transfer1. Cosmos 3 (including Sim-Transfer) uses OpenMDW 1.1, which has no guardrail clause ([Cosmos hardware check](../2026-11-09-2309-cosmos-hardware-check.md)).
- **Wan family:** Apache-2.0, so it is the safe default.
- **Prompts:** use neutral wording ("trucks", "quadcopters") for any closed API.

## 6. Critique: where the surveys agree

1. **Comparing text-only against simulator-conditioned mostly measures the extra information supplied.** Giving the full future motion and then scoring against it is close to circular. Under the lead's "do they work as advertised" framing, the C/D numbers become an **adherence audit**, and A/B show how much each control buys.
2. **Use the same model across conditions.** One VACE checkpoint, or the Wan2.1-14B family, for text-only, first-frame and controlled conditions. Cosmos is an extra comparison.
3. **Use several seeds per condition** (3–5) and report every sample, failures included. One seed per condition is a demo, not a result.
4. **Keep the raw simulator render as a reference**, and compare realism against it with blinded side-by-side ratings.
5. **Score text-only fairly.** The simulator's exact path is not the only correct reading of the prompt. Score what the prompt states (count, direction, events). Use exact path error only for controlled outputs.
6. **Add a detailed-prompt baseline** that describes the trajectory in text.
7. **Simulator labels are only candidates for the generated video.** Verify positions, identity and visibility on the output. At this scale, human annotation assisted by trackers is the primary measurement.
8. **Don't headline FVD/FID.** With 3 scenes they are statistically meaningless and biased toward content.
9. **Tone down design-doc claims:**
   - "Covers physics the simulator misses", "faster than rendering" and "closes the sim2real gap" are unsupported until measured. The Fable survey estimates that Wan-14B is *slower* than Cycles.
   - Call scripted motion "kinematically controlled."
   - NeuS-E is a refinement method, not a metric.

## 7. Points the surveys and reviews added beyond the plan

- **Static-control ablation (Astra):** feed a repeated, non-moving control to check whether gains come from the time-varying signal.
- **Pre-registered tolerances (Sol):** e.g., position within half a body length, events within 0.25 s, checked on the raw render first.
- **Log cost:** wall time, peak memory, retries and setup effort.
- **Derive NeuS-V propositions from the tracker, not a VLM (Fable survey):** we have ground truth and trackers. This also fixes NeuS-V's blind spot for hallucinated extra objects.
- **Add a closed frontier model (Fable survey):** claiming "state of the art can't do this" needs a state-of-the-art model, e.g., a Veo-class API on text and text + first frame.
- **Label validity as the headline metric (Fable review):** the fraction of simulator labels still correct in the output, shown next to edge/depth agreement on the same clips. Run 8–10 seeds of the controlled condition on one hard scene.
- **Disputed: re-entry in v0.** The Fable survey wants a drone to leave and re-enter the frame in video 2, because self-driven exit and re-entry is the one out-of-frame case not yet benchmarked. The lead deferred memory tests to later. If the simulator supplies the unseen interval, a correct re-entry shows external state plus rendering, not the model's own memory.
- **Disputed: drone size.** The Fable review wants a 64/16/8 px size axis on video 2. The lead chose clearly visible drones for v0.
- **Minimum viable experiment (Astra):** 3 scenes × VACE {text, text + first frame, text + first frame + control} × 3 seeds = 27 clips, plus the raw renders. The Fable survey's version: 3 scenes × 5 core conditions × 5 seeds = 75 clips, about 25–40 A100-hours by estimate.

## 8. Open items

- **Licence and guardrail check** for Cosmos, and confirmation that HunyuanVideo and MiniMax are excluded. This is urgent for AFRL.
- Server pre-flight: driver, glibc, EGL/Vulkan, and which swarm GPUs have about 65 GB free for Cosmos. *Answered for Cosmos (Nov 9)* in the [Cosmos hardware check](../2026-11-09-2309-cosmos-hardware-check.md); EGL/Vulkan is still unchecked.
- ~~Whether Cosmos3-Sim-Transfer runs on A100.~~ Yes (Nov 9), see the [Cosmos hardware check](../2026-11-09-2309-cosmos-hardware-check.md).
- Release status of CoDeR, DriveDreamer-2 and GraphVid code and weights.
- Whether to add RotorPy or gym-pybullet-drones dynamics, or stay kinematic for v0.
- Decide on the two disputed items: re-entry in video 2, and the drone-size axis.
- §5–6 of the earlier version are folded into the [experiment plan](../2026-11-09-1626-project-understanding.md) (§7 there). The new items in §5 and §7 here are not yet folded in.
