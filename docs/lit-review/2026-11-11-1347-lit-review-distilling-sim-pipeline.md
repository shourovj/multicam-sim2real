# Literature Review: Distilling the Simulator-Controlled Pipeline into a Fast Student (Nov 11, 2026)

**Owner:** Shourov Joarder · **Drafted:** Nov 11, 2026 with Claude Fable 5.2 and five search subagents · **Status:** first pass. Covers work posted up to Nov 11, 2026.

**The question.** Our controlled pipeline takes 13–30 minutes per clip. Could a teacher that has the whole simulator pipeline (physics state, boxes, tracks, depth, a realistic first frame) be distilled into a small, few-step student that generates from text only?

**How this was made.** Five independent searches: few-step video distillation, physics trained into generators, privileged-teacher distillation, neural simulators and label output, and a date-sorted arXiv sweep (207 API queries, 700 unique papers from Jun 1 to Nov 11, 2026, plus OpenReview keyword search of ICLR 2027 submissions). A search agent opened the arXiv, GitHub or Hugging Face page for every item below unless it is marked † (search snippet only) or ‡ (OpenReview abstract only, no arXiv version found). Pages were read through a summarising fetcher, so re-check any number before it goes into a paper. Numbers are the authors' own.

## 1. Bottom line

- **The exact idea is not published, but one paper posted two days ago is close.** PhysOPSD ([2611.12411](https://arxiv.org/abs/2611.12411), Nov 9, 2026) trains a text-only student from a teacher that sees simulator-rendered drafts, and drops the simulator at inference. Its student is not small or few-step (20 steps, 8–13B models), and it measures only general physics plausibility.
- **Each part of the idea exists separately.** Few-step students that keep the control exist (MotionStream, Cosmos3-Nano-Sim-Transfer). Few-step students with physics supervision exist, without a simulator teacher (PhyS). Nobody has combined a simulator teacher, a few-step student and text-only inference.
- **Evidence says a text-only student would lose the two things our project is about.** It cannot be told which event to produce, and its output no longer comes with simulator labels. No text-to-video model found emits boxes, identities or visibility.
- **Physics learned from simulator data generalises poorly.** The cleanest test (PISA) shows accuracy collapsing outside the simulated range, and one text-only variant (PhyCo) got worse on physics it was not trained on.
- **The speed problem has a cheaper answer.** About 12.5 of our 13.5 minutes per clip is the 30 sampling steps. Open 4-step add-ons for Wan2.1 and Wan2.2 need no training, and Cosmos3-Nano-Sim-Transfer is already a 4-step model. What nobody has measured is whether box or depth adherence survives at 4 steps.

## 2. The idea in three parts

| Part | Published? | Closest work |
|---|---|---|
| Teacher has simulator inputs, student has text only | Yes, once | PhysOPSD (Nov 9, 2026) |
| Student is few-step and keeps the control | Yes | MotionStream, Cosmos3-Nano-Sim-Transfer, Matrix-Game 4.0, AlayaRenderer-Flash |
| Student is few-step with physics supervision, no simulator | Yes | PhyS, PhysDMD‡, PhyMotion |
| Simulator teacher + few-step + text-only student | **Not found** | — |
| Any of the above scored on counts, direction, order, identity or label validity | **Not found** | — |
| Any of the above on drones, aerial or overhead scenes | **Not found** | — |

## 3. Closest prior work

| Paper | What it does | Gap to our idea |
|---|---|---|
| **PhysOPSD: Code as Privilege** · [2611.12411](https://arxiv.org/abs/2611.12411) · Nov 9, 2026 | A coding agent writes MuJoCo, Genesis or Blender programs. The renders (a dense draft and a wireframe + centroid-track draft) condition a teacher copy of the model. The student is the same model with a LoRA and sees text only. VideoPhy-2 overall 63.1 → 67.3 (UniVideo 13B) and 62.9 → 67.0 (OmniWeaving 8.3B) | 20 sampling steps, no small student, no speed claim. No count, direction, order or identity metric. Rigid, elastic and fluid scenes only. The paper gives its training size as both 3K and 33K |
| **PhyS** · [2608.07981](https://arxiv.org/abs/2608.07981) · Aug 8, 2026 | Fine-tunes Wan2.1-14B on 120K real physics clips, distils to a 1.3B causal student at 4–5 steps, then applies reinforcement learning with a motion reward | No simulator, no privileged inputs. The student keeps little of the teacher's gain (§5) |
| **MotionStream** · [2511.01266](https://arxiv.org/abs/2511.01266) · ICLR 2026 | Distils a track-controlled Wan teacher into a causal few-step student, 0.79 → 16.7 fps on the 1.3B model | The student keeps the tracks. Track error rises 46–55% after distillation. No code released |
| **Interactive World Simulation with Cosmos** · [2611.06627](https://arxiv.org/abs/2611.06627) · Nov 4, 2026 | NVIDIA's recipe behind Cosmos3-Nano-Sim-Transfer: a 4-step causal student that takes depth, segmentation, edge or blur from an external simulator | The simulator stays at inference. Code and weights are open; this is the model we already run as condition D |
| **HPSD** · [2608.13205](https://arxiv.org/abs/2608.13205) · Aug 13, 2026 | Teacher gets a synthesised first frame and an enhanced prompt; student is plain text-to-video on Wan2.2-TI2V-5B. Code and weights released | No simulator. Motion amount falls (dynamic degree 0.560 → 0.422) |
| **PDM** · [2607.24731](https://arxiv.org/abs/2607.24731) · Jul 27, 2026 | Dense pose or depth teacher, sparse-keyframe student, on Wan2.1-VACE-1.3B | The student still gets keyframes |
| **SimForcing** · [2610.06598](https://arxiv.org/abs/2610.06598) · Oct 5, 2026 | A simulation teacher's motion is distilled into a real-domain robot world model. Code open | Action-conditioned robotics, not few-step |
| **PhysDMD**‡ · OpenReview ZoFEPJyPpf · Sep 2026 | Adds supervision from a frozen latent world model to step distillation; 4-step student, Physics-IQ-Verified 25.31 → 28.10 | No simulator |

## 4. Strand A: few-step video generation

### 4.1 Open few-step checkpoints we could use

| Checkpoint | Base and steps | Reported speed | Licence |
|---|---|---|---|
| LightX2V Wan2.1 4-step LoRAs | Wan2.1 T2V-14B and I2V, 4 steps, no guidance | "4 vs 50+ steps" | Apache-2.0 |
| Wan2.2-Lightning (V2.0, Nov 2025) | Wan2.2 T2V and I2V A14B, 4 steps, high- and low-noise LoRA pair | "×20" | Apache-2.0 |
| CausVid · [2412.07772](https://arxiv.org/abs/2412.07772) | Wan2.1, 50 → 4 steps, causal | 9.4 fps | Open |
| Self-Forcing | Wan2.1-T2V-1.3B, causal | Real time on an RTX 4090 | Apache-2.0 |
| Krea Realtime 14B | Wan2.1-T2V-14B, 4 steps, causal | 11 fps on a B200 | Apache-2.0 |
| FastWan2.2-TI2V-5B (FastVideo) | Wan2.2 5B, 3 steps | ">50× denoising" | Apache-2.0 |
| TurboDiffusion · [2512.16093](https://arxiv.org/abs/2512.16093) | Wan2.1 1.3B and 14B, Wan2.2-I2V, 1–4 steps | Wan2.1-14B 480p: 1676 s → 9.9 s on an RTX 5090 | Apache-2.0 |
| Cosmos3-Nano-Sim-Transfer (Oct 20, 2026) | Cosmos3-Nano 16B, 4 steps per chunk, causal | 14.6 fps on an H100 at 480p | OpenMDW-1.1 |
| LTX-2.5 distilled | 22B, 8 steps | — | LTX community licence |

HunyuanVideo-1.5 and MiniMax H3 also have distilled versions, but our earlier review excludes both on licence grounds.

No distilled add-on is published specifically for Wan2.1-VACE or Wan2.2-VACE-Fun.

### 4.2 Few-step with control kept

| Item | What is known |
|---|---|
| Wan2.2-Fun-Control + Lightning 4-step (ComfyUI docs) | 640×640, 81 frames on an RTX 4090D: 524 s → 138 s. "May cause some loss in video dynamics" |
| Wan2.1-VACE + CausVid or lightx2v LoRA (community tool docs) | 4–12 steps, no guidance. "Lower steps = lower quality (especially motion)" |
| Wan2.2-VACE-Fun-A14B + Lightning | No primary source confirms it works |
| Wan-Move + lightx2v I2V LoRA | Listed in the official ComfyUI Wan-Move workflow; step count not stated |
| VACE on causal few-step models · [2602.14381](https://arxiv.org/abs/2602.14381) | VACE weights reused without training on a 4-step 1.3B model: 17.2 fps with depth on an RTX 5090. No comparison with normal VACE |
| MotionStream · [2511.01266](https://arxiv.org/abs/2511.01266) | Track error after distillation: 5.35 → 7.80 (DAVIS), 2.71 → 4.21 (Sora set) on the 1.3B model |
| Cosmos3-Nano-Sim-Transfer | Model card: quality and prompt following slightly below base Nano, control following better |
| 4DStreamCtrl · [2608.25479](https://arxiv.org/abs/2608.25479) | 3D point tracks → 4-step causal student, about 20 fps. No code |
| Context-Matched Distillation · [2608.13391](https://arxiv.org/abs/2608.13391) | A causal teacher gives better camera-control adherence in few-step students than a bidirectional one |
| Matrix-Game 4.0 · [2610.27999](https://arxiv.org/abs/2610.27999) · AlayaRenderer-Flash · [2607.18703](https://arxiv.org/abs/2607.18703) | Engine proxy or G-buffers → 1–2 step or few-step student (60 fps; 0.56 → 31.5 fps). The engine stays at inference |
| MotionEcho · [2506.19348](https://arxiv.org/abs/2506.19348) | Training-free motion control methods fail on distilled generators |

### 4.3 What distillation loses

- **Motion.** The Lightning maintainers acknowledged slower motion in the 4-step model and released V2.0 to fix it.
- **Diversity.** PTD ([2610.11498](https://arxiv.org/abs/2610.11498)): 4-step Wan2.1-14B students match the 50-step teacher on VBench (84.4–84.95 vs 83.90), but diversity falls from 0.126 to 0.057. Uncertainty DMD ([2609.11265](https://arxiv.org/abs/2609.11265)) reports that different seeds give near-identical first chunks.
- **Task adherence.** DyMD ([2609.31349](https://arxiv.org/abs/2609.31349)), robot videos: 14B teacher 75.3, 1.3B undistilled 47.3, 1.3B at 4 steps 34.6, while visual quality stays flat. Outputs look good and barely move.
- **Physics: sources disagree.** PhyS: Wan2.1-14B scores 26.9 on Physics-IQ against 10.2–16.9 for 1.3B causal students. PhaseLock ([2606.06361](https://arxiv.org/abs/2606.06361)) reports the opposite at equal size: a 4-step Wan2.1 scores 27.7 against 20.9 at full steps. Cutting steps at the same size is not shown to hurt physics; shrinking the model is.
- **Model size costs more than step count.** PhyS, VideoPhy-2 joint score: 14B 46.0, 1.3B at 40 steps 23.9, 1.3B at 4 steps 24.4. Neither PhyS nor DyMD runs a clean ablation.
- **Not measured anywhere:** a controlled teacher against its few-step student on box or depth adherence, counts or identity.

### 4.4 Training cost of a few-step student

- Self-Forcing on a 1.3B model: under 2 hours on 64 H100s, about 128 H100-hours.
- FastWan2.2-TI2V-5B: 12 hours on 64 H200s, about 768 H200-hours.
- PhysOPSD reports no GPU-hours.

### 4.5 Addendum: self-distillation to flexible 1–4 steps with the control kept

A follow-up question: the teacher is the controlled model at 50 steps (prompt, first frame, simulator control), and the student is the same model with the same inputs at a flexible 1–4 steps, in the style of shortcut models. Checked at abstract level with targeted arXiv queries.

| Paper | Few-step method | Control kept | Steps |
|---|---|---|---|
| FlashMotion · [2603.12146](https://arxiv.org/abs/2603.12146) · Mar 2026 | Distil the generator, then re-train the trajectory adapter | Trajectories | "Few-step". Naive distillation causes "noticeable degradation in both video quality and trajectory accuracy" |
| Genie Sim PanoWorld · [2607.26646](https://arxiv.org/abs/2607.26646) · Jul 2026 | Shortcut-model self-consistency objective | Camera path, as geometry-warped frames | 4, no guidance |
| Dreamer 4 · [2509.24527](https://arxiv.org/abs/2509.24527) · Sep 2025 | "Shortcut forcing", trained from scratch | Actions | Real time on one GPU |
| DreamerAD · [2603.24587](https://arxiv.org/abs/2603.24587) · Mar 2026 | Shortcut forcing | Driving context | 1 |
| ForgeWM · [2608.14022](https://arxiv.org/abs/2608.14022) · Aug 2026 | Causal consistency distillation, then distribution matching | Keyboard and mouse | 1, 2 and 4, apparently one student per budget |
| Causal-rCM · [2606.25473](https://arxiv.org/abs/2606.25473) · Jun 2026 | Continuous-time consistency distillation | Actions (Cosmos variant) | 1–2 |
| ZYT-World · [2609.21712](https://arxiv.org/abs/2609.21712) · Sep 2026 | 40-step teacher → one-step student | Driving layout | 1 |
| AnyFlow · [2605.13724](https://arxiv.org/abs/2605.13724) · May 2026 | Flow-map distillation of pretrained 1.3B–14B video models | None stated | Any step count |
| Shortcutting Pre-trained Flow Matching Diffusion Models · [2510.17858](https://arxiv.org/abs/2510.17858) · NeurIPS 2025 | Velocity self-distillation of a pretrained model into a shortcut model | None (images, Flux) | 3 |

- **The pattern is established.** Controlled few-step students, shortcut-style objectives under control, and shortcut self-distillation of pretrained models all exist.
- **Not found as one paper:** shortcut-style self-distillation of a pretrained box-, depth- or track-controlled video model into a single student that runs at any of 1–4 steps, with control adherence reported per step count.

## 5. Strand B: physics trained into the generator

| Paper | Teacher or data | Student, inference inputs | Result | Generalisation |
|---|---|---|---|---|
| PhysOPSD · [2611.12411](https://arxiv.org/abs/2611.12411) | Simulator drafts as teacher context | 8–13B LoRA; text only; 20 steps | VideoPhy-2 +4.1; supervised fine-tuning on WISA-80K gave about +1 | Qualitative only |
| PhyS · [2608.07981](https://arxiv.org/abs/2608.07981) | 120K real clips, then distil, then reward | 1.3B causal; text; 4–5 steps | Teacher Physics-IQ 26.9 → 31.8; student 16.9 → 17.4 after distillation → 20.9 after reward | Optics and materials regress |
| PISA · [2503.09595](https://arxiv.org/abs/2503.09595) · ICML 2025 | 5K Kubric drop videos + rewards | Open-Sora; image + text | Real-video error 0.175 → 0.076 | Outside the trained heights, IoU 0.155 → 0.049 |
| NewtonRewards · [2512.00425](https://arxiv.org/abs/2512.00425) | 50K Kubric/PyBullet clips, flow and V-JEPA2 rewards | Open-Sora; text + 4 frames | +9.75% over fine-tuning | Out of distribution +8.6%; real free fall +4.15% |
| PhyCo · [2604.28169](https://arxiv.org/abs/2604.28169) · CVPR 2026 | 100K+ simulated clips, property maps, VLM reward | Cosmos-Predict2-2B; also a text-only Wan2.2 LoRA | Physics-IQ 27.7 → 36.3; text-only 30.5 → 35.1 | Text-only variant: solids 34.3 → 42.1, **thermodynamics 36.0 → 22.1** |
| PhysAlign · [2603.13770](https://arxiv.org/abs/2603.13770) | 3K Blender rigid-body clips + V-JEPA2 alignment | Wan2.2-I2V-14B LoRA; image + text | Physics-IQ 29.6 → 38.1 | Gains mainly rigid-body |
| Objects … Slower Than They Appear · [2512.02016](https://arxiv.org/abs/2512.02016) | 100 Blender clips of one falling ball | Wan 5B LoRA; frame + text | Effective gravity 1.81 → 6.43 m/s² | Two-ball timing improves zero-shot; still 65% of real gravity |
| WROP · [2609.28654](https://arxiv.org/abs/2609.28654) | 1.5M Blender clips, keyframed with no physics solver | Cosmos3-Nano 16B; 57 context frames + text; 35 steps | Third of 14 models overall | Tested on its own generators; eighth on collisions |
| PhysMaster · [2510.13809](https://arxiv.org/abs/2510.13809) | Kubric free fall + WISA-80K, preference optimisation | Image + text; 50 steps | VideoPhy physical commonsense 0.29 → 0.40 | "Unseen" means new objects, not new physics |
| Force Prompting · [2505.19386](https://arxiv.org/abs/2505.19386) · NeurIPS 2025 | 15K Blender flags, 12K Blender balls | CogVideoX-5B; frame + text + force | Human preference only | Flags → smoke and leaves; fails where the base prior conflicts |
| Synthetic Video Enhances Physical Fidelity · [2503.20822](https://arxiv.org/abs/2503.20822) · ICCV 2025 | 33K Blender + 18K UE5 clips mixed with real data | 8B model; text only | User success 9 → 61% on gymnastics | Authors: it "still lacks an understanding of… physics" |
| VideoREPA · [2505.23656](https://arxiv.org/abs/2505.23656) · NeurIPS 2025 | Feature distillation from VideoMAEv2 | CogVideoX; text only | VideoPhy physical commonsense 32.3 → 40.1 | Judged by an automatic rater only |
| PhyGDPO · [2512.24551](https://arxiv.org/abs/2512.24551) | 135K real preference pairs | Wan2.1-14B LoRA; text only | PhyGenBench 0.40 → 0.55 | Reward model and evaluator are the same family |
| PhysRVG · [2601.11087](https://arxiv.org/abs/2601.11087) | About 700 rigid clips, SAM 2 trajectory reward | Wan2.2-5B; 5 context frames; 16 steps | Own-benchmark IoU 0.64 vs 0.27 | Rigid only, one seed |
| PhyMotion · [2605.14269](https://arxiv.org/abs/2605.14269) | MuJoCo inside the reward | **4-step** Wan 1.3B; text only | +68 Elo | Human motion only |
| DiPhy‡ · OpenReview hh9NHythnJ | A JEPA teacher mines a physics codebook | Text-only module predicts the codes | About +9% VideoPhy-2 physical commonsense | — |

**What the evidence says about generalisation:**

- **Simulator fine-tunes fix the simulated family and little else.** PISA is the cleanest negative result. PhyCo's text-only variant shows negative transfer.
- **Positive transfer claims are weak.** They are qualitative (Force Prompting, PhysOPSD), about new objects not new physics (PhysMaster), or small (NewtonRewards, +4.15% on real video).
- **No paper reports a held-out physics-law split.**
- **Distillation transfers little of a teacher's physics gain.** In PhyS the teacher gains 4.9 Physics-IQ points and the distilled students gain 0.5–1.7; a reward stage on the student recovers part of it.
- **The benchmarks are unstable.** Physics-IQ Verified ([2606.18943](https://arxiv.org/abs/2606.18943)) revised 57.6% of samples and rankings shifted. PhysWeep ([2609.06207](https://arxiv.org/abs/2609.06207)) finds generators lock onto wrong physical constants that plausibility scores do not see.

## 6. Strand C: privileged-teacher distillation

### 6.1 Diffusion and video work

| Paper | Teacher sees → student sees | Finding |
|---|---|---|
| PhysOPSD · [2611.12411](https://arxiv.org/abs/2611.12411) | Text + simulator draft → text | Staging matters: wireframe draft only +2.6, dense draft only +3.4, both +4.1. As teacher context, refined text scored 62.8 and real video 63.1 against 67.0 for the simulator draft (base 62.9) |
| HPSD · [2608.13205](https://arxiv.org/abs/2608.13205) | Text + first frame → text | Naive matching scored below the untouched base model (0.26 vs 0.53); their fix reaches 1.88 |
| PDM · [2607.24731](https://arxiv.org/abs/2607.24731) | Dense control → sparse keyframes | Errors cancel at the guidance scale used in training and appear at other scales |
| D-OPSD · [2605.05204](https://arxiv.org/abs/2605.05204) · OPSD-V · [2607.08766](https://arxiv.org/abs/2607.08766) | Target image or real video context → text or own context | Few-step students can be trained this way |
| LIFT · [2609.38146](https://arxiv.org/abs/2609.38146) | Per-frame layouts → last-frame layout | Dense-to-sparse layout distillation |
| World Model Self-Distillation · [2606.12072](https://arxiv.org/abs/2606.12072) | Image + a VLM's step-by-step solution → image + short prompt | Claims the student beats the teacher under a VLM protocol |
| D-OPCD · [2610.07250](https://arxiv.org/abs/2610.07250) | Agent-improved prompt → original query | An agent pipeline distilled into image-model weights |
| D2DF · [2607.14976](https://arxiv.org/abs/2607.14976) | Ground-truth latent as draft → no draft, 1 step | Video object removal at about 1 s per clip |
| MCPD‡ · OpenReview ZFCw6a3QKH | Clean continuations → 4-step Wan2.2 | The largest privilege budget transfers worse than an intermediate one |
| State Forcing‡ · OpenReview TwYhVGtwgW | Verified terminal states → streaming student | Strict success 37.3% vs 24.9% |

Nothing found distils VLIPP, LVD, PhysAgent, VideoCoCo, CoDeR, State World Models or Video-Coder into a single model.

### 6.2 What a first frame or a motion plan buys a text-only model

- **Factorized Video Generation** ([2512.16371](https://arxiv.org/abs/2512.16371)): giving Wan2.2-5B a generated anchor frame lifts counting 0.42 → 0.74 and spatial layout 0.58 → 0.71 on T2V-CompBench, but motion direction only 0.257 → 0.282.
- **ViMoLa** ([2610.25595](https://arxiv.org/abs/2610.25595)): a small planner predicts motion tokens from text for a frozen Wan2.2. Direction accuracy 0.29 → 0.44 (chance 0.25); even oracle motion tokens reach only 0.53.
- **T2V-Turbo-v2** ([2410.05677](https://arxiv.org/abs/2410.05677)): motion guidance in the teacher lifts student motion-direction score 22.2 → 24.4.

### 6.3 The warning from the privileged-information literature

- **The imitation gap.** Weihs et al. ([2007.12173](https://arxiv.org/abs/2007.12173)) and Warrington et al. ([2012.15566](https://arxiv.org/abs/2012.15566)): when the student cannot see what determined the teacher's behaviour, imitation discards it, and the teacher can recommend behaviour that is wrong for the student.
- **Gains may not come from the privileged content.** In language-model studies ([2608.04794](https://arxiv.org/abs/2608.04794), [2608.18271](https://arxiv.org/abs/2608.18271)), training loss falls while accuracy stalls, and a reference solution to a different problem sometimes helps as much.
- **No paper measures** whether a text-only video student keeps counts, direction, order or identity after such distillation.

## 7. Strand D: neural simulators, labels and aerial data

### 7.1 Neural simulators keep their control input

GameNGen ([2408.14837](https://arxiv.org/abs/2408.14837)), DIAMOND, Matrix-Game 2.0–4.0, WorldPlay ([2512.14614](https://arxiv.org/abs/2512.14614)), Genie 3, RealWonder ([2603.05449](https://arxiv.org/abs/2603.05449)) and PlayGen2 ([2611.07867](https://arxiv.org/abs/2611.07867), Nov 5, 2026) all distil an engine or game into a fast video model, and every one keeps actions, tracks or state as input at inference.

- **No work found distils a control-conditioned teacher into a text-only student and keeps event control.**
- RealWonder's text-only ablation gives physically implausible motion (qualitative).
- WorldExam ([2608.02603](https://arxiv.org/abs/2608.02603)), 20 models: language-driven models follow controls less faithfully, and their revisit consistency is 0.505 at best against 1.000 for camera-driven ones.
- A survey ([2608.23070](https://arxiv.org/abs/2608.23070)) counts only 6 of 163 implementation papers that expose any queryable entity state.

### 7.2 Labels

| Paper | Labels | Finding |
|---|---|---|
| Cosmos-Drive-Dreams · [2506.09042](https://arxiv.org/abs/2506.09042) | Keeps map and 3D box conditions as labels | 3D detection 0.299 → 0.337; gains shrink as real data grows |
| TrackDiffusion · [2312.00651](https://arxiv.org/abs/2312.00651) | Keeps box tracklets | Tracker accuracy on generated video 44.7 vs 45.4 on real; an uncontrolled generator hurts the tracker |
| Dream4Drive · [2510.19195](https://arxiv.org/abs/2510.19195) · ICLR 2026 | Keeps 3D boxes of inserted assets | At equal training epochs the gain is small (38.4 → 38.7 mAP at 2×) |
| NeuralSRender · [2610.29200](https://arxiv.org/abs/2610.29200) | Keeps the CARLA scene | Detector AP50 35.1 → 40.6; label consistency asserted, not measured |
| WorldReel · [2512.07821](https://arxiv.org/abs/2512.07821) · GenSight · [2610.39232](https://arxiv.org/abs/2610.39232) · UGG · [2611.14440](https://arxiv.org/abs/2611.14440) | Emit depth, flow, masks or camera pose with the video | Scored against estimated pseudo-labels; no detector trained on them |

**No text-to-video model found emits instance boxes, identities or visibility.** Box and identity labels survive only where the control is kept.

### 7.3 Generated video as aerial training data

- **Dreams Are Cheaper Than Flights** ([2610.41030](https://arxiv.org/abs/2610.41030), Oct 29, 2026): Cosmos-Predict1 seeded with drone frames, experts reject 75–88% of clips and hand-label the rest. VisDrone mAP 0.138 → 0.184. Adding the rejected clips drops bus AP from 0.211 to 0.010.
- **Generative Spatiotemporal Data Augmentation** ([2512.12508](https://arxiv.org/abs/2512.12508)): 8-step distilled LTX-Video plus SAM 2 box propagation, about 18 s per video. VisDrone mAP50 45.9 → 49.6.
- No drone-vs-bird or anti-UAV study with diffusion-generated video was found, and no paper compares generation cost against plain rendering.

## 8. What this means for our project

This section is analysis, not literature.

- **Where our time goes.** Each Wan2.2-VACE-Fun step takes about 25 s on a free A100, so 30 steps are about 12.5 of the 13.5 minutes per clip. Cutting steps is the right lever.
- **A text-only student gives up event control.** Our own result is that text cannot specify the event (0 of 8 clips). The literature agrees that the gap is in motion: a privileged first frame fixes counts and layout but not direction (§6.2).
- **A text-only student gives up labels.** Simulator labels apply to a generated clip only if the clip follows the simulator. Without the control, labelling falls back to after-the-fact tools, with the rejection rates in §7.3.
- **What a text-only student would converge to is uncertain.** If one prompt maps to many simulator runs, a velocity-matching student is trained toward their average, which would likely show as blur or weak motion. HPSD's failed naive baseline and the motion loss in §4.3 are consistent with this, but no paper states it for diffusion.
- **Data cost.** At 13.5 minutes per clip, 3,000 teacher clips are about 675 A100-hours. With a 4-step teacher at 2–3 minutes per clip, 100–150.
- **Training cost.** A full few-step distillation is 128 H100-hours or more (§4.4), on top of the teacher data.

## 9. Options, in order of cost

| Option | Training | Keeps event control and labels | What is new |
|---|---|---|---|
| 1. Add a 4-step LoRA to the models we already run, and score box and depth adherence at 4, 8 and 30 steps | None | Yes | Nobody has measured control adherence, counts or identity after step distillation |
| 2. Use Cosmos3-Nano-Sim-Transfer as the fast controlled model | None | Yes, with dense control | It is three weeks old and untested on drones or overhead scenes |
| 3. Train a few-step student that keeps sparse control (boxes or tracks), following MotionStream or the Cosmos recipe | Heavy | Yes | Sparse-control few-step students exist for tracks, not for boxes with identity |
| 4. Train a text-only few-step student from the simulator pipeline | Heavy, plus thousands of teacher clips | No | Few-step + small + event-level scoring are what PhysOPSD lacks |

Option 4 is the idea as first proposed. It is open in the literature, but it changes what the project produces: plausible video from text, not a clip that matches a simulated event with valid labels. It fits best as a comparison arm that measures how much is lost when the control is dropped.

## 10. Very recent items (Aug – Nov 2026)

- **Nov:** PhysOPSD [2611.12411](https://arxiv.org/abs/2611.12411) (Nov 9) · UGG [2611.14440](https://arxiv.org/abs/2611.14440) (Nov 10) · Condition Shifting [2611.14033](https://arxiv.org/abs/2611.14033) (Nov 10, cheaper step distillation) · PlayGen2 [2611.07867](https://arxiv.org/abs/2611.07867) (Nov 5) · Interactive World Simulation with Cosmos [2611.06627](https://arxiv.org/abs/2611.06627) (Nov 4)
- **Oct:** Dreams Are Cheaper Than Flights [2610.41030](https://arxiv.org/abs/2610.41030) · GenSight [2610.39232](https://arxiv.org/abs/2610.39232) · BoxWorld [2610.36371](https://arxiv.org/abs/2610.36371) · SF² [2610.35594](https://arxiv.org/abs/2610.35594) · DTTD [2610.33566](https://arxiv.org/abs/2610.33566) · Matrix-Game 4.0 [2610.27999](https://arxiv.org/abs/2610.27999) · FluidGen [2610.25660](https://arxiv.org/abs/2610.25660) · ViMoLa [2610.25595](https://arxiv.org/abs/2610.25595) · PTD [2610.11498](https://arxiv.org/abs/2610.11498) · D-OPCD [2610.07250](https://arxiv.org/abs/2610.07250) · SimForcing [2610.06598](https://arxiv.org/abs/2610.06598)
- **Sep:** LIFT [2609.38146](https://arxiv.org/abs/2609.38146) · DyMD [2609.31349](https://arxiv.org/abs/2609.31349) · WROP [2609.28654](https://arxiv.org/abs/2609.28654) · "Why Do Video Diffusion Models Violate Physics?" [2609.23658](https://arxiv.org/abs/2609.23658) · ZYT-World [2609.21712](https://arxiv.org/abs/2609.21712) · Uncertainty DMD [2609.11265](https://arxiv.org/abs/2609.11265) · PhysWeep [2609.06207](https://arxiv.org/abs/2609.06207) · FlashRender [2609.03563](https://arxiv.org/abs/2609.03563) · PhysDMD‡ · DiPhy‡
- **Aug:** 4DStreamCtrl [2608.25479](https://arxiv.org/abs/2608.25479) · Cosmos-H-Dreams [2608.24199](https://arxiv.org/abs/2608.24199) · Context-Matched Distillation [2608.13391](https://arxiv.org/abs/2608.13391) · HPSD [2608.13205](https://arxiv.org/abs/2608.13205) · PhyS [2608.07981](https://arxiv.org/abs/2608.07981) · WorldExam [2608.02603](https://arxiv.org/abs/2608.02603) · MCPD‡ · State Forcing‡

## 11. Coverage and caveats

- **November is thinly indexed.** Web search of the last three weeks is sparse; the arXiv API sweep is the main source for that period.
- **Four broad arXiv queries hit the 100-result cap,** including "on-policy self-distillation" (225 hits). This is a fast-moving line of work, and a few-step version of PhysOPSD could appear soon.
- **NeurIPS 2026 main-track papers were not searchable on OpenReview,** and ICLR 2027 coverage is keyword search only.
- **Three ICLR 2027 submissions (‡) have no arXiv version;** their numbers come from OpenReview abstracts.
- **Community reports** on few-step LoRAs with VACE come from tool documentation, not papers.
- **Numbers are not comparable across papers.** Baselines, clip lengths and evaluators differ, and several physics gains are scored by the same model family that supplied the reward.
