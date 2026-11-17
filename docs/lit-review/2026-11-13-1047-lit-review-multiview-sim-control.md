# Literature Review: The Simulator as a Multi-View Controller (Nov 13, 2026)

**Owner:** Shourov Joarder · **Drafted:** Nov 13, 2026 with Claude Opus 5.5 and four search subagents · **Status:** first pass. Covers work posted up to Nov 13, 2026.

**The question.** The simulator knows the whole 3D scene, so it can render controls (boxes, depth, tracks) from any number of cameras. Can an off-the-shelf control-to-video ("X2V") model turn each view into realistic video, so that all views agree with each other (same objects, colours and motion) and the simulator's labels hold in every view? Has anyone done this?

**How this was made.** Four independent searches: multi-camera driving and robot generators; general multi-view and 4D video; multi-view sim-to-real, aerial work and datasets; and a date-sorted arXiv sweep (78 API queries, 926 unique papers, mostly Jun 1 – Nov 13, 2026, plus OpenReview keyword search of ICLR 2027 submissions). A search agent opened the arXiv, GitHub, Hugging Face or project page for every item unless it is marked † (search snippet only) or ‡ (OpenReview record only, no arXiv version found). I checked the closest items myself (CRAFT, VideoWeaver, Physical AI Smart Spaces, BWM-Forge, the Cosmos3 Sim-Auto docs). Pages were read through a summarising fetcher, so re-check any number before it goes into a paper. Numbers are the authors' own and come from different protocols.

## 1. Bottom line

- **The basic idea has been done, in narrow forms.**
  - CRAFT ([2604.03552](https://arxiv.org/abs/2604.03552), Apr 2026) places extra cameras in a robot simulator, tiles up to four views of edge maps into one frame, and runs Wan2.1-Fun-Control 1.3B zero-shot.
  - Physical AI Smart Spaces ([2610.02580](https://arxiv.org/abs/2610.02580), NeurIPS 2026 Datasets) runs Cosmos-Transfer on about 1,800 Isaac Sim cameras.
  - BWM-Forge ([2611.16221](https://arxiv.org/abs/2611.16221), Nov 12) renders three views of a Blender scene program through a closed video model.
  - None of them measures whether the views agree.
- **The versions that do enforce agreement are trained and tied to one setting.**
  - NVIDIA's Cosmos3-Nano-Sim-Auto ([2611.06627](https://arxiv.org/abs/2611.06627)) takes simulator controls for 7–11 cameras, but its docs say it "is optimized for the reference camera rig" and "a custom rig can produce degraded results".
  - VideoWeaver ([2603.25420](https://arxiv.org/abs/2603.25420)) takes per-view depth and edges for robot cameras, with no released weights.
  - Game-engine work (CoCode-World, SyncWorld‡) handles two players or one game.
- **Running a single-view model separately per camera is known to give inconsistent views.** At least six papers report it with numbers. None has scored it on simulator controls with simulator correspondences.
- **Nothing covers aerial scenes.** No generative multi-view work has drones, a chase camera, or a ground camera paired with an overhead camera. The re-rendering methods are tested on camera moves of at most about 90–120° around the subject.
- **What looks open:** keeping views consistent with off-the-shelf models across arbitrary, wide-baseline simulator cameras, and measuring per-object identity and label validity across views using the simulator's exact correspondences. No paper measures per-object colour or identity agreement across views.

## 2. The idea in parts

| Part | Published? | Where |
|---|---|---|
| Simulator renders controls for several cameras, a video model generates each view | Yes | CRAFT, Physical AI Smart Spaces, BWM-Forge, NVIDIA's warehouse cookbook |
| …with off-the-shelf models, no training | Yes | CRAFT (Wan2.1-Fun-Control, zero-shot), Smart Spaces and the NVIDIA cookbook (Cosmos-Transfer per camera) |
| …with views kept consistent by a trained joint model | Yes, for fixed rigs | Cosmos3 Sim-Auto (car rig), Cosmos-Transfer2.5 multiview, VideoWeaver and RoboTransfer (robot rigs), CoCode-World and SyncWorld‡ (games) |
| …with consistency measured | Partly | Matched pixels, Sampson error or MEt3R on robot and driving rigs. Never per object, never against simulator correspondences |
| …with labels checked per view | No | Labels are reused, not verified. Smart Spaces projects 3D boxes into each view for manual checking |
| Arbitrary, wide-baseline cameras (ground + chase + overhead) | No | Nearest: SynCamMaster's hemisphere of cameras, with text control only |
| Aerial or drone scenes | No | — |

## 3. Closest prior work

| Paper | Controls and views | How views are kept consistent | What is missing for us |
|---|---|---|---|
| **CRAFT** · [2604.03552](https://arxiv.org/abs/2604.03552) · Apr 2026 · USC | RoboTwin simulator → Canny edges. "We place additional cameras inside the simulator and tile up to four simultaneous views into a single image." Wan2.1-Fun-Control 1.3B, zero-shot, with a reference image | Tiling: one generation holds all views. "Tiling ensures spatial consistency across all viewpoints" (asserted, not measured) | No consistency metric; judged by robot-policy success. "Distant views produce noisy Canny-edge representations", so cameras must stay close. Tiling shrinks each view's resolution |
| **Physical AI Smart Spaces** · [2610.02580](https://arxiv.org/abs/2610.02580) · Oct 2026 | Isaac Sim, about 1,800 cameras, 280 h of 1080p. Cosmos-Transfer for appearance. 3D boxes and cross-camera identities kept as labels | Per camera, no consistency step (per the search agent's full-text read) | No cross-view consistency measure. The label tool projects world boxes into each view "for cross-camera verification" |
| **BWM-Forge** · [2611.16221](https://arxiv.org/abs/2611.16221) · Nov 12, 2026 | A coding agent writes a Blender scene program; 3 synchronized views (per full-text read) rendered as proxy videos through closed Seedance 2.5 | Only the shared scene program | Keyframed motion, no physics; no cross-view measure |
| **CoCode-World** · [2611.15602](https://arxiv.org/abs/2611.15602) · Nov 11, 2026 | GTA V map and engine state → proxy videos for 2 players | Side-by-side in one generation, LoRA on MiniMax H3 33B; player identity across views is measured | Two views, one game. MiniMax H3's licence is a problem for us (earlier review) |
| **SyncWorld: Many Views, One World**‡ · OpenReview 38wMvpoPBG | CS:GO engine state → per-observer projections → one shared renderer; 10 concurrent views | LoRA and adapters; proposes SymMVC, a reprojection consistency metric | One game; no arXiv version |
| **Cosmos3-Nano-Sim-Auto / Transfer-Auto** · [2611.06627](https://arxiv.org/abs/2611.06627) · Nov 4, 2026 | Per-camera scenario renders (boxes, lanes, signs) + LiDAR from an external driving simulator; 7–11 cameras | Joint model with per-camera embeddings; Sim-Auto is 4-step and causal. Cross-view Sampson error is reported | Fixed car rig; a new rig needs post-training on paired data. Driving controls only. Open weights (OpenMDW-1.1) |
| **Cosmos-Transfer2.5 multiview** · [2511.00062](https://arxiv.org/abs/2511.00062) | Driving: per-view scenario maps for a 7-camera rig. Robot: one third-person video → 2 new views | Views concatenated along time with per-view embeddings | Fixed rigs. NVIDIA Open Model License, which ends if guardrails are disabled (earlier review) |
| **VideoWeaver** · [2603.25420](https://arxiv.org/abs/2603.25420) · CVPR 2026† | Per-view depth + Canny; 3 views natively, more added one at a time; heterogeneous robot cameras with "limited or even negligible overlap" | Shared 4D latent from a 3D foundation model (Pi3); in-house 11B model | No weights. Robot scenes; fails on small objects |
| **RoboTransfer** · [2505.23171](https://arxiv.org/abs/2505.23171) | Per-view depth + normals + appearance references; 3 robot cameras | Views concatenated in one latent | Sim-to-real results are qualitative only; code "will be released" |
| **RoboRender** · [2610.09254](https://arxiv.org/abs/2610.09254) · Oct 2026 | Simulator depth + robot mask → Wan2.1-T2V-1.3B fine-tuned on about 130K real robot clips; 3 views | One stacked multi-view latent | No cross-view metric. Real-robot success 71%, 7.1× the raw-simulator baseline |
| **NVIDIA Cosmos Cookbook, warehouse multi-view** ([link](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-warehouse-mv/inference.html)) | Isaac Sim, 6 cameras → Cosmos-Transfer1, depth 0.5 + edge 0.5 | Independent per view: same prompt, same weights | Consistency claimed, not measured. Detector mAP50: simulator 0.627, Cosmos 0.686, both 0.742 |

## 4. Evidence that per-view generation disagrees

| Source | Setting | Per view vs joint |
|---|---|---|
| SynCamMaster · [2412.07760](https://arxiv.org/abs/2412.07760) · ICLR 2025 | Per-view image-to-video from a shared first frame | Matched pixels 117–173K vs 527K joint |
| Cosmos-Transfer2.5 robot multiview · [2511.00062](https://arxiv.org/abs/2511.00062) | Robot cameras | Cross-view Sampson error 26.61 vs 19.73 px |
| EMMA · [2509.22407](https://arxiv.org/abs/2509.22407) | Cosmos-Transfer1 run per view vs joint | Matched pixels 2097 vs 3270 |
| CVD · [2405.17414](https://arxiv.org/abs/2405.17414) · NeurIPS 2024 | Per-pair baselines | Epipolar precision 16.5–23.5 vs 51.0 |
| MV-Forcing · [2607.05376](https://arxiv.org/abs/2607.05376) · ECCV 2026 | One view re-rendered per view | Matched pixels 144–147K vs 239–244K |
| CoDrive · [2609.34749](https://arxiv.org/abs/2609.34749) | Wan2.2-I2V with per-view first frames | Static geometry close (1.82 vs 1.64), same-vehicle hit rate 0.302 vs 0.482 |
| Config + AWS blog, Jul 2026 ([link](https://aws.amazon.com/blogs/physical-ai/how-config-scales-robot-training-data-without-scaling-data-collection/)) | Cosmos-Transfer per camera on robot demos | "Colors drift between the separately generated views", ghost objects; moved to a joint model |
| VideoWeaver | — | "Naively applying single-view models independently to each camera leads to inconsistent appearance across views" (no number) |

This matches our single-view results. Depth control kept geometry but invented colours that changed with the seed, and box control took identity only from frame 0. Generating each view separately would amplify both problems.

## 5. Ways to keep views consistent

### 5.1 Trained joint models

- **Driving rigs:** MagicDrive / MagicDrive-V2, Panacea, DiVE, Delphi, DriveScape, UniMLVG, Drive-WM, DriveArena + DreamForge (closed loop with a traffic simulator), HelloWorld ([2609.28931](https://arxiv.org/abs/2609.28931)), Cosmos-Drive-Dreams ([2506.09042](https://arxiv.org/abs/2506.09042)), and Cosmos3 Sim-Auto. Most enforce agreement only between neighbouring cameras on a fixed rig. MagicDrive lists "limited camera-parameter generalization".
- **Arbitrary cameras with text or camera control only:**
  - SynCamMaster: open Wan2.1-1.3B module, trained on 3.4K Unreal scenes × 10 cameras.
  - MV-Forcing (Wan2.1-1.3B).
  - SyncVid ([2610.31765](https://arxiv.org/abs/2610.31765)): LoRA on Wan with noise shared in 3D.
  - CoDrive (two vehicles, 8–34 m apart).
  - Prisma-World ([2606.09507](https://arxiv.org/abs/2606.09507)).
  - Khora ([2608.08600](https://arxiv.org/abs/2608.08600)).
  - None takes per-view control maps.
- **Re-rendering one video to other cameras:** ReCamMaster, TrajectoryCrafter, EX-4D, GEN3C, Reshoot-Anything ([2604.21776](https://arxiv.org/abs/2604.21776), LoRA on Wan2.2-I2V-A14B) and Track2View ([2606.15534](https://arxiv.org/abs/2606.15534)). Track2View takes paired 3D point tracks, which our simulator gives exactly, but its weights are not released. All are tested on moves of about 90–120° at most, and depth or tracking errors are their main stated failure.

### 5.2 Training-free options that fit our models

| Option | What it fixes | Catch |
|---|---|---|
| Tile the views into one frame (CRAFT) | One generation sees all views | Each view gets a fraction of 832×480. Our drones are 70–109 px wide and the convoy vehicles 30–50 px, so tiling would push them below a few tokens (see the first review on token size) |
| Same prompt, seed and reference images per view (VACE reference mode, Phantom-Wan) | Object class and colour scheme | Not exact texture agreement; untested on our scenes |
| Wan-Move tracks projected from the same 3D surface points | Motion agrees across views | First frames must already agree |
| Consistent first frames across views | Appearance, for the frame-0 conditions | Our frame 0 is hand-edited for one camera; editing several views to match by hand is not realistic |
| Generate one "hero" view, then re-render it to the others, with the re-rendering anchor built from simulator geometry instead of estimated depth | Exact agreement where the hero view saw the object | Surfaces the hero view never saw must be invented; wide baselines unproven |
| Feed the simulator's albedo (base colour) as an extra control, as NeuralSRender ([2610.29200](https://arxiv.org/abs/2610.29200)) does | Colours fixed by the simulator in every view | NeuralSRender is a trained LoRA on Wan2.2-TI2V-5B, single view, and does not measure colour retention |
| Shared or mesh-attached noise (EquiVDM [2504.09789](https://arxiv.org/abs/2504.09789), Go-with-the-Flow) or latent aggregation in texture space (Tex4D [2410.10821](https://arxiv.org/abs/2410.10821)) | Agreement anchored to the 3D surface | Every working DiT version used a fine-tune or LoRA; untested on plain Wan |
| Generate, fit a 4D Gaussian scene, re-render, regenerate at low control strength (4DGS-Fixer [2609.21176](https://arxiv.org/abs/2609.21176), Difix3D+) | Repairs disagreements | Plausible, unproven with Wan; costly |

### 5.3 Texture the simulator instead

Generate realistic textures for the 3D assets once, then render every view in the simulator, which makes agreement automatic.

- **Texturing methods:** TEXTure, Text2Tex, Paint3D, SyncMVD, MV-Adapter, SceneTex, SceneHI ([2609.10363](https://arxiv.org/abs/2609.10363)), and Tex4D (animated meshes).
- **Comparisons with video models exist only for robots and still images:**
  - Augmentron ([2611.09145](https://arxiv.org/abs/2611.09145), Nov 6) edits backgrounds once and renders them into 4 cameras. Out-of-distribution success is 66.0 vs 53.3 for a per-frame generative method, at 0.93× source runtime vs 16–21× per view for generative methods. Single seed, no consistency metric.
  - SynthRender ([2602.21141](https://arxiv.org/abs/2602.21141)): generated asset textures did not beat random textures for training a detector (mAP50 85.9–86.8 vs 95.1).
- **No one compares this route with controlled video models on realism or label validity for video.**

## 6. Aerial scenes and validation data

- **No generative multi-view work on drones, aerial or air-to-ground scenes was found.**
  - Cross-view aerial work generates static ground views from satellite or aerial images: Sat2Scene, Streetscapes, SatDreamer360, AerialGo, Skyfall-GS, Top2Ground.
  - Simulators already render aligned air and ground cameras without any generative step: CARLA-Air ([2603.28032](https://arxiv.org/abs/2603.28032)), Griffin ([2503.06983](https://arxiv.org/abs/2503.06983)), MATRIX ([2511.08615](https://arxiv.org/abs/2511.08615)).
- **Real multi-camera drone data that could validate a method:**
  - DPJAIT (Scientific Data 2025): 4 synchronized indoor cameras with 1–4 drones and Vicon ground truth, plus 13 AirSim sequences with 4–8 virtual cameras. CC BY 4.0. The best paired sim/real multi-view drone set found.
  - ETH multi-view drone tracking ([GitHub](https://github.com/CenekAlbl/drone-tracking-datasets)): 4–7 outdoor cameras with RTK ground truth.
  - MAVREC (CVPR 2024): one ground and one drone camera, synchronized, with ground targets.
  - MDMT and MDMOT: two to five drones viewing the same ground scene.

## 7. Metrics

- **Matched pixels** (the SynCamMaster protocol, reused by most papers above) and **cross-view Sampson error** (Cosmos-Transfer2.5). With the simulator's exact cameras, Sampson error becomes a direct epipolar error.
- **MEt3R** ([2501.06336](https://arxiv.org/abs/2501.06336)) is common, but papers use opposite sign conventions. One study (2605.18754†) finds it unreliable and COLMAP-based metrics up to 4× better correlated with human judgement.
- **Conflicting numbers:** Cosmos-Transfer2.5's cross-view Sampson error is 2.310 in its own paper and 6.549 in HelloWorld's protocol.
- **No standard cross-view identity metric.** CLIP-based similarity is too coarse. With the simulator we could crop each object ID in every view at the same time step and compare colour, embeddings and counts. This would be new.

## 8. What this means for our project

This section is analysis, not literature.

- **Our simulator is nearly ready.** The renderer already saves camera matrices and world-frame object poses, and computes boxes and tracks by projecting world points, so extra cameras are extra render passes. The two drone scenes use a 3D sky and ground and can take new viewpoints. The convoy's ground is a flat aerial photo, so oblique views of it would look flat.
- **A method claim alone would be weak.** CRAFT already does the zero-shot version on Wan, and NVIDIA ships the trained version for cars.
- **The measurement is open and fits our framing.** The earlier framing was "Do simulator labels survive generative sim-to-real?". Here it becomes: do they survive across views, and does the same drone keep its identity and colour in every view? We have what other papers lack: exact correspondences, per-object identities and visibility in every view, from the simulator.
- **Cost multiplies with views.** Three views at 13.5 minutes each is about 40 minutes per scene. Few-step models (first distillation review) matter more here.
- **The real-data check** could be DPJAIT, which has synchronized drone cameras and an AirSim twin.

## 9. Options, in order of cost

| Option | Training | What it tests |
|---|---|---|
| 1. Render 2–3 cameras of `drones_air_phys`; run VACE or Wan-Move per view with the same prompt and seed; score cross-view identity, colour and count against simulator correspondences | None | The baseline nobody has measured on simulator controls |
| 2. Add the cheap fixes one at a time: shared reference crops, tracks from the same 3D points, tiling (CRAFT) | None | Which training-free fix helps, and what tiling costs in object size |
| 3. Hero view + re-render with a simulator-built anchor | None, if Reshoot-Anything or EX-4D weights work | Whether exact geometry fixes re-rendering's main failure |
| 4. Albedo control or textured simulator assets | LoRA, or none for texturing | Whether fixing colour in the simulator beats asking the model to keep it |
| 5. A joint multi-view model with per-view controls | Heavy | Only if 1–4 fail |

## 10. Very recent items (Jun – Nov 2026)

- **Nov:** BWM-Forge [2611.16221](https://arxiv.org/abs/2611.16221) (Nov 12) · vid2vid-long [2611.17298](https://arxiv.org/abs/2611.17298) (Nov 12, single view; reuses earlier output at matching locations) · CoCode-World [2611.15602](https://arxiv.org/abs/2611.15602) (Nov 11) · Augmentron [2611.09145](https://arxiv.org/abs/2611.09145) (Nov 6) · Interactive World Simulation with Cosmos [2611.06627](https://arxiv.org/abs/2611.06627) (Nov 4)
- **Oct:** SyncVid [2610.31765](https://arxiv.org/abs/2610.31765) · VPM [2610.28594](https://arxiv.org/abs/2610.28594) · OverSeer [2610.28554](https://arxiv.org/abs/2610.28554) · WorldCast [2610.12412](https://arxiv.org/abs/2610.12412) · RoboRender [2610.09254](https://arxiv.org/abs/2610.09254) · Artemis [2610.07031](https://arxiv.org/abs/2610.07031) · PROWBench [2610.02205](https://arxiv.org/abs/2610.02205) · Physical AI Smart Spaces [2610.02580](https://arxiv.org/abs/2610.02580) · EMMA v3 [2509.22407](https://arxiv.org/abs/2509.22407)
- **Sep:** CoDrive [2609.34749](https://arxiv.org/abs/2609.34749) · HelloWorld [2609.28931](https://arxiv.org/abs/2609.28931) · TriWorldBench [2609.26314](https://arxiv.org/abs/2609.26314) · 4DGS-Fixer [2609.21176](https://arxiv.org/abs/2609.21176) · SceneHI [2609.10363](https://arxiv.org/abs/2609.10363) · CamTrol++ [2609.03639](https://arxiv.org/abs/2609.03639)
- **Aug:** 4DAnyone [2608.20335](https://arxiv.org/abs/2608.20335) · Stream4D [2608.19556](https://arxiv.org/abs/2608.19556) · CamWorldQA [2608.18710](https://arxiv.org/abs/2608.18710) · Khora [2608.08600](https://arxiv.org/abs/2608.08600) · MASS [2608.06257](https://arxiv.org/abs/2608.06257) · UniWorld-View [2608.04701](https://arxiv.org/abs/2608.04701)
- **Jul:** MV-Forcing [2607.05376](https://arxiv.org/abs/2607.05376) · UniCaMo [2607.02798](https://arxiv.org/abs/2607.02798)
- **Jun:** IOI [2606.23296](https://arxiv.org/abs/2606.23296) · OmniDrive [2606.17536](https://arxiv.org/abs/2606.17536) · Track2View [2606.15534](https://arxiv.org/abs/2606.15534) · HarmoView [2606.10839](https://arxiv.org/abs/2606.10839) · Prisma-World [2606.09507](https://arxiv.org/abs/2606.09507)
- **ICLR 2027 submissions, OpenReview only‡:** SyncWorld (38wMvpoPBG) · SCENORBIT (REb18bqu83, possibly the same work as OmniDrive)

## 11. Coverage and caveats

- **November is thinly indexed;** the arXiv API sweep is the main source for it. Two directly relevant papers appeared on Nov 11 and 12, so more may follow soon.
- **Some queries returned nothing useful.** Two Isaac/MuJoCo multi-view queries returned zero, possibly a query-parsing problem, and several broad queries were mostly noise.
- **ICLR 2027 coverage is keyword search only.** Several submissions found there have no arXiv version (View4D, MultiLive, ViewWeaver, Dual-Sim, Sim-∞) and were not assessed.
- **Some details come from full-text reads by a search agent, not from abstracts:** BWM-Forge's view count and video model, and Smart Spaces' per-camera processing.
- **Metrics differ in protocol and sign convention across papers,** so the numbers in §4 compare per-view against joint within each paper, not across papers.
