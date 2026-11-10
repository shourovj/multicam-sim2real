# Literature Survey and Project Review: Claude Fable 5.2

**Source:** Claude Code 2.1.321 in headless mode (`claude -p`), model `claude-fable-5-2`, `--effort high`. Allowed tools: Read, Grep, Glob, WebSearch, WebFetch, and Agent (it used 7 subagents). Write, Edit, and Bash were blocked. Run on Nov 9, 2026, finished 19:35. **Input:** the same [prompt](2026-11-09-1735-codex-review-prompt.md) as the Codex runs, word for word, and a copy of the [v0 design doc](../2026-11-09-1416-design-doc-v0.md), in an isolated directory. It did not see the repo, the other surveys, or the earlier [Fable paper-direction review](2026-11-09-1918-paper-direction-fable.md). These settings mirror the Codex runs as closely as possible.

The report below is unedited except for this header and the removal of seven progress-update lines ("N of 7 reports are in…") that appeared before it. See the [synthesis](2026-11-09-1750-lit-review-synthesis.md) for how it compares with the other surveys.

---

# Second opinion: simulator-conditioned video generation for drone and overhead scenes

**Bottom line.** The idea is sound, but "a simulator step improves motion over text-to-video" is already established, and the out-of-frame failure was quantified by three mid-2026 benchmarks. What is open is the aerial/overhead domain, drones as subjects, and a controlled conditioning ladder on one backbone scored against simulator ground truth. For this week I recommend Blender (Cycles on CUDA) with a lightweight dynamics script, paired with the Wan2.1-14B family (T2V, I2V, Wan-Move, VACE). Cosmos-Transfer2.5 is an optional second backbone.

**How this was verified.** Seven subagents searched and loaded the pages. I then independently re-fetched 14 load-bearing links (marked ✔). Entries marked (S) were confirmed from search results only. Most 2026 papers post-date my training knowledge, so every fact about them comes from fetched pages. Numbers were extracted by a summarising fetch tool; spot-check any you intend to quote. Nothing was run on the servers.

---

## 1. Literature survey

### A. Code-planned and simulator-in-the-loop generation (closest to the project)

| Paper | Simulator step and what the video model receives | Baselines and key result | Relevance |
|---|---|---|---|
| **CoDeR: Code Plans, Diffusion Renders** — Fang, arXiv Sep 2026 ✔ [2609.26458](https://arxiv.org/abs/2609.26458) | LLM agents write Three.js "whitebox" worlds. The renderer gets whitebox RGB video, ground-truth depth and a text prompt. The renderer is trained (LoRA plus ControlNet branch). | WorldScore 87.88 vs LingBot-World 74.79 and Matrix-Game 3.0 74.20; motion accuracy 82.76 vs 60.79. Code not released. | Closest architecture, but baselines are interactive world models. No text-only or first-frame arm; not runnable. |
| **Code World Model** — Chen, arXiv Aug 2026 [2608.25927](https://arxiv.org/abs/2608.25927) | Coding agent adapts game-engine code. The renderer gets a 336×192 proxy video (depth plus semantic IDs, boxes as fallback). LoRA trained on GTA V. | Qualitative only. | Shows a coarse proxy is enough for entity and camera control. |
| **VideoCoCo** — Li, arXiv Jul 2026 ✔ [2607.27380](https://arxiv.org/abs/2607.27380) | Agent writes Blender Python, which renders a "white-clay" draft video. The draft goes to a video editor. | PhyGenBench 0.475 (base) to 0.506 (tuning-free draft) to 0.558 (LoRA). | Cleanest same-backbone evidence that a simulator draft helps. No first-frame arm. |
| **PSIVG: Physical Simulator In-the-Loop Video Generation** — Foo, CVPR 2026 ✔ [2603.06408](https://arxiv.org/abs/2603.06408) | A T2V template is reconstructed to meshes and simulated (Taichi MPM). Simulator optical flow goes to Go-with-the-Flow as warped noise, plus first frame. Training-free. | User preference for physical plausibility 82.3% vs CogVideoX 7.2%. No code. | Admits MPM fails on vehicles and the flow route struggles with small or thin objects. |
| **PhysAgent** — Li, arXiv Jul 2026 ✔ [2607.16355](https://arxiv.org/abs/2607.16355) | VLM writes a physics program, with a verify-and-repair loop. Simulated optical flow goes to the generator. | 0.714 vs Wan2.2-14B 0.633. Without the repair loop: 0.631, level with plain Wan2.2. | Caution: a one-shot simulator pass may not beat the baseline. |
| **GPT4Motion** — Lv, CVPR 2024 Workshops [2311.12631](https://arxiv.org/abs/2311.12631) | GPT-4 writes Blender scripts. Canny and depth maps go to SDXL ControlNets. Training-free. | Wins its user study vs AnimateDiff; three scenario types only. | The original text → code → Blender → diffusion recipe. |
| **PhysGen** — Liu, ECCV 2024 [2409.18964](https://arxiv.org/abs/2409.18964) | 2D rigid-body simulation (Pymunk) from an image. Composite video plus masks are refined by a video model. Code public. | Baselines include I2VGen-XL, SEINE, Kling (numbers not read). | 2D rigid top-down motion is structurally close to the convoy. |
| **PhysGen3D** — Chen, CVPR 2025 [2503.20746](https://arxiv.org/abs/2503.20746) | Single-image 3D reconstruction, MPM, Mitsuba render. The render is the output. | Physical realism 3.71 vs Kling 2.81; photorealism lower than Gen-3 (3.41 vs 3.58). | Simulation alone wins physics and loses realism. |
| **WonderPlay** — Li, ICCV 2025 [2505.18151](https://arxiv.org/abs/2505.18151) | Genesis solvers on a scene from one image. Flow (warped noise), coarse RGB and mask go to Go-with-the-Flow. | Physics preference 80.2% vs CogVideoX-I2V. Realism drops from 0.700 to 0.650 without RGB. | Direct evidence against a text+first-frame baseline. No code found. |
| **NewtonGen** — Yuan, ICLR 2026 [2509.21309](https://arxiv.org/abs/2509.21309) | Learned neural ODE predicts motion; optical flow goes to Go-with-the-Flow. | Physical Invariance Score 0.980 vs 0.910 (Sora) on parabolic motion. | Text-only baselines; an invariance metric that suits "constant convoy speed". |
| **ChronoAgentic** — Wang, arXiv May 2026 [2605.14398](https://arxiv.org/abs/2605.14398) | Agents write PyChrono programs. No video diffusion. | 82.5% joint correctness vs 52.5% for the best of ten T2V models. | Chrono is a vehicle-dynamics engine: the natural simulator for a convoy later. |

Also verified in this family: RealWonder ([2603.05449](https://arxiv.org/abs/2603.05449)), PhysCtrl (NeurIPS 2025, [2509.20358](https://arxiv.org/abs/2509.20358)), MotionCraft (NeurIPS 2024, [2405.13557](https://arxiv.org/abs/2405.13557)) and the prompt-only control PhyT2V (CVPR 2025, [2412.00596](https://arxiv.org/abs/2412.00596)).

### B. Generators conditioned on scene state rather than a rendered video

| Paper | What the model ingests | Baselines and key result | Relevance |
|---|---|---|---|
| **Wan-Move** — Chu, NeurIPS 2025 ✔ [2512.08765](https://arxiv.org/abs/2512.08765), [code](https://github.com/ali-vilab/Wan-Move) ✔ | First frame plus point tracks and visibility (`.npy`). First-frame latent features are copied along each track. Wan2.1-I2V-14B, 832×480, Apache-2.0. | MoveBench end-point error 2.6 vs MagicMotion 3.2, Tora 3.3. Multi-object 2.2 vs Tora 3.5. | Best direct fit: simulator positions project straight to tracks. Array format is undocumented. |
| **ATI: Any Trajectory Instruction** — Wang, arXiv 2025 [2505.22944](https://arxiv.org/abs/2505.22944) | First frame plus point tracks; Gaussian-weighted feature injection. Wan2.1-I2V-14B, Apache-2.0. | No baseline table. A third party (WorldCanvas) measured ObjMC 127.2 vs plain Wan2.2 I2V 139.6. | Backup to Wan-Move; trained on only 1–20 tracks. |
| **Time-to-Move** — Singer, ICLR 2026 [2511.08633](https://arxiv.org/abs/2511.08633) | First frame, a crude warped RGB animation and a mask video. Training-free on Wan2.2 I2V. | Object control distance 13.7 vs Go-with-the-Flow 27.8. | Works on Wan2.2 with no training, but it is the closest to "feeding a rendered video". |
| **Diffusion as Shader** — Gu, SIGGRAPH 2025 [2501.03847](https://arxiv.org/abs/2501.03847) | A "3D tracking video" (points coloured by initial xyz) rendered from Blender meshes. CogVideoX-5B. | Camera rotation error 5.97 vs CameraCtrl 7.82. | Most direct mesh-to-model path; first frame must match exactly. |
| **VerseCrafter** — Zheng, CVPR 2026 ✔ [2601.05138](https://arxiv.org/abs/2601.05138) | Camera path, background point cloud and one 3D Gaussian trajectory per object. Wan2.1-14B, research-only licence. | ObjMC 2.51 vs Uni3C 5.88. | Good fit for a convoy with a moving camera; about 57 GB only in offload mode, heavy setup. |
| **3DTrajMaster** — Fu, ICLR 2025 [2412.07759](https://arxiv.org/abs/2412.07759) | Per-entity 6-DoF pose sequences as tokens in self-attention. Text-to-video, no first frame. | Open release is a CogVideoX LoRA trained on 50 animals, at most 3 entities. | Conceptually what the lead asks for; drones and vehicles are out of distribution. |
| **WorldCanvas** — Wang, arXiv Dec 2025 [2512.16924](https://arxiv.org/abs/2512.16924) | Tracks plus a box and caption per agent. Wan2.2 I2V. | ObjMC 91.1 vs ATI 127.2. | The only entry trained for objects entering mid-clip. No code found. |
| **Go-with-the-Flow** — Burgert, CVPR 2025 [2501.08331](https://arxiv.org/abs/2501.08331) | Optical flow, used only to warp the initial noise. CogVideoX-5B. | Back end of PSIVG, WonderPlay and NewtonGen. Time-to-Move beats it on both of that paper's benchmarks. | Loose adherence; weak on small objects. |
| **VACE** — Jiang, ICCV 2025 [2503.07598](https://arxiv.org/abs/2503.07598) | Control videos: depth, flow, pose, scribble, boxes. Wan2.1 1.3B/14B, Apache-2.0. | Trails specialised methods in two third-party sim2real tests. Open issues: box-track mode broken, quality drop vs base. | The dense-control arm on the same backbone as Wan-Move. |
| **Video Models as Native 4D Renderers** — Chen, arXiv Jul 2026 [2608.00094](https://arxiv.org/abs/2608.00094) | Tracking, world-position and normal buffers into a Wan2.2 adapter. | +1.54 dB PSNR over Wan2.2 depth control. Swapping world position for depth costs 1.26–1.55 dB. | Evidence that depth is a weaker motion carrier than tracking. No code. |

Also verified: MagicMotion (ICCV 2025, native box input, [code](https://github.com/quanhaol/MagicMotion)), Tora ([code](https://github.com/alibaba/Tora)), Motion Prompting (CVPR 2025, [2412.02700](https://arxiv.org/abs/2412.02700), closed), LooseControlVideo ([2606.19495](https://arxiv.org/abs/2606.19495), no weights).

### C. Simulator render to realistic video

| Paper | Simulator and signals passed | Baselines and key result | Relevance |
|---|---|---|---|
| **Cosmos-Transfer1** — Alhaija, arXiv 2025 [2503.14492](https://arxiv.org/abs/2503.14492) | Omniverse / Isaac Lab. Blurred RGB, Canny edge, depth, segmentation. | Edge F1 is only 0.28 under edge-only control. Depth 0.14, segmentation 0.11. No text-only baseline. | Thin structure is not pixel-preserved even with edge control. |
| **Cosmos-Predict2.5 / Transfer2.5** — Ali, arXiv Oct 2025 [2511.00062](https://arxiv.org/abs/2511.00062) | Edge, blur, depth, segmentation. 2B model, 65.4 GB, A100 listed as test hardware. | Edge F1 0.49 vs 0.38 for Transfer1. Robot policy 24/30 vs 5/30 with standard augmentation. | Strongest verified off-the-shelf sim2real model that fits your GPUs. |
| **Cosmos 3** — NVIDIA, Jun 2026 [2606.02800](https://arxiv.org/abs/2606.02800); [Sim-Transfer card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) ✔ | Simulator-rendered depth, segmentation, edge or blur. OpenMDW-1.1. | Own card: edge F1 0.341, segmentation mIoU 0.738. | Card lists Hopper and Blackwell only. A100 support is a gamble. |
| **NeuralSRender** — Lu, arXiv Oct 2026 ✔ [2610.29200](https://arxiv.org/abs/2610.29200) | CARLA / UE5 G-buffers (base colour, depth, normal, metallic, roughness). LoRA on Wan2.2-TI2V-5B. | FVD 599.7 vs 981.8 for Cosmos-Transfer2.5 (edge). Detector AP50 35.1 to 40.6 on nuImages. | Closest published analogue of the planned pipeline; driving only. |
| **Zero-shot Synthetic Video Realism Enhancement** — Wang, arXiv Nov 2025 [2511.14719](https://arxiv.org/abs/2511.14719) | CARLA and GTA-V video inverted and re-denoised with Cosmos-Transfer1 controls. | LPIPS 0.368 vs VACE 0.414 and Cosmos-Transfer1 0.418. | Only small-object-focused evaluation found: traffic lights and signs come out wrong or blurred in the baselines. |
| **RealMaster** — Cohen-Bar, arXiv Mar 2026 [2603.23462](https://arxiv.org/abs/2603.23462) | GTA-V RGB render as in-context reference for a Wan2.2 LoRA. | Depth-conditioned ablation worse than edge. | Admits conservative edits on small distant objects and failure when objects enter mid-sequence. |
| **Cosmos-Drive-Dreams** — Ren, arXiv Jun 2025 [2506.09042](https://arxiv.org/abs/2506.09042) | HD map plus 3D cuboids; no RGB render. | Lane detection F1 0.428 to 0.451. Detection mixed (0.446 to 0.439 at 1:1 ratio). | Downstream gains are not uniform. |
| **SimGen** — Zhou, NeurIPS 2024 [2406.09386](https://arxiv.org/abs/2406.09386); **Dreamland** — Mo, arXiv 2025 [2506.08006](https://arxiv.org/abs/2506.08006) | MetaDrive depth and segmentation, first converted to "real-like" conditions. | Dreamland FID 50.6 vs SimGen 106.0. | Lesson: raw simulator conditions conflict with real priors. |
| **LucidSim** — Yu, CoRL 2024 [2411.00083](https://arxiv.org/abs/2411.00083) | MuJoCo depth and masks to an image ControlNet; simulator flow warps frames. | Real robot: 100% vs 70% and 85% vs 35% against domain randomisation. | Strong conditioning reduces detail; weak conditioning drifts. |

Also verified: DiffusionRenderer (CVPR 2025, [2501.18590](https://arxiv.org/abs/2501.18590)), RoboTransfer ([2505.23171](https://arxiv.org/abs/2505.23171); says Cosmos-Transfer "degrades notably for dynamic viewpoints").

### D. Evidence that generators fail at motion, and metrics

| Paper | What it measures | Headline finding | Relevance |
|---|---|---|---|
| **Physics-IQ** — Motamed, arXiv 2025 [2501.09038](https://arxiv.org/abs/2501.09038) | Continuation of 396 real physics clips, scored by motion-mask IoU. | Best model 29.5/100; Sora 10.0. Realism and physics uncorrelated. | Best "looks real, moves wrong" citation. Copy its noise-ceiling idea. |
| **VideoPhy-2** — Bansal, arXiv 2025 [2503.06800](https://arxiv.org/abs/2503.06800) | Human-rated adherence and physical commonsense. | Best model (Wan2.1-14B) 32.6%. Auto-evaluator correlates with humans at only 0.37. | Supports the claim; shows VLM judges are weak on physics. |
| **T2V-CompBench** — Sun, arXiv 2024 [2407.14505](https://arxiv.org/abs/2407.14505) | Motion direction, counts, spatial relations via detector plus tracker. | Every model below 0.29 on motion direction; counting fails above about 3 objects. | Direct evidence for "convoy turns right" and count failures. Code reusable. |
| **VBench-2.0** — Zheng, arXiv 2025 [2503.21755](https://arxiv.org/abs/2503.21755) | 18 dimensions via VLM question-answering. | Dynamic spatial relationship about 20%; motion order 15–29%. | Event order is documented as weak. |
| **WRBench** ("Current World Models Lack a Persistent State Core") — Lu, arXiv Jun 2026 ✔ [2606.20545](https://arxiv.org/abs/2606.20545) | 9,600 videos from 23 models; targets leave view by camera motion. | Returning targets resume in their exit state; scaling Wan2.1 from 1.3B to 14B did not help. | Closest prior statement of the lead's out-of-frame claim. |
| **MemoBench** — Chen, arXiv Jun 2026 ✔ [2606.27537](https://arxiv.org/abs/2606.27537) | Disappear-and-reappear with Unreal ground truth; SAM 3 reappearance score. | No model exceeds 0.6 (Wan2.2 0.328). | Quantified out-of-frame failure; metric reusable. Camera-induced exit only. |
| **PersistBench** — He, arXiv Sep 2026 [2609.20819](https://arxiv.org/abs/2609.20819) | 360° ground truth, so off-screen objects stay known. | Consistency "degrades significantly once objects leave the field of view" (abstract only). | Third out-of-frame benchmark. |
| **WorldBench** — Upadhyay, arXiv Jan 2026 [2601.21282](https://arxiv.org/abs/2601.21282) | Kubric scenes; SAM 2 with ground-truth boxes, then mask IoU. | Cosmos models 0.31–0.37 mIoU on permanence. | A ground-truth-referenced recipe to copy. |
| **Video models are zero-shot learners and reasoners** — Wiedemer, arXiv 2025 [2509.20328](https://arxiv.org/abs/2509.20328) | Qualitative Veo 3 study. | Optimistic counter-position. | Cite as the opposing view. |
| **NeuS-V** — Sharan, CVPR 2025 [2411.16718](https://arxiv.org/abs/2411.16718) | Prompt to temporal logic; VLM proposition confidences; model checking. | Pearson 0.48–0.62 with humans. Cannot penalise hallucinated extra objects. | Fits event order. Here, derive propositions from tracks instead of a VLM. |
| **On the Content Bias in FVD** — Ge, CVPR 2024 [2404.12391](https://arxiv.org/abs/2404.12391); **JEDi** — Luo, ICLR 2025 [2410.05203](https://arxiv.org/abs/2410.05203) | Critiques of FVD. | FVD barely moves under temporal corruption and needs large samples. | Do not report FVD on 30 videos. |

Also verified: Kang et al. (ICML 2025, [2411.02385](https://arxiv.org/abs/2411.02385)), MotionCtrl's ObjMC metric ([2312.03641](https://arxiv.org/abs/2312.03641)), CoTracker3 ([2410.11831](https://arxiv.org/abs/2410.11831)), SAM 3 ([2511.16719](https://arxiv.org/abs/2511.16719)).

### E. Drone and overhead domain

| Paper | What is generated, from what | Baselines and key result | Relevance |
|---|---|---|---|
| **AirScape** — Zhao, arXiv 2025 [2507.08885](https://arxiv.org/abs/2507.08885) | Egocentric UAV video from frame plus motion intention; CogVideoX fine-tune. | Intention alignment 84.5% vs 31.8% for the best open baseline. | Open models follow aerial motion poorly zero-shot. Drone is the camera. |
| **MotionScape** — Guo, arXiv 2026 ✔ [2604.07991](https://arxiv.org/abs/2604.07991) | Real UAV-view benchmark, 228 clips. | PSNR drops 27–35% from low to high motion on Wan2.2, Cosmos-Predict2.5 and others. | Zero-shot weakness on aerial views quantified. |
| **Aero-World** — Al Radi, arXiv 2026 [2605.19728](https://arxiv.org/abs/2605.19728) | Egocentric aerial video from inertial action tokens. | FVD 596.5 vs AirScape 1058.6. | Nearest "drone dynamics" benchmark, still egocentric. |
| **SynDroneVision** — Lenhard, WACV 2025 [2411.05633](https://arxiv.org/abs/2411.05633) | Unreal 5 renders of drones as targets; no generative model. | mAP50 on real data: synthetic-only 0.716, real 0.896, both 0.944. | The simulator-only baseline for drones as subjects. |
| **Drone Detection … Pure Synthetic Data** — Wisniewski, arXiv 2024 [2411.09077](https://arxiv.org/abs/2411.09077) | Blender Cycles with domain randomisation. | Synthetic vs real AP50: MAV-VID 97.0 vs 97.8; Anti-UAV 67.8 vs 97.7. | Gap is small for large close drones, large for tiny ones. |
| **Vehicle Perception from Satellite** — Zhao, TPAMI [2402.00703](https://arxiv.org/abs/2402.00703) | 12 real satellite videos plus 14 GTA-V synthetic ones. | No sim2real ablation in the abstract. | The only synthetic satellite vehicle video found. |
| **RarePlanes** — Shermeyer, WACV 2021 [2006.02963](https://arxiv.org/abs/2006.02963) | Simulator-rendered overhead aircraft. | mAP: real 73.3, synthetic-only 54.9, synthetic plus 10% real 69.2. | Defense-adjacent precedent for the sim2real gap. |
| **AeroGen** — Tang, CVPR 2025 (S) [2411.15497](https://arxiv.org/abs/2411.15497) | Remote-sensing images from box layouts. | mAP gains of 3.7 and 4.3 on DIOR variants. | Possible first-frame generator; weights release unconfirmed. |

**Gap.** No verified work pairs a simulator with video diffusion for drones as subjects or for overhead convoys. No T2V benchmark was found with drone-subject or top-down traffic categories (absence of evidence only).

---

## 2. The lead's open question: scene state versus rendered video

**Yes, models can take scene motion without a rendered video.** But the strong open ones still place that state on the pixel or latent grid, and all the practical ones need a first frame. The useful distinction is how much appearance and how dense a signal the condition carries, not whether it arrives as a video file.

From least to most constraining:

1. **Token-level state.** 3DTrajMaster injects pose embeddings into attention; nothing is rasterised. It needs paired training, and the open weights cover animals only.
2. **Sparse tracks on the latent grid.** Wan-Move, ATI and WorldCanvas. The condition carries no rendered appearance; everything off-track (dust, shadows, rotor blur, background) is generated.
3. **Noise warping.** Go-with-the-Flow. A flow field shapes only the noise prior; adherence is loose.
4. **Rasterised geometry.** Diffusion as Shader, VerseCrafter, VACE boxes. No texture, but dense structure.
5. **Depth, segmentation, edge or RGB control.** Cosmos-Transfer, VACE depth. This is the regime the lead calls style transfer.

**The lead's intuition half holds.** Dense control fixes every silhouette and leaves only texture, so "style transfer" is fair. Three qualifications:

- **Even dense control is not faithful.** Edge F1 is 0.28–0.49 under edge-only control, and third parties report spurious objects. It is not reliable style transfer for small objects either.
- **The published "code world models" are in the rendered camp.** CoDeR passes whitebox RGB plus depth; Code World Model passes a low-resolution proxy video. Both train their renderer.
- **Appearance leaks through the first frame.** Wan-Move and ATI are image-to-video, so if frame 0 is a simulator render, the simulator's look enters anyway.

**Open models that take simulator state most directly, no training:**

| Model | Simulator export | Notes |
|---|---|---|
| Wan-Move | Project object points to (x, y, t, visibility) | Apache-2.0, 40 GB with offload. Array format undocumented; a user reports about 30 min per clip on an H100. |
| ATI | Same tracks, YAML | Apache-2.0. Use few tracks per object. |
| Time-to-Move | Sprites dragged along simulator paths, plus masks | Training-free on Wan2.2; the input is really a crude RGB video. |
| Diffusion as Shader | Blender tracking video or 4×4 pose matrices | 32 GB, CogVideoX base, so it breaks a single-backbone design. |
| VerseCrafter | Camera path plus per-object ellipsoid trajectories | Research-only licence; fits A100 only with offload. |

My recommendation: make this gradient the experiment. Run sparse tracks and dense control side by side on one backbone and measure what each preserves and what each lets the model add.

---

## 3. Simulator and generator pairings

**Simulator for all three: Blender Cycles on CUDA, from the tarball (no sudo).** The alternatives are blocked or risky:

- **Isaac Sim** is officially unsupported: "GPUs without RT Cores (A100, H100) are not supported" (confirmed via a tool summary of the requirements page; not checked verbatim). This removes the v0 doc's "fits Cosmos Transfer best" option.
- **Unreal-based simulators** (CARLA, CARLA-Air, Project AirSim) depend on a working Vulkan install you cannot fix without root.
- **EEVEE** depends on NVIDIA's EGL libraries being present; a user report on an A100 shows it failing while Cycles worked.

Run a 15-minute pre-flight first: driver version, `ldd --version`, and whether the EGL and Vulkan vendor files exist.

**Generator for all three: the Wan2.1-14B family at 832×480, 81 frames.** T2V-14B, I2V-14B-480P, Wan-Move and VACE-14B share the same base model, so the comparison isolates conditioning. All are Apache-2.0. The v0 doc mixes Wan2.2 baselines with a Wan2.1 control model, which confounds backbone with conditioning.

| Video | Simulator recipe | Setup cost | Main risks |
|---|---|---|---|
| **1. Take-off** | RotorPy (pip, CPU) for quadrotor dynamics, or a spline with attitude derived from acceleration. Export poses, keyframe a CC-BY drone model in Blender over a Poly Haven HDRI and ground. | About 1 day to first clip | Rotor meshes may need splitting by hand. Tracks sit on the body, so rotor behaviour is up to the model. |
| **2. In the air** | Same pipeline, static ground camera or chase camera. Have one drone leave the frame and return. | 2–3 hours after video 1 | Canny fails on a drone against sky. Wan-Move's behaviour when a track goes invisible and returns is unknown. |
| **3. Convoy** | Kinematic bicycle model along a hand-drawn centreline over a public-domain NAIP orthophoto; Blender nadir camera. Fallback: 2D sprite compositing. | 0.5–1 day | Vehicle size (see below). No evidence any open model has nadir priors. Vehicle asset licences unverified. |

**Risks across all three:**

- **Object size.** From my memory of Wan2.1's architecture, one latent token covers about 16×16 pixels at this resolution. A 10-pixel satellite-scale vehicle is smaller than a token. Render vehicles at least 40 pixels long and call the scene "overhead", not "satellite", in v0.
- **Objects must be visible in frame 1.** Wan-Move and ATI copy first-frame features, so an object entering later has nothing to copy.
- **VACE-14B on an A100** has no stated memory figure. The Wan2.2 Fun-Control card lists A100 as tested and is the fallback.
- **Wan-Move glue.** The track format is undocumented; budget half a day to read the code.

**Second backbone (optional): Cosmos-Transfer2.5.** It fits (65.4 GB), but needs driver ≥570.124 and glibc ≥2.35, which you should check. An NVIDIA collaborator says on a GitHub issue that it is trained on real scenes and degrades on simulator input. Cosmos 3 Sim-Transfer is trained on simulator renders but lists only Hopper and Blackwell.

**Licences matter here.** HunyuanVideo-1.5's licence bans use "for military purposes" ✔, and MiniMax H3's reportedly excludes the USA. Avoid both for AFRL work. Cosmos ships a prompt and output guardrail; whether "military convoy" trips it is unverified. Cosmos 2.5's licence reportedly terminates your rights if you disable the guardrail without a substitute. This is my reading, not legal advice.

---

## 4. Critique

**Is the core idea sound?** Yes, but as an evaluation study in an unexplored domain, not as a method claim. "A simulator step beats pure T2V on physics" has been shown repeatedly, from GPT4Motion (2023) through VideoCoCo and CoDeR (2026). Repeating it on three videos adds nothing.

**The headline comparison is close to circular.** If you give one arm the simulator trajectory and then grade all arms against that trajectory, the conditioned arm wins by construction. A prompt like "drones take off and fly somewhere" does not determine a trajectory, so the text-only arm cannot match it. To make the comparison informative:

- **Score the text-only and first-frame arms on what the prompt specifies.** Count, event order, turn direction, staying on the road until the turn, no teleporting or merging, correct re-entry. Not metric trajectory error.
- **Score the conditioned arms on adherence and on cost.** Trajectory error, plus hallucinated or dropped objects and what the generator added or broke.
- **Add the baselines a reviewer will ask for.** A detailed prompt that spells out the trajectory; the raw simulator render as a faithfulness ceiling; and ideally one closed frontier model.

**"SOTA cannot do this" needs a SOTA model.** Wan2.1 is not the frontier. Without a Veo-class arm, the claim rests on the lead's informal tests, which I have not seen. Use neutral wording ("trucks", "quadcopters") in case an API refuses military prompts.

**What is already known:**

- Generators fail at motion direction, counting and event order (T2V-CompBench, VBench-2.0).
- Out-of-frame failure is quantified (WRBench, MemoBench, PersistBench).
- Simulator-to-real transfer works for driving and tabletop robotics (Cosmos-Transfer and successors).

**What appears to be new:**

- Drones as subjects and overhead vehicle scenes; no simulator-plus-video-diffusion work was found for either.
- Exit and re-entry caused by the object's own motion under a known camera. The three benchmarks all hide the target by moving the camera.
- A conditioning ladder (text, first frame, sparse tracks, dense control) on one backbone, scored against simulator ground truth.
- An independent audit of small-object preservation and hallucination. No verified paper reports size-stratified results.

**Strong result versus trivial result:**

- **Trivial:** "simulator-conditioned video follows the simulator better than text-only video."
- **Strong:** a dose-response curve showing how little state is needed for faithful motion, with a measured price in realism and hallucination at each step. Or: sparse tracks with visibility flags fix re-entry while dense control does not (or the reverse). Or: Cosmos-Transfer and VACE drop or duplicate objects at a measured rate on aerial scenes, contrary to how they are advertised.

**Where I disagree with the v0 doc:**

- **"Runs faster than a full-fidelity render" is likely false.** By estimate, Wan-14B takes 15–30 minutes per 5-second clip on an A100, against seconds per frame for Cycles.
- **"Covers physics the simulator doesn't model" is a hypothesis, not a benefit.** Under dense control the model has little room to add anything; under sparse control it adds things that may be wrong. Test it: does the convoy raise dust on the field?
- **FVD or FID on about 30 videos is not meaningful.** Drop it.
- **The detector train-on-synthetic, test-on-real check does not fit this week.**
- **NeuS-V with VLM-derived propositions is the weaker option here.** You have ground truth and trackers, so derive the predicates from tracks. That also fixes its blind spot for hallucinated extra objects.
- **Under-3-pixel targets** are below what these latent grids can represent. Deferring them was right; they may need a different approach entirely.

**Why not just use the simulator?** For a sim2real programme this is the question that matters, and three videos cannot answer it. The eventual answer has to be a downstream result: a detector trained on generated data beats one trained on raw renders, tested on real footage. Say this in the report so v0 is not read as evidence for it.

---

## 5. Proposed plan for this week

**Conditions (one backbone, 832×480, 81 frames, 5 seeds each):**

| Arm | Input | Model |
|---|---|---|
| S | Raw simulator render (reference) | Blender |
| T1 | Short prompt | Wan2.1-T2V-14B |
| T2 | Detailed prompt describing the trajectory | Wan2.1-T2V-14B |
| I | Prompt plus first frame | Wan2.1-I2V-14B-480P |
| K | Prompt, first frame, sparse tracks with visibility | Wan-Move |
| D | Prompt, first frame or reference, depth or edge video | VACE-14B |
| D2 (optional) | Prompt plus edge and depth | Cosmos-Transfer2.5 |
| F (optional) | T1 and I on one closed frontier model | API |

All image-conditioned arms share the same first frame: the Cycles render of frame 0. Note the appearance leak in the report.

**Size.** Three scenes × five core arms × five seeds is 75 clips, roughly 25–40 GPU-hours at 480p by estimate. That is one overnight run on eight A100s, with room for reruns.

**Metrics:**

- **Extraction.** SAM 3 with a noun prompt for tracks; run it on the simulator renders first to get its error floor. Hand-click centroids on about 10 generated clips to validate it.
- **Count.** Per-frame count error, plus hallucinated and dropped tracks after Hungarian matching.
- **Trajectory (arms K, D only).** Average and final displacement error, normalised by image diagonal.
- **Spec-level (all arms).** Turn direction from signed cumulative heading; event order from track-derived predicates (on road, turning, on field, airborne, out of frame).
- **Re-entry (video 2).** Did the exit happen, did the object return, timing and position error, identity preserved. Report as k of n seeds.
- **Human check.** Blind pairwise choice by three raters on "which follows the described motion" and "which looks real".
- **Reporting.** Per-scene dot plots showing every seed. No significance tests across three scenes.

**Schedule:**

- **Mon–Tue morning:** pre-flight checks; Blender installed; video 3 (simplest) rendered; Wan environments built.
- **Tue 3 PM:** pairing decision as above. Launch T1, T2 and I overnight.
- **Wed:** videos 1 and 2 rendered; Wan-Move track export working; report document started.
- **Thu:** K and D runs; metric pipeline validated on simulator renders.
- **Fri:** tables, side-by-side video grid, failure gallery.

**Deliverable.** One page: a video grid (scene × arm), a table of spec-level pass rates and trajectory errors, the re-entry result, and a short list of observed failures per model.

**What to cut, in order:**

1. The Cosmos arm (D2) and the closed-model arm (F).
2. RotorPy; use a spline with attitude from acceleration.
3. NeuS-V and automated tracking; hand-annotate centroids on eight keyframes per clip, which is feasible at 75 clips.
4. Video 1; video 2 exercises the same pipeline and carries the out-of-frame test.
5. Seeds, down to three.

Do not cut the K-versus-D comparison or the re-entry test; they carry the new content.

---

## 6. What I am least sure about

- **2026 literature.** Most 2026 entries come from fetched pages summarised by a tool. Titles and authors of the 14 re-fetched items are solid; individual numbers are not double-checked.
- **Wan-Move in practice.** Whether it handles a track going invisible and returning, how it behaves with several small objects, and its runtime on an A100.
- **Memory fit.** VACE-14B on an 80 GB A100, and whether Cosmos 3 runs on Ampere at all.
- **Overhead priors.** Whether any open model produces a plausible nadir view of vehicles, with or without conditioning.
- **The measurement pipeline.** Whether SAM 3 tracks small generated vehicles and drones reliably enough to score anything.
- **Rendering on your servers.** Cycles speed, and whether EGL or Vulkan are installed.
- **Runtimes.** Every A100 figure is scaled from other hardware.
- **Licence and guardrail readings.** They need a check by whoever owns compliance for the AFRL work.
- **The lead's prior experiments.** I have not seen them, so I cannot say which models or prompts the "cannot simulate motion" observation rests on.
- **Small discrepancies between sources.** Cosmos-Transfer2.5's stated runtime (264 s vs 870 s), ATI's runtime, and the Go-with-the-Flow repository's organisation name.
