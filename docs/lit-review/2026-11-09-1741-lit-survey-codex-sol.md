# Literature Survey and Project Review: Codex (GPT-6.1 Sol)

**Source:** Codex CLI 0.167.0, model `gpt-6.1-sol`, high reasoning effort, live web search, subagents enabled, read-only sandbox. Run on Nov 9, 2026, finished 17:41. **Input:** only the [prompt](2026-11-09-1735-codex-review-prompt.md) and the [v0 design doc](../2026-11-09-1416-design-doc-v0.md). It did not see the Claude survey or review. The report below is unedited except for this header and a fixed path to the design doc. See the [synthesis](2026-11-09-1750-lit-review-synthesis.md) for how it compares with the other surveys.

---

**My recommendation is to run a small, carefully controlled feasibility study this week.** Use Blender as the common scene and rendering interface, Cosmos-Transfer2.5 as the main generator, and Wan2.1-VACE for comparisons that change the conditioning while keeping the checkpoint fixed. Add a trajectory-only method if time permits.

The core idea is sound, but **simulator-assisted video generation is already established**. The useful research question is now: **how much visual realism can these systems add while preserving specified trajectories, identities, timing, and visibility—particularly for drones and overhead vehicles?**

I read [design-doc-v0.md](../2026-11-09-1416-design-doc-v0.md) and used three focused literature subagents. This report reflects material available by **November 9, 2026**. Paper identities and links were checked against primary sources. Published results below are authors’ reports; I did not execute the generators or modify files.

**1. Literature survey**

The 37 entries below cover executable worlds, simulator-conditioned synthesis, structured motion control, physics, evaluation, and the target domains. Published numbers use different datasets and protocols and should not be compared across rows as a leaderboard. “No training” means no new model training by your team; some methods still perform substantial optimization at inference.

**Executable worlds and structured world representations**

| Paper — first author; year; venue | Intermediate step → actual video-model input | Baselines and key result | Relevance and availability |
|---|---|---|---|
| **1. [Code Plans, Diffusion Renders: Open-Ended Generative World Modeling](https://arxiv.org/html/2609.26458v1)** — Zixun Fang; 2026; arXiv. **CoDeR** | Executable Three.js world with persistent entities and rules → whitebox/appearance-registered RGB, ground-truth depth, and state-derived text. | WorldScore average **87.88**, versus LingBot-World **74.79** and Matrix-Game 3.0 **74.20**. Registration ablation reduces subject consistency **96.31→84.63**. | Essential architectural precedent. Its MiniMax H3 “Artist” receives visual controls and was adapted with LoRA/ControlNet training. Runnable public release **UNVERIFIED**; paper promises code/weights. |
| **2. [State World Models: Execute the Rules, Generate the Pixels](https://arxiv.org/html/2610.26811v1)** — Chan Yan Ki; 2026; arXiv | Executable rules and semantic state ledger, plus clay 3D scene → compiled event instruction, preview video, and visual anchor. | Same frozen MiniMax H3: **32%** PairFollow with rule-only prompts, **64%** with required endings specified, **92%** with full harness, on 50 counterfactual twin pairs. | Particularly useful evidence because it includes a same-backbone comparison. [Runtime released](https://github.com/Zhangyr2022/StateWorldModel), targeting 4/8 A800-80GB GPUs; substantial downloads and setup. Game-rule results do not establish flight fidelity. |
| **3. [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800)** — NVIDIA, corporate author; 2026; arXiv | Unified learned interfaces for text, vision, audio, actions, and spatial controls. Transfer consumes control videos; action pathways use embodiment-specific numeric representations. | PAIBench-C: Nano segmentation mIoU **0.72**, versus Transfer2.5 **0.68**; reported Nano DOVER **10.39**, versus **9.49**. | Major current player omitted from v0. [Open models/code](https://github.com/NVIDIA/Cosmos). Supported action interfaces are not arbitrary scene-state interfaces. A100 compatibility for the latest transfer deployment is **UNVERIFIED**. |
| **4. [Interactive World Simulation with Cosmos for Physical AI](https://arxiv.org/abs/2611.06627)** — NVIDIA, corporate author; 2026; arXiv, November 4 | External simulator maintains physics/state → streaming depth, segmentation, edge, or blurred-image controls → generated observations. Other variants learn action-conditioned dynamics. | [Official project](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/) reports Sim-Transfer edge F1 **59.2% higher** and depth error **26.5% lower** than its Cosmos 3 reference. Reported throughput uses GB200 hardware. | **Direct overlap with the proposed architecture.** Code/checkpoints released. This is particularly important related work for the report’s novelty statement. |
| **5. [Momentum: Persistent Dynamics for Video World Models](https://arxiv.org/abs/2610.23423)** — Nedko Savov; 2026; arXiv | CARLA data; synchronized eight-camera rigs form evolving video memory → target-camera video conditioned on that memory. | Reports **126.2% improvement** in dynamic-scene consistency over static-image conditioning on unseen street scenes. | Directly relevant to objects moving outside the field of view and reappearing. It maintains dynamic visual memory rather than authoritative physical state. Paper promises code/weights; runnable release **UNVERIFIED**. |
| **6. [InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models](https://arxiv.org/html/2412.03934v2)** — Yifan Lu; 2025; ICCV, 2024 preprint | Map/box-conditioned semantic voxel world → rendered semantic and XYZ-coordinate buffers plus initial-frame latent; subsequently lifted into dynamic 3D Gaussians. | Extends 25-frame SVD-XT to **200-frame** sequences; reports better long-horizon FID than Panacea/Vista. Their conditioning capabilities differ. | Strong precedent for geometry, instance persistence, and world-guided synthesis. [Code/checkpoints released](https://github.com/nv-tlabs/InfiniCube). Street-view training makes overhead, off-road transfer uncertain. |
| **7. [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/html/2403.06845v2)** — Guosheng Zhao; 2025; AAAI, 2024 preprint | LLM-generated Python invokes trajectory functions; trajectories condition HD maps → camera-projected structural controls, appearance text, optional initial images. | Reports nuScenes **FID 11.2 / FVD 55.7**, approximately 30%/50% relative improvements over its comparison, and downstream detection/tracking gains from augmentation. | Close code-plan → layout → video precedent. [Inference code released](https://github.com/f1yfisher/DriveDreamer2); weights linked through Baidu. Driving-camera assumptions differ from overhead convoy footage. |
| **8. [WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation](https://arxiv.org/html/2312.02934v3)** — Jiachen Lu; 2024; ECCV, 2023 preprint | Semantic 4D voxel occupancy/maps → learned sparse-3D encoder and camera sampling → ControlNet features. | Reports **FVD 417.7**, versus DriveDreamer **452.0** and DriveGAN **502.3**; capabilities differ across methods. | A useful example of a learned geometric-state interface rather than RGB editing. [Code released](https://github.com/LogosIntelligenceLab/WoVoGen); complete pretrained video checkpoints **UNVERIFIED**. Poor Friday dependency. |
| **9. [PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](https://arxiv.org/html/2509.20358v1)** — Chen Wang; 2025; NeurIPS | Learned physics model predicts 2,048-point trajectories from geometry, forces, and material parameters → trajectories projected into pixel space → trajectory-controlled generator. | Trajectory vIoU **77.03%**, versus adapted MDM **53.78%**. A small 12-video GPT evaluation reports physical-commonsense **4.5**, versus Wan2.1 **3.7**. | Shows how simulator-like dynamics can be converted into a supported generator interface. [Code/model links released](https://github.com/cwchenwang/physctrl). Material deformation differs from powered flight and coordinated driving. |
| **10. [3DPhysVideo: Consistency-Guided Flow SDE for Video Generation via 3D Scene Reconstruction and Physical Simulation](https://arxiv.org/html/2605.16795v1)** — Hwidong Kim; 2026; arXiv | Reconstructed scene point cloud and simulated particles → rendered point-cloud videos → inversion and consistency-guided optimization of pretrained I2V generation. | Reports VideoPhy score **0.266**, versus VLIPP **0.244**, PhysCtrl **0.218**, and PSIVG **0.170**. About **12.9 minutes** for its first 36-frame chunk on RTX3090. | Strong simulator-to-video evidence, but primarily material/particle examples. [Repository](https://github.com/hwidong-kim/3DPhysVideo) explicitly says code release is coming soon. Exclude from Friday dependencies. |

**Video foundations and usable control interfaces**

| Paper — first author; year; venue | Intermediate step → actual video-model input | Baselines and key result | Relevance and availability |
|---|---|---|---|
| **11. [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/html/2503.20314v2)** — Team Wan; 2025; arXiv | No simulator in ordinary T2V/I2V: text, or text plus starting image. | Wan-Bench weighted score **0.724** for 14B, versus Hunyuan **0.673** and Mochi **0.639**. These are the authors’ benchmark results. | Foundation for VACE and useful uncontrolled baselines. [Wan2.2 release](https://github.com/Wan-Video/Wan2.2) provides a unified TI2V-5B checkpoint supporting text-only and image-conditioned inference. |
| **12. [Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control](https://arxiv.org/abs/2503.14492)** — NVIDIA; 2025; arXiv | Simulator → depth/segmentation/edge/blur videos → adaptive multimodal ControlNet branches. | TransferBench compares individual controls, uniform combinations, and adaptive weighting; demonstrates adjustable controllability/diversity tradeoffs. | Explicit sim-to-real prior work. [Open pretrained models](https://github.com/nvidia-cosmos/cosmos-transfer1); supported conditioning requires no new training. Superseded practically by later releases. |
| **13. [World Simulation with Video Foundation Models for Physical AI](https://arxiv.org/abs/2511.00062)** — NVIDIA; 2025, revised 2026; arXiv. **Cosmos-Transfer2.5** | Simulator → synchronized depth/edge/segmentation/blur videos → pretrained 2B generator. | Uniform four-control comparison against Transfer1: edge F1 **0.26→0.41**, segmentation mIoU **0.74→0.76**, quality **9.24→9.31**. | Best engineering bet for this hardware/deadline. [Open implementation](https://github.com/nvidia-cosmos/cosmos-transfer2.5). Exact report title differs from the model’s marketing tagline. |
| **14. [VACE: All-in-One Video Creation and Editing](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf)** — Zeyinzi Jiang; 2025; ICCV | Text, context/control frames, aligned generation masks, optional reference images → unified Video Condition Unit. | In the paper’s **LTX-based** depth evaluation, normalized average **74.99%**, versus Control-A-Video **72.35%**, VideoComposer **70.74%**, ControlVideo **70.07%**. | Excellent for conditioning ablations. [Official Wan2.1-VACE 1.3B/14B checkpoints](https://github.com/ali-vilab/VACE). Do not attribute the LTX results to Wan14B. |
| **15. [Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control](https://arxiv.org/abs/2501.03847)** — Zekai Gu; 2025; SIGGRAPH | Animated meshes/point clouds → tracking video with persistent colors for corresponding 3D points → CogVideoX-based generator, compatible first image, and text. | Compares with MotionCtrl/CameraCtrl for camera control and TokenFlow/CCEdit for motion transfer. Tracking controls outperform depth in its reported reconstruction/video ablation. | Particularly suitable for simulator-authored motion and identity. [Checkpoint released](https://github.com/IGL-HKUST/DiffusionAsShader); documented inference requires about **32GB**. Still consumes a rasterized tracking video. |
| **16. [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.pdf)** — Zhenghao Zhang; 2025; CVPR, 2024 preprint | Simulator positions → 2D trajectories → sparse motion maps → trajectory extractor/DiT fusion; text and optional first image. | At 128 frames, trajectory error **11.72**, versus MotionCtrl **38.39** and DragNUWA **41.25**. | Useful test of whether sparse paths suffice. [Released CogVideoX implementation](https://github.com/alibaba/Tora) reports approximately **30GiB on A100**. Published experiments used OpenSora; distinguish them from released checkpoints. |
| **17. [SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation](https://arxiv.org/abs/2411.04989)** — Koichi Namekata; 2025; ICLR, 2024 preprint | First image plus numeric bounding boxes and per-frame 2D trajectories → latent optimization within frozen Stable Video Diffusion. | Outperforms its zero-shot baselines and approaches supervised DragNUWA/DragAnything on quality and motion fidelity. | Strong answer to “without rendered video.” [Actual interface uses image plus `traj.npy`](https://github.com/Kmcode1/SG-I2V). Approximately 30GB peak memory; published setup generates only 14 frames and uses test-time gradients. |
| **18. [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/html/2312.03641v2)** — Zhouxia Wang; 2024; SIGGRAPH, 2023 preprint | Numeric camera rotations/translations and object trajectories; object paths become sparse displacement maps. | Against VideoComposer: object-motion error **36.8351→28.877**, FVD **1004.99→852.15**. | Established separation of camera and object control. [Pretrained implementations released](https://github.com/TencentARC/MotionCtrl). Older visual quality; supported camera poses do not imply full scene-state understanding. |
| **19. [CameraCtrl: Enabling Camera Control for Video Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/f98fd73d59d8494489ea970747b91fe4-Paper-Conference.pdf)** — Hao He; 2025; ICLR, 2024 preprint | Camera intrinsics/extrinsics → Plücker-ray embeddings → trained controller on video backbone. | Reports better camera accuracy than MotionCtrl and AnimateDiff MotionLoRA while retaining comparable quality/dynamics. | [Open controllers](https://github.com/hehao13/CameraCtrl) accept simulator camera trajectories without new training. Camera control alone cannot prescribe drone or convoy motion. |
| **20. [TrackDiffusion: Tracklet-Conditioned Video Generation via Diffusion Models](https://openaccess.thecvf.com/content/WACV2025/papers/Li_TrackDiffusion_Tracklet-Conditioned_Video_Generation_via_Diffusion_Models_WACV_2025_paper.pdf)** — Pengxiang Li; 2025; WACV, 2023 preprint | Object bounding-box sequences, identities, descriptions → instance/location tokens and temporal instance enhancer. | TrackAP **39.4**, versus vanilla conditional generator **36.0** at 256²; higher-resolution variant reaches **44.7**. | Explicit tracklet interface fits convoy count/scale/paths. [T2V/SVD weights released](https://github.com/pixeli99/TrackDiffusion). Lower resolution and older stack reduce Friday appeal. |
| **21. [DragAnything: Motion Control for Anything using Entity Representation](https://arxiv.org/html/2403.07420v2)** — Weijia Wu; 2024; ECCV | First image and entity masks → entity features; projected 2D paths move those features/location maps → SVD ControlNet. | Versus DragNUWA on VIPSeg: FVD **494.8 vs 519.3**, object-motion error **305.7 vs 324.6**. | Simulator masks make input preparation straightforward. [Pretrained model released](https://github.com/showlab/DragAnything). Center trajectories weakly constrain orientation, shape, and occlusion. |

**Physics integration and the direction of information flow**

| Paper — first author; year; venue | Intermediate step → actual model input | Baselines and key result | Relevance |
|---|---|---|---|
| **22. [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://arxiv.org/html/2409.18964v1)** — Shaowei Liu; 2024; ECCV | Image, forces/torques, inferred materials/geometric primitives → Pymunk 2D simulation → composed/relit frames → SEINE refinement. | Human physical realism **4.14/5**, versus best comparison **2.11**; Motion-FID **30.20**, versus SEINE **38.76**. Appearance does not uniformly win. | Essential early simulation-plus-generative-refinement precedent. Training-free; approximately three minutes for 16 frames on A40. Image-plane dynamics limit application to 3D flight. |
| **23. [Physical Simulator In-the-Loop Video Generation](https://arxiv.org/html/2603.06408v1)** — Lin Geng Foo; 2026; CVPR. **PSIVG** | Generated template → reconstructed objects/background → MPM/Taichi simulation → Mitsuba renders/correspondences → flow-guided video generation and texture optimization. | SAM mIoU **0.84**, versus SG-I2V **0.75**; **82.3%** selection in a six-way physical-plausibility study with 32 participants. | Very close published precedent. [Implementation released](https://github.com/MarkHershey/PSIVG). Authors explicitly identify vehicles/articulated structures and small/thin objects as limitations. Training-free includes test-time optimization. |
| **24. [Physics-Based Interaction with 3D Objects via Video Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00270.pdf)** — Tianyuan Zhang; 2024; ECCV. **PhysDreamer** | Reconstructed 3D Gaussians → SVD reference motion → differentiable MPM estimates material/velocity fields → simulation under new forces. | Human motion-realism preference **80.8%** over PhysGaussian and **63.5%** over DreamGaussian4D. | Important distinction: **video prior → simulator identification**, rather than simulator → generative appearance. Requires reconstruction and scene optimization; unsuitable as Friday’s primary pipeline. |
| **25. [MotionCraft: Physics-Based Zero-Shot Video Generation](https://papers.nips.cc/paper_files/paper/2024/hash/ded49449e9239a47057c33ef8d42c26e-Abstract-Conference.html)** — Antonio Montanaro; 2024; NeurIPS | Physics simulation → dense optical flow → warp Stable Diffusion noise, with attention modifications. | Motion consistency **0.8655**, versus Text2Video-Zero **0.6740**; reported one-to-five-minute generation on A6000. | Early evidence that simulated motion can control synthesis without rendered RGB input. Visually dated, but conceptually relevant. Proceedings author order differs from the original arXiv ordering. |

**Evaluation, including evidence against relying on a single judge**

For these entries, the intermediate representation is an **evaluation representation**; the input is an already generated video.

| Paper — first author; year; venue | Representation/input | Baselines and key result | Relevance |
|---|---|---|---|
| **26. [VeriPhy: Agentic Physical Reasoning for World Model Evaluation and Refinement](https://arxiv.org/html/2609.03153v1)** — Wenzhuo Xu; 2026; arXiv | Prompt → typed obligations; video → tracking/counting/depth measurements → supported/contradicted/unknown verdicts. Also implements **MuJoCo depth/silhouette → frozen Wan2.2-VACE-Fun-14B** generation. | Accounts for **228/304** recorded flaws, versus question decomposition **164** and same-backbone monolithic prompting **222**. Refinement self-score rises **0.797→0.862**, but independent VideoPhy2 physical score stays **3.59→3.58**. | Direct pipeline overlap and a valuable evaluator caution. Its generation stretches simulation time **1.8–2.3×**; path/order agreement is therefore different from absolute dynamic agreement. |
| **27. [Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification](https://arxiv.org/abs/2411.16718)** — S P Sharan; 2025; CVPR, 2024 preprint. **NeuS-V** | Prompt → temporal logic; generated frames → probabilistic automaton; formal satisfaction probability. | Gen-3 human correlation **0.479**, versus VBench **0.089**; Pika **0.624**, versus **−0.087**, on its human-evaluation subset. | Natural existing framework for takeoff/translation and approach/turn/order requirements. Formal verification applies to the extracted representation; perception errors still matter. |
| **28. [We’ll Fix it in Post: Improving Text-to-Video Generation with Zero Training](https://arxiv.org/html/2504.17180v4)** — Minkyu Choi; 2025, revised 2026; arXiv. **NeuS-E** | Temporal verification locates weak propositions/failure times → trim, generate continuation, stitch. No simulator. | Pika NeuS-V **0.577→0.811**; independent T2V-CompBench **0.394→0.505**; VBench **0.789→0.772**. Human preference for edited clips **52%**. | Relevant external refinement comparator, though unnecessary for the initial pipeline. **Metadata caveat:** abstract page retains “…with Neuro-Symbolic Feedback”; linked v4 uses “…with Zero Training.” |
| **29. [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)** — Ziqi Huang; 2024; CVPR | Sixteen dimensions using embeddings, detectors, motion estimators, and quality predictors. | CogVideo scores **96.47% smoothness** but **42.22% dynamic degree**; VideoCrafter **91.79% / 89.72%**. | Useful visual diagnostics. The example shows why smoothness alone rewards insufficient motion. A total score is not physical fidelity. |
| **30. [Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://proceedings.mlr.press/v267/meng25c.html)** — Fanqing Meng; 2025; ICML, 2024 preprint. **PhyGenBench** | 160 prompts, 27 laws; staged phenomenon detection, event-order verification, and whole-video evaluation. | Original Gen-3 automated/human physical scores **0.51/0.48**; CogVideoX5B **0.45/0.37**. Prompt rewriting yields limited improvements. | Supports separate semantic and physical evaluation. Its broad laws are less targeted than drone/convoy trajectory constraints. |
| **31. [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](https://arxiv.org/abs/2503.06800)** — Hritik Bansal; 2026; ICLR, 2025 preprint | Action-centric videos; separate semantic adherence, physical commonsense, rule grounding; trained AutoEval model. | Original Wan2.1 joint success **32.6% overall / 21.9% hard**. Unseen-model AutoEval physical-score correlation **0.37**. | Useful independent secondary judge, with substantial uncertainty. Freeze benchmark version: original and current action/prompt counts differ. |
| **32. [T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation](https://arxiv.org/html/2407.14505v2)** — Kaiyue Sun; 2025; CVPR, 2024 preprint | 1,400 prompts/seven categories; MLLM, detection, depth, and tracking checks; foreground motion separated from camera motion. | Original direction accuracy peaks at **47%** for LVD; Gen-3 **42%**. Models struggle with counts above three. | Closely relevant to count, identity, motion binding, and turning direction. Borrow its principles rather than running the entire benchmark. |
| **33. [Physics-IQ Verified](https://arxiv.org/abs/2606.18943)** — Tim Rädsch; 2026; arXiv | Real experiments and observed prefixes → generated continuations → motion-mask overlap/error against real continuations; audited sample-level aggregation. | Corrects **57.6%** of samples and **34.8%** of prompts; model ranking agreement before/after correction only **Kendall τ=0.46**. | Closest evaluation analogy to known-reference dynamics. Demonstrates that reference preparation and aggregation can change conclusions substantially. |
| **34. [On the Content Bias in Frechet Video Distance](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html)** — Songwei Ge; 2024; CVPR | Controlled spatial/temporal corruptions and alternative feature encoders diagnose FVD. | Shows FVD can improve by selecting videos without meaningful motion; self-supervised video features reduce content bias. | Strong reason to avoid FVD as Friday’s headline metric, especially with three scenes and a small number of samples. |

**Narrowing to drones and satellite/overhead imagery**

| Paper — first author; year; venue | Intermediate step/input | Baselines and key result | Relevance |
|---|---|---|---|
| **35. [CARLA-Air: Fly Drones Inside a CARLA World—A Unified Infrastructure for Air-Ground Embodied Intelligence](https://arxiv.org/abs/2603.28032)** — Tianle Zeng; 2026; arXiv | CARLA and AirSim share one Unreal process, physics tick, coordinate frame, and sensor pipeline. No generative video model. | Addresses synchronization/domain separation in independent driving/aerial simulators; supports up to 18 synchronized sensor modalities. No generator comparison. | Confirms that the v0 simulator candidate exists. Useful longer-term combined air/ground infrastructure; fresh Unreal setup is unnecessary risk for Friday. |
| **36. [DiffusionSat: A Generative Foundation Model for Satellite Imagery](https://arxiv.org/html/2312.03606v2)** — Samar Khanna; 2024; ICLR, 2023 preprint | Satellite images, text, and metadata such as geolocation/time/GSD; task-specific conditioning for temporal generation, super-resolution, inpainting. | Reports improvements over prior satellite-generation methods and generic diffusion baselines across its satellite tasks. | Domain appearance precedent, **not evidence of second-by-second moving-vehicle simulation**. Its temporal imagery task should not be confused with convoy video. |
| **37. [Video Reconstruction using Diffusion-based Image-to-Video Generation with Trajectory Guidance](https://arxiv.org/html/2605.16420v1)** — Stelio Bompai; 2026; MuseKDE workshop at IEEE MDM | GPS telemetry → projected per-vessel trajectories/bounding boxes → SG-I2V plus first frame; top-down footage of boats. | Compares optical-flow extrapolation and RIFE; reports BRISQUE **25.52**, versus real footage **23.64**. GPS alignment and unequal baseline information complicate interpretation. | Unusually close overhead moving-target application with no domain fine-tuning. Small, short experiment; boats viewed from a drone differ from visible drones and satellite convoys. |

The literature narrows the gap considerably. Existing work already covers code execution, explicit state, physics simulation, render-conditioned generation, sparse trajectories, and off-screen persistence. The remaining opportunity is a credible evaluation of their reliability on your subjects and viewing geometry.

**2. The lead’s open question: scene state versus rendered video**

**Yes, models can accept structured motion or geometry without an externally rendered video. But I did not verify a general open checkpoint that accepts an arbitrary simulator’s complete state and faithfully renders arbitrary drone/convoy scenes without an adapter.**

Three distinctions matter:

| Interface | Verified examples | What this means for you |
|---|---|---|
| **Numeric trajectories, boxes, camera poses** | SG-I2V, TrackDiffusion, MotionCtrl, CameraCtrl; Tora exposes trajectory inputs. | Export supported quantities from the simulator. No RGB control video is required at the public interface, although implementations may internally build maps or embeddings. |
| **Learned geometric/action interfaces** | WoVoGen’s semantic volume encoder; Cosmos 3’s embodiment-specific action interfaces. | More direct structured conditioning exists, but requires the representation and semantics learned during training. Arbitrary drone state is not interchangeable with robot-effector or ego-driving actions. |
| **Rendered structural controls** | Cosmos Transfer, VACE, DaS; also CoDeR and State World Models upstream of their renderers. | The most practical general route today: state → depth/edges/segmentation/tracking buffers → generated video. |

For actual Friday dependencies, the clearest non-video state interface is **SG-I2V’s first image plus numeric boxes/trajectories**. Tora is another practical trajectory option; TrackDiffusion adds explicit tracklet identities. Their limitation is representational: paths and boxes do not fully specify 3D orientation, self-occlusion, rotor geometry, or contact.

The distinction between “style transfer” and “video generation” should not decide the project. A depth or correspondence sequence leaves appearance substantially unspecified; a diffusion model still generates appearance and temporal detail. Conversely, translating a fully rendered RGB clip may function mainly as appearance transfer. **Both are legitimate experimental conditions.**

For CoDeR specifically, executable state is authoritative upstream, but its Artist receives rendered RGB/depth and compiled text. It does not demonstrate a frozen generic generator directly consuming the world’s code or state dictionary. [CoDeR method](https://arxiv.org/html/2609.26458v1)

I would phrase your question operationally:

> Which representation preserves the required behavior with the least rendering effort: sparse trajectories, structural buffers, or RGB video?

That is testable and avoids arguing over terminology.

**3. Simulator ↔ generator pairings**

**Use one common Blender scene/export pipeline for all three videos.** For drones, optionally obtain physically simulated poses from a lightweight CPU quadrotor simulator; for the convoy, a scripted vehicle trajectory is sufficient for v0. Keep the state source separate from the renderer so changing it later is straightforward.

| Test video | Recommended pairing | Initial controls | Estimated setup and main risks |
|---|---|---|---|
| **Drones take off and translate** | CPU quadrotor rollout → Blender → **Cosmos-Transfer2.5**; VACE comparator. | Depth plus edges; visible ground plane, distinct bodies, fixed camera. | Roughly **one engineering day** for a minimal shared pipeline, assuming usable assets. Contact, altitude, shadows, and thin arms may be altered. A plausible-looking ascent does not establish correct flight dynamics. |
| **Drones moving in the air** | Same scene/pose pipeline → **Cosmos-Transfer2.5**; optionally **Tora** as sparse-trajectory comparator. | Edges plus depth; separate subjects clearly. Include a brief exit/re-entry only if persistence is a required claim. | Small incremental setup once the first scene works. Risks: merging, identity swaps, shape changes, and ignored depth against distant sky. |
| **Overhead convoy turns right onto a field** | Blender top-down scene with scripted vehicle paths → **Cosmos-Transfer2.5**; VACE comparator. | Edges and blurred RGB; test depth only if it carries useful signal. Keep camera/background fixed. | Approximately **half a day additional**, with simple assets. Nearly flat overhead depth is weak; risks include erased vehicles, altered road geometry, wrong headings, and invented camera movement. |

These are engineering estimates, not measured setup times or evidence that the recommended generator wins.

For drone dynamics, [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) provides existing quadrotor environments. Use a scripted controller, not reinforcement-learning training. If integrating it threatens Tuesday’s milestone, use prescribed Blender trajectories and explicitly describe the experiment as **kinematic control**, not validated flight simulation.

Hardware changes the recommendation:

- **Isaac Sim should not be the critical path.** Its requirements explicitly exclude GPUs without RT cores, including A100/H100. [Official requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- **Blender Cycles has a CUDA rendering path**, so lack of RT cores does not impose the same restriction. Use simple scenes and modest samples; verify the installed build actually selects CUDA. [Blender GPU documentation](https://docs.blender.org/manual/en/4.3/render/cycles/gpu_rendering.html)
- **Cosmos-Transfer2.5 explicitly supports Ampere/A100.** Its documentation lists **65.4GB** for single-GPU inference, and its setup requires a sufficiently recent driver/CUDA environment. No verified A100 runtime is published in the timing table. [Setup](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md), [inference](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md)
- **Newer does not automatically mean easier this week.** Cosmos3-Nano-Sim-Transfer is released, but its model card lists Hopper/Blackwell compatibility; A100 execution remains **UNVERIFIED**. [Model card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)
- Eight GPUs are not automatically one pooled-memory device. Choose parallel independent jobs or supported model sharding according to the actual node/interconnect layout.

**DaS is the best stretch pairing** if depth/edges fail to preserve identity or motion: Blender can supply persistent point correspondences directly. Its input preparation is more work than exporting ordinary passes, so I would not make it the first dependency.

**4. Honest critique**

**The project is worth doing as a scoping study. The broad method is not novel enough for a paper by itself.** Three successful videos would establish implementation feasibility. Three reproducibly measured failures could be equally useful.

The strongest existing overlap is Cosmos transfer and its newest interactive simulator integration, alongside PSIVG and VeriPhy. CoDeR and State World Models already establish explicit execution followed by generative visual realization. Your contribution therefore needs to live in the task, measurement, failure analysis, or a demonstrably better interface.

The proposed baseline comparison is informative **as a comparison of available workflows**, but it does not isolate a simulator’s benefit:

- T2V receives language.
- I2V additionally receives initial appearance/layout.
- Simulator-conditioned generation may receive the entire future trajectory, visibility, and geometry.

Better agreement with that trajectory is expected. It is not evidence that the generator learned physics, predicts hidden dynamics, or became a better world model.

Use two kinds of comparison:

1. **Practical comparison:** Given a user request, which workflow produces an acceptable clip, at what cost?
2. **Conditioning comparison:** Within one checkpoint, what changes when adding initial appearance, sparse paths, or dense structural controls?

For text-only baselines, score requirements actually stated in the prompt. An unspecified exact spline should not become a headline failure. Exact trajectory error remains useful diagnostically, provided the information difference is explicit.

Several v0 claims need tightening:

| Earlier claim/design choice | My assessment |
|---|---|
| “Simulator ground truth is free.” | Input-state annotations are available. Generated-video annotations still require measurement; the generator may invalidate the original boxes, masks, and identities. |
| “The generator covers physics the simulator misses.” | Unsupported for these experiments. It may add convincing appearance while introducing incorrect behavior. |
| “Faster than full-fidelity rendering.” | Must be measured end-to-end. Simple conventional rendering may be faster than diffusion, especially for these three scenes. |
| Track smoothness as physical plausibility | Insufficient. Frozen or uniformly drifting objects can be smooth and wrong. Powered flight also cannot be judged using an unpowered-ballistic assumption. |
| FVD/FID and synthetic-to-real detector training for v0 | Too much infrastructure and too little data. Detector training also conflicts with the stated no-training scope. |
| One-line failures establish model incapability | They establish failures under those prompts/settings. Record versions, seeds, prompts, and all attempts before making broader claims. |

A **strong result** would establish a useful boundary: for example, depth/edge conditioning preserves paths but loses identities; correspondence conditioning improves identity; overhead controls fail despite success on visible drones; or appearance gains disappear when trajectory accuracy is required.

A **trivial result** would show that a model given the complete future motion follows it better than a model given “three drones fly away,” accompanied only by selected good-looking examples.

With only these three scenarios, do not claim general drone/satellite performance. Also, prescribed trajectories can answer the initial control question without a sophisticated physics simulator. Demonstrating that a full simulator is necessary would require a later comparison against simpler trajectory authoring.

**5. Proposed experiment plan through Friday**

Keep **three canonical scenes**, with repeated seeds. Do not expand to the old 30-prompts-per-domain grid.

For each scene, save a deterministic state timeline, camera parameters, projected object centers/boxes/masks, visibility, headings, and event times. Render RGB, depth, and edges from the same timeline. Keep depth normalization fixed over the clip and match each model’s expected preprocessing; independently normalizing each frame can corrupt the motion signal.

Use clearly resolved subjects—approximately **32–96 pixels across for drones** and **24–48 pixels long for vehicles** at evaluation resolution. These are starting choices for measurable v0 objects, not domain-wide recommendations.

The scenes should make success unambiguous:

| Scene | Required behavior |
|---|---|
| **Takeoff** | Three grounded drones → visible ground separation → specified translation, with count preserved. |
| **Airborne motion** | Three drones follow distinct paths with stable identities. If testing persistence, one briefly leaves and re-enters the frame at a specified time/location. |
| **Convoy** | Three vehicles approach along the road, turn right onto the field, and retain ordering/separation. Road adherence applies before the intentional departure. |

**Conditions**

| Condition | Model/input | Purpose |
|---|---|---|
| **R: Conventional render** | Blender RGB | Motion/geometry reference and appearance baseline. |
| **T: Text only** | Wan2.1-VACE, no visual conditions | Same-checkpoint uncontrolled comparison. |
| **I: Text + first frame** | Same VACE checkpoint; subsequent frames unspecified | Isolates the initial appearance/layout contribution. |
| **D: Structural control** | Same VACE checkpoint; simulator depth, plus supported appearance reference | Measures added motion/geometry control within that checkpoint. |
| **B: Appearance transfer** | Cosmos-Transfer2.5, blur-only control from simulator RGB | Practical RGB-derived transfer comparison. |
| **M: Structural/multiple controls** | Cosmos-Transfer2.5, case-appropriate controls | Main proposed pipeline. |

VACE’s documented interface supports text-only, first-frame, structural, and composed conditioning tasks. Use the official mask conventions rather than assuming a white mask means the same thing across models. [VACE user guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md)

Run **three seeds × three scenes × five generative conditions = 45 generated clips**, plus three deterministic renders. If the 14B VACE setup fails the early smoke test, use 1.3B consistently across its conditions and label the limitation.

Add **Wan2.2-TI2V-5B I2V** as nine additional clips if practical, to check whether the VACE baseline is unusually weak. It supports both T2V/I2V and documented single-GPU execution. [Official release](https://github.com/Wan-Video/Wan2.2)

The most informative stretch experiment is **Tora’s projected trajectories**, because it tests whether dense render-derived controls are needed. SG-I2V is another option, but its short native output and optimization cost complicate matching five-second clips.

Keep duration, field of view, and event timing matched. Use native supported dimensions and record resize/crop transforms. Do not silently stretch playback time to improve apparent motion. Matching integer seeds across different architectures does not make their noise samples equivalent.

**Metrics**

Use a small dashboard, with separate axes:

| Axis | Friday measurement |
|---|---|
| **Event completion** | Binary required-event checks and timestamp errors: takeoff, translation, right turn, exit/re-entry. |
| **Trajectory compliance** | Center-position error in pixels and object-body lengths, using fixed time alignment; endpoint error where visible. |
| **Count and identity** | Correct visible count, duplication/disappearance, identity switches, convoy order changes. |
| **Geometry/contact** | Human-audited ground separation, body integrity, heading agreement, inter-object overlap. |
| **Background/camera stability** | Background-feature displacement outside moving-object masks; separately report unintended camera motion. |
| **Visual appearance** | Blinded ratings of realism and temporal artifacts; selected VBench diagnostics including dynamic degree. |
| **Practical cost** | Setup effort, generation wall time, peak memory, failed attempts, and preprocessing/rendering time. |

For this sample size, **manual annotation is a reasonable primary measurement**. Annotate about eight keyframes per clip, plus event boundaries; use tracking to assist interpolation and audit every failure. Report tracker coverage and uncertain identities. A missed detection must not silently vanish from trajectory averages.

Use NeuS-V-style temporal specifications for event order where helpful, preferably grounded in audited measurements. A VLM judge can be secondary. Keep appearance assessment separate from compliance assessment so attractive footage cannot conceal incorrect motion.

Pre-register simple pilot tolerances, such as a trajectory deviation below half a body length and event timing within 0.25 seconds, then report continuous errors as well. Those thresholds should be checked on the conventional render before judging generators.

Use three blinded raters if available. Report per-scene outcomes, all-seed results, and disagreement. Three scenes do not support broad population-level confidence claims. Skip headline FVD/FID.

**Schedule and gates**

| Deadline | Concrete output |
|---|---|
| **Tuesday, Nov 10, 3 PM** | Freeze simulator/export representation, primary generator, three scenes, and metrics. Require one custom-scene output from Cosmos and VACE before committing to both. |
| **Wednesday, Nov 11, 3 PM** | Start the report with related work, hypotheses, condition table, evaluation definitions, and first qualitative results. Freeze prompts/settings after a small pilot. |
| **Thursday, Nov 12** | Complete generation, annotation, and blinded assessment. Preserve every output and failed run. |
| **Friday, Nov 13** | Deliver a comparison page, per-scene score tables, trajectory overlays, runtime table, and a short findings memo including failures. |

The deliverable should include all seeds, not just the best examples. A useful figure is a row containing simulator RGB, structural control, T2V, I2V, VACE control, Cosmos control, and a trajectory-error overlay.

If time runs short, cut in this order:

1. Additional models and closed-model comparisons.
2. Additional backgrounds, lighting variations, and prompt grids.
3. Automated detector/tracker engineering beyond annotation assistance.
4. Any full benchmark suite or distribution metric.
5. Sophisticated dynamics integration; retain prescribed trajectories and describe that limitation clearly.

Preserve all three scenes, repeated seeds, conventional rendering, T2V/I2V comparisons, one structural-control method, and motion-versus-appearance evaluation.

**6. What I am least sure about**

- **A100 execution of current Cosmos 3 transfer checkpoints:** **UNVERIFIED**. Transfer2.5 has explicit Ampere support.
- **Exact drone and overhead-domain performance:** published demonstrations do not settle it; this is the main experiment.
- **CoDeR runnable release:** **UNVERIFIED**. Its paper was verified; the supplied project page was inaccessible through browsing.
- **Automatic identity/pose measurement:** even clearly visible targets can fool trackers after deformation or crossings.
- **Whether structural controls outperform RGB-derived transfer here:** overhead depth may carry very little information.
- **How much realism improves without domain adaptation:** a generally realistic video can still have incorrect satellite geometry, drone shape, or motion.