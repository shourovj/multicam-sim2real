# Literature Survey and Project Review: Codex (GPT-6 Astra)

**Source:** Codex CLI 0.167.0, model `gpt-6-astra`, high reasoning effort, live web search, subagents enabled, read-only sandbox. Run on Nov 9, 2026, finished 17:44. **Input:** only the [prompt](2026-11-09-1735-codex-review-prompt.md) and the [v0 design doc](../2026-11-09-1416-design-doc-v0.md). It did not see the Claude survey or review. The report below is unedited except for this header and a fixed path to the design doc. See the [synthesis](2026-11-09-1750-lit-review-synthesis.md) for how it compares with the other surveys.

---

**Independent assessment — November 9, 2026**

**The project is worth doing as a controlled feasibility study. The general idea—simulation or executable scene structure followed by generative rendering—is already established.** The useful question is now: *which conditioning representation preserves the intended motion, objects, and events while improving appearance, and where does that preservation fail?*

For Friday, I recommend **Blender → Cosmos-Transfer2.5** as the main simulator-to-video pipeline, **Wan2.1-VACE** for comparisons that keep the generator fixed, and **Wan-Move** as the strongest practical test of conditioning on motion without supplying an RGB simulation video. These are recommendations for testing, not experimentally established winners.

I read `design-doc-v0.md` and treated the three visible-target scenarios as superseding its tiny-target factor grid. This was a read-only review; no models were run. Paper identities were checked against primary sources. Results below are author-reported and are not comparable across different papers’ evaluation protocols. “Preprint” identifies the verified publication record; uncertain release or venue details are marked **UNVERIFIED**.

**1. Literature survey — 36 entries**

**A. Explicit simulation, physics, and executable-world intermediates**

**1. [GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Lv_GPT4Motion_Scripting_Physical_Motions_in_Text-to-Video_Generation_via_Blender-Oriented_GPT_CVPRW_2024_paper.pdf)** — **Jiaxi Lv**, 2024, CVPR PBDL Workshop; initial preprint 2023.  
**Intermediate/input:** GPT-generated Blender scripts run physical simulations; depth and edge sequences condition Stable Diffusion through ControlNets. **Evidence:** qualitative comparisons with AnimateDiff, ModelScope, Text2Video-Zero, and DirecT2V show better prescribed physical motion; no numerical advantage is asserted here. **Relevance:** an early, direct precedent for the proposed pipeline. [Code is released](https://github.com/jiaxilv/GPT4Motion), but its older generation backbone makes it a historical comparator.

**2. [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://arxiv.org/html/2409.18964v1)** — **Shaowei Liu**, 2024, ECCV.  
**Intermediate/input:** image decomposition and estimated properties → Pymunk planar rigid-body simulation → warped, composited, relit video → SEINE refinement. **Evidence:** human physical-realism score 4.14/5 versus 2.11 for I2VGen-XL and 1.68 for DynamiCrafter; image-FID does not improve over every baseline. **Relevance:** training-free and especially relevant to planar overhead motion. [Released implementation](https://github.com/stevenlsw/physgen) supports top-down images, but is less suitable for substantial 3D drone rotation.

**3. [PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](https://arxiv.org/html/2404.13026v1)** — **Tianyuan Zhang**, 2024, ECCV.  
**Intermediate/input:** reconstructed 3D Gaussians supply an initial rendering to Stable Video Diffusion; generated motion then supervises material and velocity optimization in a differentiable material-point-method simulator. **Evidence:** human motion-realism preference of 80.8% over PhysGaussian and 63.5% over DreamGaussian4D. **Relevance:** important adjacent work, but information flows from the video prior into simulation. Novel scenes require reconstruction and optimization, making the [released system](https://github.com/a1600012888/PhysDreamer) a poor Friday dependency.

**4. [PhysGen3D: Crafting a Miniature Interactive World from a Single Image](https://arxiv.org/html/2503.20746v1)** — **Boyuan Chen**, 2025, CVPR.  
**Intermediate/input:** reconstructed meshes, physical properties, and lighting → MPM simulation → Mitsuba rendering; video enhancement is optional. **Evidence:** physical-realism ratings exceed Kling 1.0, Gen-3, and Pika 1.5, while photorealism trails some of them. Adding VEnhancer increases the reported photorealism score but reduces semantic consistency from .796 to .745. **Relevance:** unusually useful evidence that enhancement can corrupt correctness. [Code is released](https://github.com/by-luckk/PhysGen3D); a simulator-only renderer belongs in our comparisons.

**5. [PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](https://arxiv.org/html/2509.20358v1)** — **Chen Wang**, 2025, NeurIPS.  
**Intermediate/input:** a trajectory diffusion model trained on simulated animations predicts 3D point motion from physical controls; projected trajectories drive Diffusion as Shader. **Evidence:** compares against Wan2.1, CogVideoX, DragAnything, and ObjCtrl; reports stronger physical-commonsense ratings, but the cited GPT-based comparison contains only 12 videos. **Relevance:** demonstrates physical-parameter-to-motion-to-video decomposition. [Weights and code](https://github.com/cwchenwang/physctrl) are available, but single-object/material dynamics are a poor match to the proposed multi-agent scenes.

**6. [Physical Simulator In-the-Loop Video Generation](https://arxiv.org/html/2603.06408v1)** — **Lin Geng Foo**, 2026, CVPR.  
**Intermediate/input:** reconstructed scene and camera → MPM simulation → Mitsuba RGB, masks, and correspondences → guided CogVideoX/HunyuanVideo generation. **Evidence:** motion-control mask IoU .84 versus .47 for CogVideoX, .46 for HunyuanVideo, and .61 for Image Conductor–Object. **Relevance:** very direct prior art. Its [implementation](https://github.com/MarkHershey/PSIVG) has substantial reconstruction dependencies and test-time optimization; “training-free” does not mean a simple inference-only integration.

**7. [Code Plans, Diffusion Renders: Open-Ended Generative World Modeling — CoDeR](https://arxiv.org/html/2609.26458v1)** — **Zixun Fang**, 2026, preprint.  
**Intermediate/input:** agents construct an executable Three.js world; whitebox RGB, depth, and text condition a MiniMax-H3-based “Artist,” with generated appearance mapped back onto geometry. **Evidence:** WorldScore average 87.88 versus 74.79 for LingBot-World and 74.20 for Matrix-Game 3.0. **Relevance:** the closest conceptual overlap, including persistent state outside the current observation. The Artist is trained; code and weights are promised in the paper, but their release is **UNVERIFIED**. It is not a dependable Friday baseline.

**8. [Video-Coder: Reinforcing Agentic Coding for Video Generation](https://arxiv.org/html/2611.09894v1)** — **Chaoyang Wang**, 2026, preprint, November 6.  
**Intermediate/input:** a trained coding agent writes Blender programs; rendered draft RGB and text guide downstream video generation. **Evidence:** MiniMax-H3’s reported PhyGenBench score rises from .61 to .67; the VBench-2.0 physics-category score rises from 54.58% to 78.19%. **Relevance:** directly limits novelty of “an agent codes a simulation to guide video.” The [repository](https://github.com/CYWang735/Video-Coder) announces a November 8 release, but its newly released serving stack creates deadline risk.

**9. [Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control](https://arxiv.org/html/2503.14492v2)** — **NVIDIA**, collective author, 2025, technical report.  
**Intermediate/input:** simulator-derived depth, segmentation, edges, and blurred RGB, with spatially and temporally varying control weights. **Evidence:** combined controls improve robot foreground-mask IoU to .63 versus .54–.57 for individual modalities; driving results also reveal tradeoffs between lane and object fidelity. **Relevance:** establishes multimodal generative rendering as a supported use case. [Code and checkpoints](https://github.com/nvidia-cosmos/cosmos-transfer1) are released; newer variants are preferable for this project.

**10. [World Simulation with Video Foundation Models for Physical AI](https://arxiv.org/html/2511.00062v1)** — **NVIDIA**, collective author, 2025, technical report; includes **Cosmos-Transfer2.5**.  
**Intermediate/input:** depth, edge, segmentation, or blur sequences condition a pretrained video renderer. **Evidence:** one real-robot visual-generalization experiment reports 24/30 successes with Transfer2.5 augmentation, versus 5/30 with conventional image augmentation and 1/30 without augmentation. This starts from robot demonstrations, not arbitrary drone simulations. **Relevance:** strong practical candidate and evidence of downstream value. [Released inference](https://github.com/nvidia-cosmos/cosmos-transfer2.5) supports the proposed workflow.

**11. [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/html/2606.02800v1)** — **NVIDIA**, collective author, 2026, technical report.  
**Intermediate/input:** unified multimodal tokens support generation and control; the subsequent Sim-Transfer release consumes simulator-rendered depth/segmentation/edge/blur and text. **Evidence:** the report improves several PAIBench control metrics over Transfer2.5, including segmentation IoU .72 versus .68. **Relevance:** Transfer2.5 is no longer the latest Cosmos generation. However, the October [Cosmos3-Nano-Sim-Transfer model card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) lists Hopper/Blackwell support, not A100/Ampere. A100 compatibility is **UNVERIFIED**, so I would not make it critical for Friday.

**12. [RoboRender: Robot-Oriented Video Generation for Visual Sim-to-Real Transfer](https://arxiv.org/abs/2610.09254)** — **Huang Huang**, 2026, preprint.  
**Intermediate/input:** simulated trajectories become depth and robot RGB-mask videos; these, plus text, condition an adapted Wan2.1 model. **Evidence:** compares with Wan-Fun-Control, Cosmos-Transfer2.5, RoboTransfer, raw simulation, and domain randomization. Real-robot policy success averages 71%, versus 10% with raw simulation and 20% with domain randomization. **Relevance:** a strong example of meaningful sim-to-real evidence. The [project](https://robo-render.github.io/) uses a renderer trained on substantial robot data; its result does not establish drone-domain generalization.

**B. Structured motion and geometry conditioning**

**13. [VACE: All-in-One Video Creation and Editing](https://arxiv.org/html/2503.07598v2)** — **Zeyinzi Jiang**, 2025, ICCV.  
**Intermediate/input:** a unified condition unit encodes text, reference frames, control videos, and masks. Controls include depth, flow, layout, and other image-space representations. **Evidence:** compares with I2VGen-XL, CogVideoX-I2V, Control-A-Video, VideoComposer, and task-specific editors; the unified model is competitive across tasks. **Relevance:** particularly valuable because several experimental conditions can use the same checkpoint. [Wan-based weights and inference](https://github.com/ali-vilab/VACE) are released; the original paper’s tables should not automatically be attributed to every later checkpoint.

**14. [Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance](https://arxiv.org/html/2512.08765v1)** — **Ruihang Chu**, 2025, NeurIPS.  
**Intermediate/input:** first image, text, point trajectories, and visibility; first-frame latent features propagate along those trajectories. **Evidence:** MoveBench trajectory EPE 2.6 versus 3.3 for Tora and 3.2 for MagicMotion; comparisons with Kling show approximate human-preference parity. **Relevance:** strongest practical non-RGB motion-input candidate. [Code and weights](https://github.com/ali-vilab/Wan-Move) are released. The paper acknowledges loss of control after prolonged track disappearance.

**15. [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.html)** — **Zhenghao Zhang**, 2025, CVPR; preprint 2024.  
**Intermediate/input:** trajectories become flow-like representations and hierarchical motion features injected into a video diffusion transformer. **Evidence:** compares with VideoComposer, DragNUWA, AnimateAnything, TrailBlazer, and MotionCtrl; reported 128-frame trajectory error is 11.72 versus 38.39 for MotionCtrl. **Relevance:** another viable way to turn simulator trajectories into model controls. [Released code](https://github.com/alibaba/Tora) documents an A100 inference configuration, although published long-video demonstrations and released configurations differ.

**16. [Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise](https://arxiv.org/html/2501.08331v1)** — **Ryan Burgert**, 2025, CVPR.  
**Intermediate/input:** optical flow creates temporally correlated warped noise used by a fine-tuned video model. **Evidence:** compares with MotionClone, DMT, and AnyV2V; on camera-motion transfer, tracking IoU .83 versus .71 for MotionClone, with lower FVD. It does not win every metric. **Relevance:** simulator flow can supply dense motion without photorealistic RGB. Use its [released trained weights](https://github.com/Eyeline-Labs/Go-with-the-Flow); this is not a universal training-free modification of arbitrary video models.

**17. [Time-to-Move: Training-Free Motion Controlled Video Generation via Dual-Clock Denoising](https://arxiv.org/abs/2511.08633)** — **Assaf Singer**, 2026, ICLR; preprint 2025.  
**Intermediate/input:** coarse RGB animation, masks, initial image, and text; separate denoising schedules preserve motion while refining appearance. **Evidence:** compares with DragAnything, MotionPro, SG-I2V, and Go-with-the-Flow; improves trajectory adherence in the reported CogVideoX comparison, with a dynamic-degree tradeoff. **Relevance:** useful fallback for simple animated proxies. [Released inference](https://github.com/time-to-move/TTM) supports Wan2.2, CogVideoX, and SVD without a newly trained control checkpoint.

**18. [Motion Prompting: Controlling Video Generation with Motion Trajectories](https://openaccess.thecvf.com/content/CVPR2025/html/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.html)** — **Daniel Geng**, 2025, CVPR.  
**Intermediate/input:** sparse or dense point tracks become a spatiotemporal conditioning volume; camera poses and simple 3D geometry can be projected into these tracks. **Evidence:** compares with ImageConductor and DragAnything; at 512 DAVIS tracks, EPE 4.055 versus DragAnything’s 10.948, but the advantage is not universal at sparse control densities. **Relevance:** a key conceptual precedent for simulator-surface tracks. Public inference weights were **UNVERIFIED** on the [project page](https://motion-prompting.github.io/).

**19. [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/html/2312.03641v2)** — **Zhouxia Wang**, 2024, SIGGRAPH; preprint 2023.  
**Intermediate/input:** camera rotation/translation matrices enter a camera-control branch; projected 2D trajectories enter an object-control branch. **Evidence:** improves camera and object-motion metrics over VideoComposer and compares with AnimateDiff. **Relevance:** a clear limited example of numerical geometric-state input—camera state—without an RGB control video. [Code and checkpoints](https://github.com/TencentARC/MotionCtrl) exist, but capabilities vary across its VideoCrafter, AnimateDiff, and SVD branches.

**20. [CameraCtrl: Enabling Camera Control for Video Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f98fd73d59d8494489ea970747b91fe4-Abstract-Conference.html)** — **Hao He**, 2025, ICLR; preprint 2024.  
**Intermediate/input:** camera intrinsics/extrinsics become Plücker ray maps encoded by a trained camera adapter. **Evidence:** compares with AnimateDiff MotionLoRA and MotionCtrl; SVD camera translation/rotation errors improve from 10.21/1.41 to 9.02/1.18. **Relevance:** supports geometric camera conditioning, but cannot independently prescribe drone or convoy motion. [Released inference](https://github.com/hehao13/CameraCtrl) makes it relevant if camera motion later becomes a variable.

**21. [GraphVid: Interactive Graph-Controllable Video Generation](https://arxiv.org/html/2607.21580v1)** — **Vedant Shah**, 2026; ECCV according to official project materials, proceedings confirmation **UNVERIFIED**.  
**Intermediate/input:** an interaction graph with object attributes and relation edges passes through a graph encoder and token adapter into LTX-Video. **Evidence:** compares with Motion-I2V, Tora, MagicMotion, and Wan-Move; FVD 99.42 improves over Motion-I2V’s 159.32 but trails Wan-Move’s 82.17. **Relevance:** genuine structured semantic input, not complete metric state. Weights exist, but the linked official GitHub repository returned 404; runnable inference is **UNVERIFIED**.

**22. [4Director: Controlling Video World Models with Rigid 3D Geometry](https://arxiv.org/html/2610.02160v1)** — **Wei Cao**, 2026, preprint.  
**Intermediate/input:** object meshes, rigid transforms, camera motion, and background geometry form a 4D scene, then render into depth video for a trained VACE-based adapter. **Evidence:** identity-gated IoU 60.4 versus 52.9 for SymphoMotion and 54.8 for VerseCrafter. **Relevance:** well matched to rigid drones and turning vehicles, but the generator ultimately receives depth. [Released inference](https://github.com/VVeiCao/4Director) requires substantial dependencies; documented validation is on H200, with at least 80 GB GPU memory.

**C. Driving systems, predictive world models, and foundation baselines**

**23. [MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](https://arxiv.org/abs/2411.13807)** — **Ruiyuan Gao**, 2025, ICCV; initially called MagicDriveDiT.  
**Intermediate/input:** numerical 3D boxes, camera poses, ego trajectories, semantic road maps, and text enter dedicated encoders. **Evidence:** reported FVD 94.84 versus 218.12 for 16-frame MagicDrive; detection and segmentation control metrics also improve. **Relevance:** an important released model that accepts structured geometry without requiring a rendered RGB movie. [Code and checkpoints](https://github.com/flymin/MagicDrive-V2) exist, but street-view training and nuScenes-style interfaces make satellite/drone adaptation substantial.

**24. [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/html/2403.06845v2)** — **Guosheng Zhao**, 2025, AAAI; preprint 2024.  
**Intermediate/input:** an LLM produces agent trajectories; an HD-map generator creates compatible road structure; encoded structural conditions guide multiview video synthesis. **Evidence:** first-frame-conditioned FVD 55.7 versus 139.0 for Panacea; synthetic videos also improve driving-perception training. **Relevance:** direct precedent for language → intermediate traffic scene → video. The [repository](https://github.com/f1yfisher/DriveDreamer2) contains code but retains release-in-progress wording; complete ready-to-run availability is **UNVERIFIED**.

**25. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/html/2503.20523v1)** — **Lloyd Russell**, 2025, technical report; equal-contribution author order is randomized.  
**Intermediate/input:** speed, curvature, camera parameters, agent boxes projected into views, and environmental metadata. **Evidence:** evaluates control adherence and training progression rather than providing a broad public head-to-head benchmark; includes off-road driving into fields. **Relevance:** the convoy’s road-to-field event is not itself novel. Box alignment also does not establish persistent instance identity. Open reproducible inference was **UNVERIFIED**.

**26. [Genie: Generative Interactive Environments](https://proceedings.mlr.press/v235/bruce24a.html)** — **Jake Bruce**, 2024, ICML.  
**Intermediate/input:** learned latent actions and video tokens; an initial image and actions drive predictive generation without an explicit physical simulator. **Evidence:** action-model/tokenizer ablations and CoinRun imitation comparisons against random and oracle behavior-cloning policies demonstrate actionable learned dynamics. **Relevance:** helps distinguish a predictive interactive world model from a renderer supplied with the entire future. It is conceptual context, not a practical drone generator.

**27. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/html/2501.03575v3)** — **NVIDIA**, collective author; **Niket Agarwal** first named individual, 2025, technical report.  
**Intermediate/input:** ordinary video latents/tokens; base inputs are text and initial visual observations, while specialized controls require post-training. **Evidence:** in a physics evaluation, 7B object IoU increases from .332 with one context frame to .592 with nine. **Relevance:** directly demonstrates how extra input information changes results. The evaluated objects start visible, so this does not establish out-of-frame persistence.

**28. [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/html/2503.20314v2)** — **Team Wan**, collective first author; **Ang Wang** first individual, 2025, technical report.  
**Intermediate/input:** no simulator required; text or image-conditioned video diffusion. **Evidence:** the report compares against HunyuanVideo, CogVideoX1.5, and commercial models; Wan14B reports VBench 86.22 versus HunyuanVideo’s 83.24 under those configurations. **Relevance:** a reproducible foundation baseline, not proof of physical competence. The later [Wan2.2 TI2V-5B release](https://github.com/Wan-Video/Wan2.2) supports both text-only and image-conditioned generation in one model.

**D. Evaluation and evidence about physical fidelity**

For these entries, the intermediate is generally in the **evaluator**, not in video generation.

**29. [VideoPhy: Evaluating Physical Commonsense for Video Generation](https://arxiv.org/abs/2406.03520)** — **Hritik Bansal**, 2025, ICLR; preprint 2024.  
**Input/evaluation:** material-interaction prompts and generated videos; semantic adherence and physical commonsense are assessed separately. **Evidence:** compares models including CogVideoX, Lumiere, and Dream Machine; CogVideoX-5B jointly satisfies the two criteria in 39.6% of evaluated instances. **Relevance:** supports separate semantic and physical scores. Its material-interaction rubric is not automatically valid for powered drone flight.

**30. [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](https://arxiv.org/abs/2503.06800)** — **Hritik Bansal**, 2026, ICLR; preprint 2025.  
**Input/evaluation:** action-centric prompts, generated videos, and explicit physical rules. **Evidence:** assesses Wan, Cosmos, HunyuanVideo, CogVideoX, and proprietary systems; exposes substantial failures on difficult actions. **Relevance:** useful for designing event-specific rubrics. Beware version changes: the older 22% headline and the newer conference results refer to different model sets; they should not be mixed.

**31. [Do Generative Video Models Understand Physical Principles?](https://arxiv.org/abs/2501.09038)** — **Saman Motamed**, 2026, WACV; introduces **Physics-IQ**, preprint 2025.  
**Intermediate/input:** no simulator state; models receive an image or a real-video prefix and predict a withheld continuation. **Evidence:** evaluates Sora, Runway, Pika, Lumiere, SVD, and VideoPoet; perceived realism does not reliably establish predictive physical correctness. **Relevance:** the right conceptual template for testing learned dynamics later. Its motion metrics are proxies and remain sensitive to camera and shot changes.

**32. [Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://proceedings.mlr.press/v267/meng25c.html)** — **Fanqing Meng**, 2025, ICML; introduces **PhyGenBench/PhyGenEval**.  
**Input/evaluation:** 160 prompts spanning 27 physical laws; evaluators examine key phenomena, temporal order, and whole-video plausibility. **Evidence:** compares CogVideoX, Open-Sora, LaVie, Vchitect, Pika, Gen-3, and Kling; scaling and prompt engineering do not eliminate failures. **Relevance:** useful evaluation structure, though drone-specific propositions require validation. Do not confuse this with similarly named image or physics-question benchmarks.

**33. [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/html/2311.17982v1)** — **Ziqi Huang**, 2024, CVPR; preprint 2023.  
**Input/evaluation:** generated videos and dimension-specific prompts, assessed on 16 dimensions. **Evidence:** compares LaVie, ModelScope, VideoCrafter, and CogVideo, with human-alignment studies and real-video references. **Relevance:** useful secondary measurements of flicker, subject consistency, and motion. High consistency can accompany insufficient motion; report dynamic degree and task completion alongside it. Three custom clips do not constitute an official VBench evaluation.

**34. [VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness](https://arxiv.org/html/2503.21755v2)** — **Dian Zheng**, 2025, preprint; conference venue **UNVERIFIED**.  
**Input/evaluation:** generalist and specialist evaluators for controllability, physics, commonsense, human fidelity, and creativity. **Evidence:** compares HunyuanVideo, CogVideoX-1.5, Sora, and Kling1.6, exposing differences beyond visual quality. **Relevance:** instance preservation, motion order, and dynamic spatial relationships are relevant. Its instance-preservation scores should not be treated as a controlled test of memory for a specific drone leaving and re-entering view.

**35. [On the Content Bias in Fréchet Video Distance](https://arxiv.org/abs/2404.12391)** — **Songwei Ge**, 2024, CVPR.  
**Input/evaluation:** distributional video features under controlled spatial and temporal distortions; compares conventional I3D features with alternative video representations. **Evidence:** FVD can improve without improved motion and can underreact to temporal corruption. **Relevance:** a direct reason to avoid FVD as the headline metric. Changing the feature backbone does not solve the statistical problem of having only three scenarios.

**36. [Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification](https://arxiv.org/abs/2411.16718)** — **S P Sharan**, 2025, CVPR; introduces **NeuS-V**.  
**Intermediate/input:** prompt → temporal logic; generated frames → neural proposition scores → probabilistic automaton → verification. **Evidence:** evaluates Gen-3, Pika, T2V-Turbo-v2, and CogVideoX against VBench and VLM-only alternatives; improves agreement with human temporal alignment. **Relevance:** directly applicable to takeoff-before-translation and road-before-field events. Verification remains dependent on perception and specification quality; the paper acknowledges limited penalization of unintended elements.

**What this survey says about the specific domains.** The most developed evidence comes from general motion control, object physics, driving, and manipulation. I did not verify a released model specifically validated for both externally viewed drones and satellite convoy motion. Driving results are useful precedents, but ground-level camera priors do not establish overhead competence. Likewise, drone navigation footage is not evidence about generating drones as visible subjects.

**2. Can models take scene state instead of a rendered video?**

**Yes, through specific structured interfaces. No verified model here accepts an arbitrary simulator state dump and automatically understands all its semantics.** There are three materially different interfaces:

| Interface | Verified examples | What this means for us |
|---|---|---|
| Numerical geometry/state subsets | MagicDrive-V2: 3D boxes, camera poses, ego transforms; MotionCtrl: camera rotations/translations | Closest to direct state input. Requires the model’s coordinate system, categories, and trained representation. |
| Motion or semantic representations derived from state | Wan-Move/Tora: projected tracks; Go-with-the-Flow: flow; GraphVid: interaction graphs | Avoids an RGB simulation movie, but requires an explicit conversion and discards some simulator information. |
| Rendered structural observations | Cosmos/VACE: depth, edges, segmentation; 4Director: depth rendered from 4D geometry | Avoids photorealistic rendering, but still uses rasterized, view-dependent controls. |

These distinctions are documented in [MagicDrive-V2’s formulation](https://arxiv.org/html/2411.13807v3), [MotionCtrl’s architecture](https://arxiv.org/html/2312.03641v2), and [Wan-Move’s inference interface](https://github.com/ali-vilab/Wan-Move).

For this week, **Wan-Move is the most useful answer to the lead’s practical question**. From simulator poses and camera calibration:

1. Sample surface points on each object, including points that express orientation.
2. Project them into each frame.
3. Compute visibility using frustum and occlusion tests.
4. Supply the trajectories, visibility arrays, initial image, and prompt.

This needs geometry and projection; it does not need a photorealistic RGB control video. A centroid alone is insufficient to specify a vehicle’s orientation through a turn.

MagicDrive-V2 is stronger evidence for direct numerical geometry input, but adapting its street-view representation to flying drones or an orbital viewpoint is a larger project. GraphVid is semantic rather than metric, and its incomplete verified release makes it unsuitable as a dependency.

I also disagree with excluding rendered conditioning because it is “just style transfer.” Depth- or trajectory-conditioned synthesis can generate appearance from noise and perform substantial generative work. The meaningful distinction is **what information is supplied and what must be predicted**:

- **Full future simulator controls:** the model synthesizes visual observations of supplied dynamics.
- **Past observations plus actions:** the model must predict future dynamics.
- **Persistent state across unseen intervals and interventions:** the system must maintain a world representation.

The first is a useful generative renderer. It does not, by itself, demonstrate improved learned physical prediction. The current Cosmos Sim-Transfer documentation explicitly assigns physics and state to the simulator and appearance to the generator. [Model description](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)

**3. Recommended simulator ↔ generator pairings**

**If Tuesday’s decision must name one common pipeline: choose scripted Blender → Cosmos-Transfer2.5.** Add PyBullet dynamics where actual drone dynamics matter. Run VACE on the same scene controls to obtain the essential comparison within one generator.

Setup estimates below are my engineering estimates for someone comfortable with Python and basic scene scripting, assuming usable assets. They are not measured runtimes.

| Test video | Recommended pairing and conditioning | Estimated setup | Main risks |
|---|---|---:|---|
| Drones take off and move | **gym-pybullet-drones/PID → Blender → Cosmos-Transfer2.5**, using depth and silhouette-preserving edges. VACE is the parallel comparison. | Approximately 1–2 person-days for the first end-to-end scene | Ground contact, changing silhouette, propeller appearance, and depth normalization. Physics-to-render coordinate mismatch can invalidate evaluation. |
| Drones moving in air | **Blender scene state → Wan-Move**, using projected surface tracks and visibility; also render controls for the common Cosmos/VACE comparison. | Approximately ½–1 additional day after the common scene/export code | Identity swaps, track loss, newly revealed surfaces, and loss of control during prolonged absence. |
| Overhead convoy turns right into a field | **Blender with a fixed near-orthographic camera → Cosmos-Transfer2.5**, prioritizing edges and suitable depth; compare VACE and projected tracks. | Approximately ½–1 additional day | Flat terrain gives weak depth contrast; vehicles may rotate incorrectly, merge, change spacing, or cause background deformation. |

The [PyBullet drone package](https://github.com/learnsyslab/gym-pybullet-drones) provides existing controllers, so this need not involve training. **If integration consumes Tuesday, use prescribed Blender trajectories and label the experiment “kinematically controlled.”** Do not describe keyframed motion as evidence that a physics solver helped.

Hardware implications:

- **Isaac Sim:** exclude from this week’s plan. Its requirements explicitly state that A100/H100 GPUs without RT cores are unsupported. [Official requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- **Blender:** use headless Cycles with CUDA, with CPU rendering as the fallback. Keep assets and render quality modest; the required controls do not need production-quality lighting.
- **Cosmos-Transfer2.5:** official documentation lists 65.4 GB GPU memory for its documented single-GPU configuration; its model card lists A100 among tested hardware. Additional controls and settings still need a smoke test. Published H100/B200 timings are not A100 measurements. [Inference requirements](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md), [model card](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
- **Wan-Move:** documented single-40-GB operation with offloading makes memory feasibility credible. Actual throughput remains unmeasured here. [Inference instructions](https://github.com/ali-vilab/Wan-Move)
- **VACE:** start with the small model for interface debugging, then use 14B for reported results if its smoke test passes; distributed inference is documented. [Official implementation](https://github.com/ali-vilab/VACE)
- **Cosmos3, 4Director, AirSim/Unreal:** valuable later candidates, but hardware validation or installation complexity makes them poor critical-path choices now.

Eight GPUs primarily help run seeds and conditions concurrently. They do not eliminate environment, control-format, or domain-generalization problems.

**4. Honest critique**

**The core idea is sound, but its broad novelty is already gone.** GPT4Motion, PhysGen, Cosmos Transfer, CoDeR, Video-Coder, and RoboRender collectively cover most versions of “use explicit structure or simulation to improve generated video.” A drone/satellite application alone is unlikely to establish a strong methods contribution.

There is still a worthwhile empirical question: **how much appearance improvement can we obtain before the generated video stops supporting the simulator’s labels?** A careful negative result could be valuable—for example, showing that visually convincing outputs systematically change object counts, orientations, re-entry positions, or convoy spacing.

**Text-only and first-frame baselines are informative, but they answer a limited question.** They measure whether the additional control information improves the pipeline. They do not isolate simulator quality or demonstrate improved physics knowledge.

Two confounds need explicit treatment:

- **Model confound:** Wan text-only versus Cosmos controlled generation changes both model and input. Keep a VACE checkpoint fixed while varying its inputs.
- **Information confound:** a complete future trajectory supplies answers that text or one image may not determine. Even a comparison within one checkpoint measures the value of extra information.

There is a further issue with trajectory scoring: **the simulator’s exact path is not necessarily the only correct interpretation of a text prompt.** A text-only model should be evaluated for requested direction, events, counts, and acceptable regions. Exact pixel-path error is useful for measuring control adherence, but a different valid path is not automatically a physics failure.

**These three maneuvers do not strongly test physics.** Takeoff, flight translation, and a convoy turn can largely be prescribed kinematically. To attribute gains specifically to physics, a later experiment should compare physically simulated motion against matched scripted motion, or test interventions involving forces, constraints, contact, or acceleration limits.

**Simulator annotations are candidate labels for generated video.** They become valid inherited labels only after checking the generated objects’ positions, geometry, visibility, and identities. Evaluating the simulator trace alone verifies the simulator, even if the generated convoy has lost a vehicle.

**Out-of-frame motion needs careful interpretation.** An object being invisible outside the camera is expected. Failures include incorrect re-entry timing/location, changed identity, or a duplicate appearing while the original remains absent. If simulator controls specify the entire unseen interval, successful re-entry demonstrates externally maintained state and rendering fidelity; it does not establish the video model’s own memory.

I would revise four claims in the design doc:

| Existing idea | Defensible replacement |
|---|---|
| “Covers physics the simulator doesn’t model” | May synthesize plausible-looking effects; their physical correctness needs independent evidence. |
| “Faster than full-fidelity rendering” | A hypothesis to measure, including setup, simulation, preprocessing, and generation time. |
| “Closes the sim2real gap” | Improves perceived appearance unless downstream real-data transfer is demonstrated. |
| Three-scene leaderboard/benchmark | A reproducible three-scenario case study with seed variability and documented failures. |

**A trivial result:** selected good-looking controlled clips beat weak text-only examples on exact supplied trajectories.

**A strong result:** all predeclared samples are reported; conditioning is ablated within one generator; visual gains and label corruption are measured separately; failure modes persist across seeds; and the comparison includes the raw simulator output. A subsequent strong paper would add scene diversity, controlled interventions, and downstream validation on real data.

One correction to the evaluation plan: **NeuS-E is a refinement pipeline, not an independent metric.** If used to optimize outputs, evaluate them with separate task annotations and humans rather than only the feedback score it optimizes. [NeuS-E manuscript](https://arxiv.org/html/2504.17180v4)

**5. Proposed experiment plan through Friday**

**Research question:** *For three prescribed scenarios, how reliably do released video generators preserve simulator-specified motion and events, and what appearance improvement accompanies that preservation?*

**Freeze three scene specifications.**

| Scene | Suggested specification | Primary checks |
|---|---|---|
| Takeoff | Two clearly resolved drones; initially grounded; takeoff followed by translation | Count, lift-off timing, event order, trajectory, ground penetration |
| Airborne motion | Two distinguishable drones on different paths; include a brief exit/re-entry only if it fits the intended example | Identity, crossings, expected visibility, re-entry timing/location |
| Convoy | Three vehicles; fixed overhead camera; northward road travel followed by a finite-radius right turn east into a field | Count, turn direction, orientation, spacing, road/field transition, background stability |

Use approximately five seconds per clip. Keep objects large enough to annotate reliably. Treat the overhead rendering as **satellite-like** unless ground sample distance and sensor effects are actually specified.

Save continuous or densely sampled scene states, camera calibration, object identities, visibility, projected tracks, RGB, depth, and masks. Build deterministic scene specifications manually for v0; an LLM scene-authoring layer would introduce another variable.

**Conditions and sample budget**

Use three predeclared generation seeds per scene. The following gives **54 generated videos plus three simulator references**:

| Condition | Generator and inputs | Purpose |
|---|---|---|
| R | Raw simulator RGB | Reference for exact control and baseline appearance |
| A | VACE: text only | Minimal-information baseline |
| B | Same VACE: text + first frame | Initial-layout baseline |
| C | Same VACE: text + first frame + dynamic simulator control | Main test of added motion conditioning |
| D | Same VACE: text + first frame + repeated/static control | Checks whether gains require the time-varying signal |
| E | Cosmos-Transfer2.5: text + dynamic controls | Practical simulator-to-video comparison |
| F | Wan-Move: text + same first frame + projected tracks/visibility | Non-RGB motion-conditioning comparison |

VACE’s documented interface supports text-only, reference-frame, and control-video tasks. Condition D is a diagnostic ablation; report it as such, because static conditioning can deliberately conflict with the requested motion. [VACE user guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md)

Choose one principal VACE control representation after a smoke test. Do not build a full depth × edge × mask × strength grid this week. A small RGB-control comparison is useful only after the core experiment works.

If time permits, add **Wan2.2 TI2V-5B text-only and first-frame conditions**, adding 18 outputs. This gives a stronger external reference without claiming it is the latest or best available model. The release supports both tasks. [Wan2.2 instructions](https://github.com/Wan-Video/Wan2.2)

Experimental controls:

- Use the identical first image wherever image conditioning applies. Avoid an unmatched real photograph with different geometry.
- Freeze prompts and prompt expansion; log the final text actually supplied.
- Use native supported generation settings, then apply a predeclared common evaluation window. Record frame rate, duration, resolution, and resizing.
- Keep seeds paired within compatible runs. The same integer seed across unrelated architectures does not create equivalent randomness.
- Use a separate smoke-test example for tuning. Report every planned evaluation sample, including failures.
- Record generation time, peak memory, preprocessing time, retries, and manual intervention.

**Primary metrics**

| Axis | Measurement | Important qualification |
|---|---|---|
| Count and identity | Visible-count accuracy; missing/extra-object duration; identity switches | Distinguish intended occlusion from disappearance. |
| Prescribed motion | Mean/final projected-position error, normalized by object size; orientation error where observable | Score visible matched objects and report missing-object rates separately. |
| Event completion | Takeoff-before-translation; right turn; road-before-field; event timing | Use explicit predicates and human-audited observations. |
| Re-entry | Correct identity, location, timing, and absence of duplicates | Describes externally controlled persistence when the future is supplied. |
| Convoy structure | Inter-vehicle spacing and order; correct heading through the turn | A moving centroid does not guarantee a correctly oriented vehicle. |
| Camera/background | Drift of fixed landmarks and unintended camera motion | Particularly important for the overhead scene. |
| Appearance | Blinded pairwise realism preference versus raw simulation; artifact rubric | Perceived realism is not physical validity. |
| Cost | Wall time, GPU memory, retry count, human setup time | Report end-to-end cost as well as denoising time. |

For trajectory error, establish identity once and maintain it through the sequence; independently rematching every frame can hide swaps. Report errors conditional on successful tracking **alongside** tracking and object-preservation failures.

Use automated tracking to accelerate annotation, then manually audit it. At this scale, human checks are affordable and more defensible than assuming a detector handles generated drones correctly. NeuS-V can supplement event checks, and selected VBench measurements can supplement appearance checks. Neither should replace task-specific observations.

**Do not headline FID/FVD.** Three reference videos provide only three independent video samples; additional frames or generated seeds do not create scene diversity. FVD also has known content bias. [Content-bias study](https://arxiv.org/abs/2404.12391)

Present per-scene results and seed variability. Avoid statistical claims about the broader population of drone or satellite videos.

**Schedule**

| Deadline | Concrete output |
|---|---|
| **Monday, Nov 9** | Freeze scope, references, control representations, and measurement definitions; check checkpoint access. |
| **Tuesday, Nov 10, before 3 PM** | One successful end-to-end smoke test; choose the pairing using observed compatibility. Freeze all three scene specifications and the experiment matrix. |
| **Wednesday, Nov 11, before 3 PM** | Start the report with methods, prior art, baseline limitations, and empty result tables. Finish simulator exports and annotation templates. |
| **Thursday, Nov 12** | Run the frozen sample matrix; audit generated tracks and event labels; conduct blinded appearance comparisons. |
| **Friday, Nov 13** | Deliver all samples, the score table, failure taxonomy, resource measurements, and a concise decision about the next experiment. |

**Friday deliverable:** a short report plus a complete video grid showing prompt, initial image, simulator/control input, output, and audited trajectory overlay. Include ordinary and failed samples, not only the best examples. A suitable title is **“Motion and Label Preservation in Simulator-Conditioned Video Generation: Three Case Studies.”**

**Cuts, in order:**

1. Drop new simulator installations, reconstruction-heavy methods, and newly released serving stacks.
2. Drop broad dataset downloads, detector training, closed-model comparisons, and distributional quality metrics.
3. Drop extra control modalities and the Wan2.2 external baseline.
4. If necessary, drop Wan-Move or Cosmos after a fixed integration time limit.
5. Preserve the minimum experiment: **three scenes × three VACE conditions A/B/C × three seeds = 27 outputs**, plus raw simulator references and human-audited measurements.

If only one seed is possible, label the result a demonstration. Do not disguise it as a reliable model ranking.

**6. What I am least sure about**

- **Which generator will preserve drone shape and identity best.** Public benchmarks do not settle that domain-specific question.
- **Actual A100 throughput and installation friction.** Documented memory feasibility is encouraging; this review did not execute the models.
- **How much depth helps in the overhead scene.** A nearly planar background may make edges or point tracks more informative.
- **Longer occlusion and re-entry behavior.** Track-conditioned systems have documented limitations; short successful clips would not establish durable memory.
- **The completeness of several recent releases.** CoDeR’s release, GraphVid’s runnable inference, and complete DriveDreamer-2 execution remain **UNVERIFIED**; Cosmos3 A100 support is also **UNVERIFIED**.
- **Research novelty beyond the pilot.** The survey establishes substantial overlap. A stronger claim will depend on the failure analysis and controlled evidence the team actually obtains.