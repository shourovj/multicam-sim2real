# Literature Survey v0: Simulator-in-the-Loop Video Generation

**Owner:** SP Sharan · **Drafted:** Nov 9, 2026 · **Scope:** general domain first, then drones and satellite. Builds on [project understanding](../2026-11-09-1626-project-understanding.md) and [design doc v0](../2026-11-09-1416-design-doc-v0.md).

**Verification:** every paper below was found by web search, and the key arXiv IDs were re-checked against their arXiv abstract pages. Items marked *(unverified)* came from secondary sources. Numbers are as reported by the authors; most 2026 entries are preprints.

## 0. Short version

1. **Yes, prior work exists, but none of it is aerial or overhead.** The closest pipelines are:
   - [CoDeR](https://arxiv.org/abs/2609.26458): LLM agents write a Three.js world, and a ControlNet video model renders it from whitebox RGB plus depth.
   - [Programmable World Model](https://arxiv.org/abs/2609.10540): an LLM writes a program, the program produces 3D boxes plus direction maps, and a ControlNet world model renders them.
   - [VLIPP](https://arxiv.org/abs/2503.23368): an LLM plans box trajectories, and flow derived from them becomes warped noise for the video model.
   - GPT4Motion, LVD, DriveDreamer-2, PhysGen and PhysCtrl.

   All of them are ground-level, ego-view, tabletop or driving. We found **no sim-conditioned generation of drone or satellite video**.
2. **Only VLIPP compares a structured-motion pipeline against both text-to-video (T2V) and image-to-video (I2V) baselines** on physics. Nobody reports counts, direction, trajectory error and event order across text-only, text + first frame, and sim-conditioned inputs on the same backbone. That three-way comparison, plus the testbench, is open.
3. **On your question (can the model take scene state directly, not a rendered video?):** yes. Open models take point tracks (Wan-Move, ATI), 6DoF poses (3DTrajMaster), 3D point-track videos (Diffusion as Shader), boxes (Boximator, SG-I2V, Ctrl-V), BEV/HD-map layouts (MagicDriveDiT, Cosmos-Drive) or actions (Vista, Matrix-Game, Aero-World). You are right that feeding a rendered video is close to style transfer, and the literature measures this directly:
   - Cosmos-Transfer1 with blur control: SSIM 0.96, but diversity only 0.19.
   - Driving with DINO calls it the "consistency–realism dilemma".

   Sparse state makes the model *generate*, but it follows the state less reliably. For example, MagicDriveDiT reaches only about 18 box mAP on its own generated video, and 3DTrajMaster handles at most 3 entities. No paper compares the input formats against each other on one backbone for motion fidelity. That is a natural ablation for us.
4. **Practical pairing for v0:** BlenderProc (headless, runs on our A100s) → Cosmos-Transfer2.5 (depth/seg/edge) and Wan2.1-VACE as the dense arms, plus **Wan-Move or ATI (Wan2.1-14B) fed with simulator tracks** as the sparse arm. Two constraints:
   - **Isaac Sim does not run on A100**, which has no RT cores. This is confirmed on NVIDIA's requirements page.
   - **In v0 the objects are the main subject and clearly visible.** Drones under 3 px are smaller than one latent cell (the VAEs downsample 8–16×), so no generator can hold them, and they are out of scope for now.

## 1. What the simulator hands the video model: the spectrum

```
DENSE <------------------------------------------------------------------------------------> SPARSE
rendered RGB      depth / seg / edge     rasterized boxes /     point tracks / 6DoF /    actions / IMU /   text only
(SDEdit, V2V,     (Cosmos-Transfer,      BEV / HD-map video     box coordinates          camera path       (T2V; LVD-style
 RealMaster)       VACE, GPT4Motion,      (Cosmos-Drive, DaS,    (Wan-Move, ATI,          (Vista,           planning)
                   CoDeR)                 Panacea, Prog. WM)     3DTrajMaster, Boximator)  Aero-World)
controllability:      highest ------------------------------------------------------------------> lowest
generative freedom:   lowest (re-texturing) ----------------------------------------------------> highest
typical failure:      sim look leaks through, sim artifacts kept             hallucinated counts, physics, drift
what the sim fixes:   shape, lighting, occlusion, timing | layout + timing | where/when objects move | camera only
```

How to read this for our experiment:

- **Dense end.** Faithfulness is mostly *inherited* from the simulator. Without an adherence metric and a "sim render, no generator" floor, a win over T2V is close to automatic.
  - Cosmos-Transfer1 TransferBench: blur control gives SSIM 0.96 at diversity 0.19; segmentation-only gives SSIM 0.66 at diversity 0.42.
  - Driving with DINO: edge and blur controls bake in sim artifacts, while depth, segmentation and HD-map controls look more real but hold structure less tightly.
- **Per-region control is the middle ground.** In Cosmos-Transfer1 "Setting 2", tight edge control sits on the foreground and only segmentation on the background. This gives the best quality and foreground mIoU. For us, that means tight control on drones and vehicles, and loose control on sky, terrain and field.
- **Sparse end.** It works, but adherence is middling:
  - Boximator: 2–9× higher box AP than with no boxes.
  - MagicDriveDiT: mAP 18.
  - Motion Prompting: EPE about 3.9 px with dense tracks, worse with sparse ones.
  - Wan-Move: loses control under long occlusion.
- **Few papers measure counts or event order on the output.** Programmable WM (count and state accuracy) and LVD (detection-based numeracy and sequence) are the exceptions. A count / track / temporal-logic testbench is a real contribution.

## 2. Closest prior work: prompt → planner/simulator → video

| Paper | Planner or simulator | Handoff to generator | Generator | Baselines | Key result |
|---|---|---|---|---|---|
| **CoDeR**, Fang et al. (USTC + Alibaba), arXiv Sep 2026 · [2609.26458](https://arxiv.org/abs/2609.26458) · [page](https://becauseimbatman0.github.io/CoDeR) | Five agent roles (Creator, Executors, Artist, Traveler, registration); GPT-6 Astra coding agents write Three.js "whitebox" worlds, with SMPL-H humans | Whitebox RGB + GT depth + text from world state; Canny-edge ControlNet | MiniMax H3 + LoRA + ControlNet | ABot-World, Matrix-Game 3.0, LingBot-World (quantitative); EgoSim, HyDRA, LiveWorld (qualitative) | WorldScore avg 87.9 vs 74.8 (LingBot-World). Compared against world models, **not T2V/I2V** |
| **Programmable World Model**, Huang et al., arXiv 2026 · [2609.10540](https://arxiv.org/abs/2609.10540) | LLM writes entity states and transition rules; an engine keeps a persistent state, including entities off screen | 3D oriented boxes rasterized into identity, semantic and **direction** maps (7 states) | LingBot-World-v1 + Structured Spatial ControlNet | LingBot-World-V2, YUME (state changes given by prompt) | Count/State accuracy **94/98** vs 40.8/8. Closest template for counts and direction |
| **VLIPP**, Yang et al., ICCV 2025 · [2503.23368](https://arxiv.org/abs/2503.23368) | GPT-4o + Grounded-SAM2 plan box trajectories using a physics law | Cut-and-paste motion video → RAFT flow → warped noise | Go-with-the-Flow (CogVideoX) | **T2V:** CogVideoX, LTX, OpenSora, PhyT2V, LVD. **I2V:** CogVideoX-I2V, SVD, LTX-I2V, SG-I2V | PhyGenBench 0.60 vs 0.52 (T2V) and 0.54 (I2V). **The only paper with our three-arm design** |
| **LVD**, Lian et al., ICLR 2024 · [2309.17444](https://arxiv.org/abs/2309.17444) | LLM writes per-frame boxes with IDs | Training-free attention energy guidance | ModelScope | ModelScope | Detection-based accuracy 49.4% vs 23.0%. Benchmark design is reusable (numeracy, visibility, spatial dynamics, sequential) |
| **VideoDirectorGPT**, Lin et al., COLM 2024 · [2309.15091](https://arxiv.org/abs/2309.15091) | GPT-4 multi-scene plan with layouts and consistency groups | Box-guided attention (Layout2Vid) | ModelScope-based *(unverified)* | ModelScopeT2V | Better layout, count and movement direction |
| **GPT4Motion**, Lv et al., CVPRW 2024 · [2311.12631](https://arxiv.org/abs/2311.12631) | GPT-4 writes a **Blender** physics script | Blender edge + depth | SDXL + ControlNets | AnimateDiff, ModelScope, T2V-Zero | First text → engine → video pipeline. Its metrics do not test motion fidelity |
| **DriveDreamer-2**, Zhao et al., AAAI 2025 · [2403.06845](https://arxiv.org/abs/2403.06845) | LLM writes agent trajectories, then an HD map | Structured layout | Driving world model | Earlier driving world models | Same structure as our pipeline, applied to driving. Generated data helps 3D detection |
| **MoReGen**, Bai et al., arXiv Dec 2025 · [2512.04221](https://arxiv.org/abs/2512.04221) | LLM agents write Pymunk, Blender or Manim code | None: the output **is** the sim render | — | Wan2.2, LTXV, CogVideoX, Veo3, Sora2 | Best trajectory DTW (8.93 vs Sora2 11.21). Equivalent to our "sim only" floor. Gives trajectory metrics |
| **NewtonGen**, Yuan et al., arXiv 2025 · [2509.21309](https://arxiv.org/abs/2509.21309) | Learned Newtonian ODE → 9-D state trajectory | Optical flow | Go-with-the-Flow | Sora, Veo3, CogVideoX, Wan2.2 | Physical Invariance Score, good for take-off acceleration and constant convoy speed |
| **PhysGen**, Liu et al., ECCV 2024 · [2409.18964](https://arxiv.org/abs/2409.18964) | Pymunk 2D rigid body in image space | Composited RGB → SDEdit | SEINE | SEINE, I2VGen-XL, DynamiCrafter | Physical realism 4.14 vs 2.11. Motion-FID is a cheap motion metric |
| **PhysCtrl**, Wang et al., NeurIPS 2025 · [2509.20358](https://arxiv.org/abs/2509.20358) | Generative physics model (trained on 550K simulations) → 3D point trajectories | 3D tracking video | Diffusion as Shader | Wan2.1-I2V-14B, CogVideoX, DragAnything | Beats plain I2V on physics. The point-track video format works well |
| **WonderPlay**, Li et al., ICCV 2025 · [2505.18151](https://arxiv.org/abs/2505.18151) | Genesis solvers on a 3D scene | Flow (warped noise) + coarse RGB (SDEdit) | Go-with-the-Flow | PhysGen, CogVideoX-I2V, Tora | Flow and RGB together beat either alone |
| **Coarse-to-Real**, Gomez-Nogales et al., arXiv Jan 2026 · [2601.22301](https://arxiv.org/abs/2601.22301) | Coarse 3D crowd renders | Rendered coarse video | DiT renderer (mixed CG + real training) | Wan2.1, Sora, Wan2.1 + ControlNet | T2V alone can't hold camera or trajectories in crowded scenes |
| **SimGen**, Zhou et al., NeurIPS 2024 · [2406.09386](https://arxiv.org/abs/2406.09386) | MetaDrive simulator | Sim layout (depth, segmentation) | Cascade diffusion | — | Synthetic data improves BEV detection and segmentation |
| **Dreamland**, Mo et al., arXiv 2025 · [2506.08006](https://arxiv.org/abs/2506.08006) | MetaDrive → layered world abstraction | Depth, semantic, instance, RGB | Cosmos-Transfer1-7B | SimGen | About 52% better FID. Measures alignment with si-RMSE and mIoU |

Related: PhysDreamer, PhysGen3D, PhysAnimator (physics + I2V, less relevant to us); PhyT2V, DiffPhy (prompt refinement only, which makes a good "smart text" baseline); Visual-Semantic Chain-of-Events ([2609.00656](https://arxiv.org/abs/2609.00656), a scene-graph intermediate); Magpie ([2608.27168](https://arxiv.org/abs/2608.27168), a game-engine whitebox fed to a real-time generative renderer).

## 3. Dense-control video models and sim2real rendering (the "style transfer" end)

| Model or paper | Conditioning | Open weights | Simulator pairing | Takeaway |
|---|---|---|---|---|
| **Cosmos-Transfer2.5**, NVIDIA 2025 · [2511.00062](https://arxiv.org/abs/2511.00062) · [repo](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | Text + edge, blur, segmentation, depth (multi-control, weighted) | Yes, 2B | CARLA ([Cookbook recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.html)), Isaac Sim (CosmosWriter) | Main dense candidate. **The repo is in maintenance mode; Cosmos 3 (June 2026) is the successor.** Check whether Cosmos 3 has a Transfer model |
| **Cosmos-Transfer1**, NVIDIA 2025 · [2503.14492](https://arxiv.org/abs/2503.14492) | Same, plus LiDAR and HD map (AV variant); per-pixel spatiotemporal control weights | Yes, 7B | Isaac Lab, CARLA 0.9.16 (native) | **TransferBench** measures adherence (Blur SSIM, Edge F1, Depth si-RMSE, Mask mIoU) against diversity. Copy this protocol |
| **Wan2.1-VACE**, Jiang et al., ICCV 2025 · [2503.07598](https://arxiv.org/abs/2503.07598) | Text, reference image, depth, pose, flow, gray, scribble, box layout or trajectory, masks | Yes, 1.3B and 14B | None published | Widest control range. Same family as our Wan2.2 baselines |
| **Cosmos-Drive-Dreams**, Ren et al., 2025 · [2506.09042](https://arxiv.org/abs/2506.09042) | HD map + 3D boxes projected to a condition video | Yes (NVIDIA license) | Structured driving scenarios | Generated data improves 3D detection, most in rain, fog and night |
| **Driving with DINO**, Chen et al., ACM MM 2026 · [2602.06159](https://arxiv.org/abs/2602.06159) | DINOv3 features of the sim render | *(unverified)* | Driving sim | Names the consistency–realism dilemma |
| **RealMaster**, Cohen-Bar et al., 2026 · [2603.23462](https://arxiv.org/abs/2603.23462) | Rendered RGB (IC-LoRA) | *(unverified)* | GTA-V, CARLA | Video version of Enhancing Photorealism Enhancement; keeps geometry and dynamics |
| **DriveCtrl**, 2026 · [2605.15116](https://arxiv.org/abs/2605.15116) | Depth | *(unverified)* | Driving sim | Argues depth is the best control because it carries the least appearance |
| **DiffusionRenderer**, Liang et al., CVPR 2025 · [2501.18590](https://arxiv.org/abs/2501.18590) | G-buffers + lighting | Yes (Cosmos-based) | Any rasterizer | A neural renderer; adds almost nothing generative |
| **Enhancing Photorealism Enhancement**, Richter et al., TPAMI · [2105.04619](https://arxiv.org/abs/2105.04619) | Render + G-buffers (GAN) | Code | GTA-V → Cityscapes | GAN-era reference. Unpaired translation hallucinates content |
| CyCADA, Hoffman et al., ICML 2018 · [1711.03213](https://arxiv.org/abs/1711.03213) | Unpaired translation (GAN) | Yes | GTA5 → Cityscapes | Classic evidence that a detector trained on translated sim data transfers to real |
| Older video ControlNets: Ctrl-Adapter, VideoComposer, AnimateDiff + SparseCtrl, ControlVideo, FlowVid, Rerender-A-Video | Depth, edge, pose, flow | Yes | — | Historical or cheap baselines. FlowVid and Rerender are pure re-skinning, a lower bound |

Control modalities accepted by the main open models:

| Model | Text | Image | Depth | Seg | Edge | Flow | Pose | Trajectory / boxes | Camera |
|---|---|---|---|---|---|---|---|---|---|
| Cosmos-Transfer2.5 | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | multiview only |
| Wan2.1-VACE | ✓ | ✓ | ✓ | masks | ✓ | ✓ | ✓ | ✓ (layout) | – |
| Wan-Move / ATI (Wan2.1-14B) | ✓ | ✓ (I2V) | – | – | – | – | – | ✓ point tracks | ATI ✓ |
| Diffusion as Shader | ✓ | ✓ | (3D tracks) | – | – | – | – | ✓ 3D tracks | ✓ |
| Tora (CogVideoX-5B) | ✓ | ✓ | – | – | – | – | – | ✓ | – |
| 3DTrajMaster (CogVideoX-5B) | ✓ | – | – | – | – | – | – | ✓ 6DoF, ≤3 entities | – |

## 4. Feeding scene state directly (no rendered video)

| Model | Exact conditioning | Open | Reported adherence |
|---|---|---|---|
| **Wan-Move**, Chu et al., NeurIPS 2025 · [2512.08765](https://arxiv.org/abs/2512.08765) | Dense or sparse 2D point tracks in latent space; I2V | **Yes**, 14B-480P + MoveBench | On par with Kling Motion Brush (user study); fails under long occlusion |
| **ATI**, ByteDance 2025 · [2505.22944](https://arxiv.org/abs/2505.22944) | Point trajectories for objects and camera; I2V | **Yes**, Wan2.1-14B | *(not checked)* |
| **3DTrajMaster**, Fu et al., ICLR 2025 · [2412.07759](https://arxiv.org/abs/2412.07759) | Per-entity **6DoF pose sequences** + entity text | Code (CogVideoX-5B) | At most 3 entities; global motion only |
| **Motion Prompting**, Geng et al., CVPR 2025 | Sparse or dense point tracks (ControlNet on Lumiere) | No | EPE 3.89 vs 12.49 (DragAnything) |
| **Tora**, Zhang et al., CVPR 2025 · [repo](https://github.com/alibaba/Tora) | Trajectories as motion patches; **T2V and I2V** | Partial (CogVideoX-5B) | — |
| **Boximator**, Wang et al., ICML 2024 | Box sequences as tokens | No | Box AP 1.9–8.9× higher than with no boxes |
| **SG-I2V**, Namekata et al., ICLR 2025 · [2411.04989](https://arxiv.org/abs/2411.04989) | Box + trajectory, zero-shot on SVD | Code | Narrows the gap to supervised methods |
| **Ctrl-V**, Luo et al., TMLR 2025 · [2406.05630](https://arxiv.org/abs/2406.05630) | 2D or 3D box tracks → Box2Video | *(unverified)* | Up to 15 boxes per frame |
| **LeviTor**, Wang et al., CVPR 2025 | 2.5D tracks (points + depth) | Code | Resolves which object passes in front |
| **MagicDriveDiT**, Gao et al. · [2411.13807](https://arxiv.org/abs/2411.13807) | 3D boxes, ego trajectory, camera, road map | *(unverified)* | BEVFormer mAP 18.2 on generated video |
| **GAIA-2**, Wayve 2025 · [2503.20523](https://arxiv.org/abs/2503.20523) | Ego actions, agent 3D boxes, metadata | No | Qualitative |
| **Vista**, Gao et al., NeurIPS 2024 · [2405.17398](https://arxiv.org/abs/2405.17398) | Steering, speed, trajectory, goal point | Yes | FID −55% vs the previous driving world model |
| **Aero-World**, Al Radi et al., arXiv 2026 · [2605.19728](https://arxiv.org/abs/2605.19728) | **Drone IMU actions** (acceleration, angular velocity); I2V | Not stated | Action alignment 63.6; beats AirScape |
| **Matrix-Game 2.0**, Skywork 2025 · [2508.13009](https://arxiv.org/abs/2508.13009) | Per-frame keyboard and mouse actions | Yes | Fails out of domain |
| Go-with-the-Flow, MotionCtrl, DragNUWA, DragAnything, TrackGo, Peekaboo, TrailBlazer, Direct-a-Video, Perception-as-Control, Panacea, DrivingDiffusion, GameNGen, Genie 2/3, DIAMOND, Oasis, UniSim | Flow, trajectories, boxes, layouts, actions | Mixed | Background. See the agent notes if needed |

**Answer to "is a rendered video just style transfer?":** largely yes for RGB, blur or edge; partly for depth or segmentation. Sparse state (tracks, boxes, poses) is the setting where the generator has to synthesize appearance, occlusion and secondary physics (dust, shadows, rotor blur). It is the more interesting test of a "world model", but adherence drops. The experiment should report adherence *and* realism for each level.

## 5. Evidence that T2V alone fails at motion (motivation)

| Paper | Finding |
|---|---|
| **How Far is Video Generation from World Model**, Kang et al. 2024 · [2411.02385](https://arxiv.org/abs/2411.02385) | In a controlled 2D simulator, models do not generalize out of distribution, even with more data or a bigger model. They copy the nearest training case |
| **Physics-IQ**, Motamed et al., WACV 2026 · [2501.09038](https://arxiv.org/abs/2501.09038) | Physical understanding is severely limited and unrelated to visual realism. Physics-IQ Verified ([2606.18943](https://arxiv.org/abs/2606.18943)) shows the ground-truth labels are noisy |
| **VideoPhy-2**, Bansal et al., ICLR 2026 · [2503.06800](https://arxiv.org/abs/2503.06800) | The best model (Wan2.1-14B) gets both semantics and physics right only about 33% of the time |
| **T2V-CompBench**, Sun et al., CVPR 2025 · [2407.14505](https://arxiv.org/abs/2407.14505) | Motion binding (tracker-based) and numeracy (detector-based) are among the hardest categories |
| **T2VCountBench** · [2504.04051](https://arxiv.org/abs/2504.04051) | Best counting accuracy is about 50% for counts 1–9 |
| **MemoBench**, 2026 · [2606.27537](https://arxiv.org/abs/2606.27537) | World models lose track of objects while they are out of view, which matches our "motion out of frame" result |
| **Training Object Permanence in World Models**, 2026 · [2609.28654](https://arxiv.org/abs/2609.28654) | Permanence has to be trained with simulator data, which supports putting the sim in the loop |
| PhyGenBench · [2410.05363](https://arxiv.org/abs/2410.05363), VideoPhy · [2406.03520](https://arxiv.org/abs/2406.03520), T2VPhysBench · [2505.00337](https://arxiv.org/abs/2505.00337), WorldModelBench · [2502.20694](https://arxiv.org/abs/2502.20694), WorldScore · [2504.00983](https://arxiv.org/abs/2504.00983), VBench-2.0 · [2503.21755](https://arxiv.org/abs/2503.21755) | Physics and controllability fail consistently; scaling and prompting do not fix it |

## 6. Metrics we can reuse

- **Tracking on generated video:** SAM 2 ([2408.00714](https://arxiv.org/abs/2408.00714)) for box/mask tracks and CoTracker3 ([2410.11831](https://arxiv.org/abs/2410.11831)) for point tracks. For v0, hand-correct the tracks (there are only 3 videos).
- **Trajectory:** ADE/FDE, DTW and Fréchet distance against the sim (MoReGen); Physical Invariance Score (NewtonGen); TRAJAN ([2505.00209](https://arxiv.org/abs/2505.00209)), a per-video motion score.
- **Adherence to the sim condition:** re-estimate depth, segmentation and boxes on the output and compare with the sim's ground truth (Cosmos TransferBench, Dreamland). This guards against "wins" that are copied from the sim.
- **Counts and event order:** per-frame count error; NeuS-V ([2411.16718](https://arxiv.org/abs/2411.16718)) temporal-logic specs over propositions from the tracker, e.g. `on_ground U airborne` and `on_road U (turn ∧ F on_field)`.
- **Realism:** at n=3, only per-video scores make sense. Compare with content-debiased FVD ([2404.12391](https://arxiv.org/abs/2404.12391)) or FVMD ([2407.16124](https://arxiv.org/abs/2407.16124)) later, at distribution scale. Plain FVD ignores temporal corruption.
- **Small objects:** drones are large in v0, but satellite vehicles are still only about 5–15 px. For those, match by center distance or NWD ([2110.13389](https://arxiv.org/abs/2110.13389)), not IoU@0.5. VISO uses an "any overlap" criterion. Report what fraction of objects are detectable at all. A generator that draws objects bigger than requested has made an error.
- **VLM-as-judge:** secondary only. Its reliability is weak on temporal order (Vinoground, [2503.05977](https://arxiv.org/abs/2503.05977)).
- **Downstream use:** "Dreams Are Cheaper Than Flights" ([2610.41030](https://arxiv.org/abs/2610.41030)) adds world-model drone video to real data and gets +4.3% mAP. An expert rejected about 87.5% of the generated clips, so curation matters.

## 7. Drone and satellite domains, simulators, and pairings

**Domain work (all still images or first-person views; nothing sim-conditioned):**
- Satellite: DiffusionSat ([2312.03606](https://arxiv.org/abs/2312.03606)), CRS-Diff ([2403.11614](https://arxiv.org/abs/2403.11614)), GeoSynth ([2404.06637](https://arxiv.org/abs/2404.06637); OpenStreetMap → satellite tile, works in diffusers), Text2Earth ([2501.00895](https://arxiv.org/abs/2501.00895)), MetaEarth.
- Aerial first-person: AirScape ([2507.08885](https://arxiv.org/abs/2507.08885)), Aero-World, FlightDiffusion ([2509.14082](https://arxiv.org/abs/2509.14082)).
- Synthetic drone targets: SynDroneVision ([2411.05633](https://arxiv.org/abs/2411.05633)), SimD3 ([2601.14742](https://arxiv.org/abs/2601.14742)), SynthSwarm.
- RSVideo ([2608.02039](https://arxiv.org/abs/2608.02039)) is a UAV and satellite video understanding benchmark. It could supply real reference clips.

**Simulators:**

| Tool | Exports | Fit for us |
|---|---|---|
| **BlenderProc** ([repo](https://github.com/DLR-RM/BlenderProc)) | RGB, depth, normals, segmentation, boxes, poses; headless Cycles on CUDA | **Best for v0.** Runs on our A100s; orthographic overhead camera; exact trajectories |
| **CARLA-Air** ([2603.28032](https://arxiv.org/abs/2603.28032), [repo](https://github.com/louiszengCN/CarlaAir)) | CARLA + AirSim multirotors in one UE4 process; 18 sensor types | Covers all 3 videos in one world; prebuilt binary. A100 off-screen speed *(unverified)*. Non-commercial license |
| CARLA 0.9.16 | RGB, depth, segmentation, flow, boxes; **native Cosmos-Transfer integration** | Overhead convoy possible (precedent: SkyScenes), but driving a vehicle off the road is awkward |
| Cosys-AirSim | RGB, depth, segmentation, poses (UE5) | Real multirotor physics, but building a UE5 environment takes days |
| **Isaac Sim + Pegasus** | CosmosWriter → ready-made Transfer inputs | **Blocked: NVIDIA does not support A100/H100 (no RT cores).** Needs an RTX machine |
| 2D compositing | Exact masks, boxes, tracks | Quick fallback for the satellite view |

**Documented simulator → generator pairings:** CARLA → Cosmos-Transfer1/2.5 (native integration plus a Cookbook recipe; suggested multi-control weights depth 0.4, edge 0.1, seg 0.5, vis 0.1); Isaac Sim → Cosmos-Transfer (warehouse and robotics); LLM → CARLA → Transfer1 ([2507.01264](https://arxiv.org/abs/2507.01264)). **None of them is aerial or overhead.**

**Recommended pairing for each test video:**

| Video | Simulator | Dense arm | Sparse arm | Main risk |
|---|---|---|---|---|
| 1. Drones take off and fly away | BlenderProc: drone meshes on a spline (rest → lift → translate), ground or Infinigen terrain, fixed ground camera, drone at least 30 px. Alternative: CARLA-Air | Cosmos-Transfer2.5 (depth ~0.5, seg ~0.3, edge ~0.2); VACE depth + first frame | Wan-Move/ATI with sim point tracks; 3DTrajMaster with 6DoF poses | The generator turns drones into birds or blur, or adds extra drones. Sweep the control weights |
| 2. Drones moving in the air | Blender camera + sky or landscape; 1–3 drones as the main subject, clearly visible (at least about 30 px), crossing or converging | Transfer2.5 / VACE | Wan-Move/ATI with sim point tracks; 3DTrajMaster with 6DoF poses | The generator merges or swaps drones where their paths cross, or a generated background adds bird-like specks that confuse the detector |
| 3. Satellite convoy, road → right turn onto field | Blender orthographic camera over a real satellite tile or a GeoSynth tile; 4–6 low-poly vehicles on a road spline with a turn-off; 0.3–0.5 m ground sample distance | Transfer2.5 (seg/edge/depth, vis ~0.2–0.3 to keep the tile); VACE with tile as first frame + layout | Wan-Move/ATI with vehicle tracks on the same tile (top-down makes pixel ↔ world trivial) | Little nadir video in the models' training data. Expect added parallax or camera motion; say "static orthographic" in the prompt |

## 8. Proposed v0 experiment arms (for each of the 3 videos)

Use one backbone family wherever possible, so the comparison isolates the conditioning:

| Arm | Input | Model |
|---|---|---|
| A. Text only | prompt | Wan2.2 T2V (and Tora T2V) |
| B. Text + first frame | prompt + sim's first frame (or a real or GeoSynth frame) | Wan2.2 TI2V-5B / Wan2.1 I2V |
| C. Sparse sim state | B + sim point tracks or poses | Wan-Move or ATI (Wan2.1-14B); 3DTrajMaster for drones |
| D. Dense sim render | prompt + sim depth/seg/edge | Wan2.1-VACE, Cosmos-Transfer2.5 |
| E. Sim only (floor) | the raw sim render, with no generator | — |

Score every arm on adherence to the sim (counts, ADE/FDE, event-order specs) **and** per-video realism. The expected result: C and D beat A and B on faithfulness; D ≈ E on faithfulness but looks more real; C is the trade-off point.

## 9. Gaps we could fill

1. **Aerial and overhead sim-conditioned generation:** nothing published for drones or satellite.
2. **A three-way comparison on one backbone** (text vs text + first frame vs sim state vs sim render): only VLIPP comes close, and only on general physics.
3. **An ablation of the sim-to-generator handoff format** on one backbone: RGB vs depth/seg vs flow vs tracks vs boxes.
4. **Tiny objects (later, not v0):** no conditioning method has been tested on objects smaller than a latent cell.
5. **Evals:** most papers use VLM judges or VBench. A ground-truth testbench (count, tracks, temporal-logic event order) is still missing, and this is where NeuS-V and NeuS-E fit.

## 10. Open items

- Check whether Cosmos 3 includes a Transfer or dense-control model before committing to Transfer2.5.
- Confirm that weights are released for PhysCtrl, Ctrl-V and MagicDriveDiT.
- Find out whether CARLA-Air renders off-screen on A100.
- The CoDeR code link was not on its project page at survey time. Check the GitHub link from the arXiv paper.
