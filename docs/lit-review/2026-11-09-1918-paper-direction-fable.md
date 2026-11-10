# Paper Direction Review: Fable 5.2

**Source:** Claude Fable 5.2 (`claude-fable-5-2`), high effort, run as a fresh subagent on Nov 9, 2026. The prompt asked it to act as a senior research advisor and give a candid, high-level opinion on whether this project can become a top-tier paper (CVPR, ECCV, ICCV, NeurIPS, ICLR, ICML), its direction, and where it can go after this week's results. **Input:** it read the [v0 design doc](../2026-11-09-1416-design-doc-v0.md), the [project understanding doc](../2026-11-09-1626-project-understanding.md) (including the revised §7 plan), and the [lit-review synthesis](2026-11-09-1750-lit-review-synthesis.md). It was also told about NeuS-V/NeuS-E, the compute, the AFRL context, and the surveys' conclusions. It could use web search.

**Caveats:**
- Fable verified only a few of the Sept–Nov 2026 papers itself.
- PROWBench and WBench, which it names as possible competing benchmarks, come from secondary listings and have not been checked.
- The venue dates and acceptance odds are its own estimates.
- Its suggested size axis (64/16/8 px drones) partly reverses the lead's decision to keep drones clearly visible in v0. That decision is still open.

The report below is unedited except for this header.

---

# Advisor opinion: Sim2Real-VideoGen after the scoping week

I read the three docs and skimmed the surveys. I could not independently verify the Sept–Nov 2026 papers beyond three quick searches, so I rely on the synthesis for those.

## 1. Verdict

**There is no top-tier paper in the current framing, but there is one nearby.** An "adherence audit on three scenes with a claimed-vs-observed table" is a workshop paper or an AFRL report. It was written to survive the circularity critique, not to claim anything.

Where I disagree with the current framing:

- **"Adherence audit" under-claims.** The reason to run sim-to-real is that the simulator's labels (counts, tracks, IDs, event order) come free. The useful question is whether those labels are still true after the generator repaints the scene. It is not circular if measured independently on the output, and it is not trivially 100%.
- **Pixel-level adherence is already measured.** PAI-Bench-C and TransferBench score edge F1, depth error and mask IoU against the control. NVIDIA's Cookbook says annotations are preserved; my quick search found no measurement of that. That gap is your "advertised vs. practice" hook.
- **"Claimed vs. observed" should not be the spine.** The claims come from other domains and protocols, so reviewers will read it as an out-of-distribution gotcha. Keep it as an internal memo for the lead.
- **Scoping out small objects removed the science.** With drones at 30 px or more as the main subject, this is one more object category. Object size relative to the latent grid is a domain-general variable, and it is where the AFRL use case lives.
- **Excluding downstream training is right for v0 and wrong for the paper.** Fine-tuning a detector or tracker is not training a video model, and it is the only fully non-circular endpoint.

Candidate theses, ranked:

| Rank | Working title | One-sentence claim |
|---|---|---|
| 1 | **Do Simulator Labels Survive Generative Sim-to-Real?** | Pixel-level control metrics overstate preservation: instance and event labels break at rates that grow with small size, crossings and overhead views, and a verifier built on simulator-derived temporal-logic specs catches the breaks and recovers downstream accuracy on real data. |
| 2 | **Verify, Then Repair: Spec-Guided Simulator-Conditioned Generation** | Specs generated from simulator state let a NeuS-V-style verifier localize violations and trigger targeted regeneration that beats best-of-N sampling at equal compute. |
| 3 | **How Much Must the Generator Be Told?** | On one backbone there is a knee in the faithfulness–realism curve from text, to a written plan, to a first frame, to sparse tracks, to dense maps. |

Thesis 1 is the paper. Thesis 2 is best as its method section, and VeriPhy and State World Models already crowd it as a standalone. Thesis 3 is a section unless the curve is surprising.

**Venue and timeline.** CVPR 2027 is effectively gone. NeurIPS 2027 (main or Datasets & Benchmarks, typically May) is realistic. ICCV 2027 (typically March) is a stretch, only if the downstream result lands by February. Put out a preprint early to stake the claim. My rough odds for a main-track accept: 30–40% for thesis 1 with the downstream result, under 10% without it.

## 2. What this week has to show

| Thesis | Lives if | Dies or redirects if |
|---|---|---|
| 1 | Under dense control (C, D), a real share of clips (roughly 20% or more in some scene) break a label: missing, extra, merged or swapped objects, or a mistimed event. Those clips still score well on edge/depth agreement, and failures vary with a factor. | Label validity is near 100% everywhere, including a stressed scene. Then go to the small-object and sparse regime, or to "does this help downstream at all?". If outputs collapse completely, the project becomes domain adaptation, which needs training. |
| 2 | Failures vary by seed (some seeds pass on the same input) and are localized in time or object. | Failures are systematic across seeds, so resampling cannot fix them, or there are too few failures to repair. |
| 3 | The curve is non-monotonic or has a clear knee, for example tracks (E) get the events right with better realism than dense maps, or the written plan (A+) is no better than text (A). | It is strictly monotonic, or Wan-Move cannot render drones at all. |

The most valuable outcome this week is one figure: the same clips with high pixel-level control agreement and low label validity.

## 3. Path from v0 to a paper

Lead with the benchmark and analysis, and make the verifier the method. That uses what only this team has: specs generated automatically from simulator state, NeuS-V, and the drone and satellite setting. A pure method paper competes head-on with NVIDIA and loses.

- **Scale.** Turn the three scenes into a parametric family of about 200–500 scenes over count, size, crossings, occlusion, viewpoint and background. Add one in-distribution control domain, such as ground-level vehicles.
- **Automated verifier.** Tracker plus temporal-logic specs with a three-valued verdict, validated against human labels on a few hundred clips. Human annotation is the right primary measure this week, but in the paper it has to become the validation set.
- **Generators.** Four to six, including Cosmos 3 Sim-Transfer once it runs on your hardware.
- **Real-data validation.** Train a detector or tracker on raw simulation, on unfiltered generated video, and on verifier-filtered generated video, then test on Anti-UAV and VISO. The lit survey cites a drone paper where an expert rejected about 87.5% of generated clips; replacing that expert with a formal verifier is the result that makes this top-tier.
- **Method.** Verifier-guided filtering and repair, zero-training, with best-of-N as the baseline.
- **Training.** Optional and later: a LoRA for nadir views only if collapse forces it.
- **Human studies.** Only for verifier validation and realism.

## 4. Main risks

- **Scooped by NVIDIA.** They will keep publishing pixel-level adherence on their own domains. They are unlikely to publish event-level label failure rates on out-of-distribution scenes or downstream harm. Treat Cosmos as a subject of evaluation, not a competitor. The faster threat is another benchmark (PROWBench and WBench surfaced in my search), so preprint early.
- **"Circular."** Headline the residual failure under full control, measured independently on the output. Score A, A+ and B only on prompt-level specs. The downstream result ends the argument.
- **"Only three scenes."** A parametric family with factor-wise curves. Never present the three as the dataset.
- **"Niche domain."** Frame it as a stress test of small objects and out-of-distribution viewpoints, with size in latent cells as the axis and the in-distribution control alongside.
- **"You just applied your own metric."** Show that simulator-grounded specs predict downstream utility better than a VLM judge or pixel metrics.
- **Evaluator reliability on small generated objects.** Report verifier precision and recall against humans, and keep the "unknown" verdict.
- **Model churn.** The harness has to be automated so new releases are a rerun.
- **AFRL release approval.** Check now, and use only public datasets.

## 5. Changes to this week's plan

1. **Rename the headline metric to label validity.** Report the fraction of simulator-labelled object-frames and events still correct in the output, plus a clip-level "all specs hold" rate. Put edge F1 and depth error next to it on the same clips.
2. **Add one difficulty axis to scene 2.** Render it at three drone sizes, about 64, 16 and 8 px, where 8–16 px is around one latent cell or token for Wan-family models. Run only C (and D if it is up). This is 6–12 extra clips, and humans can still annotate 8 px with the simulator overlay.
3. **Use more seeds where it matters.** Run 8–10 seeds of C on the stressed scene, annotated pass/fail only. Three seeds cannot estimate a failure rate or seed-dependence, and compute is not the bottleneck.
4. **Log the automated verifier in parallel.** Run SAM 2 or CoTracker with the NeuS-V specs and record agreement with the human audit. It is not primary this week.
5. **Add the static-control ablation.** The synthesis recommends it but §7 leaves it out. It is one condition on one scene.
6. **Keep the cut order, but timebox Cosmos.** If the smoke test is not passing by Tuesday night, drop D for the week.
7. **Demote two items.** "Claimed vs. observed" becomes a memo. Realism against the raw render stays but is light; real footage is the reference that matters later.
8. **Write the lives/dies criteria from section 2 into the doc before Thursday's runs.**

Sources it cited:
- [PAI-Bench: A Comprehensive Benchmark For Physical AI](https://arxiv.org/pdf/2512.01989)
- [NVIDIA Cosmos Cookbook blog post](https://developer.nvidia.com/blog/how-to-scale-data-generation-for-physical-ai-with-the-nvidia-cosmos-cookbook)
- [Cosmos Cookbook Sim2Real recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-x-mobility/inference.html)
- [Interactive World Simulation with Cosmos for Physical AI (secondary listing)](https://www.teahose.com/paper/arXiv%20Physical%20AI/Interactive%20World%20Simulation%20with%20Cosmos%20for%20Physical%20AI)
- [PROWBench listing](https://papers.cool/arxiv/2610.02205)
- [WBench digest](https://hfdaily-paper-digest-site.pages.dev/2026_05/papers/p043-wbench-a-comprehensive-multi-turn-benchmark-for-interactive-video-world/digest)
