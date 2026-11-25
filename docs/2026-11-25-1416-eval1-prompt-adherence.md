# Eval1: prompt adherence for the simulation, and the SceneAgent loop around it

Branch `raghav/sim-evals` (from `raghav/physx-sim` at `f156407`). Code is in `sim/physx/spec`, `sim/physx/agent` and
`sim/physx/src/check/{Formula,Adherence,Visibility}.*`.

Eval1 in the [research proposal](2026-11-25-1223-research-proposal.pdf) has two halves:

- **Physics validity**: the physics checks of `p2s_check`.
- **Prompt adherence**: does the simulation X show what the prompt asks for? The proposal plans to extend
  NeuS-V [23] to 3D for this.

This doc covers the second half, and the agent loop that uses both halves as feedback.

## The gap

Until now, prompt adherence was checked only against the scene's own `expect` list, which the SceneAgent writes. So
the report can only confirm what the agent remembered to state. In the Haiku round-3 T-bone run the report said
PASS, yet:

- the sedan had no crumple zone, so its expectation was gone too;
- the oil came out 0.4 m ahead of the sedan's nose, in mid-air;
- "runs the red light" was not measurable, because the sedan had no signal;
- the vehicles kept coasting after the crash (this one was seen only by eye);
- nothing said the sedan is red or the van white.

The proposal's own Figure 3 artefact (tank_truck oil appearing in mid-air beside the wheel) passed the checks for the
same reason.

## The method: NeuS-V with the simulation as ground truth

NeuS-V (Sharan et al., CVPR 2025) works in three steps:

1. An LLM turns the prompt into atomic propositions and temporal-logic formulas (PULS).
2. A VLM scores each proposition per frame, with calibrated confidence.
3. STORM model-checks a video automaton built from those scores.

We keep step 1, done once per prompt by a model that never sees the scene. We replace steps 2 and 3. A simulation
already knows every pose, contact, crush, particle and signal state, and what each camera sees (from geometry,
occlusion and on-screen size). So propositions become **predicates measured exactly in X**. Formulas are evaluated
with robust (quantitative) semantics on the finite trace: each requirement gets a robustness in [-1, 1], is satisfied
above 0, and comes with the measured reason when it fails.

There is no VLM judge in this path, and the same S and X always give the same verdict. Two NeuS-V limitations go
away:

- the VLM's confidence calibration;
- the code's rewriting of `G` into `F` and `U` into `& F`: here `G`, `U` and windows have their true semantics.

The spec also does what NeuS-V says it cannot: it penalises unintended content, such as a prompt object struck
without the prompt saying so.

| part | what it checks | how |
|---|---|---|
| objects | present, count, size against the class (`spec/classes.json`), class and colour (new `appearance` key on S) | bind prompt objects to entities by kind, size, appearance and id words; colours compared in CIELAB families |
| parts | "the sedan's crumpled front" exists | the front region of the bound entity, and the soft bodies mounted there |
| cameras | type, the body a camera rides or is mounted on, height | bind to the scene's cameras |
| requirements | what happens, as formulas | 33 predicates: `hits`, `deformed`, `turned`/`moved`/`slid_sideways` with `since=` anchors, `runs_red`, `leaks_from`, `fills`, `on_ground`, `visible(cam, o, px)` with a minimum on-screen size, ... |
| unprompted contact | a person touched although the prompt does not say so | from the contact table |

The spec format, grammar and predicates are documented in `sim/physx/spec/README.md`.

## Results

### The Haiku T-bone run that passed

The spec was written by Qwen3.8-27B from the prompt alone. Its objects, parts and cameras match a hand-written
reference, and both have 15 requirements.

| | before (scene's own expect list) | with the prompt spec |
|---|---|---|
| status | PASS | FAIL (9 failures) |
| eval1 / physics | - | 0.64 / 0.95 |

What the spec caught:

- no crumple zone on the sedan;
- the oil emitter 0.4 m off the car's surface;
- `runs_red` not measurable;
- no appearance class or colour on any entity;
- the vehicles never stop (implied requirement, a warning).

### The example scenes

- All 11 still PASS without a spec, and all 31 tests pass (3 of them new: formula semantics and the spec linter).
- Against its hand-written spec, `tank_truck` caught the Figure 3 artefact: the oil nozzle was 12 cm off the truck's
  surface.
- Moving the nozzle under the crumple zone exposed a subtler version. The nozzle rides the rigid frame, the front
  crushed 0.45 m back, and so the first oil appeared 0.26 m in front of the crushed nose.
- With the nozzle behind the crush depth, all 13 requirements hold (eval1 0.95, physics 0.95), and the oil pools on
  the road.

### The SceneAgent loop: one open model, a 15-minute budget

Qwen3.8-27B (FP8, MTP speculative decoding, one A100 via vLLM) plays three roles:

- spec writer;
- SceneAgent;
- render judge: 256-px clay renders, about 50 tokens per image.

The simulator and Blender run on a second GPU. Each iteration's report r.md goes back to the agent, together with the
judge's text. After a run, `--reflect` rewrites `agent/lessons.md` from the run's failures, and every later run starts
from those lessons: that is how the prompt improves across runs. The handed-over result is the best iteration, with
its **eval1** and **physics** scores.

T-bone prompt, 900 s budget:

| run | change | iterations (valid X) | handed over | eval1 | physics | requirements | render audit |
|---|---|---|---|---|---|---|---|
| r2 | BF16 + MTP, 53 tok/s | 4 (1) | it3 | 0.73 | 0.56 | 10/15 | 0.50 |
| r3 | + JSON-constrained answers, thinking caps, lessons v1 | 4 (3) | it1 | 0.77 | 0.83 | 9/15 | 1.00 |
| r4 | + FP8 (83 tok/s), lessons v2 | 5 (4) | it4 | 0.91 | 0.75 | 14/15 | 1.00 |
| bus_r1 (new prompt*) | + worse iterations set aside | 6 (4) | it5 | 1.00 | 0.90 | 8/8 | 1.00 |

\* "A blue pickup truck rear-ends a yellow school bus that is stopped at a red light; the pickup's front crumples and
the bus lurches forward a little. Filmed by a traffic camera on a pole at the corner and by a phone held by a
pedestrian on the sidewalk." There is no example scene like it. The spec took 35 s.

After these runs, `visible` was made stricter. It used to accept any unhidden sliver at least 24 px tall; it now also
needs 40% of the side facing the camera unhidden. The renders showed why: the pickup is behind the bus in both views
at the moment of impact. Rescored with the stricter rule:

| result | eval1 | requirements |
|---|---|---|
| bus_r1 | 0.88 | 6/8 |
| r4 | 0.88 | 13/15 |
| r3 | 0.77 | 9/15 |
| tank_truck example | 0.96 | 12/13: the tank hides the truck from the traffic-light camera at the impact |
| Haiku run | 0.64 | 10/15 |

All results and renders are under `/nas/raghav/sim2real/eval1/` (`agent/<run>/{result.json,final,view}`).

Observations:

- **Time is the bottleneck.** One SceneAgent turn is 7-12k tokens, about 2-4 minutes. Each speed fix bought
  iterations: thinking caps (an uncapped first turn thought for 24k tokens and wrote no scene), JSON-constrained
  output (a missing brace cost a whole iteration), and FP8 + MTP (3x the original 27 tok/s).
- **Prompt adherence rises faster than physics.** In r3 eval1 went 0.77 → 0.89 while physics fell 0.83 → 0.59, and
  in r4 one late iteration collapsed to 2/15. Since r4, an iteration that scores below the best is set aside, and the
  agent continues from the best scene with a note on what the failed attempt did.
- **The checker's advice steers the agent.** One wrong hint ("set ccd" on a vehicle, which has no ccd key) cost an
  iteration, and one false positive (particles "touching" their container counted as an unprompted contact) became
  a wrong lesson. Both are fixed: a runaway vehicle is now reported as energy injection.
- **The render audit is advisory and does not choose the result.**
  - At 256 px it cannot see a 10 cm dent or oil dots, so its checklist is limited to solid objects (shape, colour)
    and each camera's view of the event.
  - It can be fooled. Blender's back faces are transparent, so a CCTV camera that geometry says sits inside a
    building still "sees" the crash.

## Limits and next steps

- **Binding is heuristic** (kind, size, appearance, id words). A prompt with two similar objects ("a red car and a
  blue car") relies on the appearance colours to tell them apart.
- **Places are not modelled.** Words like "the corner", "the road" or "the intersection" have no predicates yet, so
  "waiting on the corner" checks only that the person waits.
- **Implied requirements are judgement calls of the spec writer.** They are warnings and can be acknowledged.
- **No spec-writer accuracy number yet.** One prompt was compared against a hand reference. A small labelled set
  (prompts, reference specs, and scenes that do or don't match) would measure the spec writer and the binding.
- **Fast draft mode is not built** (coarser substeps and FEM for early iterations). With a sim at 35-60 s per
  iteration it would buy about one more iteration per budget.
