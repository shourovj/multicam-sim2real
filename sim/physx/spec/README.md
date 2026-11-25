# Prompt specs: checking a simulation against its prompt

The SceneAgent loop (`docs/2026-11-24-1859-sceneagent-prompt.md`) checks two things about a scene: is it physically
valid (the physics checks), and does it do what its own `expect` list says. The `expect` list is written by the
SceneAgent itself, so it can only confirm what the agent remembered to state. In the Haiku T-bone trial the agent
dropped the sedan's crumple zone from S, and with it the expectation that the sedan's front crumples; the report said
PASS.

A **prompt spec** closes that gap. It is written from the prompt alone, once per prompt, by a separate model (`p2s_spec`)
that never sees the scene, and the SceneAgent may not edit it. `p2s_check` then checks every scene iteration against
it. This half of Eval1 asks "is this the scene the prompt describes?"; the physics checks ask "is it physically possible?".

The method follows NeuS-V (Sharan et al., *Neuro-symbolic evaluation of text-to-video models using formal
verification*, CVPR 2025):

1. An LLM turns the prompt into atomic propositions and temporal-logic formulas.
2. Each proposition is scored per frame.
3. Each formula is model-checked over the resulting trace.

NeuS-V scores propositions with a vision-language model on generated video frames, and a VLM's confidence is only an
estimate. A simulation already holds the ground truth: every pose, contact, deformation, particle and signal state, and
what each camera sees from geometry. So here the propositions are **predicates measured exactly in X**. Each formula
is evaluated with quantitative (robust) semantics: a robustness in [-1, 1] that is above 0 when the requirement holds.
That needs no vision model and no confidence calibration, and the same S and X always give the same verdict.

The binding of prompt objects to scene entities (below) also checks object existence, counts, class, size and colour.
NeuS-V notes that it cannot penalise unintended content. Here the spec lists the prompt's objects, so an unprompted
contact between them (a pedestrian struck in a prompt that never says so) is reported too.

## Files

| file | what |
|---|---|
| `classes.json` | object classes: entity kinds, size ranges, names; colour families with sRGB prototypes |
| `p2s_spec.py` | prompt -> spec with an LLM served by vLLM; lints the result with `p2s_check --lint-spec` and has the model repair it |
| `serve_llm.sh` | serves the model (Qwen3.8-27B, one A100) with vLLM |
| `render_audit.py` | optional, after PASS: renders frames and asks the VLM about each prompt object's look (not part of the loop) |
| `examples/*.spec.json` | hand-written specs, also the writer's few-shot examples |
| `../src/check/Formula.*` | formula parser and robust semantics |
| `../src/check/Adherence.*` | binding, the predicates, the report section |

## Use

```bash
source sim/physx/env.sh
CUDA_VISIBLE_DEVICES=6 sim/physx/spec/serve_llm.sh &                  # once; ~3 min to load
$P2S_PY sim/physx/spec/p2s_spec.py "<prompt>" $RUN/spec.json          # once per prompt, before iteration 1
# then the usual loop: p2s_check finds $RUN/spec.json beside the itN folders (or pass --spec <file>)
```

`p2s_check` reports the spec's results as findings like any other check. It fails on:

- a missing object, a missing appearance class or colour, a size far outside the class;
- a camera that does not match the prompt's;
- an unprompted contact with a person;
- a requirement that does not hold.

Implied requirements (below) are warnings. If the spec was written for a different prompt than `meta.prompt`, or is
malformed, the report is `STATUS: ERROR` (`PROMPT SPEC ERROR`). The SceneAgent must then stop and report to a human,
because the scene is not at fault. Without a spec the report notes that only the scene's own `expect` list was used.

## Format

```json
{
 "format": "p2s-spec", "version": 1,
 "prompt": "<the prompt, verbatim>",
 "setting": {"place": "city intersection", "time_of_day": "dusk"},
 "objects": [{"name": "sedan", "class": "sedan", "color": "red", "count": 1, "span": "a red sedan"}],
 "parts": [{"name": "sedan.front", "of": "sedan", "region": "front", "span": "the sedan's crumpled front"}],
 "cameras": [{"name": "dashcam", "type": "attached", "on": "sedan", "span": "the sedan's dashcam"},
             {"name": "cctv", "type": "fixed", "mounted_on": "pole", "min_height_m": 3, "span": "a CCTV camera on a pole"}],
 "requirements": [{"id": "sedan_runs_red", "mode": "action", "text": "the sedan runs the red light",
                   "formula": "F runs_red(sedan)"}],
 "not_simulated": ["dusk lighting"]
}
```

- **objects**: every physical thing the prompt names that the simulation should contain.
  - `name` is lower_snake_case.
  - `class` is a key of `classes.json`. Choose the most specific one, e.g. `sedan` over `car`.
  - `color` is a colour word from `classes.json`, given only when the prompt states a colour.
  - `count` defaults to 1.
  - `span` quotes the prompt.
  - Places that are not bodies ("the corner", "the road") are not objects.
- **parts**: a region of an object the prompt singles out: what crumples, leaks, is hit or breaks.
  - `name` is `<object>.<part>`.
  - `region` is one of `front rear left right side top bottom any`, in the object's own frame (+X forward).
  - A part binds to that region of the object's entity and to the soft bodies mounted there.
- **cameras**: every view the prompt names.
  - `type`: `fixed`, `attached` (rides on an object), `handheld`, `orbit`, `follow` or `any`.
  - `on`: the object an attached or handheld camera rides on.
  - `mounted_on`: the object a fixed camera is mounted on. It must be within 1.5 m of that object.
  - `min_height_m`: for views from above (CCTV, a drone, a satellite).
- **requirements**: one per thing the prompt says happens or holds. Fields:
  - `mode`: `existence`, `appearance`, `spatial`, `action`, `camera` or `consistency`.
  - `text`: the requirement in words.
  - `formula`: see the grammar and predicates below.
  - `implied: true` (optional): a consequence the prompt does not state but any real scene would show, e.g. vehicles
    that crash stop driving. It gives a warning instead of a failure.
- **not_simulated**: what the prompt asks for that the simulation cannot show: lighting, weather, expressions,
  textures, sound. The report lists these as left to the renderer and the video model.

Object existence, count, class, size and colour are checked from `objects` directly; they need no requirement.

## Formulas

```
phi := atom | !phi | phi & phi | phi | phi | phi -> phi | F phi | G phi | F[t0, t1] phi | G[t0, t1] phi
     | phi U phi | seq(phi, phi, ...) | (phi) | true | false
```

- `F`: eventually. `G`: always (until the end of the clip). `F[t0, t1]`: within t0..t1 seconds from now.
- `a U b`: a holds until b happens, and b happens.
- `seq(a, b, c)`: a, then later (or at the same time) b, then c. It is short for `F(a & F(b & F c))`.
- Precedence, tightest first: `! F G`, `U`, `&`, `|`, `->`.
- A requirement holds when its formula's robustness at t = 0 is above 0. The robustness semantics: `!` negates; `&`
  and `|` take the min and max; `F` and `G` take the max and min over the frames ahead; `a U b` takes the max over
  t' of min(b(t'), a before t').
- Each predicate's robustness is its margin in its own units, scaled and clipped to [-1, 1]. For example,
  `speed(sedan, 11, 17)` at 13.9 m/s is +1 and at 10.5 m/s is about -0.4.

## Predicates

Arguments: `o` an object (or a part, where allowed), `c` a camera, `n` a number, `w` a colour word.

| predicate | holds when |
|---|---|
| `exists(o)` | the object is in the scene |
| `moving(o, [min_mps=0.3])`, `stationary(o, [max_mps=0.1])`, `speed(o, lo, hi)` | speed above / below / within (m/s) |
| `touches(a, b)` | in contact (from the first to the last touch of the pair) |
| `hits(a, b)` | the first contact (true for 0.25 s from it) |
| `near(a, b, m)` | surface gap at most m metres |
| `upright(o, [deg=30])`, `tipped(o, [deg=30])` | tilt from vertical below / above |
| `airborne(o, [m=0.05])` | its lowest point is above the ground |
| `turned(o, deg, since=...)` | heading changed by at least deg since the anchor |
| `moved(o, m, since=...)` | displaced at least m metres since the anchor |
| `slid_sideways(o, m, since=...)` | displaced at least m metres across its heading at the anchor |
| `moving_away(a, b, [mps=0.2])`, `approaching(a, b, [mps=0.2])` | a's speed away from / towards b |
| `deformed(o or part, m)` | permanent crush (90th percentile of its soft bodies' vertices) at least m metres |
| `spins(o, rpm)` | rotation rate at least rpm |
| `crosses_on(o, w)`, `runs_red(o)` | the vehicle passes its stop line while its light shows w (red) |
| `signal_is(o, w)` | the traffic light shows w |
| `emitting(o)` | particles are being added |
| `on_ground(o, [share=0.5])` | that share of the particles lies on the ground |
| `fills(container, particles, [share=0.3])` | that share of the particles is inside the container's bounds (a pour that lands) |
| `leaks_from(o, o or part)` | the particles come from that object or part: an emitter on it with its nozzle on its surface, and the first particles at its surface |
| `in_front_of(a, b)`, `behind(a, b)`, `left_of(a, b)`, `right_of(a, b)` | a's centre beyond b's extent along b's own axes |
| `above(a, b)`, `on(a, b)`, `inside(a, b)` | a's lowest point above b's top / resting on it / a's centre inside b's bounds |
| `visible(c, o, [px=24])` | the camera sees the object, unoccluded and at least px pixels tall |
| `in_frame(c, o)` | the object is inside the camera's frame (occlusion ignored) |

- `since=` takes an atom (the anchor is the first frame it holds), or a number of seconds. The default is t = 0.
- A predicate on an object with `count > 1` must hold for all of its entities.
- `visible`'s minimum size: video models lose objects that are only a few pixels tall. 24 px is about the smallest
  object a 720p video model keeps, so ask for more for the subject of a shot.

## Writing a spec (for the writer model and for people)

- **State only what the prompt says**, plus `implied` consequences that any real version of the scene would show.
  Do not invent positions, speeds or times.
- **Vague words get generous thresholds.**
  - "about 50 km/h" becomes `speed(..., 11, 17)`.
  - "caves in" becomes `deformed(..., 0.05)`.
  - "spins partly around" becomes at least 30 degrees and less than a full turn.
  - "steps back" becomes `moved(..., 0.3, since=...)`.
- **Anchor outcomes to their cause.** Write `seq(hits(a, b), deformed(b.side, 0.05))` or
  `turned(van, 30, since=hits(sedan, van))`. A turn measured from t = 0 also counts the driving.
- **Turning or tilting.** `turned` is a heading change about the vertical: a car spinning, a door swinging. For
  something that tilts or topples (a pitcher pouring, a crate tipping over), use `tipped`.
- **Order.** Use `seq` for "then". Use `U` only for "until" or "while ... before ...", e.g. "waiting on the corner"
  until the crash.
- **Cameras.** Every camera gets at least one `visible` requirement on the main event, e.g.
  `F (hits(a, b) & visible(cam, a))`.
- **Colours and kinds** go in `objects`, not in formulas.
- **Liquids that leak from something** get `leaks_from(liquid, object.part)`, which also catches liquid that appears
  in mid-air beside the object.
