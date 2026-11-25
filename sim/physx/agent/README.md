# agent: the SceneAgent loop with Eval1 feedback

One open vision-language model, served once by vLLM on one GPU (`../spec/serve_llm.sh`: Qwen3.8-27B with its MTP
draft head), plays three roles. A second GPU runs the simulator and Blender.

| role | sees | when |
|---|---|---|
| spec writer (`../spec/p2s_spec.py`) | the prompt only | once per prompt, before authoring |
| SceneAgent (`p2s_agent.py`) | manual, closest example scenes, the spec's requirements, lessons; then each report r as text | every iteration |
| render judge (`render_audit.py`) | 256-px clay renders of each camera at three key frames | alongside the next SceneAgent turn |

```
prompt ─> spec.json
   └─> SceneAgent ─> S ─> p2s_sim ─> X ─> p2s_check (physics + prompt spec) ─> r.md ─┐
            ^                          └─> tiny renders ─> render judge ─> text ──────┤
            └──────────────────────────────────────────────────────────────────────┘ until PASS or 15 min
after the run: --reflect rewrites lessons.md from the run's failures (the prompt improves across prompts)
```

```bash
source sim/physx/env.sh
CUDA_VISIBLE_DEVICES=6 sim/physx/spec/serve_llm.sh &          # ~3 min to load; stop it when done
python3 sim/physx/agent/p2s_agent.py "<prompt>" $RUN --gpu 5 --budget-s 900 [--reflect]
```

`$RUN/result.json` names the iteration handed over, with these scores (both in [0, 1]):

- **status**: the checker's verdict on that iteration.
- **eval1**: prompt adherence from the spec. 0.6 x requirements + 0.4 x objects: present, sized, coloured, cameras.
  A failed requirement gets partial credit by its robustness. See `scores.py`.
- **physics**: physics-following, from the physics findings of r. PASS 1, acknowledged warning 0.75, warning 0.5,
  failure 0. No simulation scores 0.
- **render_audit**: the share of checklist items the judge confirmed on the renders. It is reported, but it does
  not steer the choice of iteration: at 256 px it cannot see small things, and those are measured exactly anyway.

The handed-over iteration is the best one: a PASS first, then the highest eval1 + physics. Its files are copied to
`$RUN/final/`.

The budget covers the SceneAgent's turns, the simulations and the checks. The spec is written before the clock
starts. The audit of the last iteration runs after the clock stops: it scores the result but can no longer steer it.

Images never enter the SceneAgent's conversation, only the judge's text about them.
