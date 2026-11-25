# SceneAgent: iterating on a failed report

You wrote a scene S, ran `p2s_sim` and `p2s_check`, and `r.md` says `STATUS: FAIL`. This page is the standing
instruction for each further iteration. The report itself carries everything specific to this run: the findings
with fixes, the measurement tables, the fact card, the history of earlier iterations, and a generated **Iteration
guide** that explains every check that failed and which scene keys fix it. Read the whole report before editing.

## Each iteration

1. Copy the scene into a new folder `itN` (N one higher than the last), set `meta.iteration` to N.
2. Fix in this order, because earlier problems cause later ones:
   1. `simulation` (p2s_sim rejected S: the message names the JSON path) and `finite` / `physx_errors`;
   2. `initial_overlap` (bodies overlapping at t = 0 explode apart and cause energy, acceleration and penetration
      findings downstream);
   3. other hard findings: `penetration`, `below_ground`, `escaped`, `energy`, `fem_inverted`, `particles_in_solids`;
   4. failed expectations (`expect_*`), physical ones before camera ones;
   5. warnings: fix them, or acknowledge them when the scene is right:
      `"acknowledge": [{"check": "<the bold key>", "reason": "<why this is right here, with the measured number>"}]`.
3. Change only what a finding points at. Keep everything that passed; the history table shows if a change broke
   something that used to pass.
4. Never weaken an expectation just to pass. Change it only when it was wrong about the prompt (a glass that holds
   less than you poured cannot be 80 % full), and say why in `meta.notes`.
5. Never replace physics with keyframes to force an outcome. If an outcome does not emerge, change initial
   conditions, masses, materials, controller targets or triggers.
6. Run `p2s_sim` and `p2s_check` again.

## When to stop

- `PASS`: stop. X is ready for rendering (`sim/blender/import_x.py`).
- `STOP_MAX_ITERATIONS` or `STOP_STUCK`: stop and report to a human: the last report's remaining findings, what
  each iteration changed (one line each), and your best guess at the cause. Do not render.
