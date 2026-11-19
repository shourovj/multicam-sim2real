# robot_pour: source

## User prompt (frozen)

> A white robot arm on a table in a bright lab kitchen pours water from a pitcher into an empty glass until the glass is about two-thirds full, then tilts the pitcher back. Seen from a camera on the robot's wrist, a camera on a tripod beside the table, and a CCTV camera in the corner of the room.

Origin: the user's request for this week's multi-camera test (2026-11-18): "a robot hand doing some task like pouring water into a glass from some other utensil … cameras: on the robot's arm for an egocentric view, a side view and a CCTV view of the room." The user chose the analytic water stream for now (2026-11-18), with other liquid models to plug in later.

## Who wrote the spec

- Author: Claude Code (model `claude-opus-5-5`), in session, 2026-11-18.
- Instruction it worked from: the user prompt above plus docs/2026-11-18-1739-multicam-v1-design.md §5.2, with the constraint that every number must pass the spec's `checks:` (IK error, joint limits, stream lands in the glass, final level 55–80 %).
- The robot is the Franka Panda from MuJoCo Menagerie (Apache-2.0), pinned in setup/clone.sh.

## Change log

| rev | date | change |
| --- | --- | --- |
| 1 | 2026-11-18 | First version. Tuning before rendering: the lip held 7 cm short of the glass centre (was 5.5 cm), so the strongest stream still lands inside; streams under 8 mL/s are not drawn (drips are not modelled). Simulated result: pour 2.25–4.88 s, 0.20 L poured, glass 57 % full, IK error ≤ 0.2 mm. |
| 2 | 2026-11-18 | Rendered as clay (`render.style: clay`): every surface one matte grey, no textures; only the signal lamps keep their colour. The world, motion and cameras are unchanged from rev 1. Why: with the textured render as the control (condition V), Cosmos3-Nano copied the render instead of making it look real; the simulator video should only carry the structure and the motion, and the appearance should come from the prompt. Every bundle now also has control videos (depth, seg, edge, normals, rgb). |
