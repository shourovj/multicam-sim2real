# intersection_4way: source

## User prompt (frozen)

> A four-way city intersection with traffic lights on a clear afternoon. Two cars drive through north–south on a green light while two cars wait at the red light on the east–west road. Two pedestrians cross the street in front of the waiting cars, and a third stands on the corner filming with a phone. The lights change and the waiting cars drive off. Seen from the dashcam of one waiting car, the bystander's phone, and a CCTV camera on a pole.

Origin: the user's request for this week's multi-camera test (2026-11-18): "4 cars crossing in a 4 road intersection with traffic lights, and there are 3 pedestrians also … rendered from 3 camera angles: dash cam of one car, bystander pedestrian's mobile camera, and a CCTV from above."

## Who wrote the spec

- Author: Claude Code (model `claude-opus-5-5`), in session, 2026-11-18. No API script; the spec was written by hand-in-the-loop and reviewed against the simulator's checks.
- Instruction it worked from: the user prompt above plus docs/2026-11-18-1739-multicam-v1-design.md §5.1 (timeline, colours, camera list), with the constraint that every number must pass the spec's `checks:` in sim/simulate_world.py.
- Tools: sim/world (spec v2), simulated and rendered with sim/build_world.sh.

## Change log

| rev | date | change |
| --- | --- | --- |
| 1 | 2026-11-18 | First version. Tuning before the first full render (no outputs kept from these): CCTV moved from a pole to the NE building's corner, then raised from 9 m to 15 m so the corner signal pole does not block the junction; buildings 17–28 m tall; render rate 30 → 16 fps (113 frames, the rate Cosmos3-Nano takes); dashcam moved to the top of the windscreen (x 0.55 m, z 1.34 m, pitch +4°) so the hood is a strip at the bottom. |
| 2 | 2026-11-18 | Rendered as clay (`render.style: clay`): every surface one matte grey, no textures; only the signal lamps keep their colour. The world, motion and cameras are unchanged from rev 1. Why: with the textured render as the control (condition V), Cosmos3-Nano copied the render instead of making it look real; the simulator video should only carry the structure and the motion, and the appearance should come from the prompt. Every bundle now also has control videos (depth, seg, edge, normals, rgb). |
