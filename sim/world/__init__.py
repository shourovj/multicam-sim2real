"""Scene spec v2: simulate a world once, render it from any number of cameras (sim/world).

  spec (configs/scenes/<scene>.yaml, spec_version: 2)
    -> simulate.py   numpy only: every controller on one clock; writes world.npz + world.json
                     (per-frame pose of every part of every object, every camera pose, signal states,
                     events, check results). No Blender.
    -> render.py     inside Blender: builds the static world and the object meshes, keyframes each part
                     from world.npz, renders ONE camera -> rgb/depth/instance + timeline.json
    -> export.py     per camera: mp4s, boxes, tracks, visibility (the scene bundle, as in Round 2)
  sim/build_world.sh runs all three for every camera in the spec.

The contract between the halves is world.npz. A new kind of object needs
  1. a motion model in sim/physics/ (what it does; registered in simulate.MODELS),
  2. a kinematics function in kinematics.py (its parts' poses from the model's state),
  3. a mesh builder in blender/ (what each part looks like, in the part's own frame).
A new camera kind needs one function in cameras.py. A new world type needs a layout (layout.py)
and a static builder (blender/). Nothing else changes.

Conventions: world frame Z up, metres, +X east, +Y north. Object root frames: +X forward, +Y left,
+Z up, origin on the ground (vehicles, people) or at the base (robots). Camera poses are stored in
Blender's camera convention (looks down -Z, +Y up), as cam2world 4x4.
"""
