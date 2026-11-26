"""Blender-side builders for spec v2 worlds (run inside Blender via BlenderProc).

  materials.py  PBR materials from Poly Haven texture folders, plus procedural ones (paint, glass, cloth, lamps)
  meshes.py     small mesh helpers (boxes, ellipsoids, extruded profiles) that return plain bpy objects
  street.py     the intersection: road, markings, sidewalks, buildings with windows, signals, props
  car.py        procedural cars: body, cabin (glass + roof + pillars), four wheels, lights
  human.py      procedural people: one mesh per body part, in the part's own frame (kinematics.human_frame)
  props.py      glTF import of Poly Haven models

Every builder returns {part_name: [objects]} for moving things, so render.py can parent each part's meshes to
one empty and keyframe that empty from world.npz. Mesh groups (e.g. a car's "cabin") are recorded in each
object's "s2r_group" custom property, so a camera can hide one group (a dashcam hides its own car's cabin).
"""
