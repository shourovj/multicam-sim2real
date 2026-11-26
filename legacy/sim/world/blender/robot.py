"""A MuJoCo Menagerie robot's visual meshes in Blender, one mesh group per MJCF body, each in the body's own frame
(legacy/sim/world/mjcf.py reads the same file for the kinematics). Materials follow the MJCF colours, as glossy plastic."""

import os

import bpy

from . import materials as mat
from .meshes import tag

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world.mjcf import ROBOTS, Robot  # noqa: E402


def build_robot(name, model):
    path, _, _, _ = ROBOTS[model]
    rob = Robot.load(os.path.join(os.environ["S2R_ROOT"], path))
    cache = {}

    def material(rgba):
        key = tuple(round(float(c), 3) for c in rgba[:3])
        if key not in cache:
            dark = sum(key) < 1.0
            cache[key] = mat.plain(f"{name}_{len(cache)}", key, roughness=0.45 if dark else 0.3, coat=0.0 if dark else 0.4)
        return cache[key]

    parts = {}
    for b in rob.bodies:
        objs = []
        for f, rgba in b.visuals:
            before = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=os.path.join(rob.meshdir, f), forward_axis="Y", up_axis="Z")
            for o in set(bpy.data.objects) - before:
                if o.type != "MESH":
                    continue
                o.data.materials.clear()
                o.data.materials.append(material(rgba))
                for p in o.data.polygons:
                    p.use_smooth = True
                o.name = f"{name}_{b.name}_{os.path.splitext(f)[0]}"
                objs.append(o)
        parts[b.name] = tag(objs, b.name)
    return parts
