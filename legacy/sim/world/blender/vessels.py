"""Pitcher, glass and water (inside Blender). Lathe meshes (a profile revolved about Z), origin at the bottom centre.

  pitcher       {radius_m, height_m, wall_m, color}   spout pulled out on the local +X side, lip at (R + 0.008, 0, H)
  glass         {radius_m, height_m, wall_m, base_m}  clear glass
  water_level   {radius_m}                            a unit-height water column; the simulator scales its part in Z
  water_stream  per-frame tube meshes along legacy/sim/physics/pour.py's stream, shown only on their own frame
"""

import math

import bmesh
import bpy
import numpy as np

from . import materials as mat
from .meshes import _obj, tag

SEG = 48


def lathe(name, profile_rz, material, spout=None):
    """profile_rz: [(r, z), ...] along the wall from the outer bottom, up and over the rim, down the inside.
    spout: (height_from, pull_m) pulls rim vertices near angle 0 outward (a pouring lip)."""
    bm = bmesh.new()
    rings = []
    for r, z in profile_rz:
        ring = []
        for k in range(SEG):
            a = 2 * math.pi * k / SEG
            rr = r
            if spout and z >= spout[0]:
                w = max(0.0, math.cos(a)) ** 8 * (z - spout[0]) / max(spout[2] - spout[0], 1e-6)
                rr = r + spout[1] * w
            ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), z)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(SEG):
            j = (k + 1) % SEG
            bm.faces.new([rings[i][k], rings[i][j], rings[i + 1][j], rings[i + 1][k]])
    bm.faces.new(list(reversed(rings[0])))                   # outer bottom cap
    bm.faces.new(rings[-1])                                  # inner bottom cap
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _obj(name, bm, material, smooth=True)


def build_pitcher(name, a):
    R, H, w = a["radius_m"], a["height_m"], a.get("wall_m", 0.004)
    m = mat.plain(f"{name}_ceramic", a.get("color", (0.45, 0.65, 0.82)), roughness=0.18, coat=0.6)
    prof = [(R * 0.92, 0.0), (R, 0.012), (R, H - 0.004), (R - w / 2, H), (R - w, H - 0.004), (R - w, 0.012), (R * 0.92 - w, 0.008)]
    ob = lathe(f"{name}_body", prof, m, spout=(H - 0.035, 0.012, H))
    return {"body": tag([ob], "body")}


def build_glass(name, a):
    R, H, w, b = a["radius_m"], a["height_m"], a.get("wall_m", 0.003), a.get("base_m", 0.008)
    m = mat.glass(f"{name}_glass", color=(0.95, 0.97, 0.97), roughness=0.01, ior=1.5)
    prof = [(R * 0.95, 0.0), (R, 0.004), (R, H - 0.001), (R - w / 2, H), (R - w, H - 0.001), (R - w, b), (0.0, b)]
    return {"body": tag([lathe(f"{name}_body", prof, m)], "body")}


def water_material(name):
    m = mat.glass(name, color=(0.85, 0.93, 1.0), roughness=0.02, ior=1.33)
    return m


def build_water_level(name, a):
    from .meshes import cylinder
    col = cylinder(f"{name}_column", a["radius_m"], 1.0, (0, 0, 0.5), "z", water_material(f"{name}_water"), vertices=SEG)
    return {"body": tag([col], "body")}       # unit height; the simulator scales this part to the water height


def build_stream_frames(name, points, radius, active, frames):
    """One tube mesh per rendered frame where the stream exists; visibility keyframed so each shows on its frame only."""
    m = water_material(f"{name}_water")
    objs = []
    n_seg = 12
    for i, f in enumerate(frames):
        if not active[f]:
            continue
        P, Rr = points[f], radius[f]
        bm = bmesh.new()
        rings = []
        for k in range(len(P)):
            d = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
            d /= np.linalg.norm(d) + 1e-12
            u = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.99 else np.cross(d, [1.0, 0, 0])
            u /= np.linalg.norm(u)
            v = np.cross(d, u)
            rings.append([bm.verts.new(P[k] + Rr[k] * (math.cos(2 * math.pi * s / n_seg) * u + math.sin(2 * math.pi * s / n_seg) * v))
                          for s in range(n_seg)])
        for k in range(len(rings) - 1):
            for s in range(n_seg):
                t = (s + 1) % n_seg
                bm.faces.new([rings[k][s], rings[k][t], rings[k + 1][t], rings[k + 1][s]])
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        ob = _obj(f"{name}_f{f:03d}", bm, m, smooth=True)
        for fr, hidden in ((i - 1, True), (i, False), (i + 1, True)):
            if fr < 0:
                continue
            ob.hide_render = hidden
            ob.keyframe_insert("hide_render", frame=fr)
        if i > 1:
            ob.hide_render = True
            ob.keyframe_insert("hide_render", frame=0)
        objs.append(ob)
    for ob in objs:
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
    return objs
