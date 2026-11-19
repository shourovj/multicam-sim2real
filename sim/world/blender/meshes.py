"""Mesh helpers built with bmesh (inside Blender). Each returns a new object linked to the scene."""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def _obj(name, bm, material=None, smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if material is not None:
        ob.data.materials.append(material)
    return ob


def box(name, size, center=(0, 0, 0), material=None, rot_z=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    M = Matrix.Translation(center) @ Matrix.Rotation(rot_z, 4, "Z") @ Matrix.Diagonal((*size, 1))
    bmesh.ops.transform(bm, matrix=M, verts=bm.verts)
    return _obj(name, bm, material)


def boxes(name, specs, material=None):
    """Many boxes in one mesh. specs: [(size, center, rot_z), ...]."""
    bm = bmesh.new()
    for size, center, rz in specs:
        tmp = bmesh.ops.create_cube(bm, size=1.0)
        M = Matrix.Translation(center) @ Matrix.Rotation(rz, 4, "Z") @ Matrix.Diagonal((*size, 1))
        bmesh.ops.transform(bm, matrix=M, verts=tmp["verts"])
    return _obj(name, bm, material)


def ellipsoid(name, radii, center=(0, 0, 0), material=None, segments=24, rings=14):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1)), verts=bm.verts)
    return _obj(name, bm, material, smooth=True)


def cylinder(name, radius, depth, center=(0, 0, 0), axis="z", material=None, vertices=32, smooth=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=vertices, radius1=radius, radius2=radius, depth=depth)
    rot = {"z": Matrix.Identity(4), "x": Matrix.Rotation(math.pi / 2, 4, "Y"), "y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ rot, verts=bm.verts)
    ob = _obj(name, bm, material, smooth=smooth)
    if smooth:  # keep the caps flat
        ob.data.shade_smooth_by_angle(angle=math.radians(40)) if hasattr(ob.data, "shade_smooth_by_angle") else None
    return ob


def prism(name, profile_xz, width, material=None, y0=0.0, top_scale=1.0, top_above=None):
    """Extrude a closed side profile (x, z points, counter-clockwise) along Y, centred on y0, `width` wide.
    Vertices above z = top_above are pulled toward the centre by top_scale (a roof narrower than the body)."""
    bm = bmesh.new()
    rings = []
    for side in (-1, 1):
        ring = []
        for x, z in profile_xz:
            y = side * width / 2
            if top_above is not None and z > top_above:
                y *= top_scale
            ring.append(bm.verts.new((x, y0 + y, z)))
        rings.append(ring)
    n = len(profile_xz)
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[1])
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([rings[0][i], rings[0][j], rings[1][j], rings[1][i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _obj(name, bm, material)


def bevel(ob, width=0.05, segments=3, angle_deg=35):
    mod = ob.modifiers.new("bevel", "BEVEL")
    mod.width, mod.segments = width, segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle_deg)
    mod.harden_normals = False
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def tag(objs, group):
    for o in objs:
        o["s2r_group"] = group
    return objs
