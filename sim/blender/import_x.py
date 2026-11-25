"""Build a Blender scene from the physics output X of p2s_sim. Fixed code: nothing in it is written per scene.

  sim/blender/.deps/blender-4.2.1-linux-x64/blender -b --python sim/blender/import_x.py -- <x_dir> <out.blend> \
      [--preview <camera_id> <frame> <out.png>]

What it builds (all in X's world frame, Z up, metres; one Blender frame per X frame):
  - every rigid part (bodies, vehicle chassis and wheels, robot links, walkers) as an empty keyframed from its pose
    stream, parenting meshes of its exact collision primitives;
  - soft bodies and cloth as meshes whose vertices follow the simulation through a point cache (.pc2, next to the
    .blend) and a Mesh Cache modifier;
  - particle matter (fluids, sand, smoke) as point clouds from a point cache, padded to a constant count;
  - every camera, keyframed, with X's field of view, clip range and resolution;
  - traffic signal heads as lamps whose colour follows the signal stream.
Each object carries the custom property "p2s_entity" and a pass index per entity (1, 2, ...), so a segmentation pass
can tell entities apart. Materials are plain clay with a per-material tint; appearance belongs to the video model.
"""

import json
import math
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

TINT = {"steel": (0.45, 0.47, 0.5), "aluminium": (0.7, 0.72, 0.75), "rubber": (0.08, 0.08, 0.08),
        "tyre": (0.08, 0.08, 0.08), "wood": (0.45, 0.32, 0.2), "concrete": (0.6, 0.6, 0.58), "asphalt": (0.2, 0.2, 0.2),
        "glass": (0.85, 0.92, 0.95), "plastic": (0.8, 0.8, 0.8), "ceramic": (0.9, 0.9, 0.88), "flesh": (0.75, 0.6, 0.5),
        "track": (0.15, 0.15, 0.12), "cloth": (0.7, 0.2, 0.2)}
PARTICLE_TINT = {"water": (0.35, 0.55, 0.85), "oil": (0.05, 0.04, 0.02), "honey": (0.8, 0.5, 0.1), "sand": (0.76, 0.66, 0.45),
                 "smoke": (0.4, 0.4, 0.42)}


class X:
    """x.json plus its streams."""

    def __init__(self, directory):
        self.dir = directory
        with open(os.path.join(directory, "x.json")) as f:
            self.meta = json.load(f)
        self.frames = self.meta["clock"]["frames"]

    def stream(self, key):
        s = self.meta["streams"][key]
        dtype = "<i4" if s.get("dtype") == "int32" else "<f4"
        a = np.fromfile(os.path.join(self.dir, s["file"]), dtype=dtype)
        width = s["width"]
        if s.get("static"):
            return a.reshape(-1, width)
        if s["rows"] == "variable":
            counts = np.fromfile(os.path.join(self.dir, s["count_file"]), dtype="<i4")
            return a.reshape(-1, width), counts
        return a.reshape(self.frames, -1, width)


def material(name, color):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return m


def mesh_object(name, verts, faces, mat, entity, index):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    ob["p2s_entity"] = entity
    ob.pass_index = index
    bpy.context.scene.collection.objects.link(ob)
    return ob


def ring(r, z, n):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def lathe(profile, z_bottom, z_top, n=32):
    """Rings around Z closed by poles (the same construction as Shape::surfaceMesh)."""
    verts, faces = [(0, 0, z_bottom)], []
    prev = None
    for r, z in profile:
        first = len(verts)
        verts += ring(r, z, n)
        for i in range(n):
            j = (i + 1) % n
            faces.append((0, first + j, first + i) if prev is None else (prev + i, prev + j, first + j, first + i))
        prev = first
    top = len(verts)
    verts.append((0, 0, z_top))
    faces += [(prev + i, prev + (i + 1) % n, top) for i in range(n)]
    return verts, faces


def shape_geometry(s):
    """Vertices and faces of one p2s shape in its own frame."""
    t = s["type"]
    if t == "box":
        hx, hy, hz = s["half_extents"]
        v = [(x * hx, y * hy, z * hz) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
        return v, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    if t == "cylinder":
        return lathe([(s["radius"], -s["half_length"]), (s["radius"], s["half_length"])], -s["half_length"], s["half_length"])
    if t in ("sphere", "capsule"):
        r, hl = s["radius"], s.get("half_length", 0.0) if t == "capsule" else 0.0
        prof = []
        for i in range(1, 16):
            a = -math.pi / 2 + math.pi * i / 16
            if hl > 0 and a >= 0 and (not prof or prof[-1][1] < 0):
                prof.append((r, -hl))
            prof.append((r * math.cos(a), r * math.sin(a) + (hl if a >= 0 else -hl)))
        return lathe(prof, -r - hl, r + hl)
    return [tuple(v) for v in s["vertices"]], [tuple(f) for f in s["triangles"]]


def place(ob, pos, quat_xyzw):
    ob.location = Vector(pos)
    ob.rotation_mode = "QUATERNION"
    x, y, z, w = quat_xyzw
    ob.rotation_quaternion = Quaternion((w, x, y, z))


def keyframe_poses(ob, poses):
    """poses: [frames, 7] position + quaternion xyzw."""
    ob.rotation_mode = "QUATERNION"
    for f, p in enumerate(poses):
        ob.location = Vector(p[:3])
        ob.rotation_quaternion = Quaternion((p[6], p[3], p[4], p[5]))
        ob.keyframe_insert("location", frame=f + 1)
        ob.keyframe_insert("rotation_quaternion", frame=f + 1)
    if ob.animation_data and ob.animation_data.action:
        for fc in ob.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"


def write_pc2(path, frames_xyz):
    """Point cache: every frame the same number of points (frames_xyz: [frames, points, 3])."""
    frames, n, _ = frames_xyz.shape
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2\0")
        f.write(struct.pack("<iiffi", 1, n, 1.0, 1.0, frames))
        f.write(np.ascontiguousarray(frames_xyz, dtype="<f4").tobytes())


def cached_mesh(name, verts0, faces, frames_xyz, pc2_dir, mat, entity, index):
    ob = mesh_object(name, verts0, faces, mat, entity, index)
    path = os.path.join(pc2_dir, name + ".pc2")
    write_pc2(path, frames_xyz)
    mod = ob.modifiers.new("p2s_cache", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = path
    mod.frame_start = 1
    mod.forward_axis = "POS_Y"
    mod.up_axis = "POS_Z"
    return ob


def build_parts(x, e, index):
    for part in e.get("parts", []):
        poses = x.stream(part["stream"])[:, part["row"], :]
        empty = bpy.data.objects.new(e["id"] + "." + part["name"], None)
        empty["p2s_entity"] = e["id"]
        bpy.context.scene.collection.objects.link(empty)
        keyframe_poses(empty, poses)
        for k, s in enumerate(part["shapes"]):
            verts, faces = shape_geometry(s)
            mat = material("p2s_" + s.get("material", "default"), TINT.get(s.get("material"), (0.7, 0.7, 0.7)))
            ob = mesh_object(f"{e['id']}.{part['name']}.{k}", verts, faces, mat, e["id"], index)
            place(ob, s["pos"], s["quat_xyzw"])
            ob.parent = empty


def build_deformable(x, e, index, pc2_dir):
    st = e["streams"]
    verts = x.stream(st["verts"])  # [frames, n, 3]
    tris = x.stream(st["triangles"])
    mat = material("p2s_" + e["kind"], TINT["cloth"] if e["kind"] == "cloth" else TINT["steel"])
    ob = cached_mesh(e["id"], verts[0], tris, verts, pc2_dir, mat, e["id"], index)
    if e["kind"] == "cloth":
        ob.modifiers.new("thickness", "SOLIDIFY").thickness = 0.004


def build_particles(x, e, index, pc2_dir):
    pos, counts = x.stream(e["streams"]["pos"])
    n = max(1, int(counts.max()))
    frames = np.full((len(counts), n, 3), -1000.0, dtype=np.float32)  # unused slots wait far below the ground
    off = 0
    for f, c in enumerate(counts):
        frames[f, :c] = pos[off:off + c]
        off += c
    mat = material("p2s_particles_" + e["preset"], PARTICLE_TINT.get(e["preset"], (0.5, 0.5, 0.5)))
    ob = cached_mesh(e["id"], frames[0], [], frames, pc2_dir, mat, e["id"], index)
    nodes = ob.modifiers.new("points", "NODES")
    tree = bpy.data.node_groups.new(e["id"] + "_points", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    to_points = tree.nodes.new("GeometryNodeMeshToPoints")
    to_points.inputs["Radius"].default_value = e.get("particle_radius_m", 0.005) * (2.0 if e["preset"] == "smoke" else 1.2)
    set_mat = tree.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    tree.links.new(gin.outputs[0], to_points.inputs["Mesh"])
    tree.links.new(to_points.outputs[0], set_mat.inputs["Geometry"])
    tree.links.new(set_mat.outputs[0], gout.inputs[0])
    nodes.node_group = tree


def build_signal(x, e, index):
    states = x.stream(e["streams"]["state"])[:, 0, 0]
    colors = {0: (1, 0.05, 0.02), 1: (1, 0.6, 0.02), 2: (0.1, 1, 0.2)}
    for k, head in enumerate(e["heads"]):
        mat = bpy.data.materials.new(f"{e['id']}_{k}_lamp")
        mat.use_nodes = True
        emit = mat.node_tree.nodes["Principled BSDF"]
        verts, faces = shape_geometry({"type": "sphere", "radius": 0.18})
        ob = mesh_object(f"{e['id']}.head{k}", verts, faces, mat, e["id"], index)
        place(ob, head["pos"], head["quat_xyzw"])
        for f, s in enumerate(states):
            emit.inputs["Emission Color"].default_value = (*colors[int(s)], 1)
            emit.inputs["Emission Strength"].default_value = 8.0
            emit.inputs["Emission Color"].keyframe_insert("default_value", frame=f + 1)


def build_cameras(x):
    scene = bpy.context.scene
    for c in x.meta["cameras"]:
        data = bpy.data.cameras.new(c["id"])
        data.sensor_fit = "HORIZONTAL"
        data.angle = math.radians(c["fov_deg"])
        data.clip_start, data.clip_end = c["clip_m"]
        ob = bpy.data.objects.new(c["id"], data)
        ob["p2s_resolution"] = c["resolution"]
        scene.collection.objects.link(ob)
        keyframe_poses(ob, x.stream("camera/" + c["id"])[:, 0, :])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    x_dir, out_blend = argv[0], argv[1]
    preview = argv[argv.index("--preview") + 1:argv.index("--preview") + 4] if "--preview" in argv else None
    x = X(x_dir)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(x.meta["clock"]["fps"]))
    scene.frame_start, scene.frame_end = 1, x.frames
    scene.unit_settings.system = "METRIC"
    world = bpy.data.worlds.new("p2s_world")
    world.color = (0.55, 0.6, 0.65)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun)
    pc2_dir = os.path.splitext(out_blend)[0] + "_cache"
    os.makedirs(pc2_dir, exist_ok=True)

    for index, e in enumerate(x.meta["entities"], start=1):
        kind = e["kind"]
        if kind in ("soft_body", "cloth"):
            build_deformable(x, e, index, pc2_dir)
        elif kind == "particles":
            build_particles(x, e, index, pc2_dir)
        elif kind == "traffic_signal":
            build_signal(x, e, index)
        elif "parts" in e:
            build_parts(x, e, index)
    build_cameras(x)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
    print(f"[import_x] {len(x.meta['entities'])} entities, {len(x.meta['cameras'])} cameras -> {out_blend}")

    if preview:
        cam_id, frame, png = preview[0], int(preview[1]), preview[2]
        cam = bpy.data.objects[cam_id]
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = cam["p2s_resolution"]
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.frame_set(frame + 1)
        scene.render.filepath = os.path.abspath(png)
        bpy.ops.render.render(write_still=True)
        print(f"[import_x] preview {cam_id} frame {frame} -> {png}")


if __name__ == "__main__":
    main()
