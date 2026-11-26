import blenderproc as bproc  # must be the first line (BlenderProc checks)

"""Render ONE camera of an accepted prompt-to-sim world (PhysX output X). Runs inside Blender:

  blenderproc run --blender-install-path $S2R_ROOT/blender p2s/2_render/render_x.py <world_dir> <camera_id> <out_dir> \
      [--samples N] [--range a:b]

The scene is built by sim/blender/import_x.py's functions (every shape exactly as PhysX collided it, clay with a tint
per material), so nothing here is written per scene. Writes the same files as legacy/sim/render_world.py, so
p2s/2_render/bundle.py, gen/ and eval/ read a p2s camera like any other: rgb.npy uint8 [T,H,W,3], depth.npy float32
[T,H,W] (planar z, metres; sky inf), instance.npy uint8 [T,H,W] (0 sky, 1-99 scored objects, 100+ scenery; the ids
are world/spec.yaml's), normals.npy, static_ids.json, timeline.json (camera K and cam2world per frame, each object's
part poses and track points), meta.json. --range renders frames a..b-1 only (a preview; not exported).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import bpy
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim", "blender"))
import import_x as ix  # noqa: E402

SIGNAL_NAMES = {0: "red", 1: "yellow", 2: "green"}
WATER_SURFACE_ID = 254          # the flat sheet import_x draws for world.water


def pose_matrices(poses):
    """[T, 7] position + quaternion xyzw -> [T, 4, 4]."""
    p = np.asarray(poses, np.float64)
    x, y, z, w = (p[:, k] for k in (3, 4, 5, 6))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    M = np.zeros((len(p), 4, 4))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    M[:, :3, 3], M[:, 3, 3] = p[:, :3], 1
    return M


def part_triangles(objs):
    """Every triangle of a part's meshes, [n, 3, 3], in the part's frame. The meshes are children of the part's empty;
    matrix_basis is their pose in the part, whatever pose the part is in now."""
    tris = []
    for ob in objs:
        me = ob.data
        me.calc_loop_triangles()
        v = np.array([vv.co[:] for vv in me.vertices])
        if len(v) == 0 or len(me.loop_triangles) == 0:
            continue
        M = np.asarray(ob.matrix_basis)
        v = v @ M[:3, :3].T + M[:3, 3]
        tris.append(v[np.array([t.vertices[:] for t in me.loop_triangles])])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def triangle_areas(t):
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def sample_points(t, n, seed):
    """n surface points, area-weighted over the triangles t (as legacy/sim/render_world.py)."""
    if len(t) == 0 or n == 0:
        return np.zeros((0, 3))
    area = triangle_areas(t)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = t[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])


def share_points(areas, total):
    """Split an object's track points over its parts by surface area, at least one per part that has a surface (a
    car's body gets most of them, each wheel one or two), by largest remainder."""
    areas = np.asarray(areas, float)
    n = np.where(areas > 0, 1, 0)
    left = total - n.sum()
    if left > 0 and areas.sum() > 0:
        want = areas / areas.sum() * left
        n = n + np.floor(want).astype(int)
        for k in np.argsort(-(want - np.floor(want)))[:total - n.sum()]:
            n[k] += 1
    return n.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world_dir"), ap.add_argument("camera"), ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--range", default=None, help="a:b, render frames a..b-1 only (preview)")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    a = ap.parse_args(argv)
    t0 = time.time()
    spec = yaml.safe_load(open(os.path.join(a.world_dir, "spec.yaml")))
    x = ix.X(a.world_dir)
    T, (W, H) = x.frames, spec["resolution"]
    assert T == spec["frames"], "world/spec.yaml does not match x.json"
    first, last = (int(v) for v in a.range.split(":")) if a.range else (0, T)
    frames = list(range(first, last))
    os.makedirs(a.out, exist_ok=True)

    bproc.init()
    scene = bpy.context.scene
    scene.render.fps = spec["fps"]
    scene.view_settings.view_transform = "AgX"            # as legacy/sim/render_world.py
    bproc.renderer.set_world_background([0.55, 0.6, 0.65], strength=1.0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy, sun.data.angle = 3.0, math.radians(1.5)
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun)

    # the scene, by import_x: one empty per rigid part keyframed from its pose stream (X frame f is Blender frame f + 1)
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    entities = {e["id"]: e for e in x.meta["entities"]}
    for index, e in enumerate(x.meta["entities"], start=1):
        kind = e["kind"]
        if kind in ("soft_body", "cloth"):
            ix.build_deformable(x, e, index, cache)
        elif kind == "particles":
            ix.build_particles(x, e, index, cache)
        elif kind == "traffic_signal":
            ix.build_signal(x, e, index)
        elif "parts" in e:
            ix.build_parts(x, e, index)
    ix.build_water(x)
    ix.build_cameras(x)
    bpy.context.view_layer.update()

    # instance ids from world/spec.yaml: BlenderProc writes each mesh object's category_id into the segmentation
    ids = {o["id"]: o["instance_id"] for o in spec["objects"] + spec["static"]}
    ids.update({soft: ids[host] for soft, host in spec.get("part_of", {}).items()})
    static_ids = {s["id"]: {"id": s["instance_id"], "what": s["what"]} for s in spec["static"]}
    for ob in scene.objects:
        if ob.type != "MESH":
            continue
        ent = ob.get("p2s_entity")
        if ob.pass_index == 0 and ent == "water":          # build_water's sheet, not an entity
            ob["category_id"] = WATER_SURFACE_ID
            static_ids["water_surface"] = {"id": WATER_SURFACE_ID, "what": "water surface"}
        else:
            assert ent in ids, f"{ob.name}: entity {ent} has no instance id in spec.yaml"
            ob["category_id"] = int(ids[ent])

    # timeline: every scored object's rigid parts, their pose per frame and a few surface points to track
    empties = {ob.name: ob for ob in scene.objects if ob.type == "EMPTY"}
    n_pts = spec.get("tracks", {}).get("points_per_object", 24)
    objects_tl = []
    for k, o in enumerate(spec["objects"]):
        e, parts = entities[o["id"]], {}
        tris = [part_triangles([c for c in empties[f"{e['id']}.{part['name']}"].children if c.type == "MESH"]) for part in e.get("parts", [])]
        share = share_points([triangle_areas(t).sum() for t in tris], n_pts)
        for part, t, n in zip(e.get("parts", []), tris, share):
            pts = sample_points(t, n, 31 * k + len(parts))
            M = pose_matrices(x.stream(part["stream"])[:, part["row"], :])
            parts[part["name"]] = {"local2world": M[frames].tolist(), "track_points_local": pts.tolist()}
        objects_tl.append({"id": o["id"], "instance_id": int(o["instance_id"]), "type": o["asset"]["type"], "hidden": False,
                           "parts": parts})

    # camera: import_x's keyframed camera object, with X's field of view (horizontal) and clip range
    cams = [ob for ob in scene.objects if ob.type == "CAMERA" and "p2s_resolution" in ob.keys() and ob.data.name.split(".")[0] == a.camera]
    assert len(cams) == 1, f"camera {a.camera} not found in X"
    scene.camera = cams[0]
    bproc.camera.set_resolution(W, H)
    cam = next(c for c in x.meta["cameras"] if c["id"] == a.camera)
    f_px = (W / 2) / math.tan(math.radians(cam["fov_deg"]) / 2)
    K = np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1]])
    assert np.allclose(K, np.asarray(bproc.camera.get_intrinsics_as_K_matrix()), atol=0.5), "camera intrinsics differ from Blender's"
    c2w = pose_matrices(x.stream("camera/" + a.camera)[:, 0, :])
    scene.frame_start, scene.frame_end = first + 1, last + 1          # BlenderProc renders frame_start .. frame_end - 1

    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"])
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for d in prefs.devices:                       # GPU only: a CPU device alongside slows a shared GPU job down
        d.use = d.type == "CUDA"
    scene.render.use_persistent_data = True
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not any(d.type == "CUDA" and d.use for d in prefs.devices):
        raise SystemExit("CUDA_VISIBLE_DEVICES is set but Cycles is not using a CUDA device")
    bproc.renderer.set_max_amount_of_samples(a.samples)
    bproc.renderer.set_noise_threshold(0.03)
    bproc.renderer.set_denoiser("INTEL")
    bproc.renderer.set_light_bounces(glossy_bounces=4, transmission_bounces=8, transparent_max_bounces=8, max_bounces=8)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
    bproc.renderer.enable_normals_output()

    t_render = time.time()
    data = bproc.renderer.render()
    t_render = time.time() - t_render
    rgb = np.stack([np.asarray(c)[..., :3] for c in data["colors"]]).astype(np.uint8)
    depth = np.stack([np.asarray(d) for d in data["depth"]]).astype(np.float32)
    depth[depth > 1e4] = np.inf
    inst = np.stack([np.asarray(s) for s in data["category_id_segmaps"]]).astype(np.uint8)
    assert rgb.shape == (len(frames), H, W, 3), rgb.shape
    np.save(os.path.join(a.out, "rgb.npy"), rgb)
    np.save(os.path.join(a.out, "depth.npy"), depth)
    np.save(os.path.join(a.out, "instance.npy"), inst)
    if "normals" in data:
        np.save(os.path.join(a.out, "normals.npy"), np.stack([np.asarray(n)[..., :3] for n in data["normals"]]).astype(np.float16))
    json.dump(static_ids, open(os.path.join(a.out, "static_ids.json"), "w"), indent=1)
    signals = {e["id"]: [SIGNAL_NAMES[int(s)] for s in x.stream(e["streams"]["state"])[frames, 0, 0]]
               for e in x.meta["entities"] if e["kind"] == "traffic_signal"}
    timeline = {"fps": spec["fps"], "frames": len(frames), "frame_indices": frames, "resolution": [W, H],
                "times_s": [f / spec["fps"] for f in frames],
                "camera": {"id": a.camera, "K": K.tolist(), "cam2world_blender": c2w[frames].tolist(),
                           "convention": "Blender camera: looks down -Z, +Y up; one cam2world per frame"},
                "objects": objects_tl, "track_points_frame": "part", "events": x.meta["events"], "signals": signals}
    json.dump(timeline, open(os.path.join(a.out, "timeline.json"), "w"))
    meta = {"source": "p2s", "world": os.path.abspath(a.world_dir), "camera": a.camera, "preview": bool(a.range), "style": "clay, tint per material",
            "x_json_sha256": hashlib.sha256(open(os.path.join(a.world_dir, "x.json"), "rb").read()).hexdigest(),
            "blender_version": bpy.app.version_string, "samples": a.samples, "render_s": round(t_render, 1),
            "total_s": round(time.time() - t0, 1), "cycles_devices": [(d.name, d.type) for d in prefs.devices if d.use]}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("[render_x]", json.dumps(meta), "ids in view:", np.unique(inst).tolist())


main()
