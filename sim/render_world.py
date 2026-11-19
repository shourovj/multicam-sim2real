import blenderproc as bproc  # must be the first line (BlenderProc checks)

"""Render ONE camera of a simulated spec v2 world (sim/world). Runs inside Blender:

  blenderproc run --blender-install-path $S2R_ROOT/blender sim/render_world.py \
      <spec.yaml> <world_dir> <camera_id> <out_dir> [--frames 0,60,120] [--samples N]

Reads world_dir/world.npz + world.json (sim/simulate_world.py). Writes to out_dir: rgb.npy uint8 [T,H,W,3],
depth.npy float32 [T,H,W] (planar z, metres; sky inf), instance.npy uint8 [T,H,W] (0 sky, 1-99 objects,
100+ static parts; static_ids.json says which), timeline.json (camera K and cam2world per frame, each object's
part poses and track points), meta.json. --frames renders a preview of those frame indices only.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assets as v1_assets  # noqa: E402  (Nishita sky with clouds, shared with Round 2)
from world import layout as lay  # noqa: E402
from world.spec import load  # noqa: E402
from world.blender import materials as mat  # noqa: E402
from world.blender.car import build_car  # noqa: E402
from world.blender.human import build_human  # noqa: E402
from world.blender.street import LAMP_ON, build_intersection  # noqa: E402
from world.blender.room import build_room  # noqa: E402
from world.blender.robot import build_robot  # noqa: E402
from world.blender import vessels  # noqa: E402

STATE_NAMES = {0: "red", 1: "yellow", 2: "green"}


def build_object(o):
    a = o["asset"]
    if a["type"] == "car":
        return build_car(o["id"], a["model"], a["color"])
    if a["type"] == "pedestrian":
        return build_human(o["id"], a, with_phone=o.get("motion", {}).get("pose") == "phone")
    if a["type"] == "robot":
        return build_robot(o["id"], a["model"])
    if a["type"] == "pitcher":
        return vessels.build_pitcher(o["id"], a)
    if a["type"] == "glass":
        return vessels.build_glass(o["id"], a)
    if a["type"] == "water_level":
        return vessels.build_water_level(o["id"], a)
    raise SystemExit(f"{o['id']}: no Blender builder for asset type {a['type']}")


def sample_points(objs, n, seed):
    """n surface points, area-weighted over a part's meshes, in the part's frame (meshes are children of the part)."""
    tris = []
    for ob in objs:
        me = ob.data
        me.calc_loop_triangles()
        M = np.asarray(ob.matrix_local)
        v = np.array([vv.co[:] for vv in me.vertices])
        if len(v) == 0:
            continue
        v = v @ M[:3, :3].T + M[:3, 3]
        tris.append(v[np.array([t.vertices[:] for t in me.loop_triangles])])
    if not tris:
        return np.zeros((0, 3))
    t = np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = t[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"), ap.add_argument("world_dir"), ap.add_argument("camera"), ap.add_argument("out")
    ap.add_argument("--frames", default=None, help="comma-separated frame indices for a preview")
    ap.add_argument("--samples", type=int, default=None)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    a = ap.parse_args(argv)
    t0 = time.time()
    spec, sha = load(a.spec)
    wj = json.load(open(os.path.join(a.world_dir, "world.json")))
    wz = np.load(os.path.join(a.world_dir, "world.npz"))
    assert wj["spec_sha256"] == sha, "world.npz was simulated from a different spec; re-run sim/simulate_world.py"
    cam = next(c for c in wj["cameras"] if c["id"] == a.camera)
    W, H = spec["resolution"]
    frames = [int(x) for x in a.frames.split(",")] if a.frames else list(range(spec["frames"]))
    os.makedirs(a.out, exist_ok=True)

    bproc.init()
    scene = bpy.context.scene
    scene.render.fps = spec["fps"]
    scene.view_settings.view_transform = "AgX"            # photographic highlight roll-off (Blender 4.x)
    world_spec = spec["world"]
    layout = lay.build(world_spec)

    # static world
    lamps = {}
    if world_spec["type"] == "intersection":
        sky = world_spec["sky"]
        v1_assets.build_world({"sky": sky})
        static_ids, lamps = build_intersection(world_spec, layout, wj, seed=spec["seed"])
        sun = bproc.types.Light()
        sun.set_type("SUN")
        sun.set_energy(sky.get("sun_strength", 3.5))
        el, az = np.deg2rad(sky["sun_elevation_deg"]), np.deg2rad(sky["sun_rotation_deg"])
        sun_dir = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        sun.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-sun_dir))
        sun.blender_obj.data.angle = np.deg2rad(sky.get("sun_angle_deg", 1.5))   # soft shadow edges
    elif world_spec["type"] == "room":
        bproc.renderer.set_world_background([0.06, 0.06, 0.065], strength=1.0)
        static_ids = build_room(world_spec, layout, seed=spec["seed"])
    else:
        raise SystemExit(f"no static builder for world type {world_spec['type']}")

    # moving objects: one empty per part, keyframed from world.npz; meshes are children of their part
    hide = set(cam.get("hide", []))
    objects_tl = []
    for k, o in enumerate(spec["objects"]):
        if o["asset"]["type"] == "water_stream":            # per-frame geometry, not posed parts
            g = lambda key: wz[f"geometry/{o['id']}/{key}"]
            for ob in vessels.build_stream_frames(o["id"], g("points"), g("radius"), g("active"), frames):
                ob["category_id"] = int(o["instance_id"])
                if a.camera in o.get("hide_from", []):
                    ob.hide_render = True
            objects_tl.append({"id": o["id"], "instance_id": int(o["instance_id"]), "type": "water_stream",
                               "hidden": a.camera in o.get("hide_from", []), "parts": {}})
            continue
        meshes = build_object(o)
        hidden_obj = a.camera in o.get("hide_from", [])
        tl_parts = {}
        for part, objs in meshes.items():
            key = f"parts/{o['id']}/{part}"
            assert key in wz, f"{key} missing from world.npz"
            emp = bpy.data.objects.new(f"{o['id']}__{part}", None)
            scene.collection.objects.link(emp)
            for ob in objs:
                ob.parent = emp
                ob["category_id"] = int(o["instance_id"])
                if hidden_obj or f"{o['id']}/{ob.get('s2r_group', part)}" in hide:
                    ob.hide_render = True
            mats = wz[key]
            if len(objs) == 0:
                continue
            for i, f in enumerate(frames):
                emp.matrix_world = Matrix(mats[f].tolist())
                emp.keyframe_insert("location", frame=i)
                emp.keyframe_insert("rotation_euler", frame=i)
                emp.keyframe_insert("scale", frame=i)              # parts are rigid except a water column
            for fc in emp.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
            vis = [ob for ob in objs if not ob.hide_render]
            pts = sample_points(vis, spec.get("tracks", {}).get("points_per_part", 4), spec["seed"] + 31 * k + len(tl_parts))
            tl_parts[part] = {"local2world": mats[frames].tolist(), "track_points_local": pts.tolist()}
        objects_tl.append({"id": o["id"], "instance_id": int(o["instance_id"]), "type": o["asset"]["type"],
                           "hidden": hidden_obj, "parts": tl_parts})

    # signal lamps: emission on for the current state, per frame
    for sid, by_colour in lamps.items():
        states = wz[f"signals/{sid}"]
        for i, f in enumerate(frames):
            on = STATE_NAMES[int(states[f])]
            for colour, m in by_colour.items():
                mat.keyframe_emission(m, i, LAMP_ON if colour == on else 0.0)
        for m in by_colour.values():
            ad = m.node_tree.animation_data
            if ad and ad.action:
                for fc in ad.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"

    # camera
    c2w = wz[f"cams/{a.camera}/cam2world"]
    bproc.camera.set_resolution(W, H)
    bproc.camera.set_intrinsics_from_blender_params(np.deg2rad(cam["fov_deg"]), W, H, lens_unit="FOV",
                                                    clip_start=cam.get("clip_start_m", 0.1), clip_end=cam.get("clip_end_m", 1000))
    for i, f in enumerate(frames):
        bproc.camera.add_camera_pose(c2w[f], frame=i)
    K = np.asarray(bproc.camera.get_intrinsics_as_K_matrix())

    # render style. textured: the materials as built. clay: one plain matte grey on every surface (the abstract draft
    # the video model re-skins); only signal lamps keep their colour, so the light state is still in the picture.
    style = spec["render"].get("style", "textured")
    if style == "clay":
        clay = mat.plain("clay", (0.72, 0.72, 0.70), roughness=0.85)
        keep = {m for by_colour in lamps.values() for m in by_colour.values()}
        for ob in scene.objects:
            if ob.type != "MESH":
                continue
            mats = ob.data.materials
            if len(mats) == 0:
                mats.append(clay)
            for i, m in enumerate(mats):
                if m not in keep:
                    mats[i] = clay
    elif style != "textured":
        raise SystemExit(f"render.style must be textured or clay, got {style}")

    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"])
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for d in prefs.devices:                       # GPU only: a CPU device alongside slows a shared GPU job down
        d.use = d.type == "CUDA"
    scene.render.use_persistent_data = True       # keep the static scene's BVH between frames
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not any(d.type == "CUDA" and d.use for d in prefs.devices):
        raise SystemExit("CUDA_VISIBLE_DEVICES is set but Cycles is not using a CUDA device")
    samples = a.samples or spec["render"]["samples"]
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.set_noise_threshold(0.03)
    bproc.renderer.set_denoiser("INTEL" if spec["render"]["denoise"] else None)
    bproc.renderer.set_light_bounces(glossy_bounces=4, transmission_bounces=8, transparent_max_bounces=8, max_bounces=8)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
    bproc.renderer.enable_normals_output()                  # surface normals, for the normal and edge controls

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
    timeline = {"fps": spec["fps"], "frames": len(frames), "frame_indices": frames, "resolution": [W, H],
                "times_s": [wj["times_s"][f] for f in frames],
                "camera": {"id": a.camera, "K": K.tolist(), "cam2world_blender": c2w[frames].tolist(),
                           "convention": "Blender camera: looks down -Z, +Y up; one cam2world per frame"},
                "objects": objects_tl, "events": wj["events"],
                "signals": {sid: [STATE_NAMES[int(s)] for s in wz[f"signals/{sid}"][frames]] for sid in lamps}}
    json.dump(timeline, open(os.path.join(a.out, "timeline.json"), "w"))
    meta = {"spec": os.path.abspath(a.spec), "spec_sha256": sha, "camera": a.camera, "preview": bool(a.frames), "style": style,
            "world_npz_sha256": hashlib.sha256(open(os.path.join(a.world_dir, "world.npz"), "rb").read()).hexdigest(),
            "blender_version": bpy.app.version_string, "samples": samples, "render_s": round(t_render, 1),
            "total_s": round(time.time() - t0, 1), "cycles_devices": [(d.name, d.type) for d in prefs.devices if d.use]}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("[render_world]", json.dumps(meta), "ids in view:", np.unique(inst).tolist())


main()
