import blenderproc as bproc  # must be the first line (BlenderProc checks)

"""Render a scene spec to the raw half of a scene bundle (plan §4). Runs inside Blender:

  blenderproc run --blender-install-path $S2R_ROOT/blender \
      legacy/sim/render_scene.py legacy/configs/scenes/<scene>.yaml <bundle_dir> [preview frame list, e.g. 0,76,152]

Writes to <bundle_dir>: rgb.npy uint8 [T,H,W,3], depth.npy float32 [T,H,W] (planar z, metres; sky
is inf), instance.npy uint8 [T,H,W] (0 = background, else the spec's instance_id), timeline.json
(per-frame object poses, camera K and cam2world, events, local track points) and meta.json.
legacy/sim/export.py turns these into the rest of the bundle (mp4/png, tracks.npz, boxes.json, overlay);
legacy/sim/build_bundle.sh runs both.
"""

import hashlib
import json
import os
import sys
import time

import bpy
import numpy as np
import yaml
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assets  # noqa: E402
import motion  # noqa: E402
import physics  # noqa: E402


def main(spec_path, out, preview=None):
    t0 = time.time()
    spec_text = open(spec_path).read()
    spec = yaml.safe_load(spec_text)
    os.makedirs(out, exist_ok=True)
    W, H = spec["resolution"]
    T, fps = spec["frames"], spec["fps"]
    assert T % 4 == 1, f"frame count must be 1 + 4n, got {T}"
    assert abs(T / fps - spec["duration_s"]) < 1e-6
    times = np.arange(T) / fps
    if preview:  # quick look: render only these frame indices (bundle is not valid for export)
        times = times[preview]
        T = len(times)

    bproc.init()
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.view_settings.view_transform = "Standard"

    bg, cam = spec["background"], spec["camera"]
    assert cam["motion"] == "static"
    overhead = bg["type"] == "aerial_image"
    if overhead:
        # Aerial photo as the ground: image centre at the origin, +X east, +Y north. Pixels -> metres.
        mpp = bg["metres_per_pixel"]
        image = bg["image"] if os.path.isabs(bg["image"]) else os.path.join(os.environ["S2R_DATA"], bg["image"])
        ground, ground_size, (wp, hp) = assets.build_image_ground(image, mpp)
        to_world = lambda px: (np.asarray(px, float) - [wp / 2, hp / 2]) * [mpp, -mpp]
        assets.build_flat_world(bg["light"]["ambient"])
        light = bg["light"]
        ground_obs = [ground]
    else:
        assets.build_world(bg)
        light = {**bg["sky"], "sun_strength": 3.0}
        ground_obs = assets.build_ground(bg, seed=spec["seed"])
    for ob in ground_obs:
        bproc.types.MeshObject(ob).set_cp("category_id", 0)

    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(light["sun_strength"])
    el, az = np.deg2rad(light["sun_elevation_deg"]), np.deg2rad(light["sun_rotation_deg"])
    sun_dir = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])  # toward the sun
    sun.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-sun_dir))

    # Camera pose. overhead: straight down from altitude_m above centre_px, covering view_width_m of
    # ground; yaw_deg turns it about the vertical (0 = north at the top of the frame, -90 = east at the top).
    bproc.camera.set_resolution(W, H)
    if cam["type"] == "overhead":
        cx, cy = to_world(cam["centre_px"])
        pos = np.array([cx, cy, cam["altitude_m"]])
        yaw = np.deg2rad(cam.get("yaw_deg", 0.0))
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        fov = 2 * np.arctan(cam["view_width_m"] / 2 / cam["altitude_m"])
        bproc.camera.set_intrinsics_from_blender_params(fov, W, H, clip_end=cam["altitude_m"] * 3, lens_unit="FOV")
    else:
        pos = np.asarray(cam["position"], float)
        rot = bproc.camera.rotation_from_forward_vec(np.asarray(cam["look_at"], float) - pos)
        bproc.camera.set_intrinsics_from_blender_params(np.deg2rad(cam["fov_deg"]), W, H, lens_unit="FOV")
    cam2world = bproc.math.build_transformation_mat(pos, rot)
    for i in range(T):
        bproc.camera.add_camera_pose(cam2world, frame=i)
    K = np.asarray(bproc.camera.get_intrinsics_as_K_matrix())

    road = motion.road_path(to_world(spec["road"]["waypoints_px"])) if "road" in spec else None

    # Static occluders: part of the background (instance 0), but in the depth and in front of the objects.
    # Positions along the road (at_m, from_m, to_m) are arc lengths from the first waypoint.
    for k, oc in enumerate(spec.get("occluders", [])):
        if oc["type"] == "tree":
            ob = assets.build_tree(f"occluder_{k}", oc["position"], oc["trunk_height_m"], oc["crown_radius_m"])
        elif oc["type"] == "bridge":
            xy, tan = motion.path_point(road, oc["at_m"])
            ob = assets.build_bridge(f"occluder_{k}", xy, np.arctan2(tan[1], tan[0]), oc["width_m"], oc["length_m"],
                                     oc["height_m"])
        elif oc["type"] == "canopy":
            xy, tan = motion.path_point(road, np.arange(oc["from_m"], oc["to_m"] + 1e-6, 1.0))
            side = np.stack([-tan[:, 1], tan[:, 0]], -1) * oc["width_m"] / 2
            ob = assets.build_canopy_patch(f"occluder_{k}", xy + side, xy - side, oc["height_m"], pos, ground_size,
                                           ground.data.materials[0])
        else:
            raise SystemExit(f"unknown occluder type {oc['type']}")
        bproc.types.MeshObject(ob).set_cp("category_id", 0)

    builders = {"quadcopter": lambda n, a: assets.build_quadcopter(n, a["arm_span_m"], a["body_color"]),
                "tank": lambda n, a: assets.build_tank(n, a["color"]),
                "truck": lambda n, a: assets.build_truck(n, a["color"])}
    obj_specs = motion.scene_objects(spec)
    # Objects with a `motion` block are moved by legacy/sim/physics, all on one clock; the rest follow their scripted trajectory.
    driven = [o for o in obj_specs if "motion" in o]
    phys = physics.simulate(driven, fps, spec["frames"], road=road, seed=spec["seed"]) if driven else None
    pick = preview if preview else slice(None)
    objects = []
    for k, o in enumerate(obj_specs):
        ob = builders[o["asset"]["type"]](o["id"], o["asset"])
        mo = bproc.types.MeshObject(ob)
        mo.set_cp("category_id", int(o["instance_id"]))
        if "motion" in o:
            pos_t, rots = phys["objects"][o["id"]]["pos"][pick], phys["objects"][o["id"]]["rot"][pick]
        elif "keyframes" in o["trajectory"]:
            pos_t, rots = motion.sample_trajectory(o["trajectory"], times, phase=1.7 * k)
        else:
            pos_t, rots = motion.sample_path(road, o["trajectory"], times)
        for i in range(T):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = rots[i], pos_t[i]
            mo.set_local2world_mat(m)
            ob.keyframe_insert("location", frame=i)
            ob.keyframe_insert("rotation_euler", frame=i)
        pts = assets.sample_surface_points(ob, spec["tracks"]["points_per_object"], seed=spec["seed"] + k)
        objects.append((o, ob, pts))
    # constant interpolation: each frame shows exactly the pose computed for it
    for _, ob, _ in objects:
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"

    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"])
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not any(d.type == "CUDA" and d.use for d in prefs.devices):
        raise SystemExit("CUDA_VISIBLE_DEVICES is set but Cycles is not using a CUDA device")
    bproc.renderer.set_max_amount_of_samples(spec["render"]["samples"])
    bproc.renderer.set_denoiser("INTEL" if spec["render"]["denoise"] else None)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})

    # Poses and local track points, read back from Blender so they match what was rendered.
    timeline = {"fps": fps, "frames": T, "resolution": [W, H], "times_s": times.tolist(),
                "camera": {"K": K.tolist(), "cam2world_blender": np.asarray(cam2world).tolist(),
                           "convention": "Blender camera: looks down -Z, +Y up"},
                "objects": [], "events": spec.get("events", []) + (phys["events"] if phys else [])}
    if phys:    # per-frame state from the physics (speed, on_ground, ...), as ground truth
        timeline["physics"] = {i: {k: v[pick].tolist() for k, v in o.items() if k not in ("pos", "rot")}
                               for i, o in phys["objects"].items()}
    for o, ob, pts in objects:
        mats = []
        for i in range(T):
            scene.frame_set(i)
            mats.append(np.asarray(ob.matrix_world).tolist())
        timeline["objects"].append({"id": o["id"], "instance_id": int(o["instance_id"]), "type": o["asset"]["type"],
                                    "local2world": mats, "track_points_local": pts.tolist()})
    scene.frame_set(0)

    t_render = time.time()
    data = bproc.renderer.render()
    t_render = time.time() - t_render

    rgb = np.stack([np.asarray(c)[..., :3] for c in data["colors"]]).astype(np.uint8)
    depth = np.stack([np.asarray(d) for d in data["depth"]]).astype(np.float32)
    depth[depth > 1e4] = np.inf
    inst = np.stack([np.asarray(s) for s in data["category_id_segmaps"]]).astype(np.uint8)
    assert rgb.shape == (T, H, W, 3), rgb.shape
    np.save(os.path.join(out, "rgb.npy"), rgb)
    np.save(os.path.join(out, "depth.npy"), depth)
    np.save(os.path.join(out, "instance.npy"), inst)
    json.dump(timeline, open(os.path.join(out, "timeline.json"), "w"))
    meta = {"spec": os.path.abspath(spec_path), "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(),
            "blender_version": bpy.app.version_string, "render_s": round(t_render, 1),
            "total_s": round(time.time() - t0, 1), "samples": spec["render"]["samples"],
            "cycles_devices": [(d.name, d.type) for d in prefs.devices if d.use]}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("[render_scene]", json.dumps(meta), "instance ids:", np.unique(inst).tolist())


if __name__ == "__main__":
    # optional 3rd arg: comma-separated frame indices for a preview render
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)
