import blenderproc as bproc  # must be the first line (BlenderProc checks)

"""BlenderProc smoke test: render a cube with RGB, depth and instance segmentation on a CUDA GPU.

  blenderproc run smoke/blender_cube.py <out_dir>
"""

import os
import sys
import time

import numpy as np

out = sys.argv[1]
t0 = time.time()
bproc.init()

cube = bproc.object.create_primitive("CUBE", location=[0, 0, 0.5], scale=[0.5, 0.5, 0.5])
cube.set_cp("category_id", 1)
ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
ground.set_cp("category_id", 2)
for obj, rgb in [(cube, [0.8, 0.2, 0.1, 1]), (ground, [0.4, 0.4, 0.4, 1])]:
    m = bproc.material.create("m")
    m.set_principled_shader_value("Base Color", rgb)
    obj.replace_materials(m)

light = bproc.types.Light()
light.set_type("SUN")
light.set_location([3, -3, 5])
light.set_energy(4)

bproc.camera.set_resolution(832, 480)
cam = bproc.math.build_transformation_mat([3.0, -3.0, 2.0],
                                          bproc.camera.rotation_from_forward_vec([-3.0, 3.0, -1.5]))
bproc.camera.add_camera_pose(cam)

bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"])
bproc.renderer.set_max_amount_of_samples(64)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"], default_values={"category_id": 0})

import bpy  # noqa: E402
prefs = bpy.context.preferences.addons["cycles"].preferences
print("CYCLES compute_device_type:", prefs.compute_device_type,
      "devices:", [(d.name, d.type, d.use) for d in prefs.devices])
print("SCENE device:", bpy.context.scene.cycles.device)
if os.environ.get("CUDA_VISIBLE_DEVICES") and not any(d.type == "CUDA" and d.use for d in prefs.devices):
    raise SystemExit("CUDA_VISIBLE_DEVICES is set but Cycles is not using a CUDA device")

data = bproc.renderer.render()
bproc.writer.write_hdf5(out, data)
depth = np.asarray(data["depth"][0])
seg = np.asarray(data["category_id_segmaps"][0])
print("rgb", np.asarray(data["colors"][0]).shape, "depth", depth.shape, float(depth.min()), float(depth[depth < 1e5].max()),
      "seg ids", np.unique(seg).tolist(), "render_s", round(time.time() - t0, 1))
