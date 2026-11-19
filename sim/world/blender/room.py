"""A room with a table (inside Blender), from layout.Room: floor, four walls, ceiling, a window, soft ceiling
lights, a Poly Haven table scaled to the layout's table, a cabinet, and any props. Static ids 100+ as in street.py."""

import math

import bpy
import numpy as np

from . import materials as mat
from . import props as props_mod
from .meshes import box

ID_FLOOR, ID_WALLS, ID_CEILING, ID_TABLE, ID_FURNITURE, ID_PROP0 = 100, 101, 102, 110, 120, 160


def _ids(objs, value, table, label):
    for o in objs:
        o["category_id"] = value
        table[o.name] = {"id": value, "what": label}


def _fit(meshes, root, size_xy, height, center_xy):
    """Scale and move an imported model so its bounding box is size_xy x height, standing on the floor at center_xy."""
    bpy.context.view_layer.update()
    pts = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in meshes])
    lo, hi = pts.min(0), pts.max(0)
    s = np.array([size_xy[0] / (hi[0] - lo[0]), size_xy[1] / (hi[1] - lo[1]), height / (hi[2] - lo[2])])
    root.scale = tuple(s)
    c = (lo + hi) / 2 * s
    root.location = (center_xy[0] - c[0], center_xy[1] - c[1], -lo[2] * s[2])


def build_room(world, lay, seed=0):
    ids = {}
    X, Y, Z = lay.size
    floor = mat.pbr(world["materials"]["floor"], scale_m=1.2)
    wall = mat.pbr(world["materials"]["walls"], scale_m=2.0, tint=(0.95, 0.94, 0.9))
    ceil = mat.plain("ceiling", (0.85, 0.85, 0.83), roughness=0.9)
    _ids([box("floor", (X, Y, 0.1), (0, 0, -0.05), floor)], ID_FLOOR, ids, "floor")
    walls = [box("wall_n", (X, 0.1, Z), (0, Y / 2 + 0.05, Z / 2), wall), box("wall_s", (X, 0.1, Z), (0, -Y / 2 - 0.05, Z / 2), wall),
             box("wall_e", (0.1, Y, Z), (X / 2 + 0.05, 0, Z / 2), wall), box("wall_w", (0.1, Y, Z), (-X / 2 - 0.05, 0, Z / 2), wall)]
    skirt = mat.plain("skirting", (0.9, 0.9, 0.88), roughness=0.4)
    walls += [box(f"skirt_{k}", sz, c, skirt) for k, (sz, c) in enumerate([((X, 0.02, 0.1), (0, Y / 2 - 0.01, 0.05)), ((X, 0.02, 0.1), (0, -Y / 2 + 0.01, 0.05)),
                                                                           ((0.02, Y, 0.1), (X / 2 - 0.01, 0, 0.05)), ((0.02, Y, 0.1), (-X / 2 + 0.01, 0, 0.05))])]
    _ids(walls, ID_WALLS, ids, "walls")
    _ids([box("ceiling", (X, Y, 0.1), (0, 0, Z + 0.05), ceil)], ID_CEILING, ids, "ceiling")

    # window on the east wall: a bright pane (daylight) in a frame, plus a sun-coloured area light just inside it
    lt = world.get("lights", {})
    if lt.get("window", True):
        sky = mat.emissive("window_sky", (0.75, 0.85, 1.0), strength=4.0, base=(0.8, 0.88, 1.0))
        frame = mat.plain("window_frame_r", (0.92, 0.92, 0.9), roughness=0.4)
        _ids([box("window_pane", (0.02, 1.6, 1.2), (X / 2 - 0.005, -0.6, 1.55), sky),
              box("window_frame_t", (0.06, 1.7, 0.06), (X / 2 - 0.03, -0.6, 2.18), frame),
              box("window_frame_b", (0.08, 1.7, 0.06), (X / 2 - 0.03, -0.6, 0.92), frame),
              box("window_frame_m", (0.06, 0.05, 1.2), (X / 2 - 0.03, -0.6, 1.55), frame)], ID_WALLS, ids, "window")
        win = bpy.data.lights.new("window_light", "AREA")
        win.energy, win.size, win.size_y, win.shape = 350.0, 1.6, 1.2, "RECTANGLE"
        win.color = (1.0, 0.96, 0.9)
        ob = bpy.data.objects.new("window_light", win)
        bpy.context.scene.collection.objects.link(ob)
        ob.location, ob.rotation_euler = (X / 2 - 0.1, -0.6, 1.55), (0, -math.pi / 2, 0)
    for k, (x, y) in enumerate([(-0.6, 0.0), (1.0, 0.0)]):          # two big soft ceiling panels
        L = bpy.data.lights.new(f"ceiling_light_{k}", "AREA")
        L.energy, L.size, L.size_y, L.shape = lt.get("ceiling_w", 220.0), 1.2, 0.6, "RECTANGLE"
        L.color = (1.0, 0.97, 0.93)
        ob = bpy.data.objects.new(f"ceiling_light_{k}", L)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = (x, y, Z - 0.02)
        _ids([box(f"ceiling_panel_{k}", (1.2, 0.6, 0.02), (x, y, Z - 0.005), mat.emissive(f"panel_{k}", (1, 0.97, 0.93), 2.0, (0.9, 0.9, 0.9)))],
             ID_CEILING, ids, "ceiling light")

    t = world["table"]
    root, meshes = props_mod.import_model(t.get("model", "wooden_table_02"), "table")
    _fit(meshes, root, lay.table_size, lay.table_h, lay.table_c)
    _ids(meshes, ID_TABLE, ids, "table")

    # a low cabinet along the north wall and a shelf, so the CCTV and side views have a room, not a void
    wood = mat.plain("cabinet", (0.42, 0.30, 0.2), roughness=0.55, noise=0.25, noise_scale=12, bump=0.05)
    top = mat.plain("counter", (0.88, 0.87, 0.85), roughness=0.3)
    _ids([box("cabinet", (2.2, 0.55, 0.85), (0.2, Y / 2 - 0.3, 0.425), wood), box("counter", (2.25, 0.6, 0.04), (0.2, Y / 2 - 0.3, 0.87), top),
          box("shelf", (1.4, 0.25, 0.03), (0.2, Y / 2 - 0.13, 1.65), wood), box("shelf2", (1.4, 0.25, 0.03), (0.2, Y / 2 - 0.13, 2.0), wood)],
         ID_FURNITURE, ids, "cabinet")
    rng = np.random.default_rng(seed + 3)
    jars = []
    for k in range(7):
        c = rng.uniform(0.2, 0.9, 3)
        jars.append(box(f"shelf_item_{k}", (0.12, 0.12, rng.uniform(0.12, 0.25)), (-0.4 + 0.2 * k, Y / 2 - 0.13, 1.72 + 0.35 * (k % 2)),
                        mat.plain(f"shelf_item_m{k}", c, roughness=0.5)))
    _ids(jars, ID_FURNITURE, ids, "shelf items")

    for k, p in enumerate(world.get("props", [])):
        pos = list(p["pos"])
        if len(pos) == 2:
            pos.append(float(lay.ground_z(*pos)))
        r, ms = props_mod.import_model(p["asset"], f"prop_{k}_{p['asset']}")
        r.location, r.rotation_euler = pos, (0, 0, math.radians(p.get("yaw_deg", 0.0)))
        _ids(ms, ID_PROP0 + k, ids, f"prop {p['asset']}")
    return ids
