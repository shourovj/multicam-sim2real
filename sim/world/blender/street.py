"""The intersection's static world (inside Blender), from the same numbers the simulator used (layout.Intersection).

Returns (static_ids, lamps): static_ids maps each static object name to the instance id it gets in instance.npy
(100+, so a later segmentation control can tell road, sidewalk, buildings and signals apart), and
lamps[signal_id][colour] is the material render.py keyframes on and off.
"""

import math

import bpy
import numpy as np

from . import materials as mat
from . import props as props_mod
from .meshes import box, boxes, cylinder

ID_ROAD, ID_SIDEWALK, ID_MARKINGS, ID_BUILDING0, ID_SIGNAL0, ID_PROP0, ID_BACKDROP = 100, 101, 102, 110, 140, 160, 200
LAMP_ON = 7.0          # emission strength of a lit lamp: bright, but still red / amber / green under AgX
LAMP_RGB = {"red": (1.0, 0.06, 0.03), "yellow": (1.0, 0.55, 0.02), "green": (0.05, 1.0, 0.45)}


def _ids(objs, value, table, label):
    for o in objs:
        o["category_id"] = value
        table[o.name] = {"id": value, "what": label}


def build_intersection(world, lay, world_json, seed=0):
    ids = {}
    road_m = mat.pbr(world["materials"]["road"], scale_m=4.0)
    walk_m = mat.pbr(world["materials"]["sidewalk"], scale_m=2.0)
    paint_w = mat.plain("paint_white", (0.82, 0.82, 0.80), roughness=0.55, noise=0.15, noise_scale=6)
    paint_y = mat.plain("paint_yellow", (0.85, 0.62, 0.08), roughness=0.55, noise=0.15, noise_scale=6)

    # road: one large asphalt slab; the corner blocks sit on it, raised by the kerb height
    E = lay.L + 40
    _ids([box("road", (2 * E, 2 * E, 0.2), (0, 0, -0.1), road_m)], ID_ROAD, ids, "road")
    blocks = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * (lay.half + E) / 2, sy * (lay.half + E) / 2, lay.kerb / 2 - 0.05)
            blocks.append(box(f"block_{sx}{sy}", (E - lay.half, E - lay.half, lay.kerb + 0.1), c, walk_m))
    _ids(blocks, ID_SIDEWALK, ids, "sidewalk")

    # markings: crosswalk stripes, stop lines, double yellow centre lines (2-4 mm above the road)
    stripes, stops, centre = [], [], []
    a, b, h = lay.cw_in, lay.cw_out, lay.half
    n = int(round(2 * h / 1.0))
    for leg, (dx, dy) in {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}.items():
        mid = (a + b) / 2
        for i in range(n):
            off = -h + 0.5 + i * 1.0                      # 0.5 m stripes, 0.5 m gaps, centred on the road
            if dx == 0:
                stripes.append(((0.5, b - a, 0.006), (off, dy * mid, 0.003), 0))
            else:
                stripes.append(((b - a, 0.5, 0.006), (dx * mid, off, 0.003), 0))
        # stop line across the inbound lane (the lane on the right of a car heading toward the centre)
        r = (-dy, dx)                       # right-hand side of a car entering from this leg (heading -d)
        s = lay.stop - 0.2
        if dx == 0:
            stops.append(((lay.lane_w, 0.4, 0.006), (r[0] * lay.lane_w / 2, dy * s, 0.003), 0))
            for k in (-1, 1):
                centre.append(((0.12, lay.L - b, 0.006), (k * 0.12, dy * (b + lay.L) / 2, 0.003), 0))
        else:
            stops.append(((0.4, lay.lane_w, 0.006), (dx * s, r[1] * lay.lane_w / 2, 0.003), 0))
            for k in (-1, 1):
                centre.append(((lay.L - b, 0.12, 0.006), (dx * (b + lay.L) / 2, k * 0.12, 0.003), 0))
    _ids([boxes("crosswalks", stripes, paint_w), boxes("stop_lines", stops, paint_w), boxes("centre_lines", centre, paint_y)],
         ID_MARKINGS, ids, "markings")

    _buildings(world, lay, ids, seed)
    lamps = _signals(world_json, ids)
    _props(world, lay, ids)
    _backdrop(world, lay, ids, seed)
    return ids, lamps


def _buildings(world, lay, ids, seed):
    bw = world.get("buildings", {})
    rng = np.random.default_rng(bw.get("seed", seed))
    start = lay.half + lay.sidewalk + bw.get("setback_m", 4.0)
    walls = bw.get("walls", ["painted_plaster_wall"])
    lo, hi = bw.get("height_m", [12, 24])
    glass = mat.plain("window_glass", (0.04, 0.05, 0.06), roughness=0.04, specular=1.0)
    frame = mat.plain("window_frame", (0.55, 0.55, 0.53), roughness=0.5)
    store = mat.plain("storefront_glass", (0.06, 0.07, 0.07), roughness=0.03, specular=1.0)
    roof = mat.plain("roof", (0.35, 0.34, 0.33), roughness=0.9, noise=0.2, noise_scale=3)
    k = 0
    for sx in (-1, 1):
        for sy in (-1, 1):
            lots = [(start, start, 14.0, 14.0), (start + 15.0, start, 13.0, 14.0), (start, start + 15.0, 14.0, 12.0)]
            for x0, y0, wx, wy in lots[:bw.get("per_corner", 2)]:
                hgt = float(rng.uniform(lo, hi))
                cx, cy = sx * (x0 + wx / 2), sy * (y0 + wy / 2)
                wall = mat.pbr(walls[k % len(walls)], scale_m=3.0, tint=rng.uniform(0.75, 1.0, 3).tolist(), name=f"wall_{k}")
                body = box(f"building_{k}", (wx, wy, hgt), (cx, cy, hgt / 2 + lay.kerb), wall)
                cap = box(f"building_{k}_roof", (wx + 0.3, wy + 0.3, 0.4), (cx, cy, hgt + lay.kerb + 0.2), roof)
                win, frm, shop = [], [], []
                for face in range(4):                       # windows on all four facades
                    ax = face % 2                           # 0: facade normal along x, 1: along y
                    sgn = 1 if face < 2 else -1
                    span = wy if ax == 0 else wx
                    nx = max(1, int((span - 2.0) // 3.0))
                    for i in range(nx):
                        u = -span / 2 + (span - (nx - 1) * 3.0) / 2 + i * 3.0
                        for fl in range(int((hgt - 4.0) // 3.3)):
                            z = lay.kerb + 4.6 + fl * 3.3
                            if ax == 0:
                                p = (cx + sgn * (wx / 2 + 0.03), cy + u, z)
                                win.append(((0.05, 1.3, 1.6), p, 0)); frm.append(((0.04, 1.5, 0.12), (p[0], p[1], z - 0.86), 0))
                            else:
                                p = (cx + u, cy + sgn * (wy / 2 + 0.03), z)
                                win.append(((1.3, 0.05, 1.6), p, 0)); frm.append(((1.5, 0.04, 0.12), (p[0], p[1], z - 0.86), 0))
                        z = lay.kerb + 1.75                  # ground floor: tall storefront glass
                        if ax == 0:
                            shop.append(((0.05, 2.5, 2.7), (cx + sgn * (wx / 2 + 0.03), cy + u, z), 0))
                        else:
                            shop.append(((2.5, 0.05, 2.7), (cx + u, cy + sgn * (wy / 2 + 0.03), z), 0))
                parts = [body, cap, boxes(f"building_{k}_windows", win, glass), boxes(f"building_{k}_frames", frm, frame),
                         boxes(f"building_{k}_store", shop, store)]
                _ids(parts, ID_BUILDING0 + k, ids, f"building {k}")
                k += 1


def _signals(world_json, ids):
    housing = mat.plain("signal_housing", (0.03, 0.03, 0.03), roughness=0.5)
    metal = mat.plain("signal_pole", (0.22, 0.23, 0.22), roughness=0.45, metallic=0.8, noise=0.1, noise_scale=20)
    lamps = {}
    j = 0
    for sig in world_json["signals"]:
        lamps[sig["id"]] = {c: mat.emissive(f"{sig['id']}_{c}", LAMP_RGB[c], 0.0, base=tuple(0.12 * np.array(LAMP_RGB[c])))
                            for c in ("red", "yellow", "green")}
        for hd in sig["heads"]:
            hx, hy, hz = hd["pos"]
            px, py = hd["pole"]
            yaw = hd["yaw"]
            objs = [cylinder(f"sig_pole_{j}", 0.13, hz + 0.9, (px, py, (hz + 0.9) / 2), "z", metal, vertices=20)]
            dx, dy = hx - px, hy - py
            arm_len = math.hypot(dx, dy)
            arm = cylinder(f"sig_arm_{j}", 0.07, arm_len + 0.2, (0, 0, 0), "x", metal, vertices=16)
            arm.location = ((px + hx) / 2, (py + hy) / 2, hz + 0.75)
            arm.rotation_euler = (0, 0, math.atan2(dy, dx))
            objs.append(arm)
            head = bpy.data.objects.new(f"sig_head_{j}", None)            # head built facing +X, then turned to `yaw`
            bpy.context.scene.collection.objects.link(head)
            head.location, head.rotation_euler = (hx, hy, hz), (0, 0, yaw)
            parts = [box(f"sig_box_{j}", (0.30, 0.38, 1.05), (0, 0, 0), housing),
                     box(f"sig_back_{j}", (0.03, 0.62, 1.30), (-0.16, 0, 0), housing),
                     cylinder(f"sig_hang_{j}", 0.03, 0.5, (0, 0, 0.75), "z", metal, vertices=8)]
            for c, z in (("red", 0.33), ("yellow", 0.0), ("green", -0.33)):
                parts.append(cylinder(f"sig_{c}_{j}", 0.115, 0.04, (0.16, 0, z), "x", lamps[sig["id"]][c], vertices=24))
                visor = box(f"sig_visor_{c}_{j}", (0.22, 0.30, 0.03), (0.27, 0, z + 0.14), housing)
                parts.append(visor)
            for p in parts:
                p.parent = head
            _ids(objs + parts, ID_SIGNAL0 + j, ids, f"signal {sig['id']} head {hd['lane']}")
            j += 1
    return lamps


def _props(world, lay, ids):
    for k, p in enumerate(world.get("props", [])):
        pos = list(p["pos"])
        if len(pos) == 2:
            pos.append(float(lay.ground_z(*pos)))
        root, meshes = props_mod.import_model(p["asset"], f"prop_{k}_{p['asset']}")
        root.location, root.rotation_euler = pos, (0, 0, math.radians(p.get("yaw_deg", 0.0)))
        _ids(meshes, ID_PROP0 + k, ids, f"prop {p['asset']}")


def _backdrop(world, lay, ids, seed):
    """A ring of plain tall blocks 75-140 m out, so long views end in a city, not an empty horizon."""
    rng = np.random.default_rng(seed + 7)
    walls = world.get("buildings", {}).get("walls", ["painted_plaster_wall"])
    objs = []
    for k in range(36):
        ang = 2 * np.pi * k / 36 + rng.uniform(-0.05, 0.05)
        rad = rng.uniform(78, 140)
        if min(abs(math.cos(ang)), abs(math.sin(ang))) < 0.12:     # keep the four road corridors open
            rad += 30
        w, d, hgt = rng.uniform(14, 30), rng.uniform(14, 30), rng.uniform(15, 55)
        m = mat.pbr(walls[k % len(walls)], scale_m=5.0, tint=rng.uniform(0.6, 0.9, 3).tolist(), name=f"backdrop_{k}")
        objs.append(box(f"backdrop_{k}", (w, d, hgt), (rad * math.cos(ang), rad * math.sin(ang), hgt / 2), m, rot_z=ang))
    _ids(objs, ID_BACKDROP, ids, "backdrop")
