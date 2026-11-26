"""Procedural cars (inside Blender). Dimensions come from kinematics.CAR_MODELS so the wheels sit exactly where
the simulator rolls them. Part frames: body = car root (origin on the ground at the centre, +X forward);
each wheel's origin at its hub, spinning about local Y.

Mesh groups in the body part: "shell" (lower body, paint), "cabin" (glass greenhouse, roof, pillars),
"lights", "trim". A dashcam on this car hides "cabin" (spec camera `hide: [<car>/cabin]`).
"""

import math

from . import materials as mat
from .meshes import bevel, box, boxes, cylinder, prism, tag

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world.kinematics import CAR_MODELS  # noqa: E402

SHAPES = {   # side profile shape per model, as fractions: hood/trunk heights, cabin span, roof span
    "sedan":     dict(belt=0.68, nose=0.50, tail=0.66, ws_base=0.20, rw_base=-0.27, roof_front=0.04, roof_back=-0.16),
    "suv":       dict(belt=0.60, nose=0.52, tail=0.60, ws_base=0.25, rw_base=-0.44, roof_front=0.10, roof_back=-0.43),
    "hatchback": dict(belt=0.65, nose=0.48, tail=0.62, ws_base=0.20, rw_base=-0.42, roof_front=0.05, roof_back=-0.36),
}


def _arch_bottom(xs_wheels, r, L, clear):
    """Bottom edge of the side profile from rear to front, rising over each wheel in a half-circle arch."""
    pts = [(-L / 2 + 0.12, clear)]
    for xw in sorted(xs_wheels):
        ra = r + 0.05
        pts.append((xw - ra, clear))
        for k in range(1, 12):
            a = math.pi - k * math.pi / 12
            pts.append((xw + ra * math.cos(a), r + ra * math.sin(a) * 0.95))
        pts.append((xw + ra, clear))
    pts.append((L / 2 - 0.12, clear))
    return pts


def build_car(name, model, color):
    d, sh = CAR_MODELS[model], SHAPES[model]
    L, W, H, r = d["length"], d["width"], d["height"], d["wheel_r"]
    clear = 0.22
    belt = H * sh["belt"]                                     # beltline: top of the doors, base of the windows
    paint = mat.plain(f"{name}_paint", color, roughness=0.28, metallic=0.55, coat=1.0, noise=0.03, noise_scale=8)
    dark = mat.plain(f"{name}_trim", (0.02, 0.02, 0.022), roughness=0.5)
    rubber = mat.plain(f"{name}_rubber", (0.025, 0.025, 0.025), roughness=0.85, noise=0.2, noise_scale=60, bump=0.2)
    alloy = mat.plain(f"{name}_alloy", (0.6, 0.61, 0.63), roughness=0.25, metallic=1.0)
    win = mat.glass(f"{name}_glass", color=(0.75, 0.8, 0.82), roughness=0.02, tint_dark=0.55)
    head = mat.emissive(f"{name}_headlamp", (1.0, 0.97, 0.9), strength=2.0, base=(0.85, 0.85, 0.85))
    tail = mat.emissive(f"{name}_taillamp", (1.0, 0.05, 0.02), strength=1.0, base=(0.35, 0.02, 0.02))

    # lower body: rear bumper -> trunk -> (cabin sits on the belt) -> hood -> nose, bottom with wheel arches
    xw = [-d["wheelbase"] / 2, d["wheelbase"] / 2]
    top = [(L / 2, H * sh["nose"] * 0.72), (L / 2 - 0.08, H * sh["nose"]), (L / 2 - 0.55, belt * 0.97),
           (L * sh["ws_base"], belt), (L * sh["rw_base"], belt), (-L / 2 + 0.35, H * sh["tail"]),
           (-L / 2, H * sh["tail"] * 0.92), (-L / 2, clear + 0.18)]
    bottom = _arch_bottom(xw, r, L, clear)
    # one closed outline: from the nose bottom, up and over to the rear, then along the bottom (arches) to the front
    profile = [(L / 2, clear + 0.15)] + top + bottom
    shell = bevel(prism(f"{name}_shell", profile, W, paint), width=0.06, segments=3)

    # cabin: windscreen base -> roof front -> roof back -> rear window base (glass), roof slab and pillars in paint
    rf, rb = L * sh["roof_front"], L * sh["roof_back"]
    cab = [(L * sh["ws_base"], belt - 0.02), (rf, H - 0.04), (rb, H - 0.04), (L * sh["rw_base"] - 0.05, belt - 0.02)]
    glass = prism(f"{name}_cabin_glass", cab[::-1], W * 0.94, win, top_scale=0.86, top_above=belt + 0.05)
    roof = bevel(box(f"{name}_roof", (rf - rb + 0.06, W * 0.80, 0.05), ((rf + rb) / 2, 0, H - 0.02), paint), width=0.02)
    pillars = []
    for x0, x1, nm in ((L * sh["ws_base"], rf, "a"), (L * sh["rw_base"] - 0.05, rb, "c")):
        for s in (-1, 1):
            xm, zm = (x0 + x1) / 2, (belt + H) / 2
            ln = math.hypot(x1 - x0, H - belt)
            p = box(f"{name}_pillar_{nm}{'lr'[s > 0]}", (ln, 0.07, 0.07), (0, 0, 0), paint)
            p.location = (xm, s * W * 0.43, zm)
            p.rotation_euler = (0, -math.atan2(H - belt, x1 - x0), 0)     # local +X along the pillar
            pillars.append(p)
    bpil = [box(f"{name}_pillar_b{'lr'[s > 0]}", (0.08, 0.06, H - belt), ((rf + rb) / 2 - 0.1, s * W * 0.44, (belt + H) / 2), paint)
            for s in (-1, 1)]

    lights = [box(f"{name}_head{s}", (0.06, 0.32, 0.10), (L / 2 - 0.06, s * (W / 2 - 0.25), H * sh["nose"] * 0.86), head) for s in (-1, 1)]
    lights += [box(f"{name}_tail{s}", (0.05, 0.30, 0.10), (-L / 2 + 0.02, s * (W / 2 - 0.22), H * sh["tail"] * 0.85), tail) for s in (-1, 1)]
    trim = [box(f"{name}_grille", (0.04, W * 0.45, 0.14), (L / 2 - 0.02, 0, H * sh["nose"] * 0.62), dark),
            box(f"{name}_plate_f", (0.02, 0.50, 0.12), (L / 2 + 0.005, 0, clear + 0.18), mat.plain(f"{name}_plate", (0.9, 0.9, 0.88), 0.4)),
            box(f"{name}_plate_r", (0.02, 0.50, 0.12), (-L / 2 - 0.005, 0, H * sh["tail"] * 0.6), mat.plain(f"{name}_plate2", (0.9, 0.9, 0.88), 0.4)),
            boxes(f"{name}_bumpers", [((0.12, W * 0.98, 0.16), (L / 2 - 0.03, 0, clear + 0.1), 0),
                                      ((0.12, W * 0.98, 0.16), (-L / 2 + 0.03, 0, clear + 0.1), 0)], dark),
            box(f"{name}_mirror_l", (0.12, 0.18, 0.10), (L * sh["ws_base"] - 0.15, W / 2 + 0.06, belt + 0.08), paint),
            box(f"{name}_mirror_r", (0.12, 0.18, 0.10), (L * sh["ws_base"] - 0.15, -W / 2 - 0.06, belt + 0.08), paint)]
    parts = {"body": tag([shell], "shell") + tag([glass, roof] + pillars + bpil, "cabin") + tag(lights, "lights") + tag(trim, "trim")}

    # wheels: tyre + rim + five spokes, in the hub frame (spin axis Y)
    for wn in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        side = 1 if wn.endswith("l") else -1
        tyre = cylinder(f"{name}_{wn}_tyre", r, 0.22, (0, 0, 0), "y", rubber, vertices=40)
        rim = cylinder(f"{name}_{wn}_rim", r * 0.62, 0.225, (0, 0, 0), "y", alloy, vertices=32)
        spokes = boxes(f"{name}_{wn}_spokes", [((r * 1.15, 0.03, 0.06), (0, side * 0.115, 0), 0)], alloy)
        sp = [spokes]
        for k in range(1, 5):
            s2 = boxes(f"{name}_{wn}_spoke{k}", [((r * 1.15, 0.03, 0.06), (0, side * 0.115, 0), 0)], alloy)
            s2.rotation_euler = (0, k * 2 * math.pi / 5, 0)
            sp.append(s2)
        parts[wn] = tag([tyre, rim] + sp, "wheel")
    return parts
