"""Procedural people (inside Blender): one smooth mesh group per body part, built in the part's own frame as
posed by kinematics.human_frame (limbs hang along the part's -Z from the joint at its origin).

asset: {type: pedestrian, height_m: 1.75, sleeves: long | short,
        colors: {top, bottom, skin, hair, shoes}}      RGB 0..1
"""

import os
import sys

from . import materials as mat
from .meshes import box, cylinder, ellipsoid, tag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world.kinematics import human_dims  # noqa: E402

DEFAULT_COLORS = dict(top=(0.3, 0.3, 0.35), bottom=(0.1, 0.1, 0.12), skin=(0.7, 0.52, 0.4), hair=(0.08, 0.06, 0.04),
                      shoes=(0.06, 0.06, 0.06))


def build_human(name, asset, with_phone=False):
    h = asset.get("height_m", 1.75)
    dm, k = human_dims(h), h / 1.75
    c = {**DEFAULT_COLORS, **{a: tuple(b) for a, b in asset.get("colors", {}).items()}}
    top = mat.plain(f"{name}_top", c["top"], roughness=0.85, noise=0.12, noise_scale=90, bump=0.15)
    bottom = mat.plain(f"{name}_bottom", c["bottom"], roughness=0.8, noise=0.15, noise_scale=120, bump=0.2)
    skin = mat.plain(f"{name}_skin", c["skin"], roughness=0.5, noise=0.05, noise_scale=40)
    hair = mat.plain(f"{name}_hair", c["hair"], roughness=0.6, noise=0.3, noise_scale=200, bump=0.3)
    shoes = mat.plain(f"{name}_shoes", c["shoes"], roughness=0.6)
    arm_mat = top if asset.get("sleeves", "long") == "long" else skin
    P = {}
    P["pelvis"] = [ellipsoid(f"{name}_pelvis", (0.11 * k, 0.165 * k, 0.12 * k), (0, 0, 0.0), bottom)]
    P["torso"] = [ellipsoid(f"{name}_chest", (0.12 * k, 0.185 * k, 0.27 * k), (0, 0, 0.27 * k), top),
                  ellipsoid(f"{name}_shoulders", (0.10 * k, 0.215 * k, 0.085 * k), (0, 0, dm["torso"] - 0.04 * k), top),
                  cylinder(f"{name}_neck", 0.05 * k, dm["neck"] + 0.06 * k, (0, 0, dm["torso"] + dm["neck"] / 2), "z", skin, vertices=16)]
    P["head"] = [ellipsoid(f"{name}_head", (0.095 * k, 0.08 * k, 0.115 * k), (0.01 * k, 0, 0.11 * k), skin),
                 ellipsoid(f"{name}_hair", (0.098 * k, 0.086 * k, 0.085 * k), (-0.012 * k, 0, 0.155 * k), hair),
                 ellipsoid(f"{name}_nose", (0.02 * k, 0.012 * k, 0.025 * k), (0.1 * k, 0, 0.1 * k), skin, segments=10, rings=6)]
    for s in "lr":
        P[f"upper_arm_{s}"] = [ellipsoid(f"{name}_uarm_{s}", (0.052 * k, 0.052 * k, dm["upper_arm"] / 2 + 0.03 * k),
                                         (0, 0, -dm["upper_arm"] / 2), top)]
        P[f"forearm_{s}"] = [ellipsoid(f"{name}_farm_{s}", (0.043 * k, 0.043 * k, dm["forearm"] / 2 + 0.02 * k),
                                       (0, 0, -dm["forearm"] / 2), arm_mat),
                             ellipsoid(f"{name}_hand_{s}", (0.03 * k, 0.045 * k, dm["hand"] / 2),
                                       (0, 0, -dm["forearm"] - dm["hand"] / 2 + 0.01 * k), skin)]
        P[f"thigh_{s}"] = [ellipsoid(f"{name}_thigh_{s}", (0.075 * k, 0.075 * k, dm["thigh"] / 2 + 0.035 * k),
                                     (0, 0, -dm["thigh"] / 2), bottom)]
        P[f"shin_{s}"] = [ellipsoid(f"{name}_shin_{s}", (0.055 * k, 0.055 * k, dm["shin"] / 2 + 0.03 * k),
                                    (0, 0, -dm["shin"] / 2), bottom)]
        P[f"foot_{s}"] = [ellipsoid(f"{name}_foot_{s}", (dm["foot"] / 2, 0.048 * k, dm["ankle"] / 2 + 0.012 * k),
                                    (dm["foot"] / 2 - 0.06 * k, 0, -dm["ankle"] / 2 + 0.005), shoes)]
    if with_phone:
        P["phone"] = [box(f"{name}_phone", (0.009, 0.155, 0.075), (0, 0, 0), mat.plain(f"{name}_phone_m", (0.02, 0.02, 0.025), 0.2, coat=1.0))]
    return {p: tag(objs, p) for p, objs in P.items()}
