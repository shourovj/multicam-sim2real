"""Load and check a scene spec v2 (pure Python + numpy, runs inside Blender too)."""

import hashlib

import numpy as np
import yaml

SPEC_VERSION = 2
OBJECT_IDS = range(1, 100)          # instance ids of moving objects; 0 = sky, 100+ = static scene parts


class SpecError(ValueError):
    pass


def load(path):
    """Returns (spec dict, sha256 of the file's bytes)."""
    text = open(path, "rb").read()
    spec = yaml.safe_load(text)
    check(spec)
    return spec, hashlib.sha256(text).hexdigest()


def check(spec):
    def need(cond, msg):
        if not cond:
            raise SpecError(f"{spec.get('name', '?')}: {msg}")

    need(spec.get("spec_version") == SPEC_VERSION, f"spec_version must be {SPEC_VERSION}")
    for k in ("name", "rev", "fps", "frames", "duration_s", "resolution", "seed", "world", "objects", "cameras"):
        need(k in spec, f"missing key {k}")
    T, fps = spec["frames"], spec["fps"]
    need(T % 4 == 1, f"frames must be 1 + 4n, got {T}")
    need(abs(T / fps - spec["duration_s"]) < 1e-6, f"duration_s must equal frames / fps = {T / fps}")
    ids = [o["id"] for o in spec["objects"]]
    need(len(set(ids)) == len(ids), "object ids must be unique")
    inst = [o["instance_id"] for o in spec["objects"]]
    need(len(set(inst)) == len(inst) and all(i in OBJECT_IDS for i in inst), "instance_id must be unique, 1..99")
    cams = [c["id"] for c in spec["cameras"]]
    need(len(set(cams)) == len(cams), "camera ids must be unique")
    for c in spec["cameras"]:
        need(c["type"] in ("fixed", "attached", "handheld"), f"camera {c['id']}: unknown type {c['type']}")
        if c["type"] != "fixed":
            need(c.get("parent") in ids, f"camera {c['id']}: parent must be an object id")
    for o in spec["objects"]:
        for h in o.get("hide_from", []):
            need(h in cams, f"{o['id']}: hide_from names unknown camera {h}")


def times(spec):
    return np.arange(spec["frames"]) / spec["fps"]
