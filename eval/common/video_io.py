"""Read a clip as RGB frames, and the cache files the readers leave next to it."""

import json
import os

import cv2
import numpy as np


def read_video(path, size=None):
    """uint8 [T, H, W, 3] RGB. size (W, H) resizes every frame (a model that returns another size than the simulator's)."""
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if size and (f.shape[1], f.shape[0]) != tuple(size):
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        out.append(f)
    assert out, f"no frames in {path}"
    return np.stack(out)


def cache_ok(est_dir, name, stamp):
    """Is the reader output `name` in est_dir still the one made from this video by this reader version?"""
    p = os.path.join(est_dir, f"{name}.json")
    return os.path.exists(p) and json.load(open(p)).get("stamp") == stamp


def cache_mark(est_dir, name, stamp, **info):
    json.dump({"stamp": stamp, **info}, open(os.path.join(est_dir, f"{name}.json"), "w"), indent=1)
