"""Simulate a spec v2 world once: every object, every camera, every frame (pure numpy).

  $S2R_ROOT/envs/eval/.venv/bin/python sim/simulate_world.py configs/scenes/<scene>.yaml <world_dir>

Writes <world_dir>/world.npz and world.json (see sim/world/__init__.py) and exits non-zero if a check fails.

Order of work:
  1. motion models on one clock (1 ms steps, sampled at frame times): root pose + state per object
  2. kinematics: the pose of every part of every object (sim/world/kinematics.py)
  3. objects attached to a part of another object (a pitcher in a gripper)
  4. derived geometry (a water stream from the pitcher's pose; sim/physics/pour.py)
  5. cameras (sim/world/cameras.py), which may ride on any part
  6. checks (sim/world/checks.py)
"""

import json
import math
import os
import subprocess

import numpy as np

from . import cameras as cams
from . import checks as chk
from . import kinematics as kin
from . import layout as lay
from .signals import Signal
from .spec import load, times as frame_times

from physics.arm import Arm
from physics.lane_vehicle import LaneVehicle
from physics.pedestrian import Pedestrian
from physics.pour import simulate_pour

MAX_DT = 1e-3
LIGHT = {"red": 0, "yellow": 1, "green": 2}


class Static:
    """An object that does not move. pose: {pos: [x, y] or [x, y, z], yaw_deg}; z defaults to the ground."""

    def __init__(self, spec, world, index):
        p = spec.get("pose", {})
        pos = list(p.get("pos", [0, 0]))
        if len(pos) == 2:
            pos.append(float(world.layout.ground_z(*pos)))
        self.pos, self.R = np.asarray(pos, float), kin.Rz(np.deg2rad(p.get("yaw_deg", 0.0)))

    def step(self, t, dt, world):
        pass

    def pose(self):
        return self.pos, self.R

    def state(self):
        return {}


MODELS = {"lane_vehicle": LaneVehicle, "pedestrian": Pedestrian, "arm": Arm, "static": Static}
DERIVED = ("attached", "pour", "level")     # objects computed after the motion models, from other objects


class World:
    def __init__(self, layout, signals, seed):
        self.layout, self.signals, self.seed = layout, signals, seed
        self.bodies, self.events, self.t = {}, [], 0.0

    def event(self, name, **extra):
        self.events.append({"name": name, "t": round(self.t, 3), **extra})


def git_state(repo):
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def simulate(spec_path, out):
    spec, sha = load(spec_path)
    t_frames = frame_times(spec)
    n, fps = spec["frames"], spec["fps"]
    layout = lay.build(spec["world"])
    signals = {s["id"]: Signal(s) for s in spec.get("signals", [])}
    world = World(layout, signals, spec["seed"])

    # 1. motion models
    for k, o in enumerate(spec["objects"]):
        model = o.get("motion", {}).get("model", "static")
        if model in DERIVED:
            continue
        world.bodies[o["id"]] = MODELS[model](o, world, k)
    for b in world.bodies.values():
        if hasattr(b, "link"):
            b.link(world)
    for sid, s in signals.items():
        for t_change, st in s.changes():
            world.events.append({"name": f"{sid}_{st}", "t": round(t_change, 3)})
    sub = math.ceil(1.0 / (fps * MAX_DT))
    dt = 1.0 / (fps * sub)
    roots = {i: np.empty((n, 4, 4)) for i in world.bodies}
    log = {i: {"xy": np.empty((n, 2))} for i in world.bodies}
    for f in range(n):
        for i, b in world.bodies.items():
            p, R = b.pose()
            roots[i][f] = kin.T(p, R)
            log[i]["xy"][f] = p[:2]
            for key, val in b.state().items():
                log[i].setdefault(key, np.empty(n))[f] = val
        if f == n - 1:
            break
        for s in range(sub):
            world.t = (f * sub + s) * dt
            for b in world.bodies.values():
                b.step(world.t, dt, world)
    world.events.sort(key=lambda e: e["t"])

    # 2. parts
    by_id = {o["id"]: o for o in spec["objects"]}
    parts = {i: kin.parts_for(by_id[i], roots[i], log[i]) for i in world.bodies}

    # 3. attached objects: root = parent part x offset (pos, rpy in the parent part's frame)
    for o in spec["objects"]:
        m = o.get("motion", {})
        if m.get("model") != "attached":
            continue
        off = m.get("offset")
        if off is None:                     # held by an arm: use the arm's hold offset, so it is written once
            pm = by_id[m["parent"]].get("motion", {})
            assert pm.get("hold", {}).get("object") == o["id"], f"{o['id']}: attached without offset, and {m['parent']} does not hold it"
            off = pm["hold"]["offset"]
        local = kin.T(off.get("pos", [0, 0, 0]), kin.rpy(*off.get("rpy_deg", [0, 0, 0])))
        root = parts[m["parent"]][m.get("part", "body")] @ local
        roots[o["id"]], log[o["id"]] = root, {"xy": root[:, :2, 3]}
        parts[o["id"]] = kin.parts_for(o, root, log[o["id"]])

    # 4. derived geometry: a water stream from a pitcher's pose (sim/physics/pour.py), and the water column it fills
    geometry, extra_checks = {}, {}
    for o in spec["objects"]:
        m = o.get("motion", {})
        if m.get("model") != "pour":
            continue
        geo, ev, pc = simulate_pour(m, by_id, roots, t_frames)
        world.events = sorted(world.events + ev, key=lambda e: e["t"])
        for key, v in geo.items():
            geometry[f"{o['id']}/{key}"] = v
        roots[o["id"]], log[o["id"]], parts[o["id"]] = roots[m["target"]], {"xy": roots[m["target"]][:, :2, 3]}, {}
        lvl_id = m.get("level_object")
        if lvl_id:
            dst = by_id[m["target"]]["asset"]
            cols = []
            for f in range(n):
                M = roots[m["target"]][f] @ kin.T((0, 0, dst.get("base_m", 0.008)))
                M[:3, 2] *= max(geo["level_m"][f], 1e-4)        # unit-height column scaled to the water height
                cols.append(M)
            roots[lvl_id], log[lvl_id] = roots[m["target"]], {"level_m": geo["level_m"]}
            parts[lvl_id] = {"body": np.stack(cols)}
        for name, c in pc.items():
            extra_checks[name] = (lambda c: (lambda spec_c: c))(c)
    for o in spec["objects"]:
        if o.get("motion", {}).get("model") == "arm":
            st = log[o["id"]]
            extra_checks["ik_error"] = lambda c, st=st: {"ok": bool(st["ik_error_m"].max() <= c.get("max_m", 0.005)
                                                                 and st["ik_error_deg"].max() <= c.get("max_deg", 2.0)),
                                                         "detail": {"max_m": round(float(st["ik_error_m"].max()), 5),
                                                                    "max_deg": round(float(st["ik_error_deg"].max()), 3)}}
            rob = world.bodies[o["id"]]
            q = np.stack([st[f"q{i + 1}"] for i in range(len(rob.lo))], 1)
            extra_checks["joint_limits"] = lambda c, q=q, rob=rob: {"ok": bool(((q >= rob.lo) & (q <= rob.hi)).all()),
                                                                    "detail": {"min_margin_rad": round(float(np.minimum(q - rob.lo, rob.hi - q).min()), 4)}}
    if "final_level" in extra_checks:
        base = extra_checks["final_level"]

        def final_level(c, base=base):
            r = base(c)
            frac = r["detail"]["fraction_of_inner_height"]
            return {"ok": c.get("min", 0) <= frac <= c.get("max", 1), "detail": r["detail"]}
        extra_checks["final_level"] = final_level

    # 5. cameras
    W, H = spec["resolution"]
    cam_out = {}
    for c in spec["cameras"]:
        cam_out[c["id"]] = {"cam2world": cams.camera_poses(c, parts, t_frames), "K": cams.intrinsics(c["fov_deg"], W, H)}

    # 6. checks, and a framing report: how many frames each object's centre projects inside each camera
    #    (ignores occlusion; the rendered instance masks decide the `visible` checks later)
    results = chk.run(spec, roots, log, world.events, layout, extra=extra_checks)
    framing = {}
    for c in spec["cameras"]:
        K, c2w = cam_out[c["id"]]["K"], cam_out[c["id"]]["cam2world"]
        framing[c["id"]] = {}
        for o in spec["objects"]:
            if c["id"] in o.get("hide_from", []) or (c.get("parent") == o["id"]):
                continue
            ps = parts.get(o["id"]) or {"root": roots[o["id"]]}
            centre = np.mean([m[:, :3, 3] for m in ps.values()], axis=0)      # mean of the part origins
            if o["asset"]["type"] == "water_stream":
                pts = geometry.get(f"{o['id']}/points")
                if pts is not None:
                    centre = pts.mean(1)
            w2c = np.linalg.inv(c2w)
            pc = np.einsum("tij,tj->ti", w2c[:, :3, :3], centre) + w2c[:, :3, 3]
            z = -pc[:, 2]
            u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            v = K[1, 1] * -pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            framing[c["id"]][o["id"]] = int(inside.sum())

    # signal states per frame, and the head positions the renderer needs
    sig_frames = {sid: np.array([LIGHT[s.state(t)] for t in t_frames], np.int8) for sid, s in signals.items()}
    sig_info = []
    for sid, s in signals.items():
        heads = []
        for lane in s.lanes:
            pos, yaw, pole = layout.signal_head(lane)
            heads.append({"lane": lane, "pos": pos.tolist(), "yaw": float(yaw), "pole": pole.tolist()})
        sig_info.append({"id": sid, "lanes": s.lanes, "heads": heads})

    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "spec.yaml"), "wb").write(open(spec_path, "rb").read())   # this world's spec, byte for byte
    arrays = {}
    for i, ps in parts.items():
        for name, m in ps.items():
            arrays[f"parts/{i}/{name}"] = m.astype(np.float64)
    for i, lg in log.items():
        for key, v in lg.items():
            arrays[f"state/{i}/{key}"] = np.asarray(v)
    for c, v in cam_out.items():
        arrays[f"cams/{c}/cam2world"] = v["cam2world"]
        arrays[f"cams/{c}/K"] = v["K"]
    for sid, v in sig_frames.items():
        arrays[f"signals/{sid}"] = v
    for key, v in geometry.items():
        arrays[f"geometry/{key}"] = v
    np.savez_compressed(os.path.join(out, "world.npz"), **arrays)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    info = {"spec": os.path.abspath(spec_path), "spec_sha256": sha, "name": spec["name"], "rev": spec["rev"],
            "git": git_state(repo), "fps": fps, "frames": n, "times_s": t_frames.tolist(), "resolution": [W, H],
            "objects": [{"id": o["id"], "instance_id": o["instance_id"], "asset": o["asset"],
                         "parts": sorted(parts[o["id"]]), "hide_from": o.get("hide_from", [])} for o in spec["objects"]],
            "cameras": [{**c, "K": cam_out[c["id"]]["K"].tolist()} for c in spec["cameras"]],
            "signals": sig_info, "events": world.events, "checks": results, "framing_frames": framing}
    json.dump(info, open(os.path.join(out, "world.json"), "w"), indent=1)
    failed = [c for c in results if c["ok"] is False]
    for c in results:
        print(f"[check] {c['type']:24s} {'ok' if c['ok'] else ('pending' if c['ok'] is None else 'FAILED')}  {json.dumps(c['detail'])[:200]}")
    for c, f in framing.items():
        print(f"[framing] {c:10s} " + "  ".join(f"{o}:{k}" for o, k in f.items()) + f"   (of {n} frames)")
    print(f"[simulate] {spec['name']} r{spec['rev']}: {n} frames, {len(parts)} objects, {len(cam_out)} cameras, "
          f"{len(world.events)} events -> {out}")
    return info, failed
