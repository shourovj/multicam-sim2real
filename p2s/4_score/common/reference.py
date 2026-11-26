"""A view's simulator reference, the same for every scene: everything the metrics compare a generated clip against.

A view is one camera's clip or a cut that joins several cameras. Its reference is read from the camera bundles
(legacy/sim/render_world.py + p2s/2_render/bundle.py) and the world file, never from anything scene-specific:
  depth [T, H, W]      metres, inf where there is nothing (sky)
  inst [T, H, W]       instance id per pixel: 0 nothing, 1-99 simulated objects, 100+ static parts
  K [T, 3, 3], c2w [T, 4, 4]   the camera of each frame (Blender convention: looks down -Z, +Y up)
  shots                [(first frame, last frame, camera)]: one entry for a camera, one per shot for a cut
  objects              the simulated objects, each with the cameras it is scored in
  tracks               surface points on every moving part, projected into each frame, with visibility
An object is not scored in a camera that it carries (a car's own dashcam shows only its hood) or that hides it, and
not at all if its spec entry says `eval: false`.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "3_generate", "common"))
import multicam_common as mc  # noqa: E402

VIS_DEPTH_TOL_M = 0.08          # a point is visible if the rendered surface at its pixel is this close to it


class View:
    pass


def _bundle(root, cam):
    return os.path.join(root, "cams", cam, "bundle")


def project(K, c2w, world):
    """world [N, T, 3], K [3, 3], c2w [T, 4, 4] -> pixel uv [N, T, 2] and depth along the view axis [N, T]."""
    w2c = np.linalg.inv(c2w)
    pc = np.einsum("tij,ntj->nti", w2c[:, :3, :3], world) + w2c[None, :, :3, 3]
    z = -pc[..., 2]
    zz = np.maximum(z, 1e-6)
    return np.stack([K[0, 0] * pc[..., 0] / zz + K[0, 2], K[1, 1] * -pc[..., 1] / zz + K[1, 2]], -1), z


def part_points(timeline, part):
    """A part's track points in the part's own frame. Renders made before 2026-11-23 stored them relative to the
    part's pose at the last rendered frame (timeline has no `track_points_frame`); undo that here."""
    loc = np.asarray(part["track_points_local"], float).reshape(-1, 3)
    if len(loc) and timeline.get("track_points_frame") != "part":
        M = np.asarray(part["local2world"])[-1]
        loc = loc @ M[:3, :3].T + M[:3, 3]
    return loc


def camera_tracks(timeline, depth, inst, skip=()):
    """Every track point of one camera: uv [N, T, 2], visible [N, T], instance id [N]."""
    K, c2w = np.asarray(timeline["camera"]["K"]), np.asarray(timeline["camera"]["cam2world_blender"])
    T, H, W = depth.shape
    pts, vis, ids = [], [], []
    for o in timeline["objects"]:
        if o["hidden"] or o["id"] in skip:
            continue
        for part in o["parts"].values():
            loc = part_points(timeline, part)
            if len(loc) == 0:
                continue
            M = np.asarray(part["local2world"])
            uv, z = project(K, c2w, np.einsum("tij,nj->nti", M[:, :3, :3], loc) + M[None, :, :3, 3])
            ui = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
            tt = np.broadcast_to(np.arange(T), ui.shape)
            inb = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1) & (z > 0)
            pts.append(uv)
            vis.append(inb & (inst[tt, vi, ui] == o["instance_id"]) & (np.abs(z - depth[tt, vi, ui]) < VIS_DEPTH_TOL_M))
            ids += [o["instance_id"]] * len(loc)
    if not pts:
        return np.zeros((0, T, 2), np.float32), np.zeros((0, T), bool), np.zeros(0, int)
    return np.concatenate(pts).astype(np.float32), np.concatenate(vis), np.asarray(ids)


def load_view(exp, j):
    """The reference for one job of an experiment (multicam_common.jobs): a camera view or a cut."""
    root = mc.scene_root(exp, j["scene"])
    spec = mc.scene_spec(exp, j["scene"])
    world = json.load(open(os.path.join(root, "world", "world.json")))
    parent = {c["id"]: c.get("parent") for c in world["cameras"]}
    n = spec["frames"]
    shots = [(0, n - 1, j["view"])] if j["kind"] == "cam" else [(s["frames"][0], s["frames"][1], s["cam"]) for s in exp["cuts"][j["scene"]][j["view"]]]
    v = View()
    v.scene, v.view, v.kind, v.shots, v.fps = j["scene"], j["view"], j["kind"], shots, spec["fps"]
    depth, inst, K, c2w, tracks = [], [], [], [], []
    scored_in = {o["id"]: set() for o in spec["objects"]}
    for a, b, cam in shots:
        B = _bundle(root, cam)
        tl = json.load(open(os.path.join(B, "timeline.json")))
        d, i = np.load(os.path.join(B, "depth.npy"), mmap_mode="r"), np.load(os.path.join(B, "instance.npy"), mmap_mode="r")
        depth.append(np.asarray(d[a:b + 1]))
        inst.append(np.asarray(i[a:b + 1]))
        K += [np.asarray(tl["camera"]["K"])] * (b - a + 1)
        c2w.append(np.asarray(tl["camera"]["cam2world_blender"])[a:b + 1])
        skip = {o["id"] for o in spec["objects"] if o.get("eval") is False or o["id"] == parent[cam]}
        for o in tl["objects"]:
            if not o["hidden"] and o["id"] not in skip:
                scored_in[o["id"]].add(cam)
        uv, vis, ids = camera_tracks(tl, np.asarray(d), np.asarray(i), skip)
        keep = vis[:, a:b + 1].any(1)                        # points seen at some time in this shot
        t0 = a + np.argmax(vis[keep][:, a:b + 1], axis=1)    # the first frame of the shot each one is visible
        tracks.append(dict(points=uv[keep], visible=vis[keep], object_id=ids[keep], query_t=t0, first=a, last=b))
    v.depth, v.inst = np.concatenate(depth), np.concatenate(inst)
    v.K, v.c2w = np.stack(K), np.concatenate(c2w)
    v.T, v.H, v.W = v.depth.shape
    v.camera_of_frame = [cam for a, b, cam in shots for _ in range(b - a + 1)]
    v.objects = [dict(id=o["id"], instance_id=o["instance_id"], type=o["asset"]["type"], cameras=sorted(scored_in[o["id"]]))
                 for o in spec["objects"] if scored_in[o["id"]]]
    v.tracks = tracks
    return v


def scored(view, obj, t):
    """Is this object scored at frame t of the view (its camera at that frame neither carries nor hides it)?"""
    return view.camera_of_frame[t] in obj["cameras"]
