"""Scripted trajectories (plan §1, legacy/sim/motion.py). Pure numpy, so it runs inside Blender too.

A trajectory spec is {keyframes: [[t, x, y, z, yaw_deg], ...]}. Positions are interpolated with a
centripetal-free (uniform) Catmull-Rom spline through the keys; yaw linearly. Attitude is yaw, then a
lean toward the direction of travel proportional to horizontal speed, the way a quadcopter tilts to
fly forward.

A road-following spec is {path: road, start_m, speed_mps}: the object drives along the scene's road
(road_path) at constant speed, flat on the ground, heading along the road. `expand_convoy` turns the
scene's `convoy` block into one such object per vehicle.

Optional `ground_z` (metres): the height of the object's origin when it rests on the ground. The path
never goes below it, and the hover bob fades in over the first 0.5 m above it, so an object can sit
still on the ground and take off.
"""

import numpy as np

LEAN_DEG_PER_MPS = 7.0
BOB_AMPLITUDE_M = 0.025
BOB_HZ = 0.9


def _catmull_rom(keys_t, keys_p, t):
    """Uniform Catmull-Rom through (keys_t, keys_p) evaluated at times t. keys_p: [K, D]."""
    k = len(keys_t)
    p = np.concatenate([2 * keys_p[:1] - keys_p[1:2], keys_p, 2 * keys_p[-1:] - keys_p[-2:-1]])
    out = np.empty((len(t), keys_p.shape[1]))
    for n, tt in enumerate(t):
        i = int(np.clip(np.searchsorted(keys_t, tt, side="right") - 1, 0, k - 2))
        u = (tt - keys_t[i]) / (keys_t[i + 1] - keys_t[i])
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out[n] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3)
    return out


def _axis_angle(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def sample_trajectory(spec, times, phase=0.0):
    """Return (positions [T, 3], rotations [T, 3, 3]) in the world frame at the given times (s)."""
    keys = np.asarray(spec["keyframes"], dtype=float)
    kt, kp, kyaw = keys[:, 0], keys[:, 1:4], np.deg2rad(keys[:, 4])
    times = np.asarray(times, dtype=float)
    pos = _catmull_rom(kt, kp, times)
    bob = BOB_AMPLITUDE_M * np.sin(2 * np.pi * BOB_HZ * times + phase)
    ground_z = spec.get("ground_z")
    if ground_z is not None:
        pos[:, 2] = np.maximum(pos[:, 2], ground_z)
        bob = bob * np.clip((pos[:, 2] - ground_z) / 0.5, 0.0, 1.0)
    pos[:, 2] += bob
    yaw = np.interp(times, kt, kyaw)
    vel = np.gradient(pos, times, axis=0)
    rots = np.empty((len(times), 3, 3))
    for i in range(len(times)):
        rz = _axis_angle(np.array([0.0, 0.0, 1.0]), yaw[i])
        vh = np.array([vel[i, 0], vel[i, 1], 0.0])
        speed = np.linalg.norm(vh)
        if speed > 1e-6:
            lean = np.deg2rad(LEAN_DEG_PER_MPS * speed)
            rots[i] = _axis_angle(np.cross([0.0, 0.0, 1.0], vh), lean) @ rz  # tips the up axis toward v
        else:
            rots[i] = rz
    return pos, rots


def road_path(waypoints_xy, per_segment=40):
    """Smooth centre line through the waypoints (world x, y in metres).
    Returns (s [M] arc length from the first waypoint, xy [M, 2])."""
    wp = np.asarray(waypoints_xy, dtype=float)
    k = len(wp)
    xy = _catmull_rom(np.arange(k, dtype=float), wp, np.linspace(0, k - 1, per_segment * (k - 1) + 1))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return s, xy


def path_point(road, s):
    """Position [.., 2] and unit tangent [.., 2] of the road at arc length s (metres)."""
    s_path, xy = road
    at = lambda q: np.stack([np.interp(q, s_path, xy[:, 0]), np.interp(q, s_path, xy[:, 1])], -1)
    s = np.asarray(s, dtype=float)
    d = at(np.clip(s + 1.5, 0, s_path[-1])) - at(np.clip(s - 1.5, 0, s_path[-1]))   # chord about one wheelbase long
    return at(s), d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)


def sample_path(road, spec, times):
    """Return (positions [T, 3], rotations [T, 3, 3]) for an object driving along the road.
    The object's local +X is its forward direction and its origin sits on the ground."""
    s = spec["start_m"] + spec["speed_mps"] * np.asarray(times, dtype=float)
    assert s.min() >= 0 and s.max() <= road[0][-1], f"path runs off the road: {s.min():.1f}..{s.max():.1f} m of {road[0][-1]:.1f}"
    xy, tan = path_point(road, s)
    pos = np.concatenate([xy, np.zeros((len(s), 1))], axis=1)
    rots = np.stack([_axis_angle(np.array([0.0, 0.0, 1.0]), a) for a in np.arctan2(tan[:, 1], tan[:, 0])])
    return pos, rots


def expand_convoy(convoy):
    """The `convoy` block as a list of object specs: vehicle i starts i * spacing_m behind the lead.
    With `model: ground_vehicle` the vehicles are driven by legacy/sim/physics (each follows the one ahead);
    without it they hold `speed_mps` exactly (scripted)."""
    out = []
    for i, v in enumerate(convoy["vehicles"]):
        o = {"id": v.get("id", f"vehicle_{i + 1}"), "instance_id": i + 1, "asset": v}
        start = convoy["lead_start_m"] - i * convoy["spacing_m"]
        if "model" in convoy:
            o["motion"] = {"model": convoy["model"], "path": "road", "start_m": start,
                           "params": {**convoy.get("params", {}), **v.get("params", {})}}
            if i > 0:
                o["motion"]["leader"] = out[-1]["id"]
        else:
            o["trajectory"] = {"path": "road", "start_m": start, "speed_mps": convoy["speed_mps"]}
        out.append(o)
    return out


def scene_objects(spec):
    """Every moving object of a scene spec: its `objects` list plus the expanded `convoy` block."""
    return list(spec.get("objects", [])) + (expand_convoy(spec["convoy"]) if "convoy" in spec else [])


def png_size(path):
    """(width, height) of a PNG, read from its header."""
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def scene_road(spec, data_root):
    """The scene's road as (s, xy) in world metres, or None. Photo pixels map to the world with the image
    centre at the origin, +X east (image right), +Y north (image up)."""
    if "road" not in spec:
        return None
    import os
    bg = spec["background"]
    wp, hp = png_size(bg["image"] if os.path.isabs(bg["image"]) else os.path.join(data_root, bg["image"]))
    mpp = bg["metres_per_pixel"]
    return road_path((np.asarray(spec["road"]["waypoints_px"], dtype=float) - [wp / 2, hp / 2]) * [mpp, -mpp])
