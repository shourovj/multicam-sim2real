"""Per-frame camera poses from the spec's `cameras:` list and the simulated parts (pure numpy).

  cameras:
    - {id: cctv, type: fixed, position: [x, y, z], look_at: [x, y, z], fov_deg: 80}
    - {id: dashcam, type: attached, parent: car_ego, part: body,
       offset: {pos: [x, y, z], forward: [1, 0, 0], up: [0, 0, 1], yaw_deg: 0, pitch_deg: 2, roll_deg: 0}, fov_deg: 100}
         looks along `forward` of the part (default its +X), `up` toward the part's `up` (default +Z), then turned
         by yaw (left +), pitch (up +) and roll in the camera's own frame; pos is in the part's frame
    - {id: phone, type: handheld, parent: ped_phone, part: phone, offset: {pos: [x, y, z]},
       look_at: [x, y, z], shake: {rot_deg: 0.6, pos_m: 0.004, hz: [1.0, 3.0], seed: 11}, fov_deg: 66}
         sits on the part, aims at a world point, plus a small hand shake (sum of sines, fixed seed)

Optional on any camera: roll_deg, clip_start_m (default 0.1), clip_end_m (default 1000).
Returns cam2world [T, 4, 4] in Blender's camera convention (looks down -Z, +Y up) and the pinhole K.
"""

import numpy as np

from .kinematics import T, Rx, Ry, Rz


def look_rotation(forward, up=(0, 0, 1)):
    """Blender camera rotation (columns: right, up, back) for a forward direction."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:                               # looking straight up or down
        r = np.cross(f, (0, 1, 0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def intrinsics(fov_deg, W, H):
    fx = W / 2 / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])


def _shake(sh, times):
    """Small rotation (rad, about camera x/y/z) and translation (m) per frame: a few incommensurate sines."""
    rng = np.random.default_rng(sh.get("seed", 0))
    lo, hi = sh.get("hz", [1.0, 3.0])
    rot, pos = np.zeros((len(times), 3)), np.zeros((len(times), 3))
    for k in range(3):
        for _ in range(3):
            f, ph, a = rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0)
            rot[:, k] += a * np.sin(2 * np.pi * f * times + ph)
            f, ph, a = rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0)
            pos[:, k] += a * np.sin(2 * np.pi * f * times + ph)
    rot *= np.deg2rad(sh.get("rot_deg", 0.5)) / np.maximum(np.abs(rot).max(0), 1e-9)
    pos *= sh.get("pos_m", 0.003) / np.maximum(np.abs(pos).max(0), 1e-9)
    return rot, pos


def camera_poses(cam, parts, times):
    """cam: one spec entry. parts: {object_id: {part: [T, 4, 4]}}. Returns cam2world [T, 4, 4]."""
    n = len(times)
    out = np.empty((n, 4, 4))
    roll = np.deg2rad(cam.get("roll_deg", 0.0))
    if cam["type"] == "fixed":
        pos = np.asarray(cam["position"], float)
        R = look_rotation(np.asarray(cam["look_at"], float) - pos) @ Rz(roll)
        out[:] = T(pos, R)
        return out
    par = parts[cam["parent"]][cam.get("part", "body")]
    off = cam.get("offset", {})
    p_local = np.asarray(off.get("pos", [0, 0, 0]), float)
    if cam["type"] == "attached":
        y, p, r = np.deg2rad([off.get("yaw_deg", 0.0), off.get("pitch_deg", 0.0), off.get("roll_deg", 0.0)])
        local = look_rotation(off.get("forward", [1, 0, 0]), off.get("up", [0, 0, 1])) @ Ry(y) @ Rx(p) @ Rz(r + roll)
        for t in range(n):
            M = par[t]
            out[t] = T(M[:3, :3] @ p_local + M[:3, 3], M[:3, :3] @ local)
        return out
    # handheld: position on the part, aimed at a world point, then shaken in the camera frame
    rot, dpos = _shake(cam.get("shake", {}), np.asarray(times))
    target = np.asarray(cam["look_at"], float)
    for t in range(n):
        M = par[t]
        p = M[:3, :3] @ p_local + M[:3, 3]
        R = look_rotation(target - p) @ Rz(roll) @ Rx(rot[t, 0]) @ Ry(rot[t, 1]) @ Rz(rot[t, 2])
        out[t] = T(p + R @ dpos[t], R)
    return out
