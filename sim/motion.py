"""Scripted trajectories (plan §1, sim/motion.py). Pure numpy, so it runs inside Blender too.

A trajectory spec is {keyframes: [[t, x, y, z, yaw_deg], ...]}. Positions are interpolated with a
centripetal-free (uniform) Catmull-Rom spline through the keys; yaw linearly. Attitude is yaw, then a
lean toward the direction of travel proportional to horizontal speed, the way a quadcopter tilts to
fly forward.
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
    pos[:, 2] += BOB_AMPLITUDE_M * np.sin(2 * np.pi * BOB_HZ * times + phase)
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
