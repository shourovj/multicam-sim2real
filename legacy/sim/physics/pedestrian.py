"""A person who walks a path at a steady pace, or stands (legacy/sim/physics/pedestrian.py). Kinematic: no forces.

motion block, walking:
    model: pedestrian
    path: [[x, y], [x, y], ...]     waypoints on the ground; height comes from the layout (kerbs)
    speed_mps: 1.4
    start_offset_m: 0.0            already this far along the path at t = 0
    start_s: 0.0                   starts walking at this time (stands before)
motion block, standing:
    model: pedestrian
    idle_at: [x, y]
    facing: [x, y]                 a point to face (or facing_deg)
    pose: stand | phone            phone = holding a phone up in both hands
    sway_m: 0.02                   slow sway of the body, for life

State per frame: walk_phase (0..1, one full gait cycle per stride), walking (0/1), speed, pose_id.
The body's parts are posed from this state by legacy/sim/world/kinematics.py (human).
"""

import numpy as np

STRIDE_M = 1.5       # one gait cycle (two steps); phase advances by distance, so feet roughly keep pace
POSES = {"walk": 0, "stand": 1, "phone": 2}


class Pedestrian:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        self.id, self.layout, self.k = spec["id"], world.layout, index
        if "path" in m:
            self.pts = np.asarray(m["path"], float)
            seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
            self.knots = np.concatenate([[0], np.cumsum(seg)])
            self.speed, self.offset, self.t0 = float(m.get("speed_mps", 1.4)), float(m.get("start_offset_m", 0.0)), float(m.get("start_s", 0.0))
            self.mode = "walk"
        else:
            self.pts = np.asarray([m["idle_at"]], float)
            if "facing" in m:
                d = np.asarray(m["facing"], float) - self.pts[0]
                self.yaw0 = float(np.arctan2(d[1], d[0]))
            else:
                self.yaw0 = np.deg2rad(m.get("facing_deg", 0.0))
            self.mode = m.get("pose", "stand")
            self.sway = float(m.get("sway_m", 0.02))
        self.t = 0.0

    def step(self, t, dt, world):
        self.t = t + dt

    def _dist(self):
        return min(self.offset + self.speed * max(0.0, self.t - self.t0), self.knots[-1])

    def xy(self):
        if self.mode != "walk":
            return self.pts[0]
        d = self._dist()
        return np.array([np.interp(d, self.knots, self.pts[:, 0]), np.interp(d, self.knots, self.pts[:, 1])])

    def walking(self):
        return self.mode == "walk" and self.t >= self.t0 and self._dist() < self.knots[-1] - 1e-6

    def pose(self):
        xy = self.xy()
        if self.mode == "walk":
            d = self._dist()
            i = int(np.clip(np.searchsorted(self.knots, d, side="right") - 1, 0, len(self.pts) - 2))
            v = self.pts[i + 1] - self.pts[i]
            yaw = np.arctan2(v[1], v[0])
            sway = np.zeros(2)
        else:
            yaw = self.yaw0 + 0.04 * np.sin(2 * np.pi * 0.23 * self.t + self.k)
            sway = self.sway * np.array([np.sin(2 * np.pi * 0.31 * self.t + self.k), np.sin(2 * np.pi * 0.19 * self.t + 2 * self.k)])
        x, y = xy + sway
        z = float(self.layout.ground_z(x, y))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([x, y, z]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def state(self):
        if self.mode == "walk":
            phase = (self._dist() / STRIDE_M) % 1.0
            return {"walk_phase": phase, "walking": float(self.walking()), "speed": self.speed * self.walking(),
                    "pose_id": POSES["walk"] if self.walking() else POSES["stand"]}
        return {"walk_phase": 0.0, "walking": 0.0, "speed": 0.0, "pose_id": POSES[self.mode]}
