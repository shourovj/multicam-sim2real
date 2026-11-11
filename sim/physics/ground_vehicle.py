"""A vehicle driving along the scene's road: it speeds up and brakes within limits, slows for bends and keeps
its distance from the vehicle ahead.

The vehicle is a point on the road centre line with a speed. Each step it picks an acceleration:
- toward its cruise speed, but never faster than the bend ahead allows: the speed at which sideways
  acceleration stays under `lat_accel_max`, reached by braking at a comfortable rate before the bend;
- less, or braking, when the gap to its leader is short (the Intelligent Driver Model, a standard
  car-following rule: desired gap = gap_min + speed x headway, plus a term for closing speed).

motion block:
    model: ground_vehicle
    path: road
    start_m: 59.5              # road metres (arc length from the first waypoint) at t = 0
    start_mps: 9.0             # optional; default is what the road allows at start_m, up to cruise,
                               # and no faster than the leader
    leader: tank               # optional id of the vehicle ahead; list the leader before this object
    params: {cruise_mps: 13, ...}   # any of DEFAULTS
The object's local +X is forward and its origin sits on the ground.
"""

import numpy as np

DEFAULTS = dict(
    cruise_mps=13.0,
    accel_max=1.2,          # m/s^2
    brake_max=3.5,          # m/s^2, hardest braking
    brake_comfort=1.5,      # m/s^2, braking used when slowing for a bend
    lat_accel_max=2.5,      # m/s^2 sideways in a bend
    headway_s=1.0,          # time gap kept to the leader
    gap_min_m=4.0,          # bumper-to-bumper gap at standstill
    length_m=7.0,
    curve_window_m=10.0,    # road heading is smoothed over this length before taking its curvature
)


def speed_limit(road, P):
    """Highest speed allowed at every point of the road: the bend limit, lowered ahead of each bend so that
    comfortable braking reaches it in time. Returns [M], on the road's own samples."""
    s, xy = road
    d = np.diff(xy, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ds = np.diff(s)
    n = max(1, int(P["curve_window_m"] / ds.mean()))
    pad = np.concatenate([np.full(n, heading[0]), heading, np.full(n, heading[-1])])
    smooth = np.convolve(pad, np.ones(2 * n + 1) / (2 * n + 1), mode="same")[n:-n]
    kappa = np.abs(np.gradient(smooth, s[:-1]))
    v = np.sqrt(P["lat_accel_max"] / np.maximum(kappa, 1e-6))
    v = np.minimum(np.append(v, v[-1]), P["cruise_mps"])
    for i in range(len(v) - 2, -1, -1):                     # backward pass: brake in time for what lies ahead
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * P["brake_comfort"] * (s[i + 1] - s[i])))
    return v


class GroundVehicle:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        assert world.road is not None, f"{spec['id']}: a ground_vehicle needs the scene's road"
        self.id, self.P, self.road = spec["id"], {**DEFAULTS, **m.get("params", {})}, world.road
        self.limit = speed_limit(self.road, self.P)
        self.s = float(m["start_m"])
        self.v = float(m.get("start_mps", np.interp(self.s, self.road[0], self.limit)))
        self.leader_id, self.leader, self.accel = m.get("leader"), None, 0.0
        self._given_speed = "start_mps" in m

    def link(self, world):
        if self.leader_id is not None:
            self.leader = world.bodies[self.leader_id]
            if not self._given_speed:                       # a convoy starts at one pace, not closing on its leader
                self.v = min(self.v, self.leader.v)

    def step(self, t, dt, world):
        P = self.P
        v_des = max(float(np.interp(self.s, self.road[0], self.limit)), 0.1)
        acc = P["accel_max"] * (1 - (self.v / v_des) ** 4)
        if self.leader is not None:
            gap = (self.leader.s - self.leader.P["length_m"] / 2) - (self.s + P["length_m"] / 2)
            want = P["gap_min_m"] + self.v * P["headway_s"] + \
                self.v * (self.v - self.leader.v) / (2 * np.sqrt(P["accel_max"] * P["brake_comfort"]))
            acc -= P["accel_max"] * (max(want, 0.0) / max(gap, 0.1)) ** 2
        self.accel = float(np.clip(acc, -P["brake_max"], P["accel_max"]))
        self.v = max(0.0, self.v + self.accel * dt)
        self.s = min(self.s + self.v * dt, self.road[0][-1])

    def pose(self):
        s_path, xy = self.road
        at = lambda q: np.array([np.interp(q, s_path, xy[:, 0]), np.interp(q, s_path, xy[:, 1])])
        d = at(min(self.s + 1.5, s_path[-1])) - at(max(self.s - 1.5, 0.0))     # chord about one wheelbase long
        yaw = np.arctan2(d[1], d[0])
        c, sn = np.cos(yaw), np.sin(yaw)
        return np.array([*at(self.s), 0.0]), np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])

    def state(self):
        return {"speed": self.v, "road_m": self.s, "accel": self.accel}
