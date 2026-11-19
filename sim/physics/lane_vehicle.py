"""A car driving along one lane of the layout: it keeps its cruise speed, follows the car ahead, stops for a red
light and waits for pedestrians on a crosswalk in its lane (sim/physics/lane_vehicle.py).

The car is a point on the lane centre line with a speed, like ground_vehicle.py, and uses the same
Intelligent Driver Model (IDM). A red light and a pedestrian are handled the standard way: as a stopped
"virtual car" at the stop line or at the crosswalk edge, so one rule gives smooth braking for all three.
On yellow the car stops if it can do so with at most `brake_yellow` m/s^2, otherwise it goes; that choice
is kept until the light changes again.

motion block:
    model: lane_vehicle
    lane: north_in                 a lane of the layout (layout.Intersection names them <leg>_in)
    start_m: 30.0                  arc length along the lane at t = 0 (the car's centre)
    start_mps: 10.0
    signal: tl_NS                  optional; the signal that governs this lane
    yield_to: [ped_a, ped_b]       optional; pedestrians to wait for at crosswalks in this lane
    leader: car_x                  optional; the car ahead in the same lane (list it first)
    params: {cruise_mps: 10, ...}  any of DEFAULTS

State per frame: speed, lane_m, accel, wheel_angle (rad, integrated), and the signal state it saw.
Events: <id>_stops, <id>_moves (first start after a stop), <id>_passes_stop_line (with the light's state).
"""

import numpy as np

from .ground_vehicle import speed_limit

DEFAULTS = dict(
    cruise_mps=10.0,
    accel_max=2.0,          # m/s^2
    brake_max=6.0,          # m/s^2, hardest braking
    brake_comfort=2.0,      # m/s^2, IDM comfortable deceleration
    brake_yellow=3.0,       # m/s^2; on yellow, stop only if this is enough
    lat_accel_max=2.5,
    headway_s=1.2,
    gap_min_m=2.0,          # bumper-to-obstacle gap at standstill
    length_m=4.6,
    wheel_radius_m=0.33,
    curve_window_m=6.0,
    crosswalk_margin_m=1.0, # a pedestrian this close to the lane band counts as in it
)


class LaneVehicle:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        self.id, self.P = spec["id"], {**DEFAULTS, **m.get("params", {})}
        a = spec.get("asset", {})
        if a.get("type") == "car" and "length_m" not in m.get("params", {}):
            from world.kinematics import CAR_MODELS
            self.P["length_m"] = CAR_MODELS[a["model"]]["length"]
        lay = world.layout
        self.lane = m["lane"]
        self.path = lay.lanes[self.lane]
        self.info = lay.lane_info[self.lane]
        self.limit = speed_limit(self.path, self.P)
        self.s, self.v = float(m["start_m"]), float(m.get("start_mps", 0.0))
        self.signal_id, self.yield_ids, self.leader_id = m.get("signal"), m.get("yield_to", []), m.get("leader")
        self.accel, self.wheel, self.light = 0.0, 0.0, "none"
        self.yellow_go = None            # decision taken on the current yellow
        self.was_stopped = self.v < 0.05
        self.passed_stop = self.front() > self.info["stop_s"]
        self.layout = lay

    def link(self, world):
        self.signal = world.signals.get(self.signal_id) if self.signal_id else None
        self.peds = [world.bodies[i] for i in self.yield_ids]
        self.leader = world.bodies.get(self.leader_id) if self.leader_id else None

    def front(self):
        return self.s + self.P["length_m"] / 2

    def _obstacles(self, t):
        """Arc lengths of stopped virtual obstacles ahead of the front bumper."""
        out = []
        front = self.front()
        if self.signal is not None:
            st = self.signal.state(t)
            self.light = st
            if st != "yellow":
                self.yellow_go = None
            if not self.passed_stop:
                dist = self.info["stop_s"] - front
                if st == "red":
                    out.append(self.info["stop_s"])
                elif st == "yellow":
                    if self.yellow_go is None:
                        need = self.v ** 2 / (2 * max(dist, 0.01))
                        self.yellow_go = need > self.P["brake_yellow"]
                    if not self.yellow_go:
                        out.append(self.info["stop_s"])
        for s0, s1 in self.info["crosswalks"].values():
            if front > s0 - 0.3:                                    # already on (or past) this crosswalk
                continue
            if any(self._ped_in_band(p, s0, s1) for p in self.peds):
                out.append(s0 - 0.5)
        return out

    def _ped_in_band(self, ped, s0, s1):
        """Is the pedestrian on this crosswalk, within (or about to enter) this lane's width?"""
        p = ped.xy()
        d, r = self.info["dir"], self.info["right"]
        origin = self.layout.lane_point(self.lane, 0.0) - r * self.layout.lane_w / 2  # centre line at s = 0
        rel = p - origin
        along, side = rel @ d, rel @ r
        m = self.P["crosswalk_margin_m"]          # only on the centre-line side: past the kerb is the sidewalk
        return (s0 - 0.2 <= along <= s1 + 0.2) and (-m <= side <= self.layout.lane_w)

    def step(self, t, dt, world):
        P = self.P
        v0 = max(float(np.interp(self.s, self.path[0], self.limit)), 0.1)
        acc = P["accel_max"] * (1 - (self.v / v0) ** 4)
        root = np.sqrt(P["accel_max"] * P["brake_comfort"])
        gaps = [(s_obs - self.front(), 0.0) for s_obs in self._obstacles(t)]
        if self.leader is not None:
            gaps.append(((self.leader.s - self.leader.P["length_m"] / 2) - self.front(), self.leader.v))
        for gap, v_lead in gaps:
            want = P["gap_min_m"] + self.v * P["headway_s"] + self.v * (self.v - v_lead) / (2 * root)
            acc -= P["accel_max"] * (max(want, 0.0) / max(gap, 0.05)) ** 2
        self.accel = float(np.clip(acc, -P["brake_max"], P["accel_max"]))
        self.v = max(0.0, self.v + self.accel * dt)
        self.s = min(self.s + self.v * dt, self.path[0][-1])
        self.wheel += self.v * dt / P["wheel_radius_m"]
        # events
        if self.v < 0.05 and not self.was_stopped:
            self.was_stopped = True
            world.event(f"{self.id}_stops")
        elif self.v > 0.3 and self.was_stopped:
            self.was_stopped = False
            world.event(f"{self.id}_moves")
        if not self.passed_stop and self.front() > self.info["stop_s"]:
            self.passed_stop = True
            world.event(f"{self.id}_passes_stop_line", light=self.signal.state(t) if self.signal else "none")

    def xy(self):
        return self.layout.lane_point(self.lane, self.s)

    def pose(self):
        sl = self.path[0]
        d = self.layout.lane_point(self.lane, min(self.s + 1.4, sl[-1])) - self.layout.lane_point(self.lane, max(self.s - 1.4, 0))
        yaw = np.arctan2(d[1], d[0])
        c, sn = np.cos(yaw), np.sin(yaw)
        return np.array([*self.xy(), 0.0]), np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])

    def state(self):
        return {"speed": self.v, "lane_m": self.s, "accel": self.accel, "wheel_angle": self.wheel,
                "light": {"none": -1, "red": 0, "yellow": 1, "green": 2}[self.light]}
