"""Point-mass quadrotor with attitude and ground contact (ported from the pilot's simvid/physics/flight.py).

The model is deliberately small. A drone is a mass with a thrust vector:
- each goal becomes a smooth reference path (a quintic) from the drone's position and velocity when the goal
  starts to the goal position, and a position controller asks for the acceleration that follows it;
- the thrust direction can tilt at most `tilt_max_deg` from vertical and follows the request with a lag;
- the thrust force is capped and follows the request with a motor lag;
- gravity, linear drag and (optionally) gusts act on it, and the ground holds it up until thrust exceeds weight.

motion block:
    model: quadrotor
    start: {pos: [x, y, z], vel: [vx, vy, vz], on_ground: false}   # vel and on_ground are optional
    yaw_deg: 0                                                     # heading, fixed for the clip
    goals: [{from_t: 0.7, t: 5.1, pos: [x, y, z], vel: [0, 0, 0]}]  # be at pos at time t; from_t defaults to the
                                                                   # previous goal's t, vel (on arrival) to zero
    params: {mass_kg: 1.5, ...}                                    # any of DEFAULTS
World frame: Z up, metres and seconds. The object's local +Z is the thrust axis and +X its heading.
"""

import numpy as np

G = np.array([0.0, 0.0, -9.81])
UP = np.array([0.0, 0.0, 1.0])

DEFAULTS = dict(
    mass_kg=1.5,
    tilt_max_deg=35.0,
    thrust_max_g=2.0,       # maximum thrust, in multiples of the drone's own weight
    drag=0.3,               # linear drag, 1/s
    tau_attitude_s=0.15,
    tau_motor_s=0.12,
    rest_height_m=0.035,    # height of the object's origin when it rests on the ground
    omega=2.2,              # bandwidth of the position controller, rad/s (critically damped)
    gust_mps2=0.0,          # RMS of a slowly varying random acceleration (wind gusts); 0 = still air
    gust_tau_s=0.8,
)


class Quadrotor:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        self.id, self.P = spec["id"], {**DEFAULTS, **m.get("params", {})}
        P, st = self.P, m["start"]
        self.p = np.array(st["pos"], float)
        self.v = np.array(st.get("vel", [0.0, 0.0, 0.0]), float)
        self.on_ground = bool(st.get("on_ground", False))
        if self.on_ground:
            self.p[2] = P["rest_height_m"]
            self.v[:] = 0.0
        self.left_ground = not self.on_ground
        self.home = self.p.copy()
        self.yaw = np.radians(m.get("yaw_deg", 0.0))
        self.m = P["mass_kg"]                                  # payloads add their mass in link()
        self.f_max = P["thrust_max_g"] * P["mass_kg"] * 9.81
        trim = -G + P["drag"] * self.v                         # thrust acceleration that holds the starting velocity
        self.n = UP.copy() if self.on_ground else trim / np.linalg.norm(trim)
        self.F = None if not self.on_ground else 0.0           # set on the first step, once the carried mass is known
        self._trim = np.linalg.norm(trim)
        self.goals, prev_t = [], 0.0
        for g in m.get("goals", []):
            self.goals.append({"from_t": g.get("from_t", prev_t), "t": g["t"], "pos": np.array(g["pos"], float),
                               "vel": np.array(g.get("vel", [0.0, 0.0, 0.0]), float)})
            prev_t = g["t"]
        self.rng = np.random.default_rng(world.seed + 1000 * (index + 1))
        self.gust = np.zeros(3)

    def _reference(self, t):
        """Position, velocity and acceleration the controller should track at time t."""
        active = [g for g in self.goals if g["from_t"] <= t]
        if not active:
            return self.home, np.zeros(3), np.zeros(3)
        g = active[-1]
        if "p0" not in g:                                      # the reference starts from the drone's actual state
            g["p0"], g["v0"] = self.p.copy(), self.v.copy()
        T = max(g["t"] - g["from_t"], 1e-3)
        s = t - g["from_t"]
        if s >= T:                                             # past the goal: keep its arrival velocity
            return g["pos"] + g["vel"] * (s - T), g["vel"], np.zeros(3)
        h, v0, v1 = g["pos"] - g["p0"], g["v0"], g["vel"]      # quintic with zero acceleration at both ends
        a3 = (20 * h - (8 * v1 + 12 * v0) * T) / (2 * T**3)
        a4 = (-30 * h + (14 * v1 + 16 * v0) * T) / (2 * T**4)
        a5 = (12 * h - 6 * (v1 + v0) * T) / (2 * T**5)
        return (g["p0"] + v0 * s + a3 * s**3 + a4 * s**4 + a5 * s**5,
                v0 + 3 * a3 * s**2 + 4 * a4 * s**3 + 5 * a5 * s**4,
                6 * a3 * s + 12 * a4 * s**2 + 20 * a5 * s**3)

    def step(self, t, dt, world):
        P = self.P
        if self.F is None:                                     # first step: thrust that holds the starting state,
            self.F = self.m * self._trim                       # with any payload's mass already added
        p_ref, v_ref, a_ref = self._reference(t)
        idle = self.on_ground and not any(g["from_t"] <= t for g in self.goals)   # on the ground, motors off
        # thrust acceleration asked for: follow the reference, cancel gravity
        want = a_ref + P["omega"] ** 2 * (p_ref - self.p) + 2 * P["omega"] * (v_ref - self.v) - G
        mag = np.linalg.norm(want)
        d = want / mag if mag > 1e-9 else UP
        cos_max, sin_max = np.cos(np.radians(P["tilt_max_deg"])), np.sin(np.radians(P["tilt_max_deg"]))
        if d @ UP < cos_max:                                   # lean no further than the tilt limit
            side = d - (d @ UP) * UP
            side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-9 else np.zeros(3)
            d = cos_max * UP + sin_max * side
        self.n += (d - self.n) * dt / P["tau_attitude_s"]
        self.n /= np.linalg.norm(self.n)
        f_des = 0.0 if idle else min(self.m * mag, self.f_max)
        self.F += (f_des - self.F) * dt / P["tau_motor_s"]
        if P["gust_mps2"] > 0:                                 # Ornstein-Uhlenbeck gust acceleration
            tau = P["gust_tau_s"]
            self.gust += -self.gust * dt / tau + P["gust_mps2"] * np.sqrt(2 * dt / tau) * self.rng.standard_normal(3)
        a = (self.F / self.m) * self.n + G - P["drag"] * self.v
        if self.p[2] <= P["rest_height_m"] + 1e-9 and a[2] <= 0:   # the ground carries the weight
            a[:] = 0.0
            self.v[:] = 0.0
            self.p[2] = P["rest_height_m"]
            self.on_ground = True
            self.n = UP.copy()
        else:
            if self.on_ground and not self.left_ground:
                world.event(f"{self.id}_lift_off")
                self.left_ground = True
            self.on_ground = False
            a = a + self.gust
        self.v += a * dt
        self.p += self.v * dt
        if self.p[2] < P["rest_height_m"]:
            self.p[2] = P["rest_height_m"]
            self.v[2] = max(self.v[2], 0.0)

    def pose(self):
        zb = self.n / np.linalg.norm(self.n)
        xc = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        return self.p.copy(), np.column_stack([np.cross(yb, zb), yb, zb])

    def state(self):
        return {"speed": float(np.linalg.norm(self.v)), "on_ground": bool(self.on_ground),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(self.n @ UP, -1, 1))))}
