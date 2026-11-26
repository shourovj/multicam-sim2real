"""A box that rides under a carrier until released, then falls freely with drag until it rests on the ground
(ported from the pilot's payload in simvid/physics/flight.py).

motion block:
    model: rigid_body
    start: {pos: [x, y, z]}                  # if it is not attached
    attached_to: drone_1                     # optional carrier id; list the carrier before this object
    offset: [0, 0, -0.45]                    # position under the carrier, in the carrier's frame
    release_t: 1.5                           # seconds; omit to stay attached
    params: {mass_kg: 0.5, size_m: 0.35, drag: 0.3}
The carrier keeps its thrust for an instant after losing the weight, so it rises briefly.
"""

import numpy as np

G = np.array([0.0, 0.0, -9.81])
DEFAULTS = dict(mass_kg=0.5, size_m=0.35, drag=0.3)


class RigidBody:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        self.id, self.P = spec["id"], {**DEFAULTS, **m.get("params", {})}
        self.carrier_id = m.get("attached_to")
        self.attached = self.carrier_id is not None
        self.offset = np.array(m.get("offset", [0.0, 0.0, -0.45]), float)
        self.release_t = m.get("release_t", np.inf)
        self.p = None if self.attached else np.array(m["start"]["pos"], float)
        self.v, self.rot, self.on_ground = np.zeros(3), np.eye(3), False

    def link(self, world):
        if self.attached:
            self.carrier = world.bodies[self.carrier_id]
            self.carrier.m += self.P["mass_kg"]
            self._ride()

    def _ride(self):
        p, r = self.carrier.pose()
        self.p, self.rot, self.v = p + r @ self.offset, r, self.carrier.v.copy()

    def step(self, t, dt, world):
        half = self.P["size_m"] / 2
        if self.attached:
            self._ride()
            if t >= self.release_t:
                self.attached = False
                self.carrier.m -= self.P["mass_kg"]
                world.event(f"{self.id}_release")
        elif not self.on_ground:
            self.v += (G - self.P["drag"] * self.v) * dt
            self.p = self.p + self.v * dt
            if self.p[2] <= half:
                self.p[2] = half
                self.v[:] = 0.0
                self.on_ground = True
                world.event(f"{self.id}_lands")

    def pose(self):
        return self.p.copy(), self.rot.copy()

    def state(self):
        return {"speed": float(np.linalg.norm(self.v)), "on_ground": bool(self.on_ground), "attached": bool(self.attached)}
