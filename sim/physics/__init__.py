"""Physics for scene objects (sim/physics). Pure numpy, so it runs inside Blender and in plain Python.

A scene object with a `motion` block is moved by a physics model instead of scripted keyframes:

    motion: {model: quadrotor, start: {pos: [x, y, z]}, goals: [{t: 5.1, pos: [x, y, z]}]}

The scene says what the object tries to do; the model works out how it gets there. All objects in a scene
are stepped together on one clock, so they can interact (a payload under a drone, a vehicle keeping its
distance from the one ahead), and each model reports events such as lift-off as ground truth for evaluation.

Adding a kind of object means adding one model: a class with
    __init__(spec, world, index)   read the object's `motion` block
    link(world)                    optional; look up other bodies once all of them exist
    step(t, dt, world)             advance by dt seconds
    pose() -> (pos [3], rot [3, 3])  world position and orientation of the object's origin
    state() -> dict of floats / bools sampled once per frame (speed, on_ground, ...)
registered in MODELS below. The renderer, export and controls only ever see poses, so they don't change.
"""

import math

import numpy as np

from .ground_vehicle import GroundVehicle
from .quadrotor import Quadrotor
from .rigid_body import RigidBody

MODELS = {"quadrotor": Quadrotor, "ground_vehicle": GroundVehicle, "rigid_body": RigidBody}
MAX_DT = 1e-3     # seconds; the step is the largest value at or below this that divides a frame exactly


class World:
    """What a model can see besides its own object: the other bodies, the road, the clock and the event list."""

    def __init__(self, road, seed):
        self.road, self.seed, self.bodies, self.events, self.t = road, seed, {}, [], 0.0

    def event(self, name):
        self.events.append({"name": name, "t": round(self.t, 3)})


def simulate(objects, fps, frames, road=None, seed=0):
    """Roll every object forward and sample it at the clip's frame times.

    objects: scene object specs, each with `id` and a `motion` block. road: (s, xy) from motion.road_path, if any.
    Returns {"objects": {id: {"pos": [T, 3], "rot": [T, 3, 3], <state>: [T], ...}}, "events": [{name, t}], "dt": s}.
    """
    world = World(road, seed)
    for k, o in enumerate(objects):
        world.bodies[o["id"]] = MODELS[o["motion"]["model"]](o, world, k)
    for b in world.bodies.values():
        if hasattr(b, "link"):
            b.link(world)
    sub = math.ceil(1.0 / (fps * MAX_DT))
    dt = 1.0 / (fps * sub)
    rec = {i: {"pos": [], "rot": []} for i in world.bodies}
    for f in range(frames):
        for i, b in world.bodies.items():
            p, r = b.pose()
            rec[i]["pos"].append(np.array(p, float))
            rec[i]["rot"].append(np.array(r, float))
            for key, val in b.state().items():
                rec[i].setdefault(key, []).append(val)
        if f == frames - 1:
            break
        for n in range(sub):
            world.t = (f * sub + n) * dt
            for b in world.bodies.values():
                b.step(world.t, dt, world)
    return {"objects": {i: {k: np.array(v) for k, v in r.items()} for i, r in rec.items()},
            "events": world.events, "dt": dt}
