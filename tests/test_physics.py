"""Physics models (sim/physics). Run:  python -m pytest tests   or   python tests/test_physics.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import motion  # noqa: E402
import physics  # noqa: E402

FPS, FRAMES = 30, 153


def drone(**motion_block):
    return {"id": "d1", "motion": {"model": "quadrotor", **motion_block}}


def run(objects, road=None):
    return physics.simulate(objects, FPS, FRAMES, road=road)


def test_frames_land_on_the_clock():
    out = run([drone(start={"pos": [0, 8, 3]})])
    assert abs(out["dt"] * round(1 / FPS / out["dt"]) - 1 / FPS) < 1e-12
    assert out["objects"]["d1"]["pos"].shape == (FRAMES, 3) and out["objects"]["d1"]["rot"].shape == (FRAMES, 3, 3)


def test_hover_stays_put():
    pos = run([drone(start={"pos": [0, 8, 3]}, goals=[{"t": 5, "pos": [0, 8, 3]}])])["objects"]["d1"]["pos"]
    assert np.abs(pos - pos[0]).max() < 1e-3


def test_reaches_goal_and_leans_on_the_way():
    o = run([drone(start={"pos": [-2, 9, 3]}, goals=[{"t": 5, "pos": [2, 9, 3]}])])["objects"]["d1"]
    assert np.linalg.norm(o["pos"][-1] - [2, 9, 3]) < 0.15
    assert 1.0 < o["tilt_deg"].max() < physics.quadrotor.DEFAULTS["tilt_max_deg"]
    assert o["tilt_deg"][0] == 0.0
    r = o["rot"][60]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9) and np.linalg.det(r) > 0.999


def test_starts_moving_without_a_jolt():
    """A drone that starts in flight keeps its speed: no braking to zero first."""
    o = run([drone(start={"pos": [-2.5, 7, 2.6], "vel": [1.0, 0, 0]}, goals=[{"t": 5.1, "pos": [2.4, 7, 2.6]}])])["objects"]["d1"]
    assert abs(o["speed"][0] - 1.0) < 1e-6 and o["speed"][:30].min() > 0.9
    assert np.linalg.norm(o["pos"][-1] - [2.4, 7, 2.6]) < 0.15


def test_stays_on_ground_until_told_to_fly():
    out = run([drone(start={"pos": [0, 7, 0], "on_ground": True}, goals=[{"from_t": 1.0, "t": 5, "pos": [0, 7, 2]}])])
    o = out["objects"]["d1"]
    rest = physics.quadrotor.DEFAULTS["rest_height_m"]
    assert o["on_ground"][:30].all() and not o["on_ground"][-1]
    lift = [e for e in out["events"] if e["name"] == "d1_lift_off"]
    assert lift and 1.0 < lift[0]["t"] < 2.0
    assert o["pos"][:, 2].min() >= rest - 1e-6 and np.abs(o["pos"][:30] - o["pos"][0]).max() < 1e-9


def test_payload_falls_under_gravity_and_drone_rises():
    box = {"id": "b1", "motion": {"model": "rigid_body", "attached_to": "d1", "offset": [0, 0, -0.45], "release_t": 1.5,
                                  "params": {"mass_kg": 0.6, "size_m": 0.35}}}
    out = run([drone(start={"pos": [0, 8, 2.6]}, goals=[{"t": 5, "pos": [0, 8, 2.6]}]), box])
    ev = {e["name"]: e["t"] for e in out["events"]}
    ideal = np.sqrt(2 * (2.6 - 0.45 - 0.175) / 9.81)                      # box centre to its resting height, in a vacuum
    assert ideal < ev["b1_lands"] - ev["b1_release"] < ideal * 1.15       # a little slower, because of drag
    b = out["objects"]["b1"]
    assert b["attached"][0] and not b["attached"][-1] and b["on_ground"][-1]
    z = out["objects"]["d1"]["pos"][:, 2]
    assert z.max() > 2.6 + 0.03 and abs(z[-1] - 2.6) < 0.03                # rises when it loses the weight, then settles


def bend_road():
    """100 m straight, a quarter turn of radius 30 m, 100 m straight."""
    a = np.linspace(0, np.pi / 2, 12)[1:-1]
    pts = [[x, 0.0] for x in np.linspace(0, 100, 11)] + [[100 + 30 * np.sin(t), 30 - 30 * np.cos(t)] for t in a] + \
          [[130.0, y] for y in np.linspace(30, 130, 11)]
    return motion.road_path(np.array(pts))


def vehicle(i, start_m, leader=None, **params):
    m = {"model": "ground_vehicle", "path": "road", "start_m": start_m, "params": params}
    if leader:
        m["leader"] = leader
    return {"id": i, "motion": m}


def test_vehicle_slows_for_the_bend():
    road = bend_road()                                                                  # the bend is at road metres 100 - 147
    limit = physics.ground_vehicle.speed_limit(road, {**physics.ground_vehicle.DEFAULTS, "lat_accel_max": 2.5})
    at = lambda m: float(np.interp(m, road[0], limit))
    assert at(40) == 13.0 and at(200) == 13.0 and abs(at(120) - np.sqrt(2.5 * 30)) < 0.3   # cruise, bend limit, cruise
    assert at(40) > at(85) > at(100)                                                    # eased down ahead of the bend
    o = run([vehicle("v1", 80.0, cruise_mps=13.0, lat_accel_max=2.5)], road=road)["objects"]["v1"]
    in_bend = (o["road_m"] > 105) & (o["road_m"] < 140)
    assert in_bend.any() and o["speed"][in_bend].max() < np.sqrt(2.5 * 30) * 1.2        # close to the bend limit
    assert o["speed"][0] > 11.0 and o["speed"][-1] < 9.5
    assert (np.diff(o["speed"]) * FPS).min() >= -3.5 - 1e-6                             # never beyond its brakes
    assert np.all(np.diff(o["road_m"]) >= 0)


def test_follower_keeps_its_distance():
    out = run([vehicle("lead", 60.0, cruise_mps=8.0), vehicle("f", 45.0, leader="lead", cruise_mps=13.0)], road=bend_road())
    gap = out["objects"]["lead"]["road_m"] - out["objects"]["f"]["road_m"] - 7.0       # bumper to bumper
    assert gap.min() > 2.0
    assert out["objects"]["f"]["speed"][-1] < 9.5                                       # settles to the leader's pace


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
