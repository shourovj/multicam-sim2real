"""Write the round-1 scenes (configs/scenes/*_phys.yaml) as p2s scenes: drone_takeoff, drones_air, convoy_overhead.

  python sim/physx/tools/make_round1_scenes.py sim/physx/scenes

The convoy's road comes from the aerial image's pixel waypoints (0.12019 m per pixel around pixel (845, 495)).
"""

import json
import math
import os
import sys

import numpy as np

DURATION = 152 / 30  # 153 frames at 30 fps


def drone(id_, pos, goals, yaw=0, vel=None):
    shapes = [{"type": "box", "size": [0.2, 0.2, 0.08], "material": "plastic"}]
    for a in (45, 135, 225, 315):
        r = math.radians(a)
        shapes.append({"type": "cylinder", "radius": 0.11, "half_length": 0.008,
                       "pos": [round(0.22 * math.cos(r), 4), round(0.22 * math.sin(r), 4), 0.05], "material": "plastic"})
    d = {"id": id_, "pos": pos, "yaw_deg": yaw, "mass_kg": 1.2, "shapes": shapes,
         "rotors": {"layout": "quad_x", "arm_m": 0.22, "height_m": 0.05},
         "rotor": {"thrust_to_weight": 2.2, "max_rpm": 9500},
         "drag": {"cd": 1.0, "area_m2": 0.06},
         "controller": {"goals": goals, "yaw_deg": yaw}}
    if vel:
        d["velocity"] = vel
    return d


CAMERA = {"id": "ground_cam", "type": "fixed", "position": [0, 0, 1.6], "look_at": [0, 10, 2.3], "fov_deg": 50,
          "resolution": [832, 480]}
WIND = {"velocity": [1.0, 0, 0], "gust": 0.3, "turbulence": 0.4, "turbulence_length_m": 3.0}


def takeoff():
    return {"p2s_version": 1, "name": "drone_takeoff",
            "meta": {"iteration": 1, "source": "port of configs/scenes/drone_takeoff_phys.yaml",
                     "prompt": "An orange quadcopter lifts off the grass left of a small tree, climbs to the right behind "
                               "the tree and hovers. Static ground camera."},
            "clock": {"fps": 30, "duration_s": DURATION, "substeps": 16},
            "world": {"ground": {"material": "concrete"}, "wind": WIND},
            "bodies": [{"id": "tree", "type": "static", "pos": [0.3, 6.0, 0],
                        "shapes": [{"type": "cylinder", "radius": 0.12, "half_length": 1.05, "pos": [0, 0, 1.05], "material": "wood"},
                                   {"type": "sphere", "radius": 0.55, "pos": [0, 0, 2.5], "material": "wood"}]}],
            "multirotors": [drone("drone_1", [-2.2, 10.0, 0.04],
                                  [{"from_t": 0.7, "t": 2.2, "pos": [-1.6, 10.0, 1.5], "vel": [1.0, 0, 0.8]},
                                   {"t": DURATION, "pos": [2.6, 10.0, 3.9]}], yaw=20)],
            "cameras": [CAMERA],
            "expect": [{"id": "lifts_off", "type": "event", "event": "drone_1:lift_off", "within_s": [0.6, 1.3]},
                       {"id": "climbs_right", "type": "reaches", "entity": "drone_1", "point": [2.6, 10.0, 3.9],
                        "within_m": 0.5, "by_s": DURATION},
                       {"id": "visible_at_takeoff", "type": "in_view", "camera": "ground_cam", "entity": "drone_1",
                        "during_s": [0, 2.0], "min_fraction": 0.9},
                       {"id": "stays_level", "type": "upright", "entity": "drone_1", "max_tilt_deg": 40}]}


def air():
    return {"p2s_version": 1, "name": "drones_air",
            "meta": {"iteration": 1, "source": "port of configs/scenes/drones_air_phys.yaml",
                     "prompt": "Two quadcopters cross in mid-air in a light wind, one heading right and one heading left, "
                               "then hover. Static ground camera."},
            "clock": {"fps": 30, "duration_s": DURATION, "substeps": 16},
            "world": {"ground": {"material": "concrete"}, "wind": WIND},
            "multirotors": [drone("drone_1", [-2.5, 7.0, 2.6], [{"t": DURATION, "pos": [2.4, 7.3, 2.85]}], vel=[0.94, 0.12, 0.12]),
                            drone("drone_2", [3.3, 10.0, 3.25], [{"t": DURATION, "pos": [-3.2, 9.7, 3.1]}], yaw=45,
                                  vel=[-1.24, -0.12, -0.12])],
            "cameras": [CAMERA],
            "expect": [{"id": "no_collision", "type": "min_gap", "a": "drone_1", "b": "drone_2", "metres": 0.5},
                       {"id": "d1_arrives", "type": "reaches", "entity": "drone_1", "point": [2.4, 7.3, 2.85], "within_m": 0.4, "by_s": DURATION},
                       {"id": "d2_arrives", "type": "reaches", "entity": "drone_2", "point": [-3.2, 9.7, 3.1], "within_m": 0.4, "by_s": DURATION},
                       {"id": "sees_drone_1", "type": "in_view", "camera": "ground_cam", "entity": "drone_1", "min_fraction": 0.9},
                       {"id": "sees_drone_2", "type": "in_view", "camera": "ground_cam", "entity": "drone_2", "min_fraction": 0.9}]}


ROAD_PX = [[600, 118], [650, 110], [699, 106], [755, 131], [805, 159], [849, 200], [886, 244], [917, 287], [939, 331],
           [952, 375], [961, 419], [964, 444], [966, 490], [958, 535], [940, 575], [915, 612], [885, 645], [860, 672],
           [842, 697], [824, 737], [799, 781], [761, 825], [730, 862], [699, 900], [667, 937], [645, 962]]


def convoy():
    road = np.array([[(u - 845) * 0.12019, (495 - v) * 0.12019] for u, v in ROAD_PX])
    knots = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(road, axis=0), axis=1))])
    at = lambda s: np.array([np.interp(s, knots, road[:, 0]), np.interp(s, knots, road[:, 1])])
    heading = lambda s: math.degrees(math.atan2(*(at(s + 1.5) - at(s - 1.5))[::-1]))
    path = [[round(x, 2), round(y, 2)] for x, y in road]
    place = lambda s: [round(at(s)[0], 2), round(at(s)[1], 2), 0]

    def truck(id_, s, leader):
        return {"id": id_, "pos": place(s), "yaw_deg": round(heading(s), 1), "speed_mps": 10, "mass_kg": 9000,
                "shapes": [{"type": "box", "size": [2.0, 2.4, 1.6], "pos": [2.6, 0, 1.95], "material": "steel"},
                           {"type": "box", "size": [4.6, 2.4, 1.9], "pos": [-1.2, 0, 2.1], "material": "steel"}],
                "wheels": {"radius_m": 0.55, "half_width_m": 0.2, "mass_kg": 120,
                           "axles": [{"x": 2.6, "track_m": 2.0, "steer": True, "drive": True},
                                     {"x": -1.2, "track_m": 2.0, "drive": True}, {"x": -2.6, "track_m": 2.0, "drive": True}]},
                "limits": {"max_wheel_torque_nm": 3000, "max_brake_torque_nm": 6000, "max_steer_deg": 32},
                "driver": {"path": path, "cruise_mps": 13, "lat_accel_max": 3.5, "leader": leader, "gap_min_m": 6, "headway_s": 1.3}}

    th = math.radians(heading(59.5))
    tank = {"id": "tank", "pos": place(59.5), "yaw_deg": round(math.degrees(th), 1), "mass_kg": 45000,
            "velocity": [round(10 * math.cos(th), 2), round(10 * math.sin(th), 2), 0],
            "shapes": [{"type": "box", "pos": [0, -1.4, 0.5], "half_extents": [3.5, 0.35, 0.5], "material": "track"},
                       {"type": "box", "pos": [0, 1.4, 0.5], "half_extents": [3.5, 0.35, 0.5], "material": "track"},
                       {"type": "box", "pos": [0, 0, 1.25], "half_extents": [3.6, 1.75, 0.5], "material": "steel"},
                       {"type": "box", "pos": [-0.3, 0, 2.1], "half_extents": [1.8, 1.5, 0.35], "material": "steel"}],
            "traction": {"speed_mps": 13, "mu": 0.7, "track_half_length_m": 1.75, "response_s": 1.5, "path": path,
                         "lat_accel_max": 3.0}}
    bridge = {"id": "bridge", "type": "static", "pos": place(30.5), "yaw_deg": round(heading(30.5), 1),
              "shapes": [{"type": "box", "size": [8, 14, 0.6], "pos": [0, 0, 5.3], "material": "concrete"},
                         {"type": "box", "size": [8, 1.0, 5.0], "pos": [0, 6.5, 2.5], "material": "concrete"},
                         {"type": "box", "size": [8, 1.0, 5.0], "pos": [0, -6.5, 2.5], "material": "concrete"}]}
    canopy = [{"id": "tree_%02d" % k, "type": "static", "pos": place(s),
               "shapes": [{"type": "sphere", "radius": 3.4, "pos": [0, 0, 8.0], "material": "wood"}]}
              for k, s in enumerate(np.arange(68, 104, 5.0))]
    return {"p2s_version": 1, "name": "convoy_overhead",
            "meta": {"iteration": 1, "source": "port of configs/scenes/convoy_overhead_phys.yaml",
                     "prompt": "A tank leads two military trucks along a winding road, under a bridge and a tree canopy, "
                               "filmed from a drone 150 m overhead with east at the top of the frame."},
            "clock": {"fps": 30, "duration_s": DURATION, "substeps": 16},
            "world": {"ground": {"material": "asphalt"}},
            "bodies": [bridge] + canopy,
            "tracked_vehicles": [tank],
            "vehicles": [truck("truck_1", 40.0, "tank"), truck("truck_2", 20.5, "truck_1")],
            "cameras": [{"id": "overhead", "type": "fixed", "position": [0, 0, 150], "look_at": [1, 0, 0], "fov_deg": 40.3,
                         "resolution": [832, 480]}],
            "expect": [{"id": "gap_tank_truck1", "type": "min_gap", "a": "tank", "b": "truck_1", "metres": 4},
                       {"id": "gap_trucks", "type": "min_gap", "a": "truck_1", "b": "truck_2", "metres": 4},
                       {"id": "tank_drives", "type": "speed", "entity": "tank", "during_s": [1, DURATION], "min_mps": 4, "max_mps": 15},
                       {"id": "truck2_seen", "type": "in_view", "camera": "overhead", "entity": "truck_2", "min_fraction": 0.5},
                       {"id": "tank_goes_under_canopy", "type": "leaves_view", "camera": "overhead", "entity": "tank",
                        "inside_at_s": 0.0, "outside_by_s": 2.0},
                       {"id": "trucks_upright", "type": "upright", "entity": "truck_1", "max_tilt_deg": 15}]}


def main():
    out = sys.argv[1]
    for scene in (takeoff(), air(), convoy()):
        with open(os.path.join(out, scene["name"] + ".json"), "w") as f:
            json.dump(scene, f, indent=1)
        print("wrote", scene["name"])


if __name__ == "__main__":
    main()
