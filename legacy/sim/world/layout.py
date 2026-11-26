"""Static world layouts in numbers (pure numpy). The simulator drives on them and the renderer builds
meshes from the same numbers, so the road a car follows is exactly the road that is drawn.

A layout is chosen by the spec's world.type:
  intersection  two roads crossing at the origin, right-hand traffic, one signal head per approach
  room          a box room with a table; the robot and props stand on the table
Adding a world type = adding a class here with ground_z(x, y) and whatever its controllers need,
plus a static builder in blender/.
"""

import numpy as np

LEGS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}   # leg name -> direction from centre


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _polyline(points, step=0.5):
    """Resample a polyline at ~step metres. Returns (s [M], xy [M, 2])."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_knots = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, s_knots[-1], int(np.ceil(s_knots[-1] / step)) + 1)
    return s, np.stack([np.interp(s, s_knots, pts[:, 0]), np.interp(s, s_knots, pts[:, 1])], 1)


class Intersection:
    """Two straight roads crossing at the origin. Right-hand traffic: a car heading in direction d keeps to the
    right of the centre line. Lanes are named <leg>_in: entering from that leg and going straight through.

    world:
      type: intersection
      lanes_per_direction: 1        (only 1 is built for now)
      lane_width_m: 3.5
      leg_length_m: 60              centre to the end of each leg
      sidewalk_m: 3.0               the strip along the kerb; the corner blocks beyond it hold buildings
      kerb_m: 0.15                  sidewalk height above the road
      crosswalk_m: 3.0              crosswalk width (along the leg)
      crosswalk_gap_m: 0.5          from the intersection box to the crosswalk's inner edge
      stop_line_m: 8.0              stop line distance from the centre
      signal_height_m: 5.4          height of a signal head's centre
    """

    def __init__(self, w):
        self.lane_w = w.get("lane_width_m", 3.5)
        self.n_lanes = w.get("lanes_per_direction", 1)
        assert self.n_lanes == 1, "only one lane per direction is built for now"
        self.L = w.get("leg_length_m", 60.0)
        self.sidewalk = w.get("sidewalk_m", 3.0)
        self.kerb = w.get("kerb_m", 0.15)
        self.half = self.lane_w * self.n_lanes                     # half road width = half the intersection box
        self.cw_in = self.half + w.get("crosswalk_gap_m", 0.5)    # crosswalk inner edge, from the centre
        self.cw_out = self.cw_in + w.get("crosswalk_m", 3.0)      # crosswalk outer edge
        self.stop = w.get("stop_line_m", 8.0)
        assert self.stop >= self.cw_out, "the stop line must be behind the crosswalk"
        self.signal_h = w.get("signal_height_m", 5.4)
        self.lanes, self.lane_info = {}, {}
        for leg, out in LEGS.items():
            d = -np.asarray(out, float)                            # travel direction of a car entering from this leg
            r = np.array([d[1], -d[0]])                            # its right-hand side
            start = -d * self.L + r * self.lane_w / 2
            end = d * self.L + r * self.lane_w / 2
            name = f"{leg}_in"
            self.lanes[name] = _polyline([start, end])
            self.lane_info[name] = {
                "dir": d, "right": r, "leg": leg,
                "stop_s": self.L - self.stop,                      # arc length of the stop line
                # crosswalks this lane crosses, as arc-length bands: the near one (own leg), the far one (exit leg)
                "crosswalks": {f"{leg}": (self.L - self.cw_out, self.L - self.cw_in),
                               f"{_opposite(leg)}": (self.L + self.cw_in, self.L + self.cw_out)},
                "lateral": (r * 0.0, r * self.lane_w),             # lane band: centre line to kerb, along r
            }

    # --- geometry the simulator and the renderer share ---
    def ground_z(self, x, y):
        """Height of the walkable surface: road 0, sidewalks and corner blocks `kerb` (smoothed over 0.2 m)."""
        ax, ay = np.abs(x), np.abs(y)
        inside = np.minimum(ax - self.half, ay - self.half)        # > 0 on a corner block
        return self.kerb * np.clip(inside / 0.2 + 0.5, 0.0, 1.0)

    def crosswalk_rect(self, leg):
        """Axis-aligned rectangle (x0, y0, x1, y1) of the crosswalk on a leg: it spans the road width."""
        a, b, h = self.cw_in, self.cw_out, self.half
        return {"north": (-h, a, h, b), "south": (-h, -b, h, -a), "east": (a, -h, b, h), "west": (-b, -h, -a, h)}[leg]

    def signal_head(self, lane):
        """Where the head for a lane hangs: over the lane, on the far side of the junction, facing the traffic.
        Returns (head position [3], facing yaw [rad], pole base [2] at the far-right corner)."""
        li = self.lane_info[lane]
        d, r = li["dir"], li["right"]
        head_xy = d * (self.half + 1.0) + r * self.lane_w / 2
        pole_xy = d * (self.half + 0.6) + r * (self.half + 0.6)
        yaw = np.arctan2(-d[1], -d[0])                             # facing back at the approaching cars
        return np.array([*head_xy, self.signal_h]), yaw, pole_xy

    def lane_point(self, lane, s):
        sl, xy = self.lanes[lane]
        return np.array([np.interp(s, sl, xy[:, 0]), np.interp(s, sl, xy[:, 1])])


def _opposite(leg):
    return {"north": "south", "south": "north", "east": "west", "west": "east"}[leg]


class Room:
    """A box room with one table. Origin at the floor centre.

    world:
      type: room
      size_m: [5, 5, 2.8]
      table: {center: [x, y], size_m: [1.2, 0.8], height_m: 0.75}
    """

    def __init__(self, w):
        self.size = np.asarray(w.get("size_m", [5, 5, 2.8]), float)
        t = w.get("table", {})
        self.table_c = np.asarray(t.get("center", [0.0, 0.0]), float)
        self.table_size = np.asarray(t.get("size_m", [1.2, 0.8]), float)
        self.table_h = t.get("height_m", 0.75)

    def ground_z(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        on = (np.abs(x - self.table_c[0]) <= self.table_size[0] / 2) & (np.abs(y - self.table_c[1]) <= self.table_size[1] / 2)
        return np.where(on, self.table_h, 0.0)


LAYOUTS = {"intersection": Intersection, "room": Room}


def build(world_spec):
    return LAYOUTS[world_spec["type"]](world_spec)
