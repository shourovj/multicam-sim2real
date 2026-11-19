"""Checks a simulated world must pass before it is rendered (pure numpy). Listed in the spec:

  checks:
    - {type: min_gap, metres: 1.0}                    no two objects' footprints closer than this, ever
    - {type: obeys_signals}                           no car crosses its stop line on red
    - {type: yields_to_pedestrians}                   no moving car on a crosswalk while a pedestrian is in its lane there
    - {type: event_order, events: [a, b, c]}          first occurrences happen in this order
    - {type: joint_limits, object: robot}             every joint inside its range (robots)
    - {type: visible, camera: cam, objects: [...], min_frames: 20}   checked after rendering (sim/world/export.py)

Each check returns {"type", "ok", "detail"}; simulate.py writes them to world.json and fails if any is false.
Footprints: cars are three circles along their length, people one circle, other objects a circle of `radius_m`.
"""

import numpy as np

from .kinematics import CAR_MODELS


def footprints(spec_obj, root):
    """[T, K, 2] circle centres and [K] radii in the ground plane."""
    a = spec_obj["asset"]
    if a["type"] == "car":
        d = CAR_MODELS[a["model"]]
        xs = np.array([-1, 0, 1]) * (d["length"] - d["width"]) / 2
        local = np.stack([xs, np.zeros(3), np.zeros(3), np.ones(3)], 1)
        c = np.einsum("tij,kj->tki", root, local)[..., :2]
        return c, np.full(3, d["width"] / 2)
    r = 0.3 if a["type"] == "pedestrian" else a.get("radius_m", 0.0)
    return root[:, None, :2, 3], np.array([r])


def min_gap(spec, roots, metres):
    objs = [o for o in spec["objects"] if o["asset"]["type"] in ("car", "pedestrian")]
    fp = {o["id"]: footprints(o, roots[o["id"]]) for o in objs}
    worst = (np.inf, None)
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            (ca, ra), (cb, rb) = fp[a["id"]], fp[b["id"]]
            d = np.linalg.norm(ca[:, :, None] - cb[:, None, :], axis=-1) - ra[None, :, None] - rb[None, None, :]
            g = float(d.min())
            if g < worst[0]:
                worst = (g, (a["id"], b["id"], int(np.unravel_index(d.argmin(), d.shape)[0])))
    ok = worst[0] >= metres
    return {"ok": bool(ok), "detail": {"min_gap_m": round(worst[0], 3), "pair": worst[1][:2] if worst[1] else None,
                                       "frame": worst[1][2] if worst[1] else None}}


def obeys_signals(events):
    bad = [e for e in events if e["name"].endswith("_passes_stop_line") and e.get("light") == "red"]
    return {"ok": not bad, "detail": {"stop_line_crossings": [e for e in events if e["name"].endswith("_passes_stop_line")]}}


def yields_to_pedestrians(spec, bodies_log, layout):
    """For every car with yield_to: on frames where it moves (> 0.5 m/s) with its front inside a crosswalk band,
    no listed pedestrian may be inside that band within the lane."""
    bad = []
    for o in spec["objects"]:
        m = o.get("motion", {})
        if m.get("model") != "lane_vehicle" or not m.get("yield_to"):
            continue
        log = bodies_log[o["id"]]
        info = layout.lane_info[m["lane"]]
        L = 4.6
        for t in range(len(log["lane_m"])):
            front = log["lane_m"][t] + L / 2
            if log["speed"][t] < 0.5:
                continue
            for s0, s1 in info["crosswalks"].values():
                if not (s0 <= front <= s1 + L):
                    continue
                for pid in m["yield_to"]:
                    p = bodies_log[pid]["xy"][t]
                    origin = layout.lane_point(m["lane"], 0.0) - info["right"] * layout.lane_w / 2
                    along, side = (p - origin) @ info["dir"], (p - origin) @ info["right"]
                    if s0 <= along <= s1 and 0 <= side <= layout.lane_w:
                        bad.append({"car": o["id"], "ped": pid, "frame": t})
    return {"ok": not bad, "detail": {"violations": bad[:10], "count": len(bad)}}


def event_order(events, names):
    first = {}
    for e in events:
        first.setdefault(e["name"], e["t"])
    missing = [n for n in names if n not in first]
    ts = [first.get(n) for n in names]
    ok = not missing and all(a < b for a, b in zip(ts, ts[1:]))
    return {"ok": bool(ok), "detail": {"times": dict(zip(names, ts)), "missing": missing}}


def run(spec, roots, bodies_log, events, layout, extra=None):
    out = []
    for c in spec.get("checks", []):
        kind = c["type"]
        if kind == "min_gap":
            r = min_gap(spec, roots, c["metres"])
        elif kind == "obeys_signals":
            r = obeys_signals(events)
        elif kind == "yields_to_pedestrians":
            r = yields_to_pedestrians(spec, bodies_log, layout)
        elif kind == "event_order":
            r = event_order(events, c["events"])
        elif kind == "visible":
            r = {"ok": None, "detail": "checked after rendering"}
        elif extra and kind in extra:
            r = extra[kind](c)
        else:
            raise ValueError(f"unknown check {kind}")
        out.append({"type": kind, **{k: v for k, v in c.items() if k != "type"}, **r})
    return out
