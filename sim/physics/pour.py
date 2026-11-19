"""Water poured from a tilted pitcher into a glass: an analytic model, not a fluid simulation (sim/physics/pour.py).

What it computes, per frame, from the pitcher's simulated pose:
  - the water surface in the pitcher. The pitcher is a cylinder (radius R, height H) holding volume V; the
    surface stays level, so at tilt theta it reaches h_c + R tan(theta) up the downhill wall, where
    h_c = V / (pi R^2) is the height at the axis. Water spills when that passes the lip.
  - the flow over the lip: a weir, Q = C w sqrt(2 g) h^1.5, h = how far the surface stands above the lip.
  - the stream: leaves the lip at v0 = sqrt(2 g h) along the spout's horizontal direction and falls
    ballistically until it meets the water surface in the glass (or the glass bottom); its radius follows
    from continuity, r = sqrt(Q / (pi |v|)).
  - the glass: the poured volume over its inner area gives the water height.
Volumes are integrated over substeps between frames. The model sits behind one function (`simulate_pour`), so a
real fluid solver (Blender Mantaflow, ...) can replace it later and the rest of the pipeline does not change.

pour block (on the water_stream object):
    model: pour
    source: pitcher         object id; its asset gives radius_m, height_m; spout = lip point on its local +X side
    target: glass           object id; its asset gives radius_m, height_m, wall_m, base_m
    volume_l: 0.45          water in the pitcher at t = 0
    spout_width_m: 0.03
    weir_c: 0.6
    level_object: water_glass   object whose part "body" becomes the water column in the glass (scaled in z)

Outputs: stream points [T, N, 3], radius [T, N], active [T]; level_m [T]; pitcher_l [T]; glass_l [T];
events pour_start / pour_stop; checks stream_in_glass and final_level.
"""

import numpy as np

G = 9.81
N_PTS = 18
Q_MIN = 8e-6       # m^3/s (8 mL/s): below this the stream is not drawn (drips are not modelled; the volume still counts)


def simulate_pour(m, objs, roots, times):
    src, dst = objs[m["source"]]["asset"], objs[m["target"]]["asset"]
    R, H = src["radius_m"], src["height_m"]
    lip_local = np.array([R + 0.008, 0.0, H])
    rg_in = dst["radius_m"] - dst.get("wall_m", 0.003)
    base_in = dst.get("base_m", 0.008)
    area_g = np.pi * rg_in ** 2
    V = m["volume_l"] / 1000.0
    Vg = 0.0
    w, C = m.get("spout_width_m", 0.03), m.get("weir_c", 0.6)
    n = len(times)
    pts, rad, active = np.zeros((n, N_PTS, 3)), np.zeros((n, N_PTS)), np.zeros(n, bool)
    level, pl, gl = np.zeros(n), np.zeros(n), np.zeros(n)
    events, land_err = [], []
    P, Gl = roots[m["source"]], roots[m["target"]]

    def flow(Mp, V):
        """Flow rate (m^3/s) and head (m) for the pitcher pose Mp and volume V."""
        up = Mp[:3, 2]                                     # pitcher axis
        tilt = np.arccos(np.clip(up[2], -1, 1))
        hc = V / (np.pi * R * R)
        h = (hc + R * np.tan(min(tilt, np.deg2rad(85))) - H) * np.cos(tilt)
        return (C * w * np.sqrt(2 * G) * max(h, 0.0) ** 1.5, max(h, 0.0))

    sub = 20
    was = False
    for f in range(n):
        Q, h = flow(P[f], V)
        if f > 0:
            dt = (times[f] - times[f - 1]) / sub
            for s in range(sub):                           # pitcher pose held over the frame interval
                q, _ = flow(P[f - 1] if s < sub // 2 else P[f], V)
                dv = min(q * dt, V)
                V -= dv
                Vg += dv
            Q, h = flow(P[f], V)
        on = Q > Q_MIN
        if on and not was:
            events.append({"name": "pour_start", "t": round(float(times[f]), 3)})
        if was and not on:
            events.append({"name": "pour_stop", "t": round(float(times[f]), 3)})
        was = on
        lvl = min(Vg / area_g, dst["height_m"] - base_in)
        level[f], pl[f], gl[f] = lvl, V * 1000, Vg * 1000
        if not on:
            continue
        lip = P[f][:3, :3] @ lip_local + P[f][:3, 3]
        d = P[f][:3, 0].copy()
        d[2] = 0
        d /= np.linalg.norm(d) + 1e-12
        v0 = np.sqrt(2 * G * h) * d
        z_end = Gl[f][2, 3] + base_in + lvl
        t_end = (v0[2] + np.sqrt(v0[2] ** 2 + 2 * G * max(lip[2] - z_end, 1e-4))) / G
        tau = np.linspace(0, t_end, N_PTS)
        p = lip[None] + tau[:, None] * v0[None] + 0.5 * np.array([0, 0, -G])[None] * tau[:, None] ** 2
        speed = np.linalg.norm(v0[None] + tau[:, None] * np.array([0, 0, -G])[None], axis=1)
        r = np.sqrt(Q / (np.pi * np.maximum(speed, 0.05)))
        pts[f], rad[f], active[f] = p, np.clip(r, 0.0015, w / 2), True
        land_err.append(float(np.linalg.norm(p[-1, :2] - Gl[f][:2, 3]) + rad[f][-1]))
    geometry = {"points": pts, "radius": rad, "active": active, "level_m": level, "pitcher_l": pl, "glass_l": gl}
    checks = {
        "stream_in_glass": {"ok": bool(land_err) and max(land_err) < rg_in,
                            "detail": {"max_landing_offset_m": round(max(land_err), 4) if land_err else None, "glass_inner_r_m": rg_in}},
        "final_level": {"ok": None, "detail": {"level_m": round(float(level[-1]), 4),
                                               "fraction_of_inner_height": round(float(level[-1] / (dst["height_m"] - base_in)), 3),
                                               "poured_l": round(float(gl[-1]), 3)}},
    }
    return geometry, events, checks
