"""Poses of every part of an object from its motion state (pure numpy; sim/world/kinematics.py).

Each function takes the object's spec, its root poses [T, 4, 4] and its per-frame state (dict of [T] arrays)
and returns {part_name: [T, 4, 4] world matrices}. The Blender side builds one mesh per part in the part's
own frame (sim/world/blender/), so what is drawn is exactly what is computed here.

Kinds:
  car         body (pitches with acceleration) + four wheels that roll with the distance driven
  pedestrian  articulated body: pelvis, torso, head, upper arms, forearms, thighs, shins, feet (+ phone)
  rigid       one part, "body", at the root pose (props, glasses, pitchers)
  robot       one part per link of an MJCF robot, from joint angles (sim/world/mjcf.py)
"""

import numpy as np

# --- small transform helpers ---


def T(p=(0, 0, 0), R=None):
    m = np.eye(4)
    m[:3, 3] = p
    if R is not None:
        m[:3, :3] = R
    return m


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy(r, p, y):
    """Roll about x, then pitch about y, then yaw about z (all in degrees), as one rotation matrix."""
    r, p, y = np.deg2rad([r, p, y])
    return Rz(y) @ Ry(p) @ Rx(r)


# --- cars ---

CAR_MODELS = {   # metres. length/width/height of the body, wheelbase, track, wheel radius
    "sedan":     dict(length=4.70, width=1.84, height=1.45, wheelbase=2.80, track=1.58, wheel_r=0.33),
    "suv":       dict(length=4.75, width=1.94, height=1.74, wheelbase=2.85, track=1.65, wheel_r=0.37),
    "hatchback": dict(length=4.20, width=1.80, height=1.48, wheelbase=2.60, track=1.55, wheel_r=0.32),
}
PITCH_PER_ACCEL = 0.006          # rad per m/s^2: the nose dips about 2 degrees under hard braking


def car(spec, root, state):
    d = CAR_MODELS[spec["asset"]["model"]]
    n = len(root)
    pitch = -PITCH_PER_ACCEL * np.asarray(state["accel"])
    spin = np.asarray(state["lane_m"]) / d["wheel_r"]
    parts = {k: np.empty((n, 4, 4)) for k in ("body", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")}
    pivot = np.array([0, 0, d["wheel_r"]])
    for t in range(n):
        body = T(pivot) @ T(R=Ry(pitch[t])) @ T(-pivot)
        parts["body"][t] = root[t] @ body
        for name, (sx, sy) in {"wheel_fl": (1, 1), "wheel_fr": (1, -1), "wheel_rl": (-1, 1), "wheel_rr": (-1, -1)}.items():
            parts[name][t] = root[t] @ T((sx * d["wheelbase"] / 2, sy * d["track"] / 2, d["wheel_r"]), Ry(spin[t]))
    return parts


# --- people ---

def human_dims(height_m=1.75):
    k = height_m / 1.75
    return {key: v * k for key, v in dict(
        thigh=0.43, shin=0.43, ankle=0.075, torso=0.50, neck=0.06, head=0.22, hip_w=0.18, shoulder_w=0.36,
        upper_arm=0.30, forearm=0.25, hand=0.17, foot=0.25).items()}


def _walk_angles(p):
    """Joint angles (rad) for one leg and the arm on the same side, at gait phase p (0..1)."""
    c, s = np.cos(2 * np.pi * p), np.sin(2 * np.pi * p)
    hip = np.deg2rad(26) * s                                  # forward swing peaks at p = 0.25
    knee = np.deg2rad(5 + 55 * max(0.0, c) ** 2)              # bent in mid-swing (p ~ 0), straight at heel strike
    shoulder = np.deg2rad(-20 * s + 4)                        # arms swing against the legs
    elbow = np.deg2rad(16 + 8 * s)
    return hip, knee, shoulder, elbow


POSE_STAND = dict(hip=0.0, knee=np.deg2rad(3), shoulder=np.deg2rad(5), elbow=np.deg2rad(12), abd=np.deg2rad(7), head=0.0)
POSE_PHONE = dict(hip=0.0, knee=np.deg2rad(3), shoulder=np.deg2rad(60), elbow=np.deg2rad(95), abd=np.deg2rad(-24),
                  head=np.deg2rad(8))


def human_frame(dm, pose_id, phase):
    """Root-frame transforms of every part for one frame. pose_id: 0 walk, 1 stand, 2 phone (sim/physics/pedestrian.py)."""
    if pose_id == 0:
        hL, kL, sL, eL = _walk_angles(phase)
        hR, kR, sR, eR = _walk_angles(phase + 0.5)
        abdL = abdR = np.deg2rad(6)
        head, lean = 0.0, np.deg2rad(4)
        twist = np.deg2rad(5) * np.sin(2 * np.pi * phase)
    else:
        P = POSE_PHONE if pose_id == 2 else POSE_STAND
        hL = hR = P["hip"]; kL = kR = P["knee"]; sL = sR = P["shoulder"]; eL = eR = P["elbow"]
        abdL = abdR = P["abd"]; head, lean, twist = P["head"], 0.0, 0.0
    pelvis = T((0, 0, dm["thigh"] + dm["shin"] + dm["ankle"]), Rz(twist * 0.5))
    torso = pelvis @ T(R=Ry(lean) @ Rz(-twist))
    parts = {"pelvis": pelvis, "torso": torso,
             "head": torso @ T((0, 0, dm["torso"] + dm["neck"]), Ry(head))}
    for side, sg, h, k, s, e, ab in (("l", 1, hL, kL, sL, eL, abdL), ("r", -1, hR, kR, sR, eR, abdR)):
        ua = torso @ T((0, sg * dm["shoulder_w"] / 2, dm["torso"])) @ T(R=Rx(sg * ab) @ Ry(-s))
        parts[f"upper_arm_{side}"] = ua
        parts[f"forearm_{side}"] = ua @ T((0, 0, -dm["upper_arm"]), Ry(-e))
        th = pelvis @ T((0, sg * dm["hip_w"] / 2, 0), Ry(-h))
        sh = th @ T((0, 0, -dm["thigh"]), Ry(k))
        parts[f"thigh_{side}"], parts[f"shin_{side}"] = th, sh
        parts[f"foot_{side}"] = sh @ T((0, 0, -dm["shin"]), Ry(-(k - h) * 0.85))
    # keep the lower foot on the ground: shift everything so the lowest ankle sits at ankle height
    ankles = [parts[f"foot_{s}"][2, 3] for s in "lr"]
    dz = dm["ankle"] - min(ankles)
    for m in parts.values():
        m[2, 3] += dz
    if pose_id == 2:   # the phone sits between the two hands, upright, screen toward the body
        hands = [parts[f"forearm_{s}"] @ np.array([0, 0, -(dm["forearm"] + dm["hand"] * 0.5), 1]) for s in "lr"]
        mid = (hands[0][:3] + hands[1][:3]) / 2
        parts["phone"] = T(mid)
    return parts


def pedestrian(spec, root, state):
    dm = human_dims(spec["asset"].get("height_m", 1.75))
    n = len(root)
    out = {}
    for t in range(n):
        for name, m in human_frame(dm, int(state["pose_id"][t]), float(state["walk_phase"][t])).items():
            out.setdefault(name, np.empty((n, 4, 4)))[t] = root[t] @ m
    return out


# --- robots (MJCF) ---

_robots = {}


def robot(spec, root, state):
    """One part per MJCF body (link0..link7, hand, fingers) plus "tcp", from joint angles q1..q7 and the finger opening."""
    import os
    from .mjcf import ROBOTS, Robot
    model = spec["asset"]["model"]
    path, tcp_body, tcp_off, _ = ROBOTS[model]
    if model not in _robots:
        _robots[model] = Robot.load(os.path.join(os.environ["S2R_ROOT"], path))
    rob = _robots[model]
    n_arm = len(rob.arm_joints())
    n = len(root)
    out = {b.name: np.empty((n, 4, 4)) for b in rob.bodies}
    out["tcp"] = np.empty((n, 4, 4))
    for t in range(n):
        q = [state[f"q{i + 1}"][t] for i in range(n_arm)] + [state["finger"][t]] * (len(rob.joints) - n_arm)
        fr = rob.fk(np.asarray(q))
        for name, M in fr.items():
            out[name][t] = root[t] @ M
        out["tcp"][t] = root[t] @ fr[tcp_body] @ T(tcp_off)
    return out


# --- generic ---

def rigid(spec, root, state):
    return {"body": np.asarray(root)}


KINDS = {"car": car, "pedestrian": pedestrian, "robot": robot, "rigid": rigid}


def parts_for(spec, root, state):
    kind = spec["asset"]["type"]
    if kind in KINDS:
        return KINDS[kind](spec, root, state)
    return rigid(spec, root, state)
