"""A robot arm that carries a held object along keyframes (sim/physics/arm.py). Kinematic: no forces, no contact.

The keyframes say where a point on the held object should be (e.g. a pitcher's lip), which way the object
faces (yaw) and how far it is tilted about its own Y axis (tilt, spout down). From that the arm works out
where its tool centre point (TCP, between the fingertips) must be, and solves inverse kinematics for the
7 arm joints every frame: damped least squares on position + orientation, warm-started from the previous
frame, joint limits respected, with a weak pull toward the robot's home posture so the elbow stays natural.

motion block:
    model: arm
    robot: franka_panda                       sim/world/mjcf.py ROBOTS
    base: {pos: [x, y, z], yaw_deg: 0}
    hold:                                     the object in the gripper (its own motion is `attached` to this arm)
      object: pitcher
      offset: {pos: [0.08, 0, 0], rpy_deg: [0, -90, 0]}   the object's frame in the TCP frame
      gripper_m: 0.035                        finger opening (each finger, metres)
    point_local: [0.047, 0, 0.16]             the keyframed point, in the held object's frame
    keys:                                     eased between keys (smoothstep)
      - {t: 0.0, point: [x, y, z], yaw_deg: -90, tilt_deg: 0}
      ...

State per frame: q1..q7 (rad), finger (m), ik_error_m, ik_error_deg. Parts (kinematics.robot): one per link + "tcp".
"""

import os

import numpy as np

from world.kinematics import T, Ry, Rz, rpy
from world.mjcf import ROBOTS, Robot


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _rotvec(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-4:                                   # near 180 deg: take the axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        return ax * a
    return v / (2 * np.sin(a)) * a


def load_robot(model):
    path, tcp_body, tcp_off, home = ROBOTS[model]
    rob = Robot.load(os.path.join(os.environ["S2R_ROOT"], path))
    return rob, tcp_body, np.asarray(tcp_off), np.asarray(home, float)


class Arm:
    def __init__(self, spec, world, index):
        m = spec["motion"]
        self.id = spec["id"]
        self.robot, self.tcp_body, self.tcp_off, self.home = load_robot(m.get("robot", spec["asset"]["model"]))
        b = m["base"]
        self.base = T(b["pos"], Rz(np.deg2rad(b.get("yaw_deg", 0.0))))
        h = m["hold"]
        self.held = h["object"]
        self.hold_off = T(h["offset"]["pos"], rpy(*h["offset"].get("rpy_deg", [0, 0, 0])))
        self.finger = float(h.get("gripper_m", 0.04))
        self.point_local = np.asarray(m["point_local"], float)
        self.keys = m["keys"]
        self.arm = self.robot.arm_joints()
        self.lo = np.array([j[3][0] for j in self.arm])
        self.hi = np.array([j[3][1] for j in self.arm])
        self.q = self.home[:len(self.arm)].copy()
        self.t, self._solved_t, self.err = 0.0, None, (0.0, 0.0)

    # --- the target ---
    def object_frame(self, t):
        """World pose of the held object at time t, from the keys."""
        ks = self.keys
        i = int(np.clip(np.searchsorted([k["t"] for k in ks], t, side="right") - 1, 0, len(ks) - 2))
        a, b = ks[i], ks[i + 1]
        u = _smooth((t - a["t"]) / (b["t"] - a["t"]))
        lerp = lambda key: (1 - u) * np.asarray(a[key], float) + u * np.asarray(b[key], float)
        R = Rz(np.deg2rad(lerp("yaw_deg"))) @ Ry(np.deg2rad(lerp("tilt_deg")))
        p = lerp("point") - R @ self.point_local
        return T(p, R)

    def tcp_target(self, t):
        return self.object_frame(t) @ np.linalg.inv(self.hold_off)

    # --- kinematics ---
    def _full_q(self, q):
        return np.concatenate([q, [self.finger, self.finger]])

    def tcp(self, q):
        fr = self.robot.fk(self._full_q(q))
        return self.base @ fr[self.tcp_body] @ T(self.tcp_off), fr

    def _jacobian(self, q):
        """Geometric Jacobian [6, 7] of the TCP in world coordinates (hinge joints)."""
        tcp, fr = self.tcp(q)
        p = tcp[:3, 3]
        J = np.zeros((6, len(q)))
        k = 0
        for body in self.robot.bodies:
            if body.joint is None or body.joint[1] != "hinge":
                continue
            M = self.base @ fr[body.name]                     # joint frame (the joint rotates the body about its axis)
            a = M[:3, :3] @ body.joint[2]
            J[:3, k] = np.cross(a, p - M[:3, 3])
            J[3:, k] = a
            k += 1
        return J, tcp

    def solve(self, target, iters=60):
        q = self.q.copy()
        for _ in range(iters):
            J, tcp = self._jacobian(q)
            e = np.concatenate([target[:3, 3] - tcp[:3, 3], _rotvec(target[:3, :3] @ tcp[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            lam = 0.05
            JJt = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(len(q)) - np.linalg.pinv(J) @ J           # pull toward home inside the null space
            dq += N @ (0.05 * (self.home[:len(q)] - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + 1e-3, self.hi - 1e-3)
        tcp, _ = self.tcp(q)
        self.err = (float(np.linalg.norm(target[:3, 3] - tcp[:3, 3])), float(np.rad2deg(np.linalg.norm(_rotvec(target[:3, :3] @ tcp[:3, :3].T)))))
        return q

    # --- model interface (sim/world/simulate.py) ---
    def step(self, t, dt, world):
        self.t = t + dt

    def _ensure(self):
        if self._solved_t != self.t:
            self.q = self.solve(self.tcp_target(self.t), iters=200 if self._solved_t is None else 60)
            self._solved_t = self.t

    def pose(self):
        self._ensure()
        return self.base[:3, 3], self.base[:3, :3]

    def state(self):
        self._ensure()
        s = {f"q{i + 1}": float(v) for i, v in enumerate(self.q)}
        s.update(finger=self.finger, ik_error_m=self.err[0], ik_error_deg=self.err[1])
        return s
