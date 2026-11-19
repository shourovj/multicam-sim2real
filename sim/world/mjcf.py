"""A small MJCF reader for robot arms (pure Python + numpy): the body tree, joints and visual meshes of a
MuJoCo Menagerie model, enough for forward kinematics, inverse kinematics and drawing it in Blender.
No MuJoCo needed. Tested on franka_emika_panda/panda.xml; other Menagerie arms use the same elements.

  rob = Robot.load("$S2R_ROOT/mujoco_menagerie/franka_emika_panda/panda.xml")
  frames = rob.fk(q)               # {body name: 4x4 in the robot base frame}
  tcp = frames["hand"] @ T((0, 0, 0.1034))
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

ROBOTS = {   # spec asset.model -> (MJCF path under $S2R_ROOT, TCP body, TCP offset, home qpos)
    "franka_panda": ("mujoco_menagerie/franka_emika_panda/panda.xml", "hand", (0.0, 0.0, 0.1034),
                     [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]),
}


def _vec(s, default):
    return np.asarray([float(x) for x in s.split()], float) if s else np.asarray(default, float)


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, a):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class Body:
    def __init__(self, name, parent, pos, R):
        self.name, self.parent, self.local = name, parent, np.eye(4)
        self.local[:3, :3], self.local[:3, 3] = R, pos
        self.joint = None          # (name, type 'hinge'|'slide', axis [3], range [2])
        self.visuals = []          # (mesh file, rgba)


class Robot:
    def __init__(self, bodies, joints, meshdir):
        self.bodies, self.joints, self.meshdir = bodies, joints, meshdir

    @classmethod
    def load(cls, path):
        root = ET.parse(path).getroot()
        meshdir = os.path.join(os.path.dirname(path), root.find("compiler").get("meshdir", "."))
        materials = {m.get("name"): _vec(m.get("rgba"), [1, 1, 1, 1]) for m in root.iter("material")}
        meshes = {}
        for m in root.find("asset").iter("mesh"):
            f = m.get("file")
            meshes[m.get("name") or os.path.splitext(f)[0]] = f
        bodies, joints = [], []

        def walk(el, parent):
            for b in el.findall("body"):
                body = Body(b.get("name"), parent, _vec(b.get("pos"), [0, 0, 0]), quat_to_R(_vec(b.get("quat"), [1, 0, 0, 0])))
                j = b.find("joint")
                if j is not None:
                    finger = j.get("class") == "finger"
                    typ = j.get("type") or ("slide" if finger else "hinge")
                    axis = _vec(j.get("axis"), [0, 1, 0] if finger else [0, 0, 1])
                    rng = _vec(j.get("range"), [0, 0.04] if finger else [-2.8973, 2.8973])
                    body.joint = (j.get("name"), typ, axis, rng)
                    joints.append(body.joint)
                for g in b.findall("geom"):
                    if g.get("class") == "visual" and g.get("mesh"):
                        body.visuals.append((meshes[g.get("mesh")], materials.get(g.get("material"), np.array([1, 1, 1, 1.0]))))
                bodies.append(body)
                walk(b, body.name)

        walk(root.find("worldbody"), None)
        return cls(bodies, joints, meshdir)

    def fk(self, q):
        """q: one value per joint, in file order (Panda: 7 arm joints, then 2 fingers). Returns {body: 4x4}."""
        out, k = {}, 0
        for b in self.bodies:
            M = (out[b.parent] if b.parent else np.eye(4)) @ b.local
            if b.joint is not None:
                _, typ, axis, _ = b.joint
                J = np.eye(4)
                if typ == "hinge":
                    J[:3, :3] = axis_angle(axis, q[k])
                else:
                    J[:3, 3] = axis * q[k]
                M = M @ J
                k += 1
            out[b.name] = M
        return out

    def arm_joints(self):
        return [j for j in self.joints if j[1] == "hinge"]
