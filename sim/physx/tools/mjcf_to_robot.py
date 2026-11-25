"""Convert a MuJoCo Menagerie arm (MJCF) into a p2s robot description (JSON) for robot_arms in p2s_sim.

  python sim/physx/tools/mjcf_to_robot.py <panda.xml> <out.json> --tcp-body hand --tcp-offset 0 0 0.1034 \
      --home 0 0 0 -1.57079 0 1.57079 -0.7853

Each link keeps its MJCF frame, joint, mass and inertia origin. Collision meshes become capsules fitted by PCA
(the axis along the mesh's longest principal direction, radius covering every vertex), boxes stay boxes. Visual
meshes are listed for the renderer. Only numpy is needed.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np


def vec(s, default):
    return [float(x) for x in s.split()] if s else list(default)


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return [q[1], q[2], q[3], q[0]]


def quat_from_matrix(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


def mesh_vertices(path):
    """Vertices of an OBJ (text) or binary STL file."""
    if path.lower().endswith(".stl"):
        data = open(path, "rb").read()
        n = int.from_bytes(data[80:84], "little")
        tris = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return tris["v"].reshape(-1, 3).astype(float)
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                out.append([float(v) for v in line.split()[1:4]])
    return np.asarray(out)


def capsule(points):
    """Capsule along the principal axis enclosing every point: (centre, quat_xyzw, radius, half_length)."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    axis = vt[0]
    t = (points - c) @ axis
    radial = np.linalg.norm((points - c) - np.outer(t, axis), axis=1)
    r = float(radial.max())
    lo, hi = float(t.min()) + r, float(t.max()) - r
    if hi < lo:  # shorter than it is wide: a sphere-like capsule
        lo = hi = (t.min() + t.max()) / 2
    centre = c + axis * (lo + hi) / 2
    z = axis / np.linalg.norm(axis)
    x = np.cross([0, 1, 0], z) if abs(z[1]) < 0.9 else np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return centre.tolist(), quat_from_matrix(np.column_stack([x, y, z])), r, max((hi - lo) / 2, 1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mjcf")
    ap.add_argument("out")
    ap.add_argument("--tcp-body", default="hand")
    ap.add_argument("--tcp-offset", type=float, nargs=3, default=[0, 0, 0.1034])
    ap.add_argument("--home", type=float, nargs="+", default=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    a = ap.parse_args()

    root = ET.parse(a.mjcf).getroot()
    meshdir = os.path.join(os.path.dirname(a.mjcf), root.find("compiler").get("meshdir", "."))
    meshes = {}
    for m in root.find("asset").iter("mesh"):
        f = m.get("file")
        meshes[m.get("name") or os.path.splitext(f)[0]] = f
    materials = {m.get("name"): vec(m.get("rgba"), [1, 1, 1, 1]) for m in root.iter("material")}
    links = []

    def walk(el, parent):
        for b in el.findall("body"):
            link = {"name": b.get("name"), "parent": parent, "pos": vec(b.get("pos"), [0, 0, 0]),
                    "quat_xyzw": quat_wxyz_to_xyzw(vec(b.get("quat"), [1, 0, 0, 0])), "shapes": [], "visuals": []}
            j = b.find("joint")
            if j is not None:
                finger = j.get("class") == "finger"
                link["joint"] = {"name": j.get("name"), "type": "prismatic" if (j.get("type") == "slide" or finger) else "revolute",
                                 "axis": vec(j.get("axis"), [0, 1, 0] if finger else [0, 0, 1]),
                                 "range": vec(j.get("range"), [0, 0.04] if finger else [-2.8973, 2.8973])}
            inertial = b.find("inertial")
            if inertial is not None:
                link["mass_kg"] = float(inertial.get("mass"))
                link["com"] = vec(inertial.get("pos"), [0, 0, 0])
            for g in b.findall("geom"):
                cls = g.get("class") or ""
                if cls == "visual" and g.get("mesh"):
                    link["visuals"].append({"mesh": os.path.relpath(os.path.join(meshdir, meshes[g.get("mesh")]),
                                                                    os.path.dirname(os.path.dirname(a.mjcf))),
                                            "rgba": materials.get(g.get("material"), [1, 1, 1, 1])})
                elif cls == "collision" and g.get("mesh"):
                    pts = mesh_vertices(os.path.join(meshdir, meshes[g.get("mesh")]))
                    centre, q, r, hl = capsule(pts)
                    link["shapes"].append({"type": "capsule", "radius": round(r, 5), "half_length": round(hl, 5),
                                           "pos": [round(v, 5) for v in centre], "quat_xyzw": [round(v, 6) for v in q]})
                elif cls.startswith("fingertip_pad") or g.get("type") == "box":
                    continue  # millimetre pads: the finger capsules cover them
            links.append(link)
            walk(b, b.get("name"))

    walk(root.find("worldbody"), None)
    out = {"source": os.path.relpath(a.mjcf, os.path.dirname(os.path.dirname(a.mjcf))), "tcp": {"link": a.tcp_body, "offset": a.tcp_offset},
           "home": a.home, "links": links}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {len(links)} links, {sum(len(l['shapes']) for l in links)} capsules")


if __name__ == "__main__":
    main()
