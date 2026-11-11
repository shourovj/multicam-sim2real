"""Procedural assets built from Blender primitives (plan §1, sim/assets.py). Runs inside Blender.

No downloads: the quadcopter is a body, 4 arms, 4 motors and 4 rotor discs joined into one mesh;
the background is a Nishita sky with procedural cloud wisps, a noise-textured ground plane and a
tree line of squashed icospheres near the horizon.

Overhead scenes use an aerial photo as the ground (build_image_ground), box-built ground vehicles
(build_tank, build_truck) and two occluders above the road: a bridge and a canopy patch, which is a
strip of the same photo lifted to tree height so the trees in the photo hide what drives under them.
"""

import math

import bpy
import numpy as np


def _material(name, color, roughness=0.5, metallic=0.0, noise=0.0, noise_scale=40.0, bump=0.0):
    """Principled material; `noise` > 0 modulates the base colour by a noise texture for some grain."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    base = (*color[:3], 1.0)
    if noise <= 0:
        bsdf.inputs["Base Color"].default_value = base
        return m
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = noise_scale
    tex.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    lo = [c * (1 - noise) for c in color[:3]]
    hi = [min(1.0, c * (1 + noise)) for c in color[:3]]
    ramp.color_ramp.elements[0].color = (*lo, 1)
    ramp.color_ramp.elements[1].color = (*hi, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if bump > 0:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(tex.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _add(kind, location, scale, rotation=(0, 0, 0), material=None, **kw):
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    elif kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=kw.get("vertices", 24), radius=1.0, depth=1.0,
                                            location=location, rotation=rotation)
    elif kind == "ico":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=kw.get("subdivisions", 2), radius=1.0,
                                              location=location, rotation=rotation)
    ob = bpy.context.active_object
    ob.scale = scale
    if material is not None:
        ob.data.materials.append(material)
    return ob


def build_quadcopter(name, arm_span_m=0.62, body_color=(0.8, 0.3, 0.05)):
    """One joined mesh object, origin at the body centre, X/Y = arm diagonals' plane, Z up."""
    body_m = _material(f"{name}_body", body_color, roughness=0.35, noise=0.12, noise_scale=60, bump=0.05)
    carbon = _material(f"{name}_carbon", (0.03, 0.03, 0.035), roughness=0.45, noise=0.4, noise_scale=120)
    motor_m = _material(f"{name}_motor", (0.55, 0.56, 0.6), roughness=0.25, metallic=0.9)
    rotor_m = _material(f"{name}_rotor", (0.12, 0.12, 0.13), roughness=0.6, noise=0.3, noise_scale=25)
    led = _material(f"{name}_led", (0.1, 0.9, 0.2), roughness=0.3)

    r = arm_span_m / 2
    parts = [
        _add("cube", (0, 0, 0), (0.20, 0.13, 0.065), material=body_m),          # main body
        _add("cube", (0.0, 0, 0.045), (0.15, 0.09, 0.03), material=carbon),      # top plate / battery
        _add("cube", (0.105, 0, -0.005), (0.02, 0.05, 0.03), material=led),      # nose LED / camera
    ]
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        cx, cy = r * math.cos(a), r * math.sin(a)
        # arm: cylinder along its local Z, rotated to lie along the diagonal
        parts.append(_add("cyl", (cx / 2, cy / 2, 0), (0.014, 0.014, r),
                          rotation=(math.pi / 2, 0, a + math.pi / 2), material=carbon, vertices=12))
        parts.append(_add("cyl", (cx, cy, 0.025), (0.025, 0.025, 0.05), material=motor_m, vertices=16))
        parts.append(_add("cyl", (cx, cy, 0.055), (0.13, 0.13, 0.006), material=rotor_m, vertices=32))
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.active_object
    ob.name = name
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return ob


def build_tree(name, position, trunk_height_m=2.0, crown_radius_m=0.7):
    """A small foreground tree used as an occluder: a trunk and a crown of a few merged icospheres.
    `position` is the base of the trunk on the ground (x, y)."""
    x, y = position[:2]
    bark = _material(f"{name}_bark", (0.16, 0.10, 0.06), roughness=0.9, noise=0.3, noise_scale=30, bump=0.3)
    leaf = _material(f"{name}_leaf", (0.07, 0.16, 0.05), roughness=0.9, noise=0.4, noise_scale=6)
    r, h = crown_radius_m, trunk_height_m
    parts = [_add("cyl", (x, y, h / 2), (0.09 * r / 0.7, 0.09 * r / 0.7, h), material=bark, vertices=12)]
    for dx, dy, dz, k in [(0, 0, 0.25, 1.0), (0.45, 0.1, 0.0, 0.7), (-0.45, -0.1, 0.05, 0.7), (0.05, 0.1, 0.7, 0.65)]:
        parts.append(_add("ico", (x + dx * r, y + dy * r, h + dz * r), (k * r, k * r, k * r * 0.9), material=leaf,
                          subdivisions=3))
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.active_object
    ob.name = name
    return ob


def sample_surface_points(ob, n, seed=0):
    """n points sampled area-weighted on the mesh surface, in object-local coordinates [n, 3]."""
    me = ob.data
    me.calc_loop_triangles()
    tris = np.array([[me.vertices[i].co[:] for i in t.vertices] for t in me.loop_triangles])
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def build_world(bg):
    """Nishita sky with cloud wisps; returns nothing. bg: the spec's `background` dict."""
    sky = bg["sky"]
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg_node = nt.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Strength"].default_value = sky["strength"]
    sk = nt.nodes.new("ShaderNodeTexSky")
    sk.sky_type = "NISHITA"
    sk.sun_elevation = math.radians(sky["sun_elevation_deg"])
    sk.sun_rotation = math.radians(sky["sun_rotation_deg"])
    sk.sun_disc = False
    sk.air_density = 1.2
    sk.dust_density = 1.5

    # Clouds: noise on the view direction, stretched horizontally, faded out toward the horizon.
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 3.5)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.2
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[1].position = 0.75
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    horizon = nt.nodes.new("ShaderNodeMapRange")       # cloud amount by elevation (z of direction)
    horizon.inputs["From Min"].default_value = 0.02
    horizon.inputs["From Max"].default_value = 0.25
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = 0.8
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color2"].default_value = (2.2, 2.2, 2.3, 1.0)   # cloud colour in sky radiance units
    L = nt.links.new
    L(coord.outputs["Generated"], mapping.inputs["Vector"])
    L(mapping.outputs["Vector"], noise.inputs["Vector"])
    L(noise.outputs["Fac"], ramp.inputs["Fac"])
    L(coord.outputs["Generated"], sep.inputs["Vector"])
    L(sep.outputs["Z"], horizon.inputs["Value"])
    L(ramp.outputs["Color"], mul.inputs[0])
    L(horizon.outputs["Result"], mul.inputs[1])
    L(mul.outputs["Value"], mul2.inputs[0])
    L(mul2.outputs["Value"], mix.inputs["Fac"])
    L(sk.outputs["Color"], mix.inputs["Color1"])
    L(mix.outputs["Color"], bg_node.inputs["Color"])
    L(bg_node.outputs["Background"], out.inputs["Surface"])


def build_ground(bg, seed=0):
    """Ground plane + a tree line of dark squashed icospheres 250-600 m out. Returns the objects."""
    g = bg["ground"]
    c0, c1 = g["colors"]
    m = bpy.data.materials.new("ground")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    tc = nt.nodes.new("ShaderNodeTexCoord")                # object coords, so scale is per metre
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = g["noise_scale"]
    noise.inputs["Detail"].default_value = 15.0
    noise.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c0, 1)
    ramp.color_ramp.elements[1].color = (*c1, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bpy.ops.mesh.primitive_plane_add(size=g["size_m"], location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(m)

    tree_m = _material("trees", (0.05, 0.09, 0.04), roughness=0.9, noise=0.35, noise_scale=3)
    rng = np.random.default_rng(seed + 1)
    trees = []
    for _ in range(500):
        ang = rng.uniform(-1.1, 1.1)                     # radians around +Y, wider than the 50 deg FOV
        dist = rng.uniform(250, 600)
        x, y = dist * math.sin(ang), dist * math.cos(ang)
        s = rng.uniform(3, 6)
        trees.append(_add("ico", (x, y, s * 0.5), (s, s, s * rng.uniform(0.7, 1.3)), material=tree_m,
                          subdivisions=2))
    bpy.ops.object.select_all(action="DESELECT")
    for t in trees:
        t.select_set(True)
    bpy.context.view_layer.objects.active = trees[0]
    bpy.ops.object.join()
    treeline = bpy.context.active_object
    treeline.name = "treeline"
    return [ground, treeline]


def _join(parts, name):
    """Join parts into one mesh object named `name`, with its origin at the world origin."""
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.active_object
    ob.name = name
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    return ob


def build_tank(name, color=(0.20, 0.24, 0.10)):
    """A tank about 6.8 m long and 3.4 m wide, gun forward. Local +X is forward, origin on the ground."""
    hull_m = _material(f"{name}_hull", color, roughness=0.7, noise=0.25, noise_scale=8)
    dark = _material(f"{name}_track", (0.05, 0.05, 0.045), roughness=0.9, noise=0.3, noise_scale=30)
    top = _material(f"{name}_turret", [c * 0.85 for c in color], roughness=0.7, noise=0.25, noise_scale=8)
    parts = [
        _add("cube", (0, 0, 0.95), (6.6, 2.5, 0.7), material=hull_m),                      # hull
        _add("cube", (0, 1.4, 0.5), (6.8, 0.6, 1.0), material=dark),                       # tracks
        _add("cube", (0, -1.4, 0.5), (6.8, 0.6, 1.0), material=dark),
        _add("cube", (0, 1.4, 1.03), (6.9, 0.7, 0.08), material=hull_m),                   # track guards
        _add("cube", (0, -1.4, 1.03), (6.9, 0.7, 0.08), material=hull_m),
        _add("cyl", (-0.3, 0, 1.6), (1.5, 1.25, 0.6), material=top, vertices=24),          # turret
        _add("cyl", (2.9, 0, 1.65), (0.09, 0.09, 4.0), rotation=(0, math.pi / 2, 0), material=dark, vertices=12),  # gun
        _add("cyl", (-0.7, 0.45, 1.95), (0.3, 0.3, 0.12), material=hull_m, vertices=16),   # hatch
    ]
    return _join(parts, name)


def build_truck(name, color=(0.20, 0.24, 0.10)):
    """A covered cargo truck about 7 m long and 2.5 m wide. Local +X is forward, origin on the ground."""
    body = _material(f"{name}_body", color, roughness=0.6, noise=0.2, noise_scale=8)
    canvas = _material(f"{name}_canvas", [min(1.0, c * 1.35) for c in color], roughness=0.95, noise=0.3, noise_scale=5, bump=0.2)
    dark = _material(f"{name}_dark", (0.04, 0.04, 0.04), roughness=0.8)
    glass = _material(f"{name}_glass", (0.05, 0.07, 0.09), roughness=0.1)
    parts = [
        _add("cube", (0, 0, 0.75), (6.9, 2.2, 0.3), material=dark),                        # chassis
        _add("cube", (2.45, 0, 1.75), (1.9, 2.4, 1.7), material=body),                     # cab
        _add("cube", (3.42, 0, 2.05), (0.04, 2.0, 0.7), material=glass),                   # windscreen
        _add("cube", (-1.05, 0, 1.3), (4.8, 2.5, 0.8), material=body),                     # cargo bed
        _add("cube", (-1.05, 0, 2.35), (4.8, 2.5, 1.3), material=canvas),                  # canvas cover
    ]
    for x in (2.4, -1.4, -2.6):
        for y in (1.1, -1.1):
            parts.append(_add("cyl", (x, y, 0.5), (0.5, 0.5, 0.35), rotation=(math.pi / 2, 0, 0), material=dark, vertices=20))
    return _join(parts, name)


def build_flat_world(strength):
    """Uniform white ambient light, for overhead scenes where the sky is never in view."""
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def build_image_ground(image_path, metres_per_pixel):
    """A ground plane showing an aerial photo: image centre at the world origin, +X east (image right),
    +Y north (image up). Returns (object, (width_m, height_m), (width_px, height_px))."""
    img = bpy.data.images.load(image_path)
    wp, hp = img.size
    wm, hm = wp * metres_per_pixel, hp * metres_per_pixel
    m = bpy.data.materials.new("aerial_photo")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    for k in ("Specular IOR Level", "Specular"):
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = 0.0
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.scale = (wm, hm, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ground.data.materials.append(m)
    return ground, (wm, hm), (wp, hp)


def build_bridge(name, centre_xy, heading_rad, width_m, length_m, height_m):
    """A flat concrete bridge over the road. `width_m` is its size along the road, `length_m` across it,
    `height_m` the underside clearance. `heading_rad` is the road direction at the crossing."""
    conc = _material(f"{name}_concrete", (0.26, 0.25, 0.24), roughness=0.9, noise=0.15, noise_scale=3, bump=0.1)
    rail = _material(f"{name}_rail", (0.15, 0.15, 0.15), roughness=0.8)
    parts = [_add("cube", (0, 0, height_m + 0.3), (width_m, length_m, 0.6), material=conc)]              # deck
    for sx in (1, -1):
        parts.append(_add("cube", (sx * (width_m / 2 - 0.15), 0, height_m + 1.0), (0.3, length_m, 0.8), material=rail))
    for sy in (1, -1):                                                                                     # piers
        parts.append(_add("cube", (0, sy * (length_m / 2 - 0.6), height_m / 2), (width_m * 0.7, 1.2, height_m), material=conc))
    ob = _join(parts, name)
    ob.location = (centre_xy[0], centre_xy[1], 0.0)
    ob.rotation_euler = (0, 0, heading_rad)
    return ob


def build_canopy_patch(name, left_xy, right_xy, height_m, cam_xyz, ground_size_m, material):
    """A strip of the ground photo lifted to `height_m`, between the ground polylines left_xy / right_xy
    [N, 2]. Each vertex is moved along its ray toward the overhead camera and keeps the photo
    coordinate of its ground point, so from the camera the strip looks exactly like the photo beneath
    it, while anything under it is hidden and the depth sees a surface at tree height. Casts no shadow."""
    cam = np.asarray(cam_xyz, dtype=float)
    k = (cam[2] - height_m) / cam[2]
    wm, hm = ground_size_m
    g = np.concatenate([np.asarray(left_xy, dtype=float), np.asarray(right_xy, dtype=float)])
    n = len(left_xy)
    verts = [(cam[0] + (x - cam[0]) * k, cam[1] + (y - cam[1]) * k, height_m) for x, y in g]
    faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        x, y = g[loop.vertex_index]
        uv.data[loop.index].uv = (x / wm + 0.5, y / hm + 0.5)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(material)
    ob.visible_shadow = False
    return ob
