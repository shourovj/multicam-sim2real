"""Materials (inside Blender). PBR from a Poly Haven texture folder, or procedural."""

import os

import bpy

ASSETS = os.path.join(os.environ.get("S2R_DATA", ""), "assets", "polyhaven")
_cache = {}


def _bsdf(m):
    m.use_nodes = True
    return m.node_tree, m.node_tree.nodes["Principled BSDF"]


def pbr(tex_id, scale_m=2.0, tint=None, name=None, normal_strength=1.0):
    """Material from $S2R_DATA/assets/polyhaven/<tex_id>/ (diff, nor_gl, rough). Box-projected in object space,
    one texture tile per `scale_m` metres, so it works on planes and boxes without UVs."""
    key = (tex_id, scale_m, tuple(tint) if tint else None, name)
    if key in _cache:
        return _cache[key]
    folder = os.path.join(ASSETS, tex_id)
    files = {k: next((f for f in os.listdir(folder) if f"_{k}_" in f), None) for k in ("diff", "nor_gl", "rough")}
    m = bpy.data.materials.new(name or f"pbr_{tex_id}")
    nt, bsdf = _bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale_m,) * 3
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def img(kind, colorspace):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(folder, files[kind]), check_existing=True)
        n.image.colorspace_settings.name = colorspace
        n.projection = "BOX"
        n.projection_blend = 0.2
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img("diff", "sRGB")
    if tint is not None:
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color2"].default_value = (*tint[:3], 1)
        nt.links.new(d.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
    if files["rough"]:
        nt.links.new(img("rough", "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
    if files["nor_gl"]:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(img("nor_gl", "Non-Color").outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    _cache[key] = m
    return m


def plain(name, color, roughness=0.5, metallic=0.0, coat=0.0, specular=0.5, noise=0.0, noise_scale=30.0, bump=0.0):
    """Principled material with optional colour noise and bump, for paint, plastic, rubber, cloth."""
    m = bpy.data.materials.new(name)
    nt, bsdf = _bsdf(m)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.03
    bsdf.inputs["Specular IOR Level"].default_value = specular
    base = (*color[:3], 1.0)
    if noise <= 0 and bump <= 0:
        bsdf.inputs["Base Color"].default_value = base
        return m
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = noise_scale
    tex.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[c * (1 - noise) for c in color[:3]], 1)
    ramp.color_ramp.elements[1].color = (*[min(1.0, c * (1 + noise)) for c in color[:3]], 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if bump > 0:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = 0.01
        nt.links.new(tex.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def glass(name, color=(0.9, 0.95, 0.95), roughness=0.02, ior=1.5, tint_dark=0.0):
    """Clear or tinted glass. tint_dark > 0 makes it a darker car-window glass (less see-through)."""
    m = bpy.data.materials.new(name)
    nt, bsdf = _bsdf(m)
    bsdf.inputs["Base Color"].default_value = (*[c * (1 - tint_dark) for c in color], 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return m


def emissive(name, color, strength=0.0, base=(0.05, 0.05, 0.05)):
    """A lamp lens: dark when strength is 0, glowing in `color` when lit. Emission Strength can be keyframed."""
    m = bpy.data.materials.new(name)
    nt, bsdf = _bsdf(m)
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Emission Color"].default_value = (*color, 1)
    bsdf.inputs["Emission Strength"].default_value = strength
    return m


def keyframe_emission(m, frame, strength):
    sock = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    sock.default_value = strength
    sock.keyframe_insert("default_value", frame=frame)
