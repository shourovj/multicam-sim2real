"""Poly Haven glTF models (inside Blender), from $S2R_DATA/assets/polyhaven/<id>/<id>_1k.gltf."""

import glob
import os

import bpy

from .materials import ASSETS


def import_model(asset_id, name=None):
    """Import one model; returns an empty at the origin with every imported object parented to it."""
    path = sorted(glob.glob(os.path.join(ASSETS, asset_id, "*.gltf")))
    assert path, f"model {asset_id} not found under {ASSETS}; run setup/fetch_assets.py"
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path[0])
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(name or asset_id, None)
    bpy.context.scene.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    return root, [o for o in new if o.type == "MESH"]
