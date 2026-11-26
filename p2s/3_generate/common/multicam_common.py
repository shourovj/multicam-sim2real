"""Shared by everything in gen/ and by legacy/eval/score_multicam.py: the job list, the folder layout, the model families,
prompt composition and the input recipe whose hashes decide whether a finished run is still current. Stdlib + yaml,
so it imports in every env (eval, Cosmos, VideoX-Fun, LTX).

Layout under $S2R_DATA/<out_root>/<scene>/r<rev>/:
  world/                       simulate_world.py, or for a prompt-to-sim scene p2s/1_scene/write_scene.py (PhysX)
  cams/<cam>/bundle/           render_world.py + export_world.py, or p2s/2_render/render_x.py + export_x.py (rgb.npy, instance.npy, ...)
  cut/<cut>/                   rgb.npy, instance.npy, depth.npy, rgb.mp4, schedule.json   (multicam_inputs.py)
  inputs/<view>/               prompt.json, control_<cond>.mp4                              (multicam_inputs.py)
  gen/<view>/<cond>/<model>/seed<k>/   request.json, prompt.json or prompt.txt, out_video.mp4, run.log, inputs.json, run.json
Several experiments (one per model) can share an out_root: they then share the world, the bundles and the cuts, and
their runs sit side by side under gen/<view>/<cond>/<model name>/.
"""

import copy
import hashlib
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GEN = os.path.join(REPO, "p2s", "3_generate")      # this stage's folder
_sha_cache = {}

# model.family in an experiment file -> the folder under gen/ that runs it (its run.sh does one job) and the prompt
# format the model takes: "caption" = Cosmos3's structured JSON caption, "prose" = one plain paragraph.
# An experiment file with no model.family is Cosmos3-Nano (the files written before families existed).
FAMILIES = {
    "cosmos3-nano":   {"dir": "cosmos3_nano",         "prompt": "caption"},
    "sim-transfer":   {"dir": "cosmos3_sim_transfer", "prompt": "caption"},
    "wan22-vace-fun": {"dir": "wan22_vace_fun",       "prompt": "prose"},
    "ltx25":          {"dir": "ltx25",                "prompt": "prose"},
}


def family(exp):
    f = exp["model"].get("family", "cosmos3-nano")
    assert f in FAMILIES, f"unknown model.family {f!r}; known: {sorted(FAMILIES)}"
    return f


def run_script(exp):
    return os.path.join(GEN, "models", FAMILIES[family(exp)]["dir"], "run.sh")


def load_exp(path):
    return yaml.safe_load(open(path))


def sha256_file(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def scene_root(exp, scene):
    return os.path.join(os.environ["S2R_DATA"], exp["out_root"], scene, f"r{exp['scenes'][scene]['rev']}")


def scene_spec_path(exp, scene):
    """The scene's description: a spec v2 file in the repo (`spec:`), or, for a prompt-to-sim scene (`p2s:` names its
    prompt file), the spec.yaml that p2s/1_scene/write_scene.py wrote next to the accepted world."""
    s = exp["scenes"][scene]
    if "spec" in s:
        return os.path.join(REPO, s["spec"])
    p = os.path.join(scene_root(exp, scene), "world", "spec.yaml")
    assert os.path.exists(p), f"{scene}: no {p}. Build the world first: bash p2s/build_scene.sh {s.get('p2s', 'p2s/prompts/<name>.yaml')}"
    return p


def scene_spec(exp, scene):
    return yaml.safe_load(open(scene_spec_path(exp, scene)))


def world_file(exp, scene):
    """The simulated world a run depends on: world.npz (legacy/sim/simulate_world.py) or a prompt-to-sim scene's x.json."""
    return os.path.join(scene_root(exp, scene), "world", "world.npz" if "spec" in exp["scenes"][scene] else "x.json")


def jobs(exp):
    """Every run: dicts with scene, view, kind (cam | cut), cond (folder name), cond_spec, seed, probe."""
    out = []
    for scene in exp["scenes"]:
        spec = scene_spec(exp, scene)
        views = [(c["id"], "cam") for c in spec["cameras"]] + [(k, "cut") for k in exp.get("cuts", {}).get(scene, {})]
        for seed in sorted({s for c in exp["conditions"] for s in exp.get("seeds_by_condition", {}).get(c, exp["seeds"])}):
            for view, kind in views:
                for cond, cs in exp["conditions"].items():
                    if seed in exp.get("seeds_by_condition", {}).get(cond, exp["seeds"]):   # optional per-condition seeds
                        out.append(dict(scene=scene, view=view, kind=kind, cond=cond, cond_spec=cs, seed=seed, probe=None))
    for p in exp.get("probes", []):
        if "cond_spec" in p:                    # a named condition on a few views only: {name, scene, views, seed, cond, cond_spec}
            for view in p["views"]:
                kind = "cut" if view in exp.get("cuts", {}).get(p["scene"], {}) else "cam"
                out.append(dict(scene=p["scene"], view=view, kind=kind, cond=p["cond"], cond_spec=p["cond_spec"], seed=p["seed"],
                                probe=p["name"]))
            continue
        base = exp["conditions"][p["cond"]]
        kind = "cut" if p["view"] in exp.get("cuts", {}).get(p["scene"], {}) else "cam"
        for key, values in p["vary"].items():
            for v in values:
                cs = copy.deepcopy(base)
                hint = next(iter(cs))
                cs[hint][key] = v
                out.append(dict(scene=p["scene"], view=p["view"], kind=kind, cond=f"{p['cond']}_{hint}-{v}",
                                cond_spec=cs, seed=p["seed"], probe=p["name"]))
    return out


def run_dir(exp, j):
    """gen_subdir (default "gen") lets a smoke-test experiment file write next to, never into, the real runs."""
    return os.path.join(scene_root(exp, j["scene"]), exp.get("gen_subdir", "gen"), j["view"], j["cond"], exp["model"]["name"], f"seed{j['seed']}")


def run_id(exp, j):
    return f"{exp['experiment']}/{j['scene']}/r{exp['scenes'][j['scene']]['rev']}/{j['view']}/{j['cond']}/seed{j['seed']}"


def source_rgb(exp, j):
    r = scene_root(exp, j["scene"])
    return os.path.join(r, "cams", j["view"], "bundle", "rgb.npy") if j["kind"] == "cam" else os.path.join(r, "cut", j["view"], "rgb.npy")


def cut_sources(exp, j):
    r = scene_root(exp, j["scene"])
    return [os.path.join(r, "cams", s["cam"], "bundle", "rgb.npy") for s in exp["cuts"][j["scene"]][j["view"]]]


def control_path(exp, j, hint):
    """A simulator control video (depth, seg, edge, ...) for this view: the camera's bundle, or the joined cut."""
    r = scene_root(exp, j["scene"])
    if j["kind"] == "cam":
        return os.path.join(r, "cams", j["view"], "bundle", "controls", f"{hint}.mp4")
    return os.path.join(r, "cut", j["view"], "controls", f"{hint}.mp4")


def control_sources(exp, j, hint):
    """The files a control is made from: the camera's own control video, or each shot camera's for a cut."""
    r = scene_root(exp, j["scene"])
    if j["kind"] == "cam":
        return [control_path(exp, j, hint)]
    return [os.path.join(r, "cams", s["cam"], "bundle", "controls", f"{hint}.mp4") for s in exp["cuts"][j["scene"]][j["view"]]]


INPUT_CODE_VERSION = 1   # bump when the input code changes what it builds; file hashes are recorded, not compared
NOT_COMPARED = ("code",)


def _code_hashes(exp):
    """sha256 of the shared input code and of every file in the model's folder (provenance only)."""
    d = os.path.join(GEN, "models", FAMILIES[family(exp)]["dir"])
    files = [os.path.join(GEN, "common", f) for f in ("multicam_inputs.py", "multicam_common.py")]
    files += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith((".py", ".sh")))
    if family(exp) == "sim-transfer":                      # its run.sh builds the request with the Cosmos3-Nano folder's code
        files.append(os.path.join(GEN, "models", "cosmos3_nano", "request.py"))
    return {os.path.relpath(f, GEN): sha256_file(f) for f in files}


def _strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def compose_prompt(exp, j):
    """World block + the view's camera (or cut) block + the clip's format fields, as one Cosmos3 structured caption."""
    pdir = os.path.join(REPO, exp["scenes"][j["scene"]]["prompts"])
    world = _strip(json.load(open(os.path.join(pdir, "world.json"))))
    view = _strip(json.load(open(os.path.join(pdir, "cams", f"{j['view']}.json") if j["kind"] == "cam" else os.path.join(pdir, f"{j['view']}.json"))))
    cap = {}
    order = ["temporal_caption", "subjects", "background_setting", "lighting", "aesthetics", "cinematography", "style_medium",
             "artistic_style", "context", "actions", "text_and_signage_elements", "segments", "transitions"]
    for k in order:
        if k == "aesthetics":
            cap[k] = {**world.get(k, {}), **view.get(k, {})}
        elif k in view:
            cap[k] = view[k]
        elif k in world:
            cap[k] = world[k]
    spec = scene_spec(exp, j["scene"])
    W, H = spec["resolution"]
    n, fps = exp["video"]["num_frames"], exp["video"]["fps"]
    secs = int(round((n - 1) / fps))
    cap.update(resolution={"W": W, "H": H}, aspect_ratio=exp["video"]["aspect_ratio"], duration=f"0:{secs:02d}", fps=fps)
    return cap


def prose_prompt(exp, j):
    """The same world and camera blocks as one plain paragraph, for the models that take prose (Wan, LTX). In order:
    what happens in this view (the camera or cut block's temporal_caption, which names every object by its colour),
    the setting, the lighting, the medium, and the world block's realism sentence. Nothing is added or reworded, so a
    prose model and a caption model are told the same things about a view."""
    cap = compose_prompt(exp, j)
    lighting = cap.get("lighting", "")
    parts = [cap["temporal_caption"], cap.get("background_setting", ""),
             *(lighting.values() if isinstance(lighting, dict) else [lighting]), cap.get("style_medium", ""), cap.get("context", "")]
    parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(p if p.endswith((".", "!", "?")) else p + "." for p in parts)


def prompt_kind(exp):
    return FAMILIES[family(exp)]["prompt"]


def model_prompt(exp, j):
    """What this experiment's model is given as its prompt: the structured caption (dict) or the paragraph (str)."""
    return compose_prompt(exp, j) if prompt_kind(exp) == "caption" else prose_prompt(exp, j)


def composed_path(exp, j):
    """The tracked copy of model_prompt: <prompts dir>/composed/<view>.json (caption) or .txt (prose)."""
    ext = "json" if prompt_kind(exp) == "caption" else "txt"
    return os.path.join(REPO, exp["scenes"][j["scene"]]["prompts"], "composed", f"{j['view']}.{ext}")


def read_composed(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path)) if path.endswith(".json") else open(path).read().strip()


def negative_prompt_path(exp):
    """The experiment's negative prompt file, or None for `negative_prompt: none` (a model run without guidance).
    Default: the frozen copy of the Cosmos cookbook's, which every Cosmos transfer run uses."""
    p = exp.get("negative_prompt", "p2s/3_generate/negative_prompts/cosmos3_negative_prompt.json")
    return None if p in (None, "none") else os.path.join(REPO, p)


def recipe(exp, j):
    """Everything that decides a run's output, as hashes. A finished run is current only if this still matches."""
    r = scene_root(exp, j["scene"])
    spec_path = scene_spec_path(exp, j["scene"])
    src = cut_sources(exp, j) if j["kind"] == "cut" else [source_rgb(exp, j)]
    rec = {"spec": sha256_file(spec_path), "world_npz": sha256_file(world_file(exp, j["scene"])),
           "source_rgb": [sha256_file(p) for p in src],
           "cut_schedule": sha256_obj(exp["cuts"][j["scene"]][j["view"]]) if j["kind"] == "cut" else None,
           "prompt": sha256_obj(model_prompt(exp, j)), "cond": sha256_obj(j["cond_spec"]),
           "negative_prompt": sha256_file(negative_prompt_path(exp)) if negative_prompt_path(exp) else None,
           "model": sha256_obj(exp["model"]), "video": sha256_obj(exp["video"]), "sampler": sha256_obj(exp["sampler"]),
           "first_frame": exp["first_frame"], "seed": j["seed"], "input_code_version": INPUT_CODE_VERSION,
           "code": _code_hashes(exp)}
    hints = {h: ("rgb" if c.get("from") == "rgb" else h) for h, c in j["cond_spec"].items() if h != "blur"}
    if hints:
        rec["controls"] = {h: [sha256_file(p) for p in control_sources(exp, j, src)] for h, src in hints.items()}
    return rec


def same_recipe(recorded, current):
    """Compare two recipes, ignoring provenance-only keys (code file hashes); a run recorded before a key existed
    takes that key's first value (input_code_version 1)."""
    a = {k: v for k, v in recorded.items() if k not in NOT_COMPARED}
    b = {k: v for k, v in current.items() if k not in NOT_COMPARED}
    a.setdefault("input_code_version", 1)
    a.setdefault("negative_prompt", b.get("negative_prompt"))      # recorded before the hash was kept; same file content
    return a == b
