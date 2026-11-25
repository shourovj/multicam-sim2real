"""Render audit: a small look at the rendered simulation by the vision-language model, as part of Eval1.

The prompt-spec checks measure everything they can in X exactly. What they cannot measure is how the scene reads in
the image: whether a box stack reads as a van, whether a colour shows, whether the event is legible from a camera.
This audit renders each camera at a few key frames, very small (256 px wide: a few dozen image tokens each), and asks
the model a checklist made from the spec's objects and camera requirements. It reports text and a score; the
SceneAgent receives only the text, never the images.

  python3 render_audit.py <it_dir> <spec.json> <out_dir> [--url ...] [--width 256]
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSX = os.path.dirname(HERE)
BLENDER = os.path.join(PHYSX, ".deps", "blender", "blender-4.2.1-linux-x64", "blender")
IMPORTER = os.path.join(os.path.dirname(PHYSX), "blender", "import_x.py")


def key_frames(x):
    """Before, at and after the first contact between two non-ground entities (or the middle of the clip)."""
    n, fps = x["clock"]["frames"], x["clock"]["fps"]
    t = None
    for e in x.get("events", []):
        name = e["name"]
        if name.startswith("contact:") and "ground" not in name.split(":")[1:] and e["t"] >= 0:
            t = e["t"]
            break
    f = int(round((t if t is not None else n / fps / 2) * fps))
    return sorted({max(0, f - int(fps // 2)), min(n - 1, f), min(n - 1, f + int(fps))})


def render(it_dir, out_dir, width):
    x = json.load(open(os.path.join(it_dir, "x.json")))
    cams = [c["id"] for c in x.get("cameras", [])]
    frames = key_frames(x)
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "a")
    cmd = [BLENDER, "-b", "--python", IMPORTER, "--", it_dir, os.path.join(out_dir, "audit.blend"),
           "--preview", ",".join(cams), ",".join(map(str, frames)), prefix, "--preview-width", str(width)]
    with open(os.path.join(out_dir, "blender.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    images = {c: [f"{prefix}_{c}_{f:04d}.png" for f in frames if os.path.exists(f"{prefix}_{c}_{f:04d}.png")] for c in cams}
    return images, frames, x["clock"]["fps"]


SMALL = ("traffic_light", "water", "oil", "viscous_liquid", "sand", "smoke", "explosion", "projectile", "can", "cup")


def checklist(spec):
    """Only what a 256 px render can show: solid objects (shape and colour) and each camera's view of its event.
    Dents, liquids and light states are measured exactly by the prompt-spec checks; at this size a judge would
    only miss them."""
    items = []
    for o in spec.get("objects", []):
        if o["class"] in SMALL:
            continue
        what = (o.get("color", "") + " " + o["class"].replace("_", " ")).strip()
        items.append({"id": "object:" + o["name"], "ask": f"Is there something that reads as a {what} (\"{o.get('span', o['name'])}\")?"})
    for q in spec.get("requirements", []):
        if q.get("mode") == "camera":
            items.append({"id": "view:" + q["id"], "ask": f"Does this view show it: {q['text']}?"})
    return items


ASK = """These are small clay renders of a physics simulation, frames {frames} (t = {times} s) of camera "{cam}". The scene should show: "{prompt}"
The renders have no textures and plain lighting on purpose (lighting, time of day and surface detail come later); judge only shapes, colours, layout and what happens. Liquids and smoke are small dots.
For each item answer "yes", "no" (it should be visible here but is missing or looks wrong), or "not_in_view" (this camera does not show that part of the scene). Add a short note for every "no".
Items:
{items}
Answer with JSON only: {{"items": [{{"id": "...", "verdict": "yes|no|not_in_view", "note": "..."}}]}}"""


def ask(url, model, content):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "seed": 0,
                       "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"items": []}, out.get("usage", {})


def audit(it_dir, spec, out_dir, url, model, width=256):
    images, frames, fps = render(it_dir, out_dir, width)
    items = checklist(spec)
    per_item = {i["id"]: [] for i in items}
    tokens = 0
    for cam, paths in images.items():
        if not paths:
            continue
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
                   for p in paths]
        text = ASK.format(frames=", ".join(map(str, frames)), times=", ".join(f"{f / fps:.2f}" for f in frames), cam=cam,
                          prompt=spec["prompt"], items="\n".join(f"- {i['id']}: {i['ask']}" for i in items))
        content.append({"type": "text", "text": text})
        try:
            ans, usage = ask(url, model, content)
            tokens += usage.get("total_tokens", 0)
        except Exception as e:  # the audit is advisory: a failed call leaves the camera unjudged
            ans = {"items": [], "error": str(e)}
        for v in ans.get("items", []):
            if v.get("id") in per_item:
                per_item[v["id"]].append((cam, v.get("verdict", ""), v.get("note", "")))
    issues, ok, judged = [], 0, 0
    for i in items:
        votes = per_item[i["id"]]
        seen = [v for v in votes if v[1] in ("yes", "no")]
        if not seen:
            continue
        judged += 1
        if any(v[1] == "yes" for v in seen):
            ok += 1
        else:
            issues.append(f"{i['id']}: " + "; ".join(f"{c}: {n}" for c, _, n in seen if n))
    result = {"score": round(ok / judged, 4) if judged else None, "judged": judged, "held": ok, "issues": issues,
              "votes": per_item, "frames": frames, "width": width, "tokens": tokens}
    with open(os.path.join(out_dir, "audit.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("it_dir")
    ap.add_argument("spec")
    ap.add_argument("out_dir")
    ap.add_argument("--url", default=os.environ.get("P2S_LLM_URL", "http://localhost:8016/v1"))
    ap.add_argument("--model", default=os.environ.get("P2S_LLM_SERVED", "p2s-llm"))
    ap.add_argument("--width", type=int, default=256)
    a = ap.parse_args()
    r = audit(a.it_dir, json.load(open(a.spec)), a.out_dir, a.url, a.model, a.width)
    print(json.dumps({k: v for k, v in r.items() if k != "votes"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
