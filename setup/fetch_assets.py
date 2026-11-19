"""Download the CC0 assets listed in configs/assets/polyhaven.yaml into $S2R_DATA/assets/polyhaven/<id>/.

  source ~/.sim2real.env; $S2R_ROOT/envs/eval/.venv/bin/python setup/fetch_assets.py

Safe to re-run: a file already on disk with the right md5 is kept. Each asset folder gets a SOURCE.md
(where every file came from, the licence, the date, sha256) so a render can say exactly what it used.
"""
import datetime, hashlib, json, os, urllib.request

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(os.environ["S2R_DATA"], "assets", "polyhaven")
UA = {"User-Agent": "sim2real-asset-fetch/1.0"}   # generic client name; Poly Haven asks API users to set one


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def fetch(url, path, md5):
    if os.path.exists(path) and hashlib.md5(open(path, "rb").read()).hexdigest() == md5:
        return "kept"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = get(url)
    assert hashlib.md5(data).hexdigest() == md5, f"md5 mismatch for {url}"
    open(path + ".part", "wb").write(data)
    os.replace(path + ".part", path)
    return "downloaded"


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs/assets/polyhaven.yaml")))
    for kind in ("textures", "models"):
        for aid, a in cfg[kind].items():
            files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
            out = os.path.join(DEST, aid)
            todo = {}                                     # relative path -> (url, md5)
            if kind == "textures":
                for key, name in (("Diffuse", "diff"), ("nor_gl", "nor_gl"), ("Rough", "rough")):
                    f = files[key][a["res"]]["jpg"]
                    todo[f"{aid}_{name}_{a['res']}.jpg"] = (f["url"], f["md5"])
            else:
                g = files["gltf"][a["res"]]["gltf"]
                todo[os.path.basename(g["url"])] = (g["url"], g["md5"])
                for rel, f in g.get("include", {}).items():
                    todo[rel] = (f["url"], f["md5"])
            rows = []
            for rel, (url, md5) in todo.items():
                what = fetch(url, os.path.join(out, rel), md5)
                sha = hashlib.sha256(open(os.path.join(out, rel), "rb").read()).hexdigest()
                rows.append(f"| `{rel}` | {url} | `{sha}` |")
                print(f"{what:10s} {aid}/{rel}")
            info = json.loads(get(f"https://api.polyhaven.com/info/{aid}"))
            with open(os.path.join(out, "SOURCE.md"), "w") as f:
                f.write(f"# {info.get('name', aid)} ({aid})\n\nSource: https://polyhaven.com/a/{aid}\n"
                        f"Authors: {', '.join(info.get('authors', {}))}\nLicence: CC0 1.0 (Poly Haven)\n"
                        f"Fetched: {datetime.date.today().isoformat()} by setup/fetch_assets.py, resolution {a['res']}\n"
                        f"Used for: {a['for']}\n\n| File | URL | sha256 |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
