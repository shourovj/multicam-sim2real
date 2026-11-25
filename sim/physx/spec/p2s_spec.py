#!/usr/bin/env python3
"""Write the prompt spec for a prompt: p2s_spec.py "<prompt>" <out spec.json>  (or --from-scene scene.json <out>).

The spec is written from the prompt alone, by a model other than the SceneAgent, once per prompt; p2s_check then
checks every scene iteration against it (spec/README.md). The writer gets the spec format and the predicate
vocabulary (spec/README.md), the object classes and colour words (spec/classes.json) and the hand-written examples
(spec/examples), never the scene. Each draft is linted with `p2s_check --lint-spec` (the draft is kept beside the
output as <out>.draft.json); on errors the model gets them back and repairs its spec, up to --repairs times.

Needs the model served by spec/serve_llm.sh (OpenAI-compatible API, default http://localhost:8016/v1). Standard
library only.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSX = os.path.dirname(HERE)


def readme_sections():
    """The parts of spec/README.md the writer needs: format, formulas, predicates, writing rules."""
    text = open(os.path.join(HERE, "README.md")).read()
    keep = []
    for part in re.split(r"(?m)^## ", text)[1:]:
        title = part.split("\n", 1)[0].strip()
        if title in ("Format", "Formulas", "Predicates", "Writing a spec (for the writer model and for people)"):
            keep.append("## " + part.strip())
    return "\n\n".join(keep)


def vocabulary():
    classes = json.load(open(os.path.join(HERE, "classes.json")))
    lines = ["## Object classes (objects[].class must be one of these keys)", ""]
    for cid, c in classes["classes"].items():
        lines.append(f"- `{cid}`: {', '.join(c.get('names', []))} (entity kinds: {', '.join(c['kinds'])})")
    colors = classes["colors"]
    words = sorted(set(colors["families"]) | set(colors["aliases"]))
    lines += ["", "## Colour words (objects[].color, crosses_on, signal_is)", "", ", ".join(words)]
    return "\n".join(lines)


SYSTEM = """You write prompt specs for a physics simulation checker. A prompt describes a short video of one event filmed by one or more cameras. A separate agent builds the scene as a physics simulation; you never see it. Your spec states, from the prompt alone, which objects, parts and cameras the scene must contain and what must happen, as temporal-logic formulas over the measured predicates below. The checker measures every predicate exactly in the simulation.

Answer with the spec as one JSON object and nothing else. Copy the prompt verbatim into "prompt". Use only the predicates, classes and colour words listed here. Every object, part and camera a formula names must be declared. Prefer fewer, faithful requirements over many speculative ones; never invent positions, times or speeds the prompt does not give.

{readme}

{vocab}
"""


def chat(url, model, messages, temperature, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                       "seed": 0, "thinking_token_budget": max_tokens // 2, "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or "", out.get("usage", {})


def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def lint(checker, spec, draft):
    with open(draft, "w") as f:
        json.dump(spec, f, indent=1)
    r = subprocess.run([checker, "--lint-spec", draft], capture_output=True, text=True)
    return [] if r.returncode == 0 else [l for l in r.stdout.splitlines() if l.strip()] or [r.stderr.strip()]


def examples(prompt, exclude):
    out = []
    d = os.path.join(HERE, "examples")
    for name in sorted(os.listdir(d)):
        if not name.endswith(".spec.json") or name in exclude:
            continue
        ex = json.load(open(os.path.join(d, name)))
        if ex["prompt"].strip() == prompt.strip():
            continue  # never show the writer the answer to its own prompt
        ex = {k: v for k, v in ex.items() if k != "writer"}
        out += [{"role": "user", "content": "Prompt: " + ex["prompt"]},
                {"role": "assistant", "content": json.dumps(ex, indent=1)}]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="+", help='"<prompt>" <out>, or <out> with --from-scene')
    ap.add_argument("--from-scene", help="take the prompt from this scene's meta.prompt")
    ap.add_argument("--url", default=os.environ.get("P2S_LLM_URL", "http://localhost:8016/v1"))
    ap.add_argument("--model", default=os.environ.get("P2S_LLM_SERVED", "p2s-llm"))
    ap.add_argument("--checker", default=os.path.join(PHYSX, "build", "p2s_check"))
    ap.add_argument("--repairs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--exclude-example", action="append", default=[], help="example file name to leave out")
    ap.add_argument("--log", help="write the whole conversation here (JSON)")
    a = ap.parse_args()
    if a.from_scene:
        if len(a.args) != 1:
            ap.error("with --from-scene give only <out>")
        prompt, out = json.load(open(a.from_scene))["meta"]["prompt"], a.args[0]
    else:
        if len(a.args) != 2:
            ap.error('give "<prompt>" <out>')
        prompt, out = a.args
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draft = out[:-5] + ".draft.json" if out.endswith(".json") else out + ".draft.json"

    messages = [{"role": "system", "content": SYSTEM.format(readme=readme_sections(), vocab=vocabulary())}]
    messages += examples(prompt, set(a.exclude_example))
    messages.append({"role": "user", "content": "Prompt: " + prompt})
    spec, errors, log = None, ["no answer"], []
    for attempt in range(a.repairs + 1):
        text, reasoning, usage = chat(a.url, a.model, messages, a.temperature, a.max_tokens)
        log.append({"attempt": attempt + 1, "reasoning": reasoning, "answer": text, "usage": usage})
        messages.append({"role": "assistant", "content": text})
        try:
            spec = parse_json(text)
            spec["prompt"] = prompt  # verbatim, whatever the model copied
            errors = lint(a.checker, spec, draft)
        except (ValueError, KeyError, TypeError) as e:
            spec, errors = None, [f"not a JSON object: {e}"]
        print(f"attempt {attempt + 1}: {'OK' if not errors else str(len(errors)) + ' problems'}"
              f" ({usage.get('completion_tokens', '?')} tokens)", file=sys.stderr)
        if not errors:
            break
        messages.append({"role": "user", "content": "The checker rejected this spec:\n" + "\n".join(errors) +
                         "\n\nAnswer with the whole corrected spec as one JSON object."})
    if a.log:
        with open(a.log, "w") as f:
            json.dump({"prompt": prompt, "attempts": log}, f, indent=1)
    if errors:
        print("p2s_spec: no valid spec after %d attempts:\n  %s" % (a.repairs + 1, "\n  ".join(errors)), file=sys.stderr)
        return 1
    spec["writer"] = f"{a.model} via p2s_spec.py, {datetime.date.today().isoformat()}"
    keys = ["format", "version", "writer", "prompt", "setting", "objects", "parts", "cameras", "requirements", "not_simulated"]
    with open(out, "w") as f:
        json.dump({k: spec[k] for k in keys if k in spec}, f, indent=1)
        f.write("\n")
    print(f"wrote {out}: {len(spec.get('objects', []))} objects, {len(spec.get('requirements', []))} requirements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
