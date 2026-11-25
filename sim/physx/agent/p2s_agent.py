#!/usr/bin/env python3
"""SceneAgent loop with Eval1 feedback, on a time budget: prompt -> scene S -> X, scored.

  python3 p2s_agent.py "<prompt>" <run_dir> [--budget-s 900] [--gpu 5] [--reflect]

1. The prompt spec ($RUN/spec.json) is written once from the prompt (spec/p2s_spec.py) unless it exists. It is not
   counted in the authoring budget and the SceneAgent cannot change it.
2. The SceneAgent (the same model, served by spec/serve_llm.sh) writes S from the manual
   (docs/2026-11-24-1859-sceneagent-prompt.md), the closest example scenes, the spec's requirements and the lessons
   of earlier runs (agent/lessons.md).
3. Each iteration: p2s_sim -> p2s_check (physics checks, the scene's own expectations, the prompt spec) -> render
   audit (tiny renders judged by the model, run alongside the next SceneAgent turn) -> scores (agent/scores.py). The
   report r.md and the audit's text go back to the SceneAgent, which writes the next S. Images never reach it.
4. It stops at PASS with a clean audit, at the budget, or at --max-iterations, and hands over the best iteration
   (highest eval1 + physics; a PASS first): $RUN/result.json names it, with its eval1 and physics scores.
5. --reflect (after the budget): the model reads the run's history and updates agent/lessons.md, the standing
   advice every later SceneAgent run starts from. That is how the loop improves its own prompt across prompts.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import render_audit  # noqa: E402
import scores  # noqa: E402

SIM = os.path.join(PHYSX, "build", "p2s_sim")
CHECK = os.path.join(PHYSX, "build", "p2s_check")
MANUAL = os.path.join(PHYSX, "docs", "2026-11-24-1859-sceneagent-prompt.md")
ITERATION = os.path.join(PHYSX, "docs", "2026-11-24-1859-sceneagent-iteration.md")
LESSONS = os.path.join(HERE, "lessons.md")


class Budget:
    def __init__(self, seconds):
        self.t0, self.seconds = time.time(), seconds

    def used(self):
        return time.time() - self.t0

    def left(self):
        return self.seconds - self.used()


def chat(url, model, messages, max_tokens, timeout, thinking=True, think_budget=None, json_out=False):
    """One chat turn. think_budget caps the reasoning tokens (vLLM's thinking_token_budget): unbounded, the model
    can think through the whole token allowance and never write the scene. json_out constrains the answer (not the
    reasoning) to syntactically valid JSON: a 5000-character scene with one missing brace costs a whole iteration."""
    body = {"model": model, "messages": messages, "temperature": 0.3, "top_p": 0.95, "seed": 0, "max_tokens": max_tokens}
    if json_out:
        body["response_format"] = {"type": "json_object"}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif think_budget:
        body["thinking_token_budget"] = think_budget
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=max(30, timeout)) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return m.get("content") or "", out.get("usage", {}), out["choices"][0].get("finish_reason")


def parse_scene(text):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def closest_examples(prompt, k=2):
    """The example scenes whose prompts share the most words with this one."""
    words = set(re.findall(r"[a-z]+", prompt.lower())) - {"a", "an", "the", "and", "of", "in", "on", "by", "at", "to", "is", "it"}
    scored = []
    for name in sorted(os.listdir(os.path.join(PHYSX, "scenes"))):
        s = json.load(open(os.path.join(PHYSX, "scenes", name)))
        p = s.get("meta", {}).get("prompt", "")
        scored.append((len(words & set(re.findall(r"[a-z]+", p.lower()))), name, s))
    scored.sort(key=lambda t: -t[0])
    return [(n, s) for _, n, s in scored[:k]]


def spec_digest(spec):
    lines = ["Objects (each needs an entity with that appearance class and colour):"]
    for o in spec.get("objects", []):
        lines.append(f"- {o['name']}: {o.get('color', '')} {o['class']} x{o.get('count', 1)} ({o.get('span', '')})")
    for p in spec.get("parts", []):
        lines.append(f"- part {p['name']}: the {p['region']} of {p['of']} ({p.get('span', '')})")
    lines.append("Cameras:")
    for c in spec.get("cameras", []):
        extra = ", ".join(f"{k}={c[k]}" for k in ("type", "on", "mounted_on", "min_height_m") if k in c)
        lines.append(f"- {c['name']}: {extra} ({c.get('span', '')})")
    lines.append("Requirements (temporal logic over measured predicates; see the predicate list):")
    for q in spec.get("requirements", []):
        lines.append(f"- {q['id']}{' (implied)' if q.get('implied') else ''}: {q['text']}  `{q['formula']}`")
    return "\n".join(lines)


def system_prompt(prompt, spec):
    spec_readme = open(os.path.join(PHYSX, "spec", "README.md")).read()
    predicates = spec_readme[spec_readme.index("## Predicates"):spec_readme.index("## Writing a spec")]
    parts = [
        "You are the SceneAgent. You turn a prompt into a physics scene file S (JSON) for the p2s simulator, read the "
        "checker's report r on it and fix S until r says PASS. Answer every turn with the complete scene JSON and nothing "
        "else (no prose, no code fences). Keep your reasoning short and concrete: you have a time budget.",
        open(MANUAL).read(),
        open(ITERATION).read(),
        "# The prompt spec r checks your scene against\n\nWritten from the prompt by another model; you cannot change it. "
        "r binds each prompt object to one of your entities (kind, size, appearance, id words) and checks every requirement.\n\n"
        + spec_digest(spec) + "\n\n" + predicates,
    ]
    if os.path.exists(LESSONS):
        parts.append("# Lessons from earlier runs\n\n" + open(LESSONS).read())
    for name, s in closest_examples(prompt):
        parts.append(f"# Example scene that passes: {name}\n\n" + json.dumps(s, separators=(",", ":")))
    return "\n\n".join(parts)


def trim_report(md, limit=24000):
    """r.md without the passed-checks list and the iteration guide's long tail when it is too long."""
    md = re.sub(r"(?s)\n## Passed checks\n.*?(?=\n## )", "\n", md)
    return md if len(md) <= limit else md[:limit] + "\n\n[report cut]"


def run(cmd, timeout, log):
    with open(log, "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=max(5, timeout)).returncode
        except subprocess.TimeoutExpired:
            f.write("\n[timed out: authoring budget]\n")
            return 124


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("run_dir")
    ap.add_argument("--budget-s", type=float, default=900)
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
    ap.add_argument("--url", default=os.environ.get("P2S_LLM_URL", "http://localhost:8016/v1"))
    ap.add_argument("--model", default=os.environ.get("P2S_LLM_SERVED", "p2s-llm"))
    ap.add_argument("--think-first", type=int, default=8000, help="reasoning tokens for the first scene")
    ap.add_argument("--think-next", type=int, default=5000, help="reasoning tokens for each revision")
    ap.add_argument("--answer-tokens", type=int, default=10000, help="tokens for the scene JSON itself")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--reflect", action="store_true", help="update agent/lessons.md from this run afterwards")
    ap.add_argument("--reflect-only", action="store_true", help="only update the lessons from the finished run in run_dir")
    a = ap.parse_args()
    rd = os.path.abspath(a.run_dir)
    if a.reflect_only:
        reflect(a, rd, json.load(open(os.path.join(rd, "result.json")))["iterations"])
        return 0
    os.makedirs(rd, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    log = []

    def note(msg):
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.append(line)

    spec_path = os.path.join(rd, "spec.json")
    if not os.path.exists(spec_path):
        t = time.time()
        rc = subprocess.run([sys.executable, os.path.join(PHYSX, "spec", "p2s_spec.py"), a.prompt, spec_path,
                             "--url", a.url, "--model", a.model, "--log", os.path.join(rd, "spec.log.json")]).returncode
        if rc != 0:
            note("no valid prompt spec: stopping")
            return 2
        note(f"prompt spec written in {time.time() - t:.0f} s (outside the authoring budget)")
    spec = json.load(open(spec_path))

    budget = Budget(a.budget_s)
    messages = [{"role": "system", "content": system_prompt(a.prompt, spec)},
                {"role": "user", "content": f"Prompt: {a.prompt}\n\nWrite S for iteration 1 (meta.iteration 1, meta.prompt the prompt verbatim). "
                                            f"Authoring budget: {a.budget_s / 60:.0f} min in all."}]
    history, pool, pending_audit = [], cf.ThreadPoolExecutor(max_workers=1), None
    best = None
    for it in range(1, a.max_iterations + 1):
        if budget.left() < 60:
            note(f"budget spent ({budget.used():.0f} s)")
            break
        d = os.path.join(rd, f"it{it}")
        os.makedirs(d, exist_ok=True)
        t = time.time()
        try:
            think = a.think_first if it == 1 else a.think_next
            text, usage, finish = chat(a.url, a.model, messages, think + a.answer_tokens, budget.left(), think_budget=think, json_out=True)
        except Exception as e:  # out of time mid-answer, or the server went away
            note(f"it{it}: SceneAgent call failed: {e}")
            break
        gen_s = time.time() - t
        with open(os.path.join(d, "answer.txt"), "w") as f:
            f.write(text)
        entry = {"iteration": it, "gen_s": round(gen_s, 1), "tokens": usage.get("completion_tokens"), "finish": finish}
        feedback = None
        try:
            scene = parse_scene(text)
            scene.setdefault("meta", {})
            scene["meta"]["iteration"], scene["meta"]["prompt"] = it, a.prompt
            scene["meta"]["max_iterations"] = max(scene["meta"].get("max_iterations", 8), a.max_iterations)
            with open(os.path.join(d, "scene.json"), "w") as f:
                json.dump(scene, f, indent=1)
        except (ValueError, json.JSONDecodeError) as e:
            feedback = (f"Your answer was not one valid JSON object ({e}{'; it was cut off at the token limit' if finish == 'length' else ''}). "
                        "Think less and answer with the complete scene JSON only.")
        if feedback is None:
            t = time.time()
            rc = run([SIM, os.path.join(d, "scene.json"), d], budget.left(), os.path.join(d, "sim.log"))
            entry["sim_s"], entry["sim_rc"] = round(time.time() - t, 1), rc
            if rc == 2:
                err = open(os.path.join(d, "sim.log")).read()[-3000:]
                feedback = f"p2s_sim rejected S (exit 2):\n{err}\nFix the scene and answer with the complete scene JSON."
            elif rc == 124:
                note(f"it{it}: simulation cut off by the budget")
                break
            else:
                run([CHECK, os.path.join(d, "scene.json"), d, "--spec", spec_path], 120, os.path.join(d, "check.log"))
                sc = scores.score_dir(d, spec)
                entry.update(sc)
                md = open(os.path.join(d, "r.md")).read() if os.path.exists(os.path.join(d, "r.md")) else "(no report)"
                feedback = f"Report r for iteration {it}:\n\n{trim_report(md)}"
        # The previous iteration's audit ran alongside this turn; its text joins this feedback.
        if pending_audit is not None:
            prev_it, fut = pending_audit
            try:
                au = fut.result(timeout=max(1, budget.left()))
                history[prev_it - 1]["audit"] = au.get("score")
                if au.get("issues"):
                    feedback += (f"\n\n## Render audit of iteration {prev_it} (tiny renders judged by a vision model; shapes and colours only)\n\n"
                                 + "\n".join("- " + s for s in au["issues"]))
            except Exception as e:
                note(f"audit of it{prev_it} failed: {e}")
            pending_audit = None
        if not a.no_audit and entry.get("status") not in (None, "ERROR") and os.path.exists(os.path.join(d, "x.json")):
            pending_audit = (it, pool.submit(render_audit.audit, d, spec, os.path.join(d, "audit"), a.url, a.model))
        history.append(entry)
        note(f"it{it}: gen {gen_s:.0f} s ({entry.get('tokens')} tok), sim {entry.get('sim_s', '-')} s, "
             f"status {entry.get('status', 'no X')}, eval1 {entry.get('eval1', '-')}, physics {entry.get('physics', '-')}, "
             f"requirements {entry.get('requirements_held', '-')}/{entry.get('requirements', '-')}; {budget.used():.0f} s used")
        worse = None
        if "eval1" in entry:
            key = (entry["status"] == "PASS", entry["eval1"] + entry["physics"], it)
            if best is None or key > best[0]:
                best = (key, it)
                best_turn = (text, feedback)
            else:
                worse = f"eval1 {entry['eval1']}, physics {entry['physics']}"
        elif best is not None:
            worse = "no simulation"
        if entry.get("status") == "PASS":
            note("PASS")
            break
        # Keep the conversation short: the system prompt, the prompt, the last S and its report. An iteration that
        # did worse than the best so far is set aside: the agent continues from the best scene and its report, told
        # what the discarded attempt did, so fixes for the prompt do not drift into broken physics.
        if worse:
            b = best[1]
            hb = history[b - 1]
            text, feedback = best_turn[0], (
                f"Iteration {it} did worse ({worse}) than iteration {b} (eval1 {hb['eval1']}, physics {hb['physics']}), so it is "
                f"set aside. What it got wrong, in short:\n\n{feedback[:2500]}\n\n---\n\nContinue from iteration {b}'s scene "
                f"(your previous answer above) and its report:\n\n{best_turn[1]}")
        messages = messages[:2] + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": feedback + f"\n\nIteration {it + 1}: answer with the complete corrected scene JSON "
                                                                          f"(meta.iteration {it + 1}). Time left: {budget.left() / 60:.1f} min."}]
    authoring_s = budget.used()
    if pending_audit is not None:  # the last audit, outside the budget: it scores the result, it no longer steers it
        prev_it, fut = pending_audit
        try:
            au = fut.result(timeout=300)
            history[prev_it - 1]["audit"] = au.get("score")
        except Exception as e:
            note(f"audit of it{prev_it} failed: {e}")
    pool.shutdown(wait=False)

    result = {"prompt": a.prompt, "budget_s": a.budget_s, "authoring_s": round(authoring_s, 1), "iterations": history,
              "spec": spec_path}
    if best:
        it = best[1]
        h = history[it - 1]
        final = os.path.join(rd, "final")
        if os.path.exists(final):
            final = os.path.join(rd, "final_" + datetime.datetime.now().strftime("%H%M%S"))
        shutil.copytree(os.path.join(rd, f"it{it}"), final, ignore=shutil.ignore_patterns("audit"))
        result.update({"best_iteration": it, "final_dir": final, "status": h["status"], "eval1": h["eval1"],
                       "physics": h["physics"], "render_audit": h.get("audit"),
                       "requirements_held": h["requirements_held"], "requirements": h["requirements"]})
    else:
        result.update({"best_iteration": None, "status": "NO_SIMULATION", "eval1": 0.0, "physics": 0.0})
    with open(os.path.join(rd, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    note(f"result: iteration {result['best_iteration']}, {result['status']}, eval1 {result['eval1']}, physics {result['physics']}, "
         f"render audit {result.get('render_audit')}, authoring {authoring_s:.0f} s -> {rd}/result.json")
    if a.reflect:
        reflect(a, rd, history)
    with open(os.path.join(rd, "agent.log"), "a") as f:
        f.write("\n".join(log) + "\n")
    return 0


REFLECT = """You are improving the standing advice given to a SceneAgent that writes physics scenes for the p2s simulator. Below are the current lessons and the history of one run: the prompt, and for each iteration the checker's failing findings and the scores. Update the lessons: keep what still holds, add what this run teaches about getting a scene right faster (modelling choices, keys that fix recurring failures, order of fixes), drop what is wrong or redundant. Each lesson is one line starting with "- ", concrete and general (not about this prompt's objects by name), at most 40 lines. Name only scene keys that exist in the SceneAgent manual. Never advise weakening an expectation or the prompt spec to pass (wider regions, lower thresholds): the fix is always in the scene. Answer with the lessons only."""


def reflect(a, rd, history):
    lessons = open(LESSONS).read() if os.path.exists(LESSONS) else "(none yet)"
    runlog = [f"Prompt: {a.prompt}"]
    for h in history:
        d = os.path.join(rd, f"it{h['iteration']}")
        fails = []
        if os.path.exists(os.path.join(d, "r.json")):
            r = json.load(open(os.path.join(d, "r.json")))
            fails = [f"{f['key']}: {f['message'][:300]}" for f in r.get("findings", []) if f["severity"] in ("FAIL", "WARN") and not f.get("acknowledged")]
        elif os.path.exists(os.path.join(d, "sim.log")):
            fails = ["p2s_sim: " + open(os.path.join(d, "sim.log")).read()[-400:]]
        runlog.append(f"Iteration {h['iteration']}: status {h.get('status', '-')}, eval1 {h.get('eval1', '-')}, physics {h.get('physics', '-')}\n  "
                      + "\n  ".join(fails[:25]))
    text, _, _ = chat(a.url, a.model, [{"role": "system", "content": REFLECT},
                                       {"role": "user", "content": f"Current lessons:\n{lessons}\n\nRun:\n" + "\n".join(runlog)}],
                      8000, 900, think_budget=3000)
    lines = [l for l in text.splitlines() if l.startswith("- ")][:40]
    if lines:
        if os.path.exists(LESSONS):
            shutil.copy(LESSONS, os.path.join(rd, "lessons_before.md"))
        with open(LESSONS, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"lessons updated: {len(lines)} lines -> {LESSONS}")


if __name__ == "__main__":
    sys.exit(main())
