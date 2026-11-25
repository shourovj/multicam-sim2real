"""Scores of one iteration, from its report r.json (p2s_check). Both are in [0, 1], 1 best.

physics: how well X follows physics. Every physics finding of r (simulation, overlap, penetration, energy, FEM,
  acceleration, actuators, ...: everything but the scene's own expectations, the prompt-spec checks and camera
  placement) counts PASS 1, an acknowledged warning 0.75, a warning 0.5, a failure 0. No X (the simulation failed or
  S was rejected) scores 0.
eval1: how well the scene does what the prompt says, from the prompt-spec findings (spec/README.md):
  0.6 x requirements + 0.4 x objects. A requirement that holds scores 1, one that fails up to 0.5 by how close it came
  ((1 + robustness) / 4), an implied one that fails 0.5. Objects: 1 minus the share of prompt objects and cameras
  with a failing object, appearance, camera or contact finding.
"""
import json
import os

NOT_PHYSICS = ("expect_", "prompt", "camera_inside", "acknowledge")


def physics_score(r):
    if r.get("status") == "ERROR":
        return 0.0
    vals = []
    for f in r.get("findings", []):
        c = f.get("check", "")
        if c.startswith(NOT_PHYSICS) or f["severity"] == "INFO":
            continue
        if c == "simulation" and f["severity"] == "FAIL":
            return 0.0
        vals.append({"PASS": 1.0, "WARN": 0.75 if f.get("acknowledged") else 0.5, "FAIL": 0.0}[f["severity"]])
    return sum(vals) / len(vals) if vals else 0.0


def eval1_score(r, spec):
    if r.get("status") == "ERROR" or spec is None:
        return 0.0
    req, items, bad = [], set(), set()
    implied = {q["id"] for q in spec.get("requirements", []) if q.get("implied")}
    for f in r.get("findings", []):
        c, sev, subj = f.get("check", ""), f["severity"], f.get("subject", "")
        if c == "prompt":
            rho = f.get("data", {}).get("robustness", -1.0)
            if sev == "PASS":
                req.append(1.0)
            elif subj in implied:
                req.append(0.5)
            else:
                req.append(max(0.0, (1 + rho) / 4))
        elif c in ("prompt_object", "prompt_appearance", "prompt_camera", "prompt_contact") and sev == "FAIL":
            bad.add(subj)
    n_items = len(spec.get("objects", [])) + len(spec.get("cameras", []))
    objects = max(0.0, 1 - len(bad) / n_items) if n_items else 1.0
    requirements = sum(req) / len(req) if req else (1.0 if not spec.get("requirements") else 0.0)
    return 0.6 * requirements + 0.4 * objects


def score_dir(it_dir, spec):
    path = os.path.join(it_dir, "r.json")
    r = json.load(open(path)) if os.path.exists(path) else {"status": "ERROR"}
    if not os.path.exists(os.path.join(it_dir, "x.json")):
        r = {"status": "ERROR"}
    held = [f for f in r.get("findings", []) if f.get("check") == "prompt"]
    return {"status": r.get("status", "ERROR"), "physics": round(physics_score(r), 4), "eval1": round(eval1_score(r, spec), 4),
            "requirements_held": sum(f["severity"] == "PASS" for f in held), "requirements": len(held),
            "physics_fail": sum(f["severity"] == "FAIL" and not f.get("check", "").startswith(NOT_PHYSICS) for f in r.get("findings", []))}
