"""git commit + dirty flag (+ a hash of the diff when dirty) for run records."""
import hashlib
import subprocess


def git_state(repo):
    run = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout
    diff = run("diff", "HEAD")
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain").strip()),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}
