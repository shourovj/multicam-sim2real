"""What the GPUs of a shared server are doing right now (nvidia-smi), for the two job runners."""

import os
import subprocess


def _query(what, fields):
    return subprocess.run(["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()


def free_gib():
    """Free memory (GiB) per GPU index."""
    return {int(i): (int(t) - int(u)) / 1024 for i, t, u in (line.split(", ") for line in _query("gpu", "index,memory.total,memory.used"))}


def gpu_names():
    return {int(i): n.strip() for i, n in (line.split(",", 1) for line in _query("gpu", "index,name"))}


def our_mem_gib():
    """Memory (GiB) of this user's compute processes, per GPU index."""
    index = dict(line.split(", ")[::-1] for line in _query("gpu", "index,uuid"))
    mem = {}
    for line in _query("compute-apps", "gpu_uuid,pid,used_memory"):
        uuid, pid, used = line.split(", ")
        try:
            if os.stat(f"/proc/{pid}").st_uid == os.getuid():
                g = int(index[uuid])
                mem[g] = mem.get(g, 0) + int(used) / 1024
        except (OSError, KeyError, ValueError):
            pass
    return mem


def my_gpus():
    """GPUs that already hold a compute process of this user, e.g. a job from an earlier runner."""
    return set(our_mem_gib())
