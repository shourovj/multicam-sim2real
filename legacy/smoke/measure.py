#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["nvidia-ml-py"]
# ///
"""Run a command, sample the GPU memory of its process tree, and write wall time + peak to JSON.

  legacy/smoke/measure.py --out run.json -- <command ...>

Memory is the sum of NVML per-process used memory over the command's process tree, sampled
every --interval seconds on the GPUs in CUDA_VISIBLE_DEVICES (all GPUs if unset). This is what
nvidia-smi shows, so it includes the CUDA context and allocator cache, not just live tensors.
"""
import argparse, json, os, subprocess, sys, threading, time

import pynvml as nv


def tree(root):
    kids = {}
    for e in os.listdir("/proc"):
        if e.isdigit():
            try:
                with open(f"/proc/{e}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(e))
            except (OSError, IndexError, ValueError):
                pass
    out, stack = set(), [root]
    while stack:
        p = stack.pop(); out.add(p); stack.extend(kids.get(p, []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # CUDA indices = NVML / nvidia-smi indices
    nv.nvmlInit()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:  # with CUDA_DEVICE_ORDER=PCI_BUS_ID this matches NVML index order
        handles = [nv.nvmlDeviceGetHandleByIndex(int(i)) for i in cvd.split(",")]
    else:
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]

    t0 = time.time()
    child = subprocess.Popen(cmd)
    peak = {"mib": 0, "t": 0.0}
    done = threading.Event()

    def sample():
        while not done.is_set():
            pids = tree(child.pid)
            tot = 0
            for h in handles:
                for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
                    if p.pid in pids and p.usedGpuMemory:
                        tot += p.usedGpuMemory >> 20
            if tot > peak["mib"]:
                peak.update(mib=tot, t=round(time.time() - t0, 1))
            done.wait(a.interval)

    th = threading.Thread(target=sample, daemon=True); th.start()
    rc = child.wait(); done.set(); th.join()
    res = {"cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 1),
           "peak_gpu_mib": peak["mib"], "peak_at_s": peak["t"], "cuda_visible_devices": cvd,
           "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("[measure]", json.dumps(res), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
