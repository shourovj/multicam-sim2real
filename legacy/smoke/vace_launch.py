"""Run VACE's vace/vace_wan_inference.py, optionally casting the DiT weights to bf16 after loading.

  VACE_DIT_BF16=1 python legacy/smoke/vace_launch.py <vace_wan_inference.py args...>

Why: the Wan2.1-VACE-14B checkpoint stores its 14.3B base weights in fp32 (57 GB) and only the
3.05B VACE blocks in bf16. VACE loads them as stored, so 63 GB of weights sit on the GPU and the
first sampling step OOMs on an empty 80 GB A100. Sampling already runs under bf16 autocast, so
casting the weights to bf16 (~35 GB) mostly changes memory, not the matmul precision. It does put
the few ops Wan runs in fp32 (time embedding, modulation) on bf16 weights. Off by default.
Must be run with cwd = the VACE repo root.
"""
import os
import runpy
import sys

import torch

vace_dir = os.path.join(os.getcwd(), "vace")
sys.path.insert(0, vace_dir)

if os.environ.get("VACE_DIT_BF16") == "1":
    from models.wan import wan_vace  # noqa: E402

    orig = wan_vace.VaceWanModel.from_pretrained

    def from_pretrained_bf16(*args, **kwargs):
        model = orig(*args, **kwargs).to(torch.bfloat16)
        n = sum(p.numel() for p in model.parameters())
        print(f"[vace_launch] cast DiT to bf16: {n / 1e9:.2f}B params, {2 * n / 2**30:.1f} GiB", flush=True)
        return model

    wan_vace.VaceWanModel.from_pretrained = from_pretrained_bf16

sys.argv = [os.path.join(vace_dir, "vace_wan_inference.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
