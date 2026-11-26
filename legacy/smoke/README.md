# Step 1 smoke tests

Known problems and their fixes are in [the Step 1 gotchas doc](../../docs/2026-11-10-0945-step1-environment-gotchas.md). Read it before debugging a run.

Smoke tests for the v0 pipeline's tools (plan §2). Third-party code, envs and checkpoints live in `$S2R_ROOT`, outputs in `$S2R_DATA/smoke/<tool>/` (see [setup](../../setup/README.md) and [`env.sh`](../env.sh)). Wrap every GPU job in `measure.py` for wall time and peak GPU memory:

```
source env.sh; M=legacy/smoke/measure.py; D=$S2R_DATA/smoke
$M --out $D/blender/gpu/measure.json -- blenderproc run --blender-install-path $S2R_ROOT/blender legacy/smoke/blender_cube.py $D/blender/gpu
$M --out $D/cosmos/out-153/measure.json -- legacy/smoke/cosmos_simtransfer_153.sh 153
$M --out $D/eval/measure.json -- $S2R_ROOT/envs/eval/.venv/bin/python legacy/smoke/eval_tools.py --video $S2R_ROOT/co-tracker/assets/apple.mp4 --out $D/eval
$M --out $D/wanmove/measure.json -- bash legacy/smoke/wanmove_example.sh
VACE_DIT_BF16=1 MODEL=14B $M --out $D/vace/ffdepth/14B-bf16/measure.json -- bash legacy/smoke/vace_ffdepth.sh   # also vace_txt2vid.sh, vace_firstframe.sh
```

| File | What it does |
|---|---|
| `measure.py` | Runs a command, samples NVML per-process GPU memory of its process tree, writes wall time and peak to JSON |
| `blender_cube.py` | BlenderProc: cube + ground, RGB / depth / segmentation with Cycles on CUDA; fails if Cycles isn't on CUDA |
| `cosmos_simtransfer_153.sh` | Cosmos3-Nano-Sim-Transfer, example2 depth profile (full history, no sink) with the frame window set to N (default 153). The 153-frame control is example2's depth extended by a 32-frame ping-pong |
| `eval_tools.py` | SAM 2.1-L video masks from a frame-0 prompt + CoTracker3 offline tracks, with overlays and timing |
| `wanmove_example.sh` | Wan-Move-14B on its shipped example (one track) |
| `vace_common.sh`, `vace_launch.py` | Shared VACE settings; `VACE_DIT_BF16=1` casts the 14B DiT to bf16 after loading (its base weights are stored in fp32 and OOM an 80 GB A100) |
| `vace_txt2vid.sh`, `vace_firstframe.sh`, `vace_ffdepth.sh` | VACE cases: text-only, first frame (`frameref firstframe`), first frame + depth control (composition) |
| `vace_prep_inputs.{sh,py}` | Builds the VACE inputs from VACE's `assets/videos/test.mp4` |
