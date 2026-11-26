#!/usr/bin/env bash
# Render a scene spec and export the full scene bundle (plan §4).
#   bash legacy/sim/build_bundle.sh legacy/configs/scenes/<scene>.yaml <bundle_dir>
# Step 1 (render) needs Blender; step 2 (export)
# uses the eval env (numpy + OpenCV).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPEC=$1; OUT=$2
source "$REPO/env.sh"
export CUDA_CACHE_MAXSIZE=4294967296   # keep the sm_80 JIT kernel cache (gotchas §3.1)
blenderproc run --blender-install-path "$S2R_ROOT/blender" "$REPO/sim/render_scene.py" "$SPEC" "$OUT"
"$S2R_ROOT/envs/eval/.venv/bin/python" "$REPO/sim/export.py" "$OUT"
