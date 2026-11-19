#!/usr/bin/env bash
# Simulate a scene spec v2 once, then render and export every camera (sim/world).
#   bash sim/build_world.sh configs/scenes/<scene>.yaml <out_root> [gpu,gpu,...] [camera,camera,...]
# <out_root>/world/            world.npz, world.json      (sim/simulate_world.py, numpy only)
# <out_root>/cams/<cam>/bundle rgb/depth/instance, timeline, boxes, visibility, tracks, overlay, checks
# Cameras are rendered in parallel, one per listed GPU (default: GPU 0 only, one after another).
# Check nvidia-smi first: each render needs a few GB and shares the card with whatever else runs there.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPEC=$1; OUT=$2; GPUS=${3:-0}; CAMS=${4:-}
source "$REPO/env.sh"
export CUDA_CACHE_MAXSIZE=4294967296
PY=$S2R_ROOT/envs/eval/.venv/bin/python
mkdir -p "$OUT/logs"
"$PY" "$REPO/sim/simulate_world.py" "$SPEC" "$OUT/world" | tee "$OUT/logs/simulate.log"
[[ -z $CAMS ]] && CAMS=$("$PY" -c "import json,sys; print(','.join(c['id'] for c in json.load(open(sys.argv[1]))['cameras']))" "$OUT/world/world.json")
IFS=, read -ra G <<< "$GPUS"; IFS=, read -ra C <<< "$CAMS"
render_one() {   # camera gpu
  local b=$OUT/cams/$1/bundle
  mkdir -p "$b"
  CUDA_VISIBLE_DEVICES=$2 blenderproc run --blender-install-path "$S2R_ROOT/blender" "$REPO/sim/render_world.py" \
      "$SPEC" "$OUT/world" "$1" "$b" > "$OUT/logs/render_$1.log" 2>&1
  "$PY" "$REPO/sim/export_world.py" "$SPEC" "$OUT/world" "$b" > "$OUT/logs/export_$1.log" 2>&1
  echo "[build_world] $1 done on GPU $2: $(grep -o '"render_s": [0-9.]*' "$b/meta.json")"
}
pids=()
for k in "${!C[@]}"; do
  g=${G[$((k % ${#G[@]}))]}
  if (( ${#G[@]} > 1 )); then render_one "${C[$k]}" "$g" & pids+=($!); else render_one "${C[$k]}" "$g"; fi
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "visible check" "$OUT"/logs/export_*.log || true
