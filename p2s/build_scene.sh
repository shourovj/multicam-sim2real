#!/usr/bin/env bash
# From a prompt file to the camera bundles that gen/ and eval/ read: write and simulate the scene (PhysX), then render
# and export every camera (Blender).
#   source ~/.sim2real.env; bash p2s/build_scene.sh p2s/prompts/<name>.yaml [gpu,gpu,...] [camera,camera,...]
# <root> = $S2R_DATA/cv-project/p2s/<name>/r<rev>/
#   <root>/agent/              the SceneAgent's iterations                    (p2s/1_scene/write_scene.py)
#   <root>/world/              the accepted scene S, physics output X, report, spec.yaml, world.json
#   <root>/cams/<cam>/bundle/  rgb/depth/instance, timeline, boxes, visibility, tracks, overlay, controls/
# Cameras are rendered in parallel, one per listed GPU (default: GPU 0 only, one after another). Check nvidia-smi
# first: each render needs a few GB and shares the card with whatever else runs there.
# A camera whose bundle was rendered from the accepted world is skipped (P2S_FORCE_RENDER=1 renders it again).
# Extra arguments for write_scene.py (--force, --task-only, --agent-cmd "...") go in P2S_WRITE_ARGS.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROMPT=$1; GPUS=${2:-0}; CAMS=${3:-}
source "$REPO/env.sh"
export CUDA_CACHE_MAXSIZE=4294967296
PY=$S2R_ROOT/envs/eval/.venv/bin/python
IFS=, read -ra G <<< "$GPUS"
# shellcheck disable=SC2086
"$PY" "$REPO/p2s/1_scene/write_scene.py" "$PROMPT" --gpu "${G[0]}" ${P2S_WRITE_ARGS:-}
OUT=$("$PY" "$REPO/p2s/1_scene/write_scene.py" "$PROMPT" --print-root)
[[ -f $OUT/world/spec.yaml ]] || { echo "[build] no accepted world in $OUT (see the message above)"; exit 1; }
mkdir -p "$OUT/logs"
[[ -z $CAMS ]] && CAMS=$("$PY" -c "import json,sys; print(','.join(c['id'] for c in json.load(open(sys.argv[1]))['cameras']))" "$OUT/world/world.json")
IFS=, read -ra C <<< "$CAMS"
X_SHA=$(sha256sum "$OUT/world/x.json" | cut -d' ' -f1)
render_one() {   # camera gpu
  local b=$OUT/cams/$1/bundle
  if [[ -z ${P2S_FORCE_RENDER:-} && -f $b/tracks.npz && -f $b/controls/depth.mp4 ]] && grep -q "$X_SHA" "$b/meta.json" 2>/dev/null; then
    echo "[build] $1: bundle is current (rendered from this world); P2S_FORCE_RENDER=1 renders it again"
    return 0
  fi
  mkdir -p "$b"
  CUDA_VISIBLE_DEVICES=$2 blenderproc run --blender-install-path "$S2R_ROOT/blender" "$REPO/p2s/2_render/render_x.py" \
      "$OUT/world" "$1" "$b" > "$OUT/logs/render_$1.log" 2>&1
  "$PY" "$REPO/p2s/2_render/export_x.py" "$OUT/world" "$b" > "$OUT/logs/export_$1.log" 2>&1
  rm -rf "$b/cache"
  echo "[build] $1 done on GPU $2: $(grep -o '"render_s": [0-9.]*' "$b/meta.json")"
}
pids=()
for k in "${!C[@]}"; do
  g=${G[$((k % ${#G[@]}))]}
  if (( ${#G[@]} > 1 )); then render_one "${C[$k]}" "$g" & pids+=($!); else render_one "${C[$k]}" "$g"; fi
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "[build] a camera failed: see $OUT/logs/"; exit 1; }
echo "[build] $OUT"
