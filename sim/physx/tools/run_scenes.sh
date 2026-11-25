#!/usr/bin/env bash
# Simulate and check every example scene from scratch: tools/run_scenes.sh <out_root> [scene ...]
# Each scene goes to <out_root>/<scene>/it1 (simulation output and report). GPU: CUDA_VISIBLE_DEVICES as set.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?usage: run_scenes.sh <out_root> [scene ...]}
shift
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  for f in "$here"/scenes/*.json; do scenes+=("$(basename "$f" .json)"); done
fi
for s in "${scenes[@]}"; do
  d=$out/$s/it1
  mkdir -p "$d"
  cp "$here/scenes/$s.json" "$d/scene.json"
  start=$(date +%s)
  "$here/build/p2s_sim" "$d/scene.json" "$d" > "$d/sim.log" 2>&1
  sim=$?
  "$here/build/p2s_check" "$d/scene.json" "$d" > "$d/check.log" 2>&1
  echo "$s sim=$sim $(cat "$d/check.log" | tail -1) ($(( $(date +%s) - start )) s)"
done
