#!/usr/bin/env bash
# One SceneAgent iteration: simulate <run_dir>/it<N>/scene.json on PhysX and check it.
#   p2s/1_scene/iterate.sh <run_dir> <N>
# Writes into <run_dir>/it<N>/: x.json + streams/ (the physics output X), r.md + r.json (the report), sim.log.
# The history of earlier iterations is <run_dir>/history.jsonl. Exit code: 0 PASS, 1 FAIL, 2 bad input, 3 STOP.
# GPU: CUDA_VISIBLE_DEVICES if set, else the GPU with the least memory in use.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B=$REPO/sim/physx/build
[[ $# -eq 2 ]] || { echo "usage: p2s/1_scene/iterate.sh <run_dir> <N>"; exit 2; }
D=$1/it$2
[[ -f $D/scene.json ]] || { echo "missing $D/scene.json"; exit 2; }
[[ -x $B/p2s_sim && -x $B/p2s_check ]] || { echo "missing $B/p2s_sim: build it first (sim/physx/README.md)"; exit 2; }
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
  export CUDA_VISIBLE_DEVICES
fi
rm -rf "$D/x.json" "$D/streams" "$D/r.md" "$D/r.json"      # never leave an older run's output next to a new scene
"$B/p2s_sim" "$D/scene.json" "$D" > "$D/sim.log" 2>&1
s=$?
[[ $s -eq 0 ]] || grep -E "ERROR" "$D/sim.log" | tail -5
"$B/p2s_check" "$D/scene.json" "$D" --history "$1/history.jsonl"
c=$?
echo "[iterate] it$2: p2s_sim exit $s, p2s_check exit $c, report $D/r.md"
exit $c
