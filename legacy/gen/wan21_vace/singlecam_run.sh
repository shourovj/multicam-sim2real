#!/usr/bin/env bash
# VACE runner for the PoC (plan §6): conditions A (text), Aplus (detailed text, "A+"),
# B (text + frame 0), C (frame 0 + depth), C0 (depth, no frame 0), Bbox (frame 0 + box layout).
# Reuses legacy/smoke/vace_common.sh. Output: $S2R_DATA/poc/single-cam/<scene>/<cond>/14B-bf16/out_video.mp4 (+ run.log).
#   SCENE=drones_air COND=A SEED=0 bash legacy/gen/wan21_vace/singlecam_run.sh
# B and C read their src_video/src_mask from $S2R_DATA/poc/single-cam/<scene>/<cond>/inputs/ (legacy/sim/controls.py).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCENE=${SCENE:?}; COND=${COND:?}
source "$REPO/env.sh"
export SEED=${SEED:-0} VACE_DIT_BF16=${VACE_DIT_BF16:-1} MODEL=${MODEL:-14B}
export OUT_ROOT=$S2R_DATA/poc/single-cam/$SCENE
KEY=base; [[ $COND == Aplus ]] && KEY=detailed
PROMPT=$("$S2R_ROOT/VACE/.venv/bin/python" -c \
  "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$REPO/configs/prompts/$SCENE.yaml" "$KEY")
export PROMPT
source "$REPO/smoke/vace_common.sh"
IN=$OUT_ROOT/$COND/inputs
case "$COND" in
  A|Aplus) run_vace "$COND" ;;
  *) for f in "$IN/src_video.mp4" "$IN/src_mask.mp4"; do [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }; done
       run_vace "$COND" --src_video "$IN/src_video.mp4" --src_mask "$IN/src_mask.mp4" ;;
esac
