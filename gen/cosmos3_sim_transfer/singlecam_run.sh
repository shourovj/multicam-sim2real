#!/usr/bin/env bash
# Cosmos3-Nano-Sim-Transfer runner for the PoC (plan §6), condition D: depth control + text, 153 frames.
# Reuses smoke/cosmos_simtransfer_153.sh with the scene's control and frozen base prompt.
#   SCENE=drones_air SEED=0 bash gen/cosmos3_sim_transfer/singlecam_run.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCENE=${SCENE:?}
source "$REPO/env.sh"
D=$S2R_DATA/poc/single-cam/$SCENE
export SEED=${SEED:-0} NAME=D_seed${SEED:-0}
export CONTROL=$D/bundle/controls/cosmos_depth.mp4 OUT=$D/D
export PROMPT=$("$S2R_ROOT/VACE/.venv/bin/python" -c \
  "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['base'])" "$REPO/configs/prompts/$SCENE.yaml")
[[ -f $CONTROL ]] || { echo "missing $CONTROL; run sim/controls.py" >&2; exit 1; }
bash "$REPO/smoke/cosmos_simtransfer_153.sh" 153
