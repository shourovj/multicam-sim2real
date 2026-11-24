#!/usr/bin/env bash
# Wan-Move-14B runner for the PoC (plan §6), condition E: frame 0 + tracks + text.
# Tracks/visibility come from sim/controls.py ($S2R_DATA/poc/single-cam/<scene>/E/inputs/).
#   SCENE=drones_air SEED=0 bash gen/wan_move/singlecam_run.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCENE=${SCENE:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
D=$S2R_DATA/poc/single-cam/$SCENE
IN=$D/E/inputs
PROMPT=$("$S2R_ROOT/VACE/.venv/bin/python" -c \
  "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['base'])" "$REPO/configs/prompts/$SCENE.yaml")
cd "$S2R_ROOT/Wan-Move"
.venv/bin/python generate.py \
  --task wan-move-i2v --size 480*832 --ckpt_dir ./Wan-Move-14B-480P \
  --image "$D/bundle/frame0.png" --track "$IN/tracks.npy" --track_visibility "$IN/visibility.npy" \
  --prompt "$PROMPT" --frame_num 81 --base_seed "$SEED" \
  --offload_model True --t5_cpu --dtype bf16 \
  --save_file "$D/E/seed${SEED}.mp4" 2>&1 | tee "$D/E/run.log"
