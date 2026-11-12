#!/usr/bin/env bash
# Cosmos3-Nano (base) runner: the same scenes, conditions, prompts and seeds as gen/run_vacefun.sh.
#   SCENE=drones_air_phys COND=Bbox_edit SEED=0 bash gen/run_cosmos_nano.sh
# gen/cosmos_nano_request.py maps each condition to a Nano mode (text, first frame, or transfer with a depth / edge control).
# Output: $S2R_DATA/poc/<scene>/<cond>/cosmos3-nano/seed<k>/out_video.mp4 (832x480, 81 frames, 16 fps), request.json, run.log.
# Nano's sampler defaults are kept, with the step-skipping diffusion cache off and guardrails off (the guardrail model is gated).
# COSMOS_HF_CACHE holds nvidia/Cosmos3-Nano (35 GB, public); COSMOS_OVERLAY holds readable copies of four packages
# (pyyaml, markdown, pluggy, iniconfig) that are unreadable in the shared cosmos-framework env.
# The GPU comes from the caller's environment; nothing here touches CUDA_VISIBLE_DEVICES.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENE=${SCENE:?}; COND=${COND:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
OUT=$S2R_DATA/poc/$SCENE/$COND/cosmos3-nano/seed$SEED
mkdir -p "$OUT"
"$S2R_ROOT/envs/eval/.venv/bin/python" "$REPO/gen/cosmos_nano_request.py" "$SCENE" "$COND" "$SEED" "$OUT"

export HF_HUB_CACHE=${COSMOS_HF_CACHE:-/mnt/Data/sj37434/hf/hub} UV_CACHE_DIR=$S2R_CACHE/uv XDG_CACHE_HOME=$S2R_CACHE
export PYTHONPATH=${COSMOS_OVERLAY:-/mnt/Data/sj37434/sim2real-local/cosmos-overlay}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
unset LD_LIBRARY_PATH HF_HUB_OFFLINE
cd "$S2R_ROOT/cosmos-framework"
echo "[cosmos3-nano] scene=$SCENE cond=$COND seed=$SEED CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
.venv/bin/python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
  -i "$OUT/request.json" -o "$OUT/raw" --checkpoint-path Cosmos3-Nano --no-guardrails --no-diffusion-cache 2>&1 | tee "$OUT/run.log"
cp "$OUT/raw/out/vision.mp4" "$OUT/out_video.mp4"
