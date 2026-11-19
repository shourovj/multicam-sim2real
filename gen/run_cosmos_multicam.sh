#!/usr/bin/env bash
# One multicam_v1 run on Cosmos3-Nano: build its inputs (gen/multicam_inputs.py), then generate.
#   EXP=configs/experiments/multicam_v1.yaml SCENE=intersection_4way VIEW=cctv COND=V SEED=0 bash gen/run_cosmos_multicam.sh
# Smoke tests use their own experiment file (other out_root, fewer steps), so a short run never lands in the real folders.
# The GPU comes from the caller (CUDA_VISIBLE_DEVICES); gen/run_multicam.py picks it.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP=${EXP:?}; SCENE=${SCENE:?}; VIEW=${VIEW:?}; COND=${COND:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
export HF_HUB_CACHE=${COSMOS_HF_CACHE:-/mnt/Data/sj37434/hf/hub} UV_CACHE_DIR=$S2R_CACHE/uv XDG_CACHE_HOME=$S2R_CACHE
if [[ -n ${COSMOS_OVERLAY:-} ]]; then export PYTHONPATH=$COSMOS_OVERLAY; else unset PYTHONPATH; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
unset LD_LIBRARY_PATH HF_HUB_OFFLINE
cd "$S2R_ROOT/cosmos-framework"
OUT=$(.venv/bin/python "$REPO/gen/multicam_inputs.py" "$REPO/$EXP" "$SCENE" "$VIEW" "$COND" "$SEED" | sed -n 's/.*-> //p')
echo "[cosmos3-nano multicam] $SCENE $VIEW $COND seed $SEED -> $OUT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>})"
.venv/bin/python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
  -i "$OUT/request.json" -o "$OUT/raw" --checkpoint-path Cosmos3-Nano --no-guardrails --no-diffusion-cache > "$OUT/run.log" 2>&1
cp "$OUT/raw/out/vision.mp4" "$OUT/out_video.mp4"
echo "[cosmos3-nano multicam] done: $OUT/out_video.mp4"
