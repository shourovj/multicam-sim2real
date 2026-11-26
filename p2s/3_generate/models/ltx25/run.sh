#!/usr/bin/env bash
# One multicam run on LTX-2.5 (distilled): build its inputs (p2s/3_generate/common/multicam_inputs.py, eval env), then generate
# (p2s/3_generate/models/ltx25/launch.py, the LTX-2 repo's env).
#   EXP=configs/experiments/cvp_v1_ltx25.yaml SCENE=intersection_4way VIEW=cctv COND=D SEED=0 bash p2s/3_generate/models/ltx25/run.sh
# The model gets the prompt paragraph and, for a condition with a control, that simulator video through the Union
# Control IC-LoRA; there is no first frame. Length, rate, size and weight files come from the experiment file.
# LTX_DIR is the LTX-2 clone (with its .venv). The weights are read from the Hugging Face cache, offline, at the
# revisions the experiment file pins (setup/README.md). LTX_OFFLOAD=cpu streams the transformer if the GPU is short.
# The GPU comes from the caller (CUDA_VISIBLE_DEVICES); p2s/3_generate/run_multicam.py picks it.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXP=${EXP:?}; SCENE=${SCENE:?}; VIEW=${VIEW:?}; COND=${COND:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
LTX_DIR=${LTX_DIR:-$S2R_ROOT/LTX-2}
HUB=${HF_HUB_CACHE:?set HF_HUB_CACHE to the Hugging Face cache that holds the LTX-2.5 weights}
[[ -x $LTX_DIR/.venv/bin/python ]] || { echo "missing $LTX_DIR/.venv (setup/README.md: LTX-2.5)"; exit 1; }
export HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$S2R_CACHE TORCH_HOME=$S2R_CACHE/torch TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LD_LIBRARY_PATH PYTHONPATH
OUT=$("$S2R_ROOT/envs/eval/.venv/bin/python" "$REPO/p2s/3_generate/common/multicam_inputs.py" "$REPO/$EXP" "$SCENE" "$VIEW" "$COND" "$SEED" | sed -n 's/.*-> //p')
echo "[ltx-2.5 multicam] $SCENE $VIEW $COND seed $SEED -> $OUT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>})"
"$LTX_DIR/.venv/bin/python" "$REPO/p2s/3_generate/models/ltx25/launch.py" --request "$OUT/request.json" --hub "$HUB" \
  --offload "${LTX_OFFLOAD:-none}" > "$OUT/run.log" 2>&1
echo "[ltx-2.5 multicam] done: $OUT/out_video.mp4"
