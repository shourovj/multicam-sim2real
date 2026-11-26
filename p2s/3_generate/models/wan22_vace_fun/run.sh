#!/usr/bin/env bash
# One multicam run on Wan2.2-VACE-Fun-A14B (VideoX-Fun): build its inputs (p2s/3_generate/common/multicam_inputs.py, eval env),
# then generate (p2s/3_generate/models/wan22_vace_fun/launch.py, VideoX-Fun's env).
#   EXP=configs/experiments/cvp_v1_wan22_vace_fun.yaml SCENE=intersection_4way VIEW=cctv COND=D SEED=0 bash p2s/3_generate/models/wan22_vace_fun/run.sh
# The model gets the prompt paragraph and, for a condition with a control, that simulator video as the VACE control on
# every frame; there is no first frame. Length, rate, size and sampler come from the experiment file.
# VIDEOX_FUN_DIR is the VideoX-Fun clone (with its .venv); the weights are linked under its models/ folder (setup/README.md).
# MEMORY_MODE (default model_cpu_offload) is VideoX-Fun's GPU memory mode: one of the two transformers on the GPU at a time.
# The GPU comes from the caller (CUDA_VISIBLE_DEVICES); p2s/3_generate/run_multicam.py picks it.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXP=${EXP:?}; SCENE=${SCENE:?}; VIEW=${VIEW:?}; COND=${COND:?}; SEED=${SEED:-0}
source "$REPO/env.sh"
VIDEOX_FUN_DIR=${VIDEOX_FUN_DIR:-$S2R_ROOT/VideoX-Fun}
CKPT_DIR=${VACEFUN_CKPT:-$VIDEOX_FUN_DIR/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B}
[[ -d $CKPT_DIR/high_noise_model ]] || { echo "missing $CKPT_DIR (setup/README.md: Wan2.2-VACE-Fun weights)"; exit 1; }
# Keep every cache except the HF one out of ~/.cache.
export XDG_CACHE_HOME=$S2R_CACHE TORCH_HOME=$S2R_CACHE/torch MPLCONFIGDIR=$S2R_CACHE/matplotlib TOKENIZERS_PARALLELISM=false
OUT=$("$S2R_ROOT/envs/eval/.venv/bin/python" "$REPO/p2s/3_generate/common/multicam_inputs.py" "$REPO/$EXP" "$SCENE" "$VIEW" "$COND" "$SEED" | sed -n 's/.*-> //p')
echo "[wan2.2-vace-fun multicam] $SCENE $VIEW $COND seed $SEED -> $OUT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>})"
"$VIDEOX_FUN_DIR/.venv/bin/python" "$REPO/p2s/3_generate/models/wan22_vace_fun/launch.py" --videox_dir "$VIDEOX_FUN_DIR" --ckpt_dir "$CKPT_DIR" \
  --request "$OUT/request.json" --memory_mode "${MEMORY_MODE:-model_cpu_offload}" > "$OUT/run.log" 2>&1
echo "[wan2.2-vace-fun multicam] done: $OUT/out_video.mp4"
